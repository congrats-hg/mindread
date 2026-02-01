"""PyTorch dataset for DSTC2 dialogue state tracking."""

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from .dstc2 import load_split
from .schema import INFORMABLE_SLOTS, SLOT_VALUES, SPECIAL_VALUES, Dialogue, Split


@dataclass
class DSTExample:
    """A single training/evaluation example for DST."""

    dialogue_id: str
    turn_index: int
    context: str  # Concatenated dialogue history
    current_utterance: str  # Current user utterance
    labels: dict[str, int]  # Slot -> label index mapping


def build_slot_vocab() -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    """
    Build vocabulary for each slot.

    Returns:
        Tuple of (slot_values, slot_to_idx) where:
        - slot_values: {slot_name: [value1, value2, ...]}
        - slot_to_idx: {slot_name: {value: index}}
    """
    slot_values: dict[str, list[str]] = {}
    slot_to_idx: dict[str, dict[str, int]] = {}

    for slot in INFORMABLE_SLOTS:
        # Start with "none" for no value mentioned
        values = ["none"] + SPECIAL_VALUES + sorted(SLOT_VALUES.get(slot, []))
        slot_values[slot] = values
        slot_to_idx[slot] = {v: i for i, v in enumerate(values)}

    return slot_values, slot_to_idx


# Global vocabulary (built once)
SLOT_VOCAB, SLOT_TO_IDX = build_slot_vocab()


def get_num_labels(slot: str) -> int:
    """Get number of possible labels for a slot."""
    return len(SLOT_VOCAB[slot])


def dialogue_to_examples(dialogue: Dialogue) -> list[DSTExample]:
    """Convert a dialogue to a list of training examples (one per turn)."""
    examples = []
    context_parts = []

    for turn in dialogue.turns:
        # Build context from previous turns
        if turn.turn_index > 0:
            # Add previous system response to context
            prev_turn = dialogue.turns[turn.turn_index - 1]
            if prev_turn.system_utterance:
                context_parts.append(f"[SYS] {prev_turn.system_utterance}")
            if prev_turn.user_utterance:
                context_parts.append(f"[USR] {prev_turn.user_utterance}")

        context = " ".join(context_parts[-10:])  # Keep last 10 context parts

        # Current system utterance + user utterance
        current = ""
        if turn.system_utterance:
            current += f"[SYS] {turn.system_utterance} "
        current += f"[USR] {turn.user_utterance}"

        # Convert belief state to label indices
        labels = {}
        for slot in INFORMABLE_SLOTS:
            value = turn.belief_state.get(slot, "none")
            if value not in SLOT_TO_IDX[slot]:
                # Unknown value, map to "none"
                value = "none"
            labels[slot] = SLOT_TO_IDX[slot][value]

        example = DSTExample(
            dialogue_id=dialogue.dialogue_id,
            turn_index=turn.turn_index,
            context=context,
            current_utterance=current.strip(),
            labels=labels,
        )
        examples.append(example)

    return examples


class DSTC2Dataset(Dataset):
    """PyTorch Dataset for DSTC2 dialogue state tracking."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        split: Split = "train",
        max_length: int = 512,
        max_context_turns: int = 5,
        data_dir: str | None = None,
    ):
        """
        Initialize DSTC2 dataset.

        Args:
            tokenizer: HuggingFace tokenizer for encoding text.
            split: Dataset split ("train", "dev", or "test").
            max_length: Maximum sequence length for tokenization.
            max_context_turns: Maximum number of previous turns to include.
            data_dir: Path to data directory.
        """
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self.max_context_turns = max_context_turns

        # Load dialogues and convert to examples
        from pathlib import Path

        data_path = Path(data_dir) if data_dir else None
        dialogues = load_split(data_path, split)

        self.examples: list[DSTExample] = []
        for dialogue in dialogues:
            self.examples.extend(dialogue_to_examples(dialogue))

        self.slot_vocab = SLOT_VOCAB
        self.slot_to_idx = SLOT_TO_IDX

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]

        # Combine context and current utterance
        if example.context:
            text = f"{example.context} [SEP] {example.current_utterance}"
        else:
            text = example.current_utterance

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Prepare output
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "dialogue_id": example.dialogue_id,
            "turn_index": example.turn_index,
        }

        # Add token type ids if available
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        # Add labels for each slot
        for slot in INFORMABLE_SLOTS:
            item[f"label_{slot}"] = torch.tensor(example.labels[slot])

        return item

    def get_slot_sizes(self) -> dict[str, int]:
        """Get the number of possible values for each slot."""
        return {slot: len(values) for slot, values in self.slot_vocab.items()}


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for DataLoader."""
    result = {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "dialogue_ids": [x["dialogue_id"] for x in batch],
        "turn_indices": [x["turn_index"] for x in batch],
    }

    # Add token type ids if present
    if "token_type_ids" in batch[0]:
        result["token_type_ids"] = torch.stack([x["token_type_ids"] for x in batch])

    # Stack labels for each slot
    for slot in INFORMABLE_SLOTS:
        key = f"label_{slot}"
        result[key] = torch.stack([x[key] for x in batch])

    return result
