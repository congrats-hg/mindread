"""DSTC2 dataset loading and parsing."""

import json
import logging
from pathlib import Path
from typing import Iterator

from .schema import (
    INFORMABLE_SLOTS,
    Dialogue,
    DialogueAct,
    DialogueStateLabel,
    Split,
    Turn,
)

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


def parse_dialogue_act(act_dict: dict) -> DialogueAct:
    """Parse a dialogue act from DSTC2 JSON format."""
    act_type = act_dict.get("act", "null")
    slots = []

    for slot_info in act_dict.get("slots", []):
        if len(slot_info) == 2:
            slot_name, slot_value = slot_info
            slots.append((slot_name, slot_value))
        elif len(slot_info) == 1:
            # Some acts have single-element slots (e.g., request)
            slots.append((slot_info[0], None))

    return DialogueAct(act_type=act_type, slots=slots)


def extract_belief_state(turn_label: dict) -> dict[str, str]:
    """Extract belief state from turn label."""
    belief_state = {}

    # Goal labels contain the cumulative belief state
    goal_labels = turn_label.get("goal-labels", {})

    for slot in INFORMABLE_SLOTS:
        if slot in goal_labels:
            # Get the highest probability value
            slot_values = goal_labels[slot]
            if slot_values:
                # Values are stored as {value: probability}
                best_value = max(slot_values.items(), key=lambda x: x[1])[0]
                belief_state[slot] = best_value

    return belief_state


def load_dialogue(log_path: Path, label_path: Path) -> Dialogue | None:
    """Load a single dialogue from log and label files."""
    try:
        with open(log_path) as f:
            log_data = json.load(f)
        with open(label_path) as f:
            label_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning(f"Failed to load dialogue {log_path}: {e}")
        return None

    dialogue_id = log_path.parent.name
    turns = []

    log_turns = log_data.get("turns", [])
    label_turns = label_data.get("turns", [])

    # Extract goal from label data
    goal = {}
    task_goal = label_data.get("task-information", {}).get("goal", {})
    constraints = task_goal.get("constraints", [])
    for constraint in constraints:
        if len(constraint) >= 2:
            slot, value = constraint[0], constraint[1]
            if slot in INFORMABLE_SLOTS:
                goal[slot] = value

    for i, (log_turn, label_turn) in enumerate(zip(log_turns, label_turns)):
        # Parse system output (comes before user input in DSTC2)
        system_output = log_turn.get("output", {})
        system_utterance = system_output.get("transcript", "")
        system_acts = [
            parse_dialogue_act(act) for act in system_output.get("dialog-acts", [])
        ]

        # Parse user input
        user_input = log_turn.get("input", {})

        # Get ASR hypothesis (use top one)
        asr_hyps = user_input.get("live", {}).get("asr-hyps", [])
        if asr_hyps:
            user_utterance = asr_hyps[0].get("asr-hyp", "")
        else:
            user_utterance = ""

        # Parse user semantic acts from label
        user_semantics = label_turn.get("semantics", {}).get("json", [])
        user_acts = [parse_dialogue_act(act) for act in user_semantics]

        # Extract cumulative belief state
        belief_state = extract_belief_state(label_turn)

        turn = Turn(
            turn_index=i,
            user_utterance=user_utterance,
            system_utterance=system_utterance,
            user_acts=user_acts,
            system_acts=system_acts,
            belief_state=belief_state,
        )
        turns.append(turn)

    return Dialogue(dialogue_id=dialogue_id, turns=turns, goal=goal)


def iter_dialogue_dirs(data_dir: Path, split: Split) -> Iterator[Path]:
    """Iterate over dialogue directories for a given split."""
    if split in ("train", "dev"):
        # Train/dev are in dstc2_traindev
        base_dir = data_dir / "raw" / "dstc2_traindev" / "data"
    else:
        # Test is in dstc2_test
        base_dir = data_dir / "raw" / "dstc2_test" / "data"

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {base_dir}. Run `make data` to download."
        )

    # Each session has multiple dialogues
    for session_dir in sorted(base_dir.iterdir()):
        if not session_dir.is_dir():
            continue

        # Determine if this session belongs to train or dev
        # DSTC2 uses voip-* sessions for dev
        is_dev = session_dir.name.startswith("voip-")

        if split == "train" and is_dev:
            continue
        if split == "dev" and not is_dev:
            continue

        # Iterate dialogue subdirectories
        for dialogue_dir in sorted(session_dir.iterdir()):
            if dialogue_dir.is_dir():
                yield dialogue_dir


def load_split(data_dir: Path | None = None, split: Split = "train") -> list[Dialogue]:
    """
    Load all dialogues for a given split.

    Args:
        data_dir: Path to data directory. Defaults to project data/.
        split: One of "train", "dev", or "test".

    Returns:
        List of Dialogue objects.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    dialogues = []

    for dialogue_dir in iter_dialogue_dirs(data_dir, split):
        log_path = dialogue_dir / "log.json"
        label_path = dialogue_dir / "label.json"

        if log_path.exists() and label_path.exists():
            dialogue = load_dialogue(log_path, label_path)
            if dialogue is not None:
                dialogues.append(dialogue)

    logger.info(f"Loaded {len(dialogues)} dialogues for {split} split")
    return dialogues


def get_all_slot_values(dialogues: list[Dialogue]) -> dict[str, set[str]]:
    """Extract all unique slot values from dialogues."""
    values: dict[str, set[str]] = {slot: set() for slot in INFORMABLE_SLOTS}

    for dialogue in dialogues:
        for turn in dialogue.turns:
            for slot, value in turn.belief_state.items():
                if slot in values:
                    values[slot].add(value)

    return values
