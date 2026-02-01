"""BERT-based Dialogue State Tracking model."""

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel

from ..data.schema import INFORMABLE_SLOTS
from ..data.dataset import get_num_labels


class SlotClassifier(nn.Module):
    """Classification head for a single slot."""

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [batch_size, hidden_size]

        Returns:
            logits: [batch_size, num_labels]
        """
        hidden_state = self.dropout(hidden_state)
        return self.classifier(hidden_state)


class BertDST(nn.Module):
    """
    BERT-based Dialogue State Tracking model.

    This model uses a pre-trained BERT encoder and adds separate classification
    heads for each informable slot (food, area, pricerange).
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        dropout: float = 0.1,
        freeze_encoder_layers: int = 0,
    ):
        """
        Initialize BertDST model.

        Args:
            model_name: Name of the pre-trained model to use.
            dropout: Dropout probability for classification heads.
            freeze_encoder_layers: Number of encoder layers to freeze (0 = none).
        """
        super().__init__()

        self.encoder: PreTrainedModel = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Freeze encoder layers if specified
        if freeze_encoder_layers > 0:
            self._freeze_layers(freeze_encoder_layers)

        # Create a classification head for each slot
        self.slot_classifiers = nn.ModuleDict()
        for slot in INFORMABLE_SLOTS:
            num_labels = get_num_labels(slot)
            self.slot_classifiers[slot] = SlotClassifier(
                hidden_size=hidden_size,
                num_labels=num_labels,
                dropout=dropout,
            )

        self.slot_names = INFORMABLE_SLOTS

    def _freeze_layers(self, num_layers: int) -> None:
        """Freeze the first N encoder layers."""
        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

        # Freeze specified number of encoder layers
        if hasattr(self.encoder, "encoder"):
            for i, layer in enumerate(self.encoder.encoder.layer):
                if i < num_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len] (optional)

        Returns:
            Dictionary mapping slot names to logits [batch_size, num_labels]
        """
        # Encode input
        encoder_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)

        # Use [CLS] token representation
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Get logits for each slot
        slot_logits = {}
        for slot in self.slot_names:
            slot_logits[slot] = self.slot_classifiers[slot](cls_hidden)

        return slot_logits

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Get predicted slot values.

        Returns:
            Dictionary mapping slot names to predicted label indices.
        """
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        predictions = {slot: logits[slot].argmax(dim=-1) for slot in self.slot_names}
        return predictions


class HierarchicalBertDST(nn.Module):
    """
    Hierarchical BERT-based DST model.

    This model encodes system and user utterances separately, then combines
    them with a context-level attention mechanism.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        dropout: float = 0.1,
        num_context_layers: int = 2,
    ):
        """
        Initialize HierarchicalBertDST model.

        Args:
            model_name: Name of the pre-trained model to use.
            dropout: Dropout probability.
            num_context_layers: Number of transformer layers for context encoding.
        """
        super().__init__()

        self.encoder: PreTrainedModel = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Context-level transformer for combining turn representations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_context_layers,
        )

        # Slot classifiers
        self.slot_classifiers = nn.ModuleDict()
        for slot in INFORMABLE_SLOTS:
            num_labels = get_num_labels(slot)
            self.slot_classifiers[slot] = SlotClassifier(
                hidden_size=hidden_size,
                num_labels=num_labels,
                dropout=dropout,
            )

        self.slot_names = INFORMABLE_SLOTS

    def encode_utterance(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a single utterance."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        For simplicity, this implementation treats the input as a single sequence.
        A full hierarchical implementation would encode turns separately.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len] (optional)

        Returns:
            Dictionary mapping slot names to logits.
        """
        # Encode full context
        encoder_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden]

        # Apply context-level encoding
        context_output = self.context_encoder(hidden_states)

        # Use [CLS] token from context encoder
        cls_hidden = context_output[:, 0, :]

        # Get logits for each slot
        slot_logits = {}
        for slot in self.slot_names:
            slot_logits[slot] = self.slot_classifiers[slot](cls_hidden)

        return slot_logits

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Get predicted slot values."""
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        predictions = {slot: logits[slot].argmax(dim=-1) for slot in self.slot_names}
        return predictions


def create_model(
    model_type: str = "bert",
    model_name: str = "bert-base-uncased",
    dropout: float = 0.1,
    **kwargs: Any,
) -> nn.Module:
    """
    Factory function to create a DST model.

    Args:
        model_type: Type of model ("bert" or "hierarchical").
        model_name: Name of the pre-trained encoder.
        dropout: Dropout probability.
        **kwargs: Additional model-specific arguments.

    Returns:
        Initialized model.
    """
    if model_type == "bert":
        return BertDST(
            model_name=model_name,
            dropout=dropout,
            freeze_encoder_layers=kwargs.get("freeze_encoder_layers", 0),
        )
    elif model_type == "hierarchical":
        return HierarchicalBertDST(
            model_name=model_name,
            dropout=dropout,
            num_context_layers=kwargs.get("num_context_layers", 2),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
