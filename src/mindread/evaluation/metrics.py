"""Evaluation metrics for Dialogue State Tracking."""

from dataclasses import dataclass, field
from typing import Any

import torch

from ..data.schema import INFORMABLE_SLOTS


@dataclass
class DSTMetrics:
    """Metrics for evaluating DST performance."""

    # Per-slot accuracy
    slot_accuracy: dict[str, float] = field(default_factory=dict)

    # Joint goal accuracy (all slots correct)
    joint_goal_accuracy: float = 0.0

    # Number of samples evaluated
    num_samples: int = 0

    # Per-slot correct counts (for incremental updates)
    _slot_correct: dict[str, int] = field(default_factory=dict)
    _joint_correct: int = 0

    def __post_init__(self) -> None:
        for slot in INFORMABLE_SLOTS:
            if slot not in self.slot_accuracy:
                self.slot_accuracy[slot] = 0.0
            if slot not in self._slot_correct:
                self._slot_correct[slot] = 0

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to a flat dictionary for logging."""
        result = {
            "joint_goal_accuracy": self.joint_goal_accuracy,
            "num_samples": float(self.num_samples),
        }
        for slot, acc in self.slot_accuracy.items():
            result[f"slot_accuracy/{slot}"] = acc
        return result


class DSTEvaluator:
    """Evaluator for Dialogue State Tracking."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all tracked statistics."""
        self.num_samples = 0
        self.slot_correct: dict[str, int] = {slot: 0 for slot in INFORMABLE_SLOTS}
        self.joint_correct = 0

        # For detailed analysis
        self.predictions: list[dict[str, int]] = []
        self.labels: list[dict[str, int]] = []
        self.dialogue_ids: list[str] = []
        self.turn_indices: list[int] = []

    def update(
        self,
        predictions: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        dialogue_ids: list[str] | None = None,
        turn_indices: list[int] | None = None,
    ) -> None:
        """
        Update metrics with a batch of predictions.

        Args:
            predictions: Dict mapping slot names to predicted indices [batch_size]
            labels: Dict mapping slot names to ground truth indices [batch_size]
            dialogue_ids: Optional list of dialogue IDs for error analysis
            turn_indices: Optional list of turn indices for error analysis
        """
        batch_size = next(iter(predictions.values())).shape[0]
        self.num_samples += batch_size

        # Track joint correctness for each sample
        joint_mask = torch.ones(batch_size, dtype=torch.bool, device=next(iter(predictions.values())).device)

        for slot in INFORMABLE_SLOTS:
            pred = predictions[slot]
            label = labels[slot]

            # Per-slot correctness
            correct = (pred == label)
            self.slot_correct[slot] += correct.sum().item()

            # Update joint mask
            joint_mask = joint_mask & correct

        # Joint accuracy
        self.joint_correct += joint_mask.sum().item()

        # Store for detailed analysis
        if dialogue_ids is not None:
            self.dialogue_ids.extend(dialogue_ids)
        if turn_indices is not None:
            self.turn_indices.extend(turn_indices)

        # Store predictions and labels
        for i in range(batch_size):
            pred_dict = {slot: predictions[slot][i].item() for slot in INFORMABLE_SLOTS}
            label_dict = {slot: labels[slot][i].item() for slot in INFORMABLE_SLOTS}
            self.predictions.append(pred_dict)
            self.labels.append(label_dict)

    def compute(self) -> DSTMetrics:
        """Compute final metrics."""
        if self.num_samples == 0:
            return DSTMetrics()

        metrics = DSTMetrics(
            num_samples=self.num_samples,
            joint_goal_accuracy=self.joint_correct / self.num_samples,
            _joint_correct=self.joint_correct,
        )

        for slot in INFORMABLE_SLOTS:
            metrics.slot_accuracy[slot] = self.slot_correct[slot] / self.num_samples
            metrics._slot_correct[slot] = self.slot_correct[slot]

        return metrics

    def get_error_analysis(self) -> list[dict[str, Any]]:
        """
        Get detailed error analysis.

        Returns:
            List of dicts containing error information for each incorrect prediction.
        """
        errors = []

        for i, (pred, label) in enumerate(zip(self.predictions, self.labels)):
            # Check if any slot is wrong
            slot_errors = {}
            for slot in INFORMABLE_SLOTS:
                if pred[slot] != label[slot]:
                    slot_errors[slot] = {
                        "predicted": pred[slot],
                        "actual": label[slot],
                    }

            if slot_errors:
                error_info: dict[str, Any] = {
                    "index": i,
                    "slot_errors": slot_errors,
                }
                if i < len(self.dialogue_ids):
                    error_info["dialogue_id"] = self.dialogue_ids[i]
                if i < len(self.turn_indices):
                    error_info["turn_index"] = self.turn_indices[i]
                errors.append(error_info)

        return errors


def compute_joint_goal_accuracy(
    predictions: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
) -> float:
    """
    Compute joint goal accuracy for a batch.

    Args:
        predictions: Dict mapping slot names to predicted indices [batch_size]
        labels: Dict mapping slot names to ground truth indices [batch_size]

    Returns:
        Joint goal accuracy as a float.
    """
    batch_size = next(iter(predictions.values())).shape[0]
    joint_mask = torch.ones(batch_size, dtype=torch.bool, device=next(iter(predictions.values())).device)

    for slot in INFORMABLE_SLOTS:
        correct = predictions[slot] == labels[slot]
        joint_mask = joint_mask & correct

    return joint_mask.float().mean().item()
