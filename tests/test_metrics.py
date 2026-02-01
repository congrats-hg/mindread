"""Tests for evaluation metrics."""

import pytest
import torch

from mindread.data.schema import INFORMABLE_SLOTS
from mindread.evaluation.metrics import (
    DSTEvaluator,
    DSTMetrics,
    compute_joint_goal_accuracy,
)


class TestDSTMetrics:
    """Tests for DSTMetrics dataclass."""

    def test_default_values(self) -> None:
        metrics = DSTMetrics()
        assert metrics.joint_goal_accuracy == 0.0
        assert metrics.num_samples == 0
        for slot in INFORMABLE_SLOTS:
            assert slot in metrics.slot_accuracy

    def test_to_dict(self) -> None:
        metrics = DSTMetrics(
            joint_goal_accuracy=0.75,
            num_samples=100,
            slot_accuracy={"food": 0.8, "area": 0.9, "pricerange": 0.85},
        )
        d = metrics.to_dict()

        assert d["joint_goal_accuracy"] == 0.75
        assert d["num_samples"] == 100.0
        assert "slot_accuracy/food" in d


class TestDSTEvaluator:
    """Tests for DSTEvaluator class."""

    @pytest.fixture
    def evaluator(self) -> DSTEvaluator:
        return DSTEvaluator()

    def test_reset(self, evaluator: DSTEvaluator) -> None:
        evaluator.num_samples = 100
        evaluator.reset()
        assert evaluator.num_samples == 0

    def test_update_single_batch(self, evaluator: DSTEvaluator) -> None:
        predictions = {
            "food": torch.tensor([0, 1, 2]),
            "area": torch.tensor([0, 0, 0]),
            "pricerange": torch.tensor([1, 1, 1]),
        }
        labels = {
            "food": torch.tensor([0, 1, 2]),  # All correct
            "area": torch.tensor([0, 1, 0]),  # 2/3 correct
            "pricerange": torch.tensor([1, 1, 0]),  # 2/3 correct
        }

        evaluator.update(predictions, labels)

        assert evaluator.num_samples == 3
        assert evaluator.slot_correct["food"] == 3
        assert evaluator.slot_correct["area"] == 2
        assert evaluator.slot_correct["pricerange"] == 2

    def test_compute_metrics(self, evaluator: DSTEvaluator) -> None:
        # All correct batch
        predictions = {
            "food": torch.tensor([0, 1]),
            "area": torch.tensor([0, 1]),
            "pricerange": torch.tensor([0, 1]),
        }
        labels = {
            "food": torch.tensor([0, 1]),
            "area": torch.tensor([0, 1]),
            "pricerange": torch.tensor([0, 1]),
        }

        evaluator.update(predictions, labels)
        metrics = evaluator.compute()

        assert metrics.joint_goal_accuracy == 1.0
        assert metrics.num_samples == 2
        for slot in INFORMABLE_SLOTS:
            assert metrics.slot_accuracy[slot] == 1.0

    def test_joint_accuracy_partial(self, evaluator: DSTEvaluator) -> None:
        # First sample: all correct, Second sample: food wrong
        predictions = {
            "food": torch.tensor([0, 1]),
            "area": torch.tensor([0, 1]),
            "pricerange": torch.tensor([0, 1]),
        }
        labels = {
            "food": torch.tensor([0, 0]),  # Second is wrong
            "area": torch.tensor([0, 1]),
            "pricerange": torch.tensor([0, 1]),
        }

        evaluator.update(predictions, labels)
        metrics = evaluator.compute()

        assert metrics.joint_goal_accuracy == 0.5  # Only first sample fully correct
        assert metrics.slot_accuracy["food"] == 0.5
        assert metrics.slot_accuracy["area"] == 1.0
        assert metrics.slot_accuracy["pricerange"] == 1.0

    def test_get_error_analysis(self, evaluator: DSTEvaluator) -> None:
        predictions = {
            "food": torch.tensor([0, 1]),
            "area": torch.tensor([0, 0]),
            "pricerange": torch.tensor([0, 0]),
        }
        labels = {
            "food": torch.tensor([0, 0]),  # Second is wrong
            "area": torch.tensor([0, 0]),
            "pricerange": torch.tensor([0, 0]),
        }

        evaluator.update(predictions, labels)
        errors = evaluator.get_error_analysis()

        assert len(errors) == 1
        assert errors[0]["index"] == 1
        assert "food" in errors[0]["slot_errors"]


class TestComputeJointGoalAccuracy:
    """Tests for the standalone joint goal accuracy function."""

    def test_all_correct(self) -> None:
        predictions = {
            "food": torch.tensor([0, 1, 2]),
            "area": torch.tensor([0, 1, 2]),
            "pricerange": torch.tensor([0, 1, 2]),
        }
        labels = {
            "food": torch.tensor([0, 1, 2]),
            "area": torch.tensor([0, 1, 2]),
            "pricerange": torch.tensor([0, 1, 2]),
        }

        acc = compute_joint_goal_accuracy(predictions, labels)
        assert acc == 1.0

    def test_all_wrong(self) -> None:
        predictions = {
            "food": torch.tensor([0, 0, 0]),
            "area": torch.tensor([0, 0, 0]),
            "pricerange": torch.tensor([0, 0, 0]),
        }
        labels = {
            "food": torch.tensor([1, 1, 1]),
            "area": torch.tensor([1, 1, 1]),
            "pricerange": torch.tensor([1, 1, 1]),
        }

        acc = compute_joint_goal_accuracy(predictions, labels)
        assert acc == 0.0

    def test_partial_correct(self) -> None:
        predictions = {
            "food": torch.tensor([0, 1]),
            "area": torch.tensor([0, 1]),
            "pricerange": torch.tensor([0, 0]),  # Second wrong
        }
        labels = {
            "food": torch.tensor([0, 1]),
            "area": torch.tensor([0, 1]),
            "pricerange": torch.tensor([0, 1]),
        }

        acc = compute_joint_goal_accuracy(predictions, labels)
        assert acc == 0.5
