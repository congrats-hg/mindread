"""Tests for DST models."""

import pytest
import torch

from mindread.data.schema import INFORMABLE_SLOTS
from mindread.models.dst import BertDST, HierarchicalBertDST, create_model


class TestBertDST:
    """Tests for BertDST model."""

    @pytest.fixture
    def model(self) -> BertDST:
        """Create a small model for testing."""
        return BertDST(
            model_name="prajjwal1/bert-tiny",  # Use tiny model for fast tests
            dropout=0.1,
        )

    def test_model_creation(self, model: BertDST) -> None:
        assert model is not None
        assert hasattr(model, "encoder")
        assert hasattr(model, "slot_classifiers")

    def test_slot_classifiers_exist(self, model: BertDST) -> None:
        for slot in INFORMABLE_SLOTS:
            assert slot in model.slot_classifiers

    def test_forward_pass(self, model: BertDST) -> None:
        batch_size = 2
        seq_len = 32

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = model(input_ids, attention_mask)

        assert isinstance(outputs, dict)
        for slot in INFORMABLE_SLOTS:
            assert slot in outputs
            assert outputs[slot].shape[0] == batch_size

    def test_predict(self, model: BertDST) -> None:
        batch_size = 2
        seq_len = 32

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        predictions = model.predict(input_ids, attention_mask)

        for slot in INFORMABLE_SLOTS:
            assert slot in predictions
            assert predictions[slot].shape == (batch_size,)
            assert predictions[slot].dtype == torch.int64


class TestHierarchicalBertDST:
    """Tests for HierarchicalBertDST model."""

    @pytest.fixture
    def model(self) -> HierarchicalBertDST:
        """Create a small model for testing."""
        return HierarchicalBertDST(
            model_name="prajjwal1/bert-tiny",
            dropout=0.1,
            num_context_layers=1,
        )

    def test_model_creation(self, model: HierarchicalBertDST) -> None:
        assert model is not None
        assert hasattr(model, "encoder")
        assert hasattr(model, "context_encoder")
        assert hasattr(model, "slot_classifiers")

    def test_forward_pass(self, model: HierarchicalBertDST) -> None:
        batch_size = 2
        seq_len = 32

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = model(input_ids, attention_mask)

        assert isinstance(outputs, dict)
        for slot in INFORMABLE_SLOTS:
            assert slot in outputs


class TestCreateModel:
    """Tests for model factory function."""

    def test_create_bert_model(self) -> None:
        model = create_model(
            model_type="bert",
            model_name="prajjwal1/bert-tiny",
        )
        assert isinstance(model, BertDST)

    def test_create_hierarchical_model(self) -> None:
        model = create_model(
            model_type="hierarchical",
            model_name="prajjwal1/bert-tiny",
            num_context_layers=1,
        )
        assert isinstance(model, HierarchicalBertDST)

    def test_invalid_model_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model(model_type="invalid")
