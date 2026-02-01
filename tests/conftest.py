"""Pytest configuration and shared fixtures."""

import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Get the test device."""
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
