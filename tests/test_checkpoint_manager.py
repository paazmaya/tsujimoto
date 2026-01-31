#!/usr/bin/env python3
"""Unit tests for checkpoint_manager module."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from scripts.checkpoint_manager import CheckpointManager


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary checkpoint directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def model():
    """Create a simple test model."""
    return SimpleModel()


@pytest.fixture
def optimizer(model):
    """Create optimizer for test model."""
    return optim.SGD(model.parameters(), lr=0.01)


@pytest.fixture
def scheduler(optimizer):
    """Create scheduler for test optimizer."""
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)


@pytest.fixture
def checkpoint_manager(temp_checkpoint_dir):
    """Create checkpoint manager instance."""
    return CheckpointManager(temp_checkpoint_dir, approach_name="test_cnn")


class TestCheckpointManagerInit:
    """Tests for CheckpointManager initialization."""

    def test_init_creates_directory(self, temp_checkpoint_dir):
        """Test that initialization creates approach directory."""
        CheckpointManager(temp_checkpoint_dir, approach_name="cnn")
        assert (Path(temp_checkpoint_dir) / "cnn").exists()

    def test_init_with_different_approaches(self, temp_checkpoint_dir):
        """Test initialization with different approach names."""
        approaches = ["cnn", "rnn", "hiercode", "vit", "qat"]
        for approach in approaches:
            CheckpointManager(temp_checkpoint_dir, approach_name=approach)
            assert (Path(temp_checkpoint_dir) / approach).exists()

    def test_init_sets_attributes(self, checkpoint_manager):
        """Test that attributes are properly set."""
        assert checkpoint_manager.approach_name == "test_cnn"
        assert checkpoint_manager.base_dir == Path(checkpoint_manager.base_dir)
        assert checkpoint_manager.approach_dir.name == "test_cnn"


class TestCheckpointSave:
    """Tests for checkpoint saving."""

    def test_save_checkpoint_creates_file(self, checkpoint_manager, model, optimizer, scheduler):
        """Test that save_checkpoint creates a checkpoint file."""
        checkpoint_manager.save_checkpoint(
            epoch=1, model=model, optimizer=optimizer, scheduler=scheduler
        )
        checkpoint_files = list(checkpoint_manager.approach_dir.glob("*.pt"))
        assert len(checkpoint_files) >= 1
        # Check that at least one file has the epoch number in it
        assert any("001" in f.name for f in checkpoint_files)

    def test_save_checkpoint_contains_required_keys(
        self, checkpoint_manager, model, optimizer, scheduler
    ):
        """Test that saved checkpoint contains required keys."""
        checkpoint_manager.save_checkpoint(
            epoch=1, model=model, optimizer=optimizer, scheduler=scheduler
        )
        checkpoint_path = list(checkpoint_manager.approach_dir.glob("checkpoint_epoch_*.pt"))[0]
        checkpoint = torch.load(checkpoint_path)

        required_keys = {
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
        }
        assert required_keys.issubset(checkpoint.keys())

    def test_save_checkpoint_with_multiple_epochs(
        self, checkpoint_manager, model, optimizer, scheduler
    ):
        """Test saving multiple checkpoints."""
        for epoch in range(1, 4):
            checkpoint_manager.save_checkpoint(
                epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler
            )

        checkpoint_files = list(checkpoint_manager.approach_dir.glob("checkpoint_epoch_*.pt"))
        assert len(checkpoint_files) == 3


class TestCheckpointLoad:
    """Tests for checkpoint loading."""

    def test_load_checkpoint_restores_state(self, checkpoint_manager, model, optimizer, scheduler):
        """Test that loading checkpoint restores model state."""
        # Save checkpoint with modified model
        model.linear.weight.data.fill_(1.0)
        checkpoint_manager.save_checkpoint(
            epoch=1, model=model, optimizer=optimizer, scheduler=scheduler
        )

        # Create new model with different state
        new_model = SimpleModel()
        new_model.linear.weight.data.fill_(0.0)

        # Load checkpoint
        checkpoint_path = list(checkpoint_manager.approach_dir.glob("*.pt"))[0]
        checkpoint = torch.load(checkpoint_path)
        new_model.load_state_dict(checkpoint["model_state_dict"])

        # Verify weights match original model
        assert torch.allclose(model.linear.weight, new_model.linear.weight)

    def test_load_checkpoint_restores_optimizer_state(
        self, checkpoint_manager, model, optimizer, scheduler
    ):
        """Test that loading checkpoint restores optimizer state."""
        checkpoint_manager.save_checkpoint(
            epoch=1, model=model, optimizer=optimizer, scheduler=scheduler
        )

        new_optimizer = optim.SGD(model.parameters(), lr=0.01)
        checkpoint_path = list(checkpoint_manager.approach_dir.glob("*.pt"))[0]
        checkpoint = torch.load(checkpoint_path)
        new_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        assert "lr" in new_optimizer.param_groups[0]


class TestFindLatestCheckpoint:
    """Tests for finding latest checkpoint."""

    def test_find_latest_checkpoint_returns_none_when_empty(self, checkpoint_manager):
        """Test that find_latest_checkpoint returns None when no checkpoints exist."""
        latest = checkpoint_manager.find_latest_checkpoint()
        assert latest is None

    def test_find_latest_checkpoint_returns_latest(
        self, checkpoint_manager, model, optimizer, scheduler
    ):
        """Test that find_latest_checkpoint returns the latest checkpoint."""
        for epoch in [1, 2, 3]:
            checkpoint_manager.save_checkpoint(
                epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler
            )

        latest = checkpoint_manager.find_latest_checkpoint()
        assert latest is not None
        # Should return the highest epoch number
        assert "003" in latest.name

    def test_find_latest_checkpoint_with_skipped_epochs(
        self, checkpoint_manager, model, optimizer, scheduler
    ):
        """Test find_latest_checkpoint with non-sequential epochs."""
        for epoch in [1, 3, 5]:
            checkpoint_manager.save_checkpoint(
                epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler
            )

        latest = checkpoint_manager.find_latest_checkpoint()
        # Should return the highest epoch (5)
        assert latest is not None
        assert "005" in latest.name


class TestFindAndLoadLatestCheckpoint:
    """Tests for find_and_load_latest_checkpoint."""

    def test_find_and_load_latest_checkpoint_returns_none_when_empty(
        self, checkpoint_manager, model, optimizer, scheduler
    ):
        """Test returns None when no checkpoints exist."""
        checkpoint_data, start_epoch = checkpoint_manager.find_and_load_latest_checkpoint(
            model, optimizer, scheduler
        )
        assert checkpoint_data is None
        assert start_epoch == 0

    def test_find_and_load_latest_checkpoint_returns_data(
        self, checkpoint_manager, model, optimizer, scheduler
    ):
        """Test that find_and_load_latest_checkpoint returns checkpoint data."""
        checkpoint_manager.save_checkpoint(
            epoch=2, model=model, optimizer=optimizer, scheduler=scheduler
        )

        checkpoint_data, start_epoch = checkpoint_manager.find_and_load_latest_checkpoint(
            model, optimizer, scheduler
        )
        assert checkpoint_data is not None
        assert start_epoch == 3  # Should resume from next epoch (2 + 1)
        assert "epoch" in checkpoint_data

    def test_find_and_load_latest_checkpoint_loads_model_state(
        self, checkpoint_manager, model, optimizer, scheduler
    ):
        """Test that model state is properly loaded."""
        # Modify and save model
        model.linear.weight.data.fill_(1.0)
        checkpoint_manager.save_checkpoint(
            epoch=1, model=model, optimizer=optimizer, scheduler=scheduler
        )

        # Create new model with different state
        new_model = SimpleModel()
        new_model.linear.weight.data.fill_(0.0)
        new_optimizer = optim.SGD(new_model.parameters(), lr=0.01)
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=10)

        # Load checkpoint
        checkpoint_manager.find_and_load_latest_checkpoint(new_model, new_optimizer, new_scheduler)

        # Verify state was restored
        assert torch.allclose(model.linear.weight, new_model.linear.weight)


class TestCheckpointListing:
    """Tests for listing checkpoints."""

    def test_list_all_checkpoints_returns_empty_when_no_checkpoints(self, checkpoint_manager):
        """Test that list_all_checkpoints returns empty list when no checkpoints exist."""
        checkpoints = checkpoint_manager.list_all_checkpoints()
        assert checkpoints == []

    def test_list_all_checkpoints_returns_all_checkpoints(
        self, checkpoint_manager, model, optimizer, scheduler
    ):
        """Test that list_all_checkpoints returns all saved checkpoints."""
        for epoch in [1, 2, 3]:
            checkpoint_manager.save_checkpoint(
                epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler
            )

        checkpoints = checkpoint_manager.list_all_checkpoints()
        assert len(checkpoints) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
