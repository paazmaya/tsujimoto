#!/usr/bin/env python3
"""Extended tests for checkpoint manager to improve coverage."""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.lib.checkpoint import CheckpointManager


class SimpleTestModel(nn.Module):
    """Simple model for testing checkpoint operations."""

    def __init__(self, input_size=10, output_size=2):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


class TestCheckpointManagerAdvanced:
    """Advanced checkpoint manager tests for full coverage."""

    def test_save_best_checkpoint(self):
        """Test saving best checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, "test")
            model = SimpleTestModel()
            optimizer = optim.Adam(model.parameters())
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)

            # Save best checkpoint
            path = manager.save_checkpoint(
                epoch=5,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics={"accuracy": 0.95},
                is_best=True,
            )

            # Check both regular and best checkpoints exist
            assert path.exists()
            best_path = Path(tmpdir) / "checkpoint_best.pt"
            assert best_path.exists()

    def test_save_checkpoint_with_metrics(self):
        """Test saving checkpoint with custom metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, "test")
            model = SimpleTestModel()
            optimizer = optim.Adam(model.parameters())
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)

            metrics = {
                "accuracy": 0.92,
                "loss": 0.15,
                "val_accuracy": 0.89,
                "val_loss": 0.18,
            }

            path = manager.save_checkpoint(
                epoch=10,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=metrics,
            )

            # Load and verify metrics
            checkpoint = torch.load(path)
            assert checkpoint["metrics"]["accuracy"] == 0.92
            assert checkpoint["metrics"]["loss"] == 0.15

    def test_save_checkpoint_without_metrics(self):
        """Test saving checkpoint without metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, "test")
            model = SimpleTestModel()
            optimizer = optim.Adam(model.parameters())
            scheduler = None

            path = manager.save_checkpoint(
                epoch=3,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

            checkpoint = torch.load(path)
            assert checkpoint["metrics"] == {}

    def test_load_checkpoint_with_scheduler(self):
        """Test loading checkpoint with scheduler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, "test")
            model = SimpleTestModel()
            optimizer = optim.Adam(model.parameters())
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)

            # Save checkpoint
            manager.save_checkpoint(
                epoch=7,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics={"lr": 0.001},
            )

            # Load into new instances
            new_model = SimpleTestModel()
            new_optimizer = optim.Adam(new_model.parameters())
            new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=5)

            latest = manager.find_latest_checkpoint()
            epoch, metrics = manager.load_checkpoint(
                latest, new_model, new_optimizer, new_scheduler
            )

            assert epoch == 7
            assert metrics["lr"] == 0.001

    def test_load_checkpoint_without_scheduler(self):
        """Test loading checkpoint when scheduler is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, "test")
            model = SimpleTestModel()
            optimizer = optim.Adam(model.parameters())

            # Save without scheduler
            manager.save_checkpoint(
                epoch=2,
                model=model,
                optimizer=optimizer,
                scheduler=None,
            )

            # Load without scheduler
            new_model = SimpleTestModel()
            new_optimizer = optim.Adam(new_model.parameters())

            latest = manager.find_latest_checkpoint()
            epoch, metrics = manager.load_checkpoint(latest, new_model, new_optimizer, None)

            assert epoch == 2

    def test_find_latest_checkpoint_multiple_files(self):
        """Test finding latest checkpoint among multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, "test")
            model = SimpleTestModel()
            optimizer = optim.Adam(model.parameters())

            # Save multiple checkpoints
            for epoch in [1, 5, 10, 15, 20]:
                manager.save_checkpoint(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                )

            latest = manager.find_latest_checkpoint()
            assert "020" in latest.name

    def test_find_latest_checkpoint_empty_dir(self):
        """Test find_latest_checkpoint returns None for empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, "empty")
            latest = manager.find_latest_checkpoint()
            assert latest is None

    def test_load_checkpoint_for_training_no_checkpoint(self):
        """Test load_checkpoint_for_training with no_checkpoint flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, "test")
            model = SimpleTestModel()
            optimizer = optim.Adam(model.parameters())
            scheduler = None

            # Create a checkpoint but disable loading
            manager.save_checkpoint(epoch=5, model=model, optimizer=optimizer, scheduler=scheduler)

            # Load with no_checkpoint=True
            new_model = SimpleTestModel()
            new_optimizer = optim.Adam(new_model.parameters())

            start_epoch, metrics = manager.load_checkpoint_for_training(
                model=new_model,
                optimizer=new_optimizer,
                scheduler=None,
                device="cpu",
                args_no_checkpoint=True,
            )

            # Should start from epoch 0 (not load checkpoint)
            assert start_epoch == 0
            assert metrics == {}

    def test_load_checkpoint_for_training_resume_from(self):
        """Test load_checkpoint_for_training with explicit resume path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, "test")
            model = SimpleTestModel()
            optimizer = optim.Adam(model.parameters())

            # Save checkpoints
            manager.save_checkpoint(epoch=10, model=model, optimizer=optimizer, scheduler=None)
            specific_path = manager.save_checkpoint(
                epoch=5, model=model, optimizer=optimizer, scheduler=None
            )

            # Load from specific checkpoint
            new_model = SimpleTestModel()
            new_optimizer = optim.Adam(new_model.parameters())

            start_epoch, metrics = manager.load_checkpoint_for_training(
                model=new_model,
                optimizer=new_optimizer,
                scheduler=None,
                device="cpu",
                resume_from=str(specific_path),
            )

            # Should resume from epoch 5, not latest (10)
            assert start_epoch == 6

    def test_load_checkpoint_for_training_auto_detect(self):
        """Test load_checkpoint_for_training with auto-detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, "test")
            model = SimpleTestModel()
            optimizer = optim.Adam(model.parameters())

            # Save checkpoint
            manager.save_checkpoint(
                epoch=8,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                metrics={"accuracy": 0.88},
            )

            # Load with auto-detection
            new_model = SimpleTestModel()
            new_optimizer = optim.Adam(new_model.parameters())

            start_epoch, metrics = manager.load_checkpoint_for_training(
                model=new_model,
                optimizer=new_optimizer,
                scheduler=None,
                device="cpu",
            )

            assert start_epoch == 9  # Next epoch after loaded
            assert metrics["accuracy"] == 0.88

    def test_load_checkpoint_for_training_no_checkpoints_exist(self):
        """Test load_checkpoint_for_training when no checkpoints exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, "empty")
            model = SimpleTestModel()
            optimizer = optim.Adam(model.parameters())

            start_epoch, metrics = manager.load_checkpoint_for_training(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                device="cpu",
            )

            assert start_epoch == 0
            assert metrics == {}

    def test_checkpoint_epoch_formatting(self):
        """Test that epoch numbers are formatted with leading zeros."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, "test")
            model = SimpleTestModel()
            optimizer = optim.Adam(model.parameters())

            # Test various epoch numbers
            for epoch in [1, 9, 10, 99, 100]:
                path = manager.save_checkpoint(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                )

                # Check formatting
                expected_format = f"checkpoint_epoch_{epoch:03d}.pt"
                assert path.name == expected_format

    def test_cleanup_old_checkpoints_integration(self):
        """Test that keeping only recent checkpoints works as expected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, "test")
            model = SimpleTestModel()
            optimizer = optim.Adam(model.parameters())

            # Save many checkpoints
            for epoch in range(1, 21):
                manager.save_checkpoint(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                )

            # All checkpoints should exist (no automatic cleanup yet)
            checkpoints = list(Path(tmpdir).glob("checkpoint_epoch_*.pt"))
            assert len(checkpoints) == 20

    def test_checkpoint_contains_all_required_keys(self):
        """Test that saved checkpoint contains all required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, "test")
            model = SimpleTestModel()
            optimizer = optim.Adam(model.parameters())
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)

            path = manager.save_checkpoint(
                epoch=1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics={"test": 1},
            )

            checkpoint = torch.load(path)

            required_keys = [
                "epoch",
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "metrics",
            ]

            for key in required_keys:
                assert key in checkpoint
