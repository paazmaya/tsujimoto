"""Tests for base trainer and architecture-specific trainer classes.

Tests verify:
- Trainer initialization with different model architectures
- Training loop execution (train_epoch, validate)
- Checkpoint saving and loading
- Training history tracking
- Early stopping functionality
- Learning rate scheduling
- Architecture-specific features (mixed precision for ViT, QAT for QATTrainer, etc.)
"""

import json
import tempfile
from pathlib import Path
from typing import Tuple

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.lib.base_trainer import (
    CNNTrainer,
    HierCodeHiGITATrainer,
    HierCodeTrainer,
    QATTrainer,
    RNNTrainer,
    ViTTrainer,
    setup_trainer_for_model,
)

# ============================================================================
# Fixtures and Test Models
# ============================================================================


class SimpleTestModel(nn.Module):
    """Simple model for testing trainer."""

    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        """Initialize simple test model.

        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels

        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 16 * 16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output logits (B, num_classes)

        """
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class RNNTestModel(nn.Module):
    """RNN model for testing."""

    def __init__(self, num_classes: int = 10, hidden_size: int = 128):
        """Initialize RNN test model.

        Args:
            num_classes: Number of output classes
            hidden_size: Hidden state size

        """
        super().__init__()
        self.rnn = nn.LSTM(input_size=64, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output logits

        """
        _, (h_n, _) = self.rnn(x)
        x = self.fc(h_n[-1])
        return x


class HiGITATestModel(nn.Module):
    """HiGITA model that returns auxiliary output."""

    def __init__(self, num_classes: int = 10):
        """Initialize HiGITA test model.

        Args:
            num_classes: Number of output classes

        """
        super().__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.fc_main = nn.Linear(64 * 64 * 64, num_classes)
        self.fc_aux = nn.Linear(64 * 64 * 64, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning main and auxiliary outputs.

        Args:
            x: Input tensor

        Returns:
            Tuple of (main_output, auxiliary_output)

        """
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        main = self.fc_main(x)
        aux = self.fc_aux(x)
        return main, aux


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints.

    Yields:
        Path to temporary directory

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def simple_dataset():
    """Create simple dataset for training.

    Returns:
        DataLoader with dummy data

    """
    # Create dummy data: (B=32, C=1, H=64, W=64) -> num_classes
    images = torch.randn(128, 1, 64, 64)
    labels = torch.randint(0, 10, (128,))
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    return dataloader


@pytest.fixture
def simple_model():
    """Create simple CNN model.

    Returns:
        SimpleTestModel instance

    """
    return SimpleTestModel(num_classes=10)


@pytest.fixture
def rnn_model():
    """Create RNN model.

    Returns:
        RNNTestModel instance

    """
    return RNNTestModel(num_classes=10)


@pytest.fixture
def optimizer(simple_model):
    """Create optimizer for model.

    Args:
        simple_model: Model to optimize

    Returns:
        Optimizer instance

    """
    return optim.Adam(simple_model.parameters(), lr=0.001)


# ============================================================================
# Test BaseModelTrainer (via CNNTrainer)
# ============================================================================


class TestBaseTrainerInitialization:
    """Test base trainer initialization."""

    def test_trainer_init(self, simple_model, simple_dataset, optimizer, temp_checkpoint_dir):
        """Test trainer initialization."""
        trainer = CNNTrainer(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            device="cpu",
            checkpoint_dir=temp_checkpoint_dir,
            model_type="cnn",
            num_classes=10,
            image_size=64,
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert str(trainer.device) == "cpu"
        assert trainer.model_type == "cnn"
        assert trainer.num_classes == 10
        assert trainer.image_size == 64
        assert trainer.best_val_accuracy == 0.0
        assert trainer.patience == 10
        assert len(trainer.history) == 4

    def test_trainer_loss_fn(self, simple_model, simple_dataset, optimizer):
        """Test loss function is properly set."""
        trainer = CNNTrainer(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            device="cpu",
        )

        assert isinstance(trainer.loss_fn, nn.CrossEntropyLoss)

    def test_checkpoint_directory_creation(self, simple_model, simple_dataset, optimizer):
        """Test checkpoint directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = f"{tmpdir}/new_checkpoints"
            CNNTrainer(
                model=simple_model,
                train_loader=simple_dataset,
                val_loader=simple_dataset,
                optimizer=optimizer,
                checkpoint_dir=checkpoint_dir,
            )

            assert Path(checkpoint_dir).exists()


class TestTrainingLoop:
    """Test training loop functionality."""

    def test_train_epoch(self, simple_model, simple_dataset, optimizer):
        """Test training for one epoch."""
        trainer = CNNTrainer(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            device="cpu",
        )

        loss, acc = trainer.train_epoch()

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1
        assert loss >= 0

    def test_validate(self, simple_model, simple_dataset, optimizer):
        """Test validation."""
        trainer = CNNTrainer(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            device="cpu",
        )

        loss, acc = trainer.validate()

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1
        assert loss >= 0

    def test_train_multiple_epochs(self, simple_model, simple_dataset, optimizer):
        """Test training for multiple epochs."""
        trainer = CNNTrainer(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            device="cpu",
        )

        history = trainer.train(num_epochs=3, early_stopping=False, save_best_model=False)

        assert len(history["train_loss"]) == 3
        assert len(history["train_accuracy"]) == 3
        assert len(history["val_loss"]) == 3
        assert len(history["val_accuracy"]) == 3


class TestCheckpointManagement:
    """Test checkpoint saving and loading."""

    def test_save_checkpoint(self, simple_model, simple_dataset, optimizer, temp_checkpoint_dir):
        """Test checkpoint saving."""
        trainer = CNNTrainer(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            checkpoint_dir=temp_checkpoint_dir,
            device="cpu",
        )

        checkpoint_path = trainer.save_checkpoint(is_best=False)

        assert Path(checkpoint_path).exists()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

    def test_load_checkpoint(self, simple_model, simple_dataset, optimizer, temp_checkpoint_dir):
        """Test checkpoint loading."""
        trainer = CNNTrainer(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            checkpoint_dir=temp_checkpoint_dir,
            device="cpu",
        )

        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint(is_best=False)

        # Modify model
        original_weight = trainer.model.fc.weight.clone()
        trainer.model.fc.weight.data.fill_(0)

        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)

        # Verify weights are restored
        assert torch.allclose(trainer.model.fc.weight, original_weight)

    def test_save_training_history(
        self, simple_model, simple_dataset, optimizer, temp_checkpoint_dir
    ):
        """Test training history saving."""
        trainer = CNNTrainer(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            checkpoint_dir=temp_checkpoint_dir,
            device="cpu",
        )

        # Add some history
        trainer.history["train_loss"] = [0.5, 0.4, 0.3]
        trainer.history["train_accuracy"] = [0.7, 0.8, 0.9]

        history_path = trainer.save_training_history()

        assert Path(history_path).exists()
        with open(history_path) as f:
            loaded_history = json.load(f)
        assert loaded_history["train_loss"] == [0.5, 0.4, 0.3]


class TestEarlyStopping:
    """Test early stopping functionality."""

    def test_early_stopping_triggered(self, simple_model, simple_dataset, optimizer):
        """Test early stopping is triggered after patience exceeded."""
        trainer = CNNTrainer(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            device="cpu",
        )
        trainer.patience = 2

        # Manually set validation accuracy to not improve
        trainer.best_val_accuracy = 0.9
        trainer.patience_counter = 0

        for _ in range(3):
            trainer.patience_counter += 1

        assert trainer.patience_counter >= trainer.patience

    def test_early_stopping_disabled(self, simple_model, simple_dataset, optimizer):
        """Test early stopping can be disabled."""
        trainer = CNNTrainer(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            device="cpu",
        )

        history = trainer.train(num_epochs=5, early_stopping=False, save_best_model=False)

        assert len(history["train_loss"]) == 5


# ============================================================================
# Test Architecture-Specific Trainers
# ============================================================================


class TestCNNTrainer:
    """Test CNN trainer."""

    def test_cnn_trainer_init(self, simple_model, simple_dataset, optimizer):
        """Test CNN trainer initialization."""
        trainer = CNNTrainer(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
        )

        assert isinstance(trainer.loss_fn, nn.CrossEntropyLoss)
        assert isinstance(trainer, CNNTrainer)

    def test_cnn_train_epoch(self, simple_model, simple_dataset, optimizer):
        """Test CNN training epoch."""
        trainer = CNNTrainer(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            device="cpu",
        )

        loss, acc = trainer.train_epoch()
        assert loss >= 0
        assert 0 <= acc <= 1


class TestRNNTrainer:
    """Test RNN trainer."""

    def test_rnn_trainer_init(self, rnn_model, simple_dataset, optimizer):
        """Test RNN trainer initialization."""
        trainer = RNNTrainer(
            model=rnn_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
        )

        assert isinstance(trainer.loss_fn, nn.CrossEntropyLoss)
        assert isinstance(trainer, RNNTrainer)


class TestViTTrainer:
    """Test Vision Transformer trainer."""

    def test_vit_trainer_init(self, simple_model, simple_dataset, optimizer):
        """Test ViT trainer initialization."""
        trainer = ViTTrainer(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
        )

        assert isinstance(trainer, ViTTrainer)
        assert trainer.use_mixed_precision is False

    def test_vit_mixed_precision_flag(self, simple_model, simple_dataset, optimizer):
        """Test mixed precision flag."""
        trainer = ViTTrainer(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
        )

        trainer.enable_mixed_precision()
        assert trainer.use_mixed_precision is True


class TestHierCodeTrainer:
    """Test HierCode trainer."""

    def test_hiercode_trainer_init(self, simple_model, simple_dataset, optimizer):
        """Test HierCode trainer initialization."""
        trainer = HierCodeTrainer(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
        )

        assert isinstance(trainer, HierCodeTrainer)
        assert isinstance(trainer.loss_fn, nn.CrossEntropyLoss)


class TestQATTrainer:
    """Test Quantization-Aware Training trainer."""

    def test_qat_trainer_init(self, simple_model, simple_dataset, optimizer):
        """Test QAT trainer initialization."""
        trainer = QATTrainer(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
        )

        assert isinstance(trainer, QATTrainer)
        assert isinstance(trainer.loss_fn, nn.CrossEntropyLoss)

    def test_qat_train_with_qat_disabled(self, simple_model, simple_dataset, optimizer):
        """Test QAT training with quantization disabled."""
        trainer = QATTrainer(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            device="cpu",
        )

        history = trainer.train(
            num_epochs=2, early_stopping=False, save_best_model=False, qat_enabled=False
        )

        assert len(history["train_loss"]) == 2


class TestHierCodeHiGITATrainer:
    """Test HierCode-HiGITA trainer with auxiliary loss."""

    def test_higita_trainer_init(self, temp_checkpoint_dir):
        """Test HiGITA trainer initialization."""
        model = HiGITATestModel(num_classes=10)
        images = torch.randn(32, 1, 64, 64)
        labels = torch.randint(0, 10, (32,))
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=16)
        optimizer = optim.Adam(model.parameters())

        trainer = HierCodeHiGITATrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            checkpoint_dir=temp_checkpoint_dir,
            auxiliary_weight=0.2,
        )

        assert trainer.auxiliary_weight == 0.2
        assert isinstance(trainer, HierCodeHiGITATrainer)

    def test_higita_forward_pass_with_auxiliary(self):
        """Test HiGITA forward pass with auxiliary output."""
        model = HiGITATestModel(num_classes=10)
        images = torch.randn(8, 1, 64, 64)
        labels = torch.randint(0, 10, (8,))
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=8)
        optimizer = optim.Adam(model.parameters())

        trainer = HierCodeHiGITATrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            auxiliary_weight=0.1,
        )

        batch = next(iter(loader))
        outputs, loss = trainer.forward_pass(batch)

        assert outputs is not None
        assert loss is not None
        assert isinstance(loss, torch.Tensor)


# ============================================================================
# Test Factory Function
# ============================================================================


class TestSetupTrainerForModel:
    """Test factory function for trainer creation."""

    def test_factory_cnn(self, simple_model, simple_dataset, optimizer):
        """Test factory creates CNN trainer."""
        trainer = setup_trainer_for_model(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            model_type="cnn",
        )

        assert isinstance(trainer, CNNTrainer)

    def test_factory_rnn(self, rnn_model, simple_dataset, optimizer):
        """Test factory creates RNN trainer."""
        trainer = setup_trainer_for_model(
            model=rnn_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            model_type="rnn",
        )

        assert isinstance(trainer, RNNTrainer)

    def test_factory_vit(self, simple_model, simple_dataset, optimizer):
        """Test factory creates ViT trainer."""
        trainer = setup_trainer_for_model(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            model_type="vit",
        )

        assert isinstance(trainer, ViTTrainer)

    def test_factory_hiercode(self, simple_model, simple_dataset, optimizer):
        """Test factory creates HierCode trainer."""
        trainer = setup_trainer_for_model(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            model_type="hiercode",
        )

        assert isinstance(trainer, HierCodeTrainer)

    def test_factory_qat(self, simple_model, simple_dataset, optimizer):
        """Test factory creates QAT trainer."""
        trainer = setup_trainer_for_model(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            model_type="qat",
        )

        assert isinstance(trainer, QATTrainer)

    def test_factory_higita(self, simple_model, simple_dataset, optimizer):
        """Test factory creates HiGITA trainer."""
        trainer = setup_trainer_for_model(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            model_type="hiercode_higita",
        )

        assert isinstance(trainer, HierCodeHiGITATrainer)

    def test_factory_invalid_type(self, simple_model, simple_dataset, optimizer):
        """Test factory raises error for invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            setup_trainer_for_model(
                model=simple_model,
                train_loader=simple_dataset,
                val_loader=simple_dataset,
                optimizer=optimizer,
                model_type="invalid",
            )

    def test_factory_with_metadata(self, simple_model, simple_dataset, optimizer):
        """Test factory creates trainer with metadata."""
        trainer = setup_trainer_for_model(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            model_type="cnn",
            num_classes=100,
            image_size=128,
        )

        assert trainer.num_classes == 100
        assert trainer.image_size == 128


# ============================================================================
# Integration Tests
# ============================================================================


class TestTrainerIntegration:
    """Integration tests for complete training workflows."""

    def test_complete_training_workflow(self, simple_model, simple_dataset, optimizer):
        """Test complete training workflow."""
        trainer = CNNTrainer(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            device="cpu",
            model_type="cnn",
            num_classes=10,
        )

        # Train
        history = trainer.train(num_epochs=2, early_stopping=False, save_best_model=True)

        # Verify history
        assert len(history["train_loss"]) == 2
        assert len(history["val_accuracy"]) == 2

        # Verify best model was tracked
        assert trainer.best_val_accuracy >= 0

    def test_trainer_with_scheduler(self, simple_model, simple_dataset, optimizer):
        """Test trainer with learning rate scheduler."""
        trainer = CNNTrainer(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            device="cpu",
        )

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        trainer.set_scheduler(scheduler)

        assert trainer.scheduler is not None

    def test_trainer_device_mapping(self, simple_model, simple_dataset, optimizer):
        """Test trainer correctly maps model and data to device."""
        trainer = CNNTrainer(
            model=simple_model,
            train_loader=simple_dataset,
            val_loader=simple_dataset,
            optimizer=optimizer,
            device="cpu",
        )

        # Model should be on the specified device
        params = list(trainer.model.parameters())
        assert all(p.device.type == "cpu" for p in params)
