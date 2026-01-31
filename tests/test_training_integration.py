"""
Integration tests for training loops.

Tests basic functionality of training models end-to-end with small datasets.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib.config import OptimizationConfig


@pytest.fixture
def dummy_dataset():
    """Create a small dummy dataset for testing"""
    # Create 20 samples, 64x64 images
    x = torch.randn(20, 1, 64, 64)
    y = torch.randint(0, 10, (20,))
    return TensorDataset(x, y)


@pytest.fixture
def dummy_model():
    """Create a simple CNN model for testing"""

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc1 = nn.Linear(32 * 16 * 16, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return SimpleCNN


class TestTrainingLoop:
    """Test basic training loop functionality"""

    def test_single_forward_pass(self, dummy_model):
        """Test that model can perform a forward pass"""
        model = dummy_model(num_classes=10)
        x = torch.randn(4, 1, 64, 64)
        output = model(x)

        assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

    def test_single_training_step(self, dummy_model, dummy_dataset):
        """Test that model can perform a single training step"""
        model = dummy_model(num_classes=10)
        train_loader = DataLoader(dummy_dataset, batch_size=4, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Get one batch
        images, labels = next(iter(train_loader))

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0, "Loss should be positive"
        assert not np.isnan(loss.item()), "Loss is NaN"
        assert not np.isinf(loss.item()), "Loss is Inf"

    def test_multiple_training_steps(self, dummy_model, dummy_dataset):
        """Test that model can train for multiple steps without errors"""
        model = dummy_model(num_classes=10)
        train_loader = DataLoader(dummy_dataset, batch_size=4, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        losses = []

        # Train for 5 batches
        for i, (images, labels) in enumerate(train_loader):
            if i >= 5:
                break

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Check that we completed all steps
        assert len(losses) == 5, f"Expected 5 losses, got {len(losses)}"

        # Check that all losses are valid
        for i, loss_val in enumerate(losses):
            assert loss_val > 0, f"Loss at step {i} is not positive: {loss_val}"
            assert not np.isnan(loss_val), f"Loss at step {i} is NaN"
            assert not np.isinf(loss_val), f"Loss at step {i} is Inf"

    def test_evaluation_mode(self, dummy_model, dummy_dataset):
        """Test that model can switch to evaluation mode and run inference"""
        model = dummy_model(num_classes=10)
        test_loader = DataLoader(dummy_dataset, batch_size=4, shuffle=False)

        model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.tolist())
                actuals.extend(labels.tolist())

        assert len(predictions) == len(dummy_dataset), "Not all samples were predicted"
        assert all(0 <= p < 10 for p in predictions), "Predictions out of range"

    def test_loss_decreases_over_epochs(self, dummy_model, dummy_dataset):
        """Test that loss generally decreases over multiple epochs (overfitting check)"""
        model = dummy_model(num_classes=10)
        train_loader = DataLoader(dummy_dataset, batch_size=4, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Higher LR for fast convergence

        epoch_losses = []

        # Train for 3 epochs
        for _epoch in range(3):
            model.train()
            epoch_loss = 0
            num_batches = 0

            for images, labels in train_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            epoch_losses.append(avg_loss)

        # Check that loss decreased from first to last epoch
        assert epoch_losses[-1] < epoch_losses[0], (
            f"Loss did not decrease: {epoch_losses[0]:.4f} -> {epoch_losses[-1]:.4f}"
        )


class TestModelArchitectures:
    """Test different model architectures with dummy data"""

    def test_cnn_architecture_forward(self):
        """Test CNN architecture forward pass"""
        try:
            from scripts.train_cnn_model import LightweightKanjiNet

            model = LightweightKanjiNet(num_classes=10)
            x = torch.randn(4, 1, 64, 64)
            output = model(x)

            assert output.shape == (4, 10)
            assert not torch.isnan(output).any()
        except ImportError:
            pytest.skip("LightweightKanjiNet not available")

    def test_rnn_architecture_forward(self):
        """Test RNN architecture forward pass"""
        try:
            from scripts.train_rnn import KanjiRNN

            model = KanjiRNN(num_classes=10)
            x = torch.randn(4, 10, 4)  # (batch, seq_len, features)
            output = model(x)

            assert output.shape == (4, 10)
            assert not torch.isnan(output).any()
        except ImportError:
            pytest.skip("KanjiRNN not available")


class TestConfigIntegration:
    """Test training configuration integration"""

    def test_config_creation(self):
        """Test that OptimizationConfig can be created with defaults"""
        config = OptimizationConfig()

        assert config.data_dir == "dataset"
        assert config.image_size == 64
        assert config.num_classes == 43427
        assert config.batch_size == 64
        assert config.learning_rate == 0.001

    def test_config_to_dict(self):
        """Test config serialization"""
        config = OptimizationConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "data_dir" in config_dict
        assert "num_classes" in config_dict
        assert config_dict["image_size"] == 64

    def test_custom_config_values(self):
        """Test creating config with custom values"""
        config = OptimizationConfig(num_classes=100, batch_size=32, learning_rate=0.0001, epochs=10)

        assert config.num_classes == 100
        assert config.batch_size == 32
        assert config.learning_rate == 0.0001
        assert config.epochs == 10
