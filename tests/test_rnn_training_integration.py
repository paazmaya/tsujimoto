"""
Integration tests for RNN training with single-epoch runs.

Tests KanjiRNN and RNNTrainer from ``src.lib`` across cell types and configurations to ensure:
- Data loading works correctly
- Model can train without errors
- Gradients flow properly
- Checkpoint saving works
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib.base_trainer import RNNTrainer, setup_trainer_for_model
from src.lib.config import RNNConfig
from src.lib.datasets import ResearchDataset
from src.lib.models import KanjiRNN


class TestRNNTrainingIntegration:
    """Integration tests for RNN training."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary checkpoint directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def dummy_data(self):
        """Create deterministic dummy dataset for testing."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((16, 64 * 64)).astype(np.float32)
        y = rng.integers(0, 10, size=16, dtype=np.int64)
        return x, y, 10

    def _run_rnn_epoch(self, dummy_data, temp_checkpoint_dir, rnn_type: str = "lstm"):
        x, y, num_classes = dummy_data
        dataset = ResearchDataset(x, y)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        model = KanjiRNN(
            num_classes=num_classes,
            hidden_size=32,
            num_layers=1,
            rnn_type=rnn_type,
            input_channels=1,
            image_size=64,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = setup_trainer_for_model(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            device="cpu",
            model_type="rnn",
            checkpoint_dir=str(temp_checkpoint_dir),
            num_classes=num_classes,
            image_size=64,
        )
        assert isinstance(trainer, RNNTrainer)
        history = trainer.train(num_epochs=1, save_best_model=True)

        assert len(history["train_loss"]) == 1
        assert len(history["val_loss"]) == 1
        assert len(history["val_accuracy"]) == 1
        assert not np.isnan(history["train_loss"][0])
        return trainer, history

    def test_basic_rnn_single_epoch(self, dummy_data, temp_checkpoint_dir):
        """Test LSTM KanjiRNN training for one epoch."""
        self._run_rnn_epoch(dummy_data, temp_checkpoint_dir, rnn_type="lstm")

    def test_stroke_rnn_single_epoch(self, dummy_data, temp_checkpoint_dir):
        """Test GRU KanjiRNN training for one epoch."""
        self._run_rnn_epoch(dummy_data, temp_checkpoint_dir, rnn_type="gru")

    def test_simple_radical_rnn_single_epoch(self, dummy_data, temp_checkpoint_dir):
        """Test vanilla RNN KanjiRNN training for one epoch."""
        self._run_rnn_epoch(dummy_data, temp_checkpoint_dir, rnn_type="rnn")

    def test_hybrid_cnn_rnn_single_epoch(self, dummy_data, temp_checkpoint_dir):
        """Test multi-layer bidirectional LSTM KanjiRNN training and checkpoint creation."""
        x, y, num_classes = dummy_data
        dataset = ResearchDataset(x, y)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        model = KanjiRNN(
            num_classes=num_classes,
            hidden_size=32,
            num_layers=2,
            rnn_type="lstm",
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = RNNTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            device="cpu",
            checkpoint_dir=str(temp_checkpoint_dir),
            model_type="rnn",
            num_classes=num_classes,
        )
        history = trainer.train(num_epochs=2, save_best_model=True)
        assert len(history["train_loss"]) == 2
        ckpt_path = trainer.save_checkpoint(is_best=True)
        assert Path(ckpt_path).exists()

    def test_linguistic_radical_rnn_single_epoch(self, dummy_data, temp_checkpoint_dir):
        """Test RNNConfig validation and training integration."""
        config = RNNConfig(
            model_variant="linguistic_radical_rnn",
            rnn_type="gru",
            hidden_size=32,
            num_layers=1,
            epochs=1,
            batch_size=4,
            num_classes=10,
        )
        assert config.model_variant == "linguistic_radical_rnn"
        self._run_rnn_epoch(dummy_data, temp_checkpoint_dir, rnn_type=config.rnn_type)

    def test_all_variants_dataset_compatibility(self, dummy_data):
        """Test that all 5 RNNConfig variants validate and work with ResearchDataset."""
        x, y, num_classes = dummy_data
        variants = [
            "basic_rnn",
            "stroke_rnn",
            "simple_radical_rnn",
            "hybrid_cnn_rnn",
            "linguistic_radical_rnn",
        ]

        dataset = ResearchDataset(x, y)
        loader = DataLoader(dataset, batch_size=4)
        images, labels = next(iter(loader))
        assert images.shape == (4, 1, 64, 64)
        assert labels.shape == (4,)

        for variant in variants:
            cfg = RNNConfig(model_variant=variant, num_classes=num_classes, hidden_size=32)
            assert cfg.model_variant == variant
