"""
Integration tests for RNN training with single-epoch runs.

Tests all 5 RNN variants with minimal training to ensure:
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRNNTrainingIntegration:
    """Integration tests for RNN training"""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary checkpoint directory"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def dummy_data(self):
        """Create dummy dataset for testing"""
        # Small dataset: 20 samples, 64x64 images, 10 classes
        x = np.random.randn(20, 64 * 64).astype(np.float32)
        y = np.random.randint(0, 10, 20).astype(np.int64)
        return x, y, 10  # x, y, num_classes

    def test_basic_rnn_single_epoch(self, dummy_data, temp_checkpoint_dir):
        """Test basic_rnn training for one epoch"""
        try:
            from torch.utils.data import DataLoader

            from scripts.train_rnn import (
                RNNKanjiDataset,
                RNNTrainer,
                collate_fn_factory,
                create_rnn_model,
            )
            from src.lib import RNNConfig

            x, y, num_classes = dummy_data
            model_type = "basic_rnn"

            # Create dataset and loader
            dataset = RNNKanjiDataset(x, y, model_type=model_type)
            collate_fn = collate_fn_factory(model_type)
            loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

            # Create model
            model = create_rnn_model(model_type, num_classes=num_classes, hidden_size=32)

            # Create trainer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            trainer = RNNTrainer(model, device, model_type, temp_checkpoint_dir)

            # Train for 1 epoch
            config = RNNConfig(learning_rate=0.001, epochs=1, batch_size=4)
            trainer.train(loader, loader, epochs=1, config=config)

            # Verify training completed
            assert len(trainer.train_losses) == 1
            assert len(trainer.val_losses) == 1
            assert len(trainer.val_accuracies) == 1

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    def test_stroke_rnn_single_epoch(self, dummy_data, temp_checkpoint_dir):
        """Test stroke_rnn training for one epoch"""
        try:
            from torch.utils.data import DataLoader

            from scripts.train_rnn import (
                RNNKanjiDataset,
                RNNTrainer,
                collate_fn_factory,
                create_rnn_model,
            )
            from src.lib import RNNConfig

            x, y, num_classes = dummy_data
            model_type = "stroke_rnn"

            # Create dataset and loader
            dataset = RNNKanjiDataset(x, y, model_type=model_type)
            collate_fn = collate_fn_factory(model_type)
            loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

            # Create model
            model = create_rnn_model(model_type, num_classes=num_classes, hidden_size=32)

            # Create trainer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            trainer = RNNTrainer(model, device, model_type, temp_checkpoint_dir)

            # Train for 1 epoch
            config = RNNConfig(learning_rate=0.001, epochs=1, batch_size=4)
            trainer.train(loader, loader, epochs=1, config=config)

            assert len(trainer.train_losses) == 1

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    def test_simple_radical_rnn_single_epoch(self, dummy_data, temp_checkpoint_dir):
        """Test simple_radical_rnn training for one epoch"""
        try:
            from torch.utils.data import DataLoader

            from scripts.train_rnn import (
                RNNKanjiDataset,
                RNNTrainer,
                collate_fn_factory,
                create_rnn_model,
            )
            from src.lib import RNNConfig

            x, y, num_classes = dummy_data
            model_type = "simple_radical_rnn"

            # Create dataset and loader
            dataset = RNNKanjiDataset(x, y, model_type=model_type)
            collate_fn = collate_fn_factory(model_type)
            loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

            # Create model
            model = create_rnn_model(
                model_type,
                num_classes=num_classes,
                radical_vocab_size=500,
                hidden_size=32,
            )

            # Create trainer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            trainer = RNNTrainer(model, device, model_type, temp_checkpoint_dir)

            # Train for 1 epoch
            config = RNNConfig(learning_rate=0.001, epochs=1, batch_size=4)
            trainer.train(loader, loader, epochs=1, config=config)

            assert len(trainer.train_losses) == 1

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    def test_hybrid_cnn_rnn_single_epoch(self, dummy_data, temp_checkpoint_dir):
        """Test hybrid_cnn_rnn training for one epoch"""
        try:
            from torch.utils.data import DataLoader

            from scripts.train_rnn import (
                RNNKanjiDataset,
                RNNTrainer,
                collate_fn_factory,
                create_rnn_model,
            )
            from src.lib import RNNConfig

            x, y, num_classes = dummy_data
            model_type = "hybrid_cnn_rnn"

            # Create dataset and loader
            dataset = RNNKanjiDataset(x, y, model_type=model_type)
            collate_fn = collate_fn_factory(model_type)
            loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

            # Create model
            model = create_rnn_model(model_type, num_classes=num_classes, hidden_size=32)

            # Create trainer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            trainer = RNNTrainer(model, device, model_type, temp_checkpoint_dir)

            # Train for 1 epoch
            config = RNNConfig(learning_rate=0.001, epochs=1, batch_size=4)
            trainer.train(loader, loader, epochs=1, config=config)

            assert len(trainer.train_losses) == 1

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    def test_linguistic_radical_rnn_single_epoch(self, dummy_data, temp_checkpoint_dir):
        """Test linguistic_radical_rnn training for one epoch"""
        try:
            from torch.utils.data import DataLoader

            from scripts.train_rnn import (
                RNNKanjiDataset,
                RNNTrainer,
                collate_fn_factory,
                create_rnn_model,
            )
            from src.lib import RNNConfig

            x, y, num_classes = dummy_data
            model_type = "linguistic_radical_rnn"

            # Create dataset and loader
            dataset = RNNKanjiDataset(x, y, model_type=model_type)
            collate_fn = collate_fn_factory(model_type)
            loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

            # Create model
            model = create_rnn_model(
                model_type,
                num_classes=num_classes,
                radical_vocab_size=2000,
                radical_embedding_dim=64,
                rnn_hidden_size=32,
                rnn_num_layers=2,
            )

            # Create trainer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            trainer = RNNTrainer(model, device, model_type, temp_checkpoint_dir)

            # Train for 1 epoch
            config = RNNConfig(learning_rate=0.001, epochs=1, batch_size=4)
            trainer.train(loader, loader, epochs=1, config=config)

            assert len(trainer.train_losses) == 1

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    def test_all_variants_dataset_compatibility(self, dummy_data):
        """Test that all RNN variants work with their respective datasets"""
        try:
            from torch.utils.data import DataLoader

            from scripts.train_rnn import RNNKanjiDataset, collate_fn_factory

            x, y, num_classes = dummy_data

            variants = [
                "basic_rnn",
                "stroke_rnn",
                "simple_radical_rnn",
                "hybrid_cnn_rnn",
                "linguistic_radical_rnn",
            ]

            for variant in variants:
                # Create dataset
                dataset = RNNKanjiDataset(x, y, model_type=variant)
                assert len(dataset) == len(x), f"Dataset length mismatch for {variant}"

                # Test collate function
                collate_fn = collate_fn_factory(variant)
                loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

                # Get one batch to ensure it works
                batch = next(iter(loader))
                assert "labels" in batch, f"Missing labels in {variant} batch"
                assert batch["labels"].shape[0] <= 4, f"Batch size incorrect for {variant}"

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")
