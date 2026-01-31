#!/usr/bin/env python3
"""Extended tests for configuration modules to reach 100% coverage."""

import json
import tempfile
from pathlib import Path

from src.lib.config import (
    CNNConfig,
    HierCodeConfig,
    QATConfig,
    RadicalRNNConfig,
    ViTConfig,
)


class TestCNNConfig:
    """Tests for CNNConfig class."""

    def test_cnn_config_inherits_defaults(self):
        """Test that CNNConfig inherits OptimizationConfig defaults."""
        config = CNNConfig()
        assert config.num_classes == 43427
        assert config.batch_size == 64
        assert config.epochs == 30

    def test_cnn_config_to_dict(self):
        """Test CNNConfig serialization."""
        config = CNNConfig(num_classes=1000, batch_size=32)
        config_dict = config.to_dict()
        assert config_dict["num_classes"] == 1000
        assert config_dict["batch_size"] == 32


class TestRNNConfig:
    """Tests for RNNConfig class."""

    def test_rnn_config_inherits_defaults(self):
        """Test that RNNConfig inherits OptimizationConfig defaults."""
        from src.lib.config import RNNConfig

        config = RNNConfig()
        assert config.num_classes == 43427
        assert config.epochs == 30


class TestQATConfig:
    """Tests for QATConfig class."""

    def test_qat_config_defaults(self):
        """Test QAT-specific defaults."""
        config = QATConfig()
        assert config.qat_backend == "fbgemm"
        assert config.qat_bits == 8
        assert config.qat_calibration_batches == 32
        assert config.qat_freeze_bn is True
        assert config.qat_start_epoch == 5

    def test_qat_config_custom_values(self):
        """Test QATConfig with custom values."""
        config = QATConfig(
            qat_backend="qnnpack",
            qat_bits=4,
            qat_start_epoch=10,
        )
        assert config.qat_backend == "qnnpack"
        assert config.qat_bits == 4
        assert config.qat_start_epoch == 10

    def test_qat_config_to_dict(self):
        """Test QATConfig serialization includes QAT fields."""
        config = QATConfig(qat_backend="x86", qat_bits=16)
        config_dict = config.to_dict()
        assert config_dict["qat_backend"] == "x86"
        assert config_dict["qat_bits"] == 16
        assert "qat_calibration_batches" in config_dict
        assert "qat_freeze_bn" in config_dict
        assert "qat_start_epoch" in config_dict


class TestRadicalRNNConfig:
    """Tests for RadicalRNNConfig class."""

    def test_radical_rnn_defaults(self):
        """Test RadicalRNN-specific defaults."""
        config = RadicalRNNConfig()
        assert config.radical_vocab_size == 2000
        assert config.radical_embedding_dim == 128
        assert config.radical_encoding_type == "binary_tree"
        assert config.rnn_type == "lstm"
        assert config.rnn_hidden_size == 256
        assert config.rnn_num_layers == 2
        assert config.rnn_dropout == 0.3
        assert config.cnn_channels == (32, 64, 128)

    def test_radical_rnn_custom_values(self):
        """Test RadicalRNNConfig with custom values."""
        config = RadicalRNNConfig(
            radical_vocab_size=3000,
            rnn_type="gru",
            rnn_hidden_size=512,
            rnn_num_layers=3,
        )
        assert config.radical_vocab_size == 3000
        assert config.rnn_type == "gru"
        assert config.rnn_hidden_size == 512
        assert config.rnn_num_layers == 3

    def test_radical_rnn_to_dict(self):
        """Test RadicalRNNConfig serialization."""
        config = RadicalRNNConfig(
            radical_vocab_size=1500,
            rnn_type="gru",
        )
        config_dict = config.to_dict()
        assert config_dict["radical_vocab_size"] == 1500
        assert config_dict["rnn_type"] == "gru"
        assert "radical_embedding_dim" in config_dict
        assert "rnn_hidden_size" in config_dict
        assert "rnn_num_layers" in config_dict


class TestHierCodeConfig:
    """Tests for HierCodeConfig class."""

    def test_hiercode_defaults(self):
        """Test HierCode-specific defaults."""
        config = HierCodeConfig()
        assert config.codebook_total_size == 1024
        assert config.codebook_dim == 128
        assert config.hierarch_depth == 10
        assert config.multi_hot_k == 5
        assert config.temperature == 0.1
        assert config.backbone_type == "lightweight_cnn"
        assert config.backbone_output_dim == 256
        assert config.enable_prototype_learning is True
        assert config.prototype_learning_weight == 0.1
        assert config.enable_zero_shot is True
        assert config.zero_shot_radical_aware is True

    def test_hiercode_custom_values(self):
        """Test HierCodeConfig with custom values."""
        config = HierCodeConfig(
            codebook_total_size=2048,
            codebook_dim=256,
            hierarch_depth=11,
            multi_hot_k=10,
            enable_prototype_learning=False,
        )
        assert config.codebook_total_size == 2048
        assert config.codebook_dim == 256
        assert config.hierarch_depth == 11
        assert config.multi_hot_k == 10
        assert config.enable_prototype_learning is False

    def test_hiercode_to_dict(self):
        """Test HierCodeConfig serialization."""
        config = HierCodeConfig(
            codebook_total_size=512,
            multi_hot_k=3,
        )
        config_dict = config.to_dict()
        assert config_dict["codebook_total_size"] == 512
        assert config_dict["codebook_dim"] == 128
        assert config_dict["hierarch_depth"] == 10
        assert config_dict["multi_hot_k"] == 3
        assert "enable_prototype_learning" in config_dict
        assert "enable_zero_shot" in config_dict


class TestViTConfig:
    """Tests for ViTConfig class."""

    def test_vit_defaults(self):
        """Test ViT-specific defaults."""
        config = ViTConfig()
        assert config.patch_size == 8
        assert config.embedding_dim == 256
        assert config.num_heads == 8
        assert config.num_transformer_layers == 12
        assert config.mlp_dim == 1024
        assert config.use_tokens_to_tokens is True
        assert config.t2t_kernel_sizes == (3, 3, 3)
        assert config.dropout == 0.1
        assert config.attention_dropout == 0.0

    def test_vit_custom_values(self):
        """Test ViTConfig with custom values."""
        config = ViTConfig(
            patch_size=16,
            embedding_dim=512,
            num_heads=16,
            num_transformer_layers=24,
            use_tokens_to_tokens=False,
        )
        assert config.patch_size == 16
        assert config.embedding_dim == 512
        assert config.num_heads == 16
        assert config.num_transformer_layers == 24
        assert config.use_tokens_to_tokens is False

    def test_vit_to_dict(self):
        """Test ViTConfig serialization."""
        config = ViTConfig(
            patch_size=4,
            embedding_dim=128,
            num_heads=4,
        )
        config_dict = config.to_dict()
        assert config_dict["patch_size"] == 4
        assert config_dict["embedding_dim"] == 128
        assert config_dict["num_heads"] == 4
        assert "num_transformer_layers" in config_dict
        assert "use_tokens_to_tokens" in config_dict


class TestConfigSaveLoad:
    """Tests for configuration saving and loading."""

    def test_save_config_to_file(self):
        """Test saving configuration to JSON file."""
        config = QATConfig(num_classes=1000, batch_size=32)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config.save(str(config_path))

            assert config_path.exists()

            # Verify content
            with open(config_path) as f:
                loaded = json.load(f)
                assert loaded["num_classes"] == 1000
                assert loaded["batch_size"] == 32

    def test_save_config_creates_parent_dirs(self):
        """Test that save creates parent directories."""
        config = CNNConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nested" / "deep" / "config.json"
            config.save(str(config_path))

            assert config_path.exists()
            assert config_path.parent.exists()

    def test_all_config_types_can_save(self):
        """Test that all config types can be saved."""
        configs = [
            CNNConfig(),
            QATConfig(),
            RadicalRNNConfig(),
            HierCodeConfig(),
            ViTConfig(),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, config in enumerate(configs):
                config_path = Path(tmpdir) / f"config_{i}.json"
                config.save(str(config_path))
                assert config_path.exists()
