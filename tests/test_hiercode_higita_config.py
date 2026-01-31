#!/usr/bin/env python3
"""Unit tests for hiercode_higita_config module."""

import pytest

from scripts.hiercode_higita_config import (
    HiGITAContrastiveLossConfig,
    HiGITAImageEncoderConfig,
    HiGITATextEncoderConfig,
    HiGITATrainingConfig,
    get_higita_balanced_config,
    get_higita_full_config,
    get_higita_lite_config,
    get_standard_hiercode_config,
)


class TestHiGITAImageEncoderConfigDefaults:
    """Tests for HiGITA image encoder configuration defaults."""

    def test_default_stroke_dim(self):
        """Test default stroke dimension."""
        config = HiGITAImageEncoderConfig()
        assert config.stroke_dim == 128

    def test_default_radical_dim(self):
        """Test default radical dimension."""
        config = HiGITAImageEncoderConfig()
        assert config.radical_dim == 256

    def test_default_character_dim(self):
        """Test default character dimension."""
        config = HiGITAImageEncoderConfig()
        assert config.character_dim == 512

    def test_default_num_radicals(self):
        """Test default number of radicals."""
        config = HiGITAImageEncoderConfig()
        assert config.num_radicals == 16

    def test_default_num_patches(self):
        """Test default number of patches."""
        config = HiGITAImageEncoderConfig()
        assert config.num_patches == 64

    def test_all_defaults(self):
        """Test all default values together."""
        config = HiGITAImageEncoderConfig()
        assert config.stroke_dim == 128
        assert config.radical_dim == 256
        assert config.character_dim == 512
        assert config.num_radicals == 16
        assert config.num_patches == 64


class TestHiGITAImageEncoderCustomization:
    """Tests for HiGITA image encoder customization."""

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = HiGITAImageEncoderConfig(
            stroke_dim=256, radical_dim=512, character_dim=1024, num_radicals=32, num_patches=128
        )

        assert config.stroke_dim == 256
        assert config.radical_dim == 512
        assert config.character_dim == 1024
        assert config.num_radicals == 32
        assert config.num_patches == 128

    def test_partial_custom_values(self):
        """Test creating config with partial custom values."""
        config = HiGITAImageEncoderConfig(stroke_dim=256, num_radicals=8)

        assert config.stroke_dim == 256
        assert config.num_radicals == 8
        # Defaults should still be used
        assert config.radical_dim == 256
        assert config.character_dim == 512


class TestHiGITATextEncoderConfigDefaults:
    """Tests for HiGITA text encoder configuration defaults."""

    def test_default_num_strokes(self):
        """Test default number of stroke types."""
        config = HiGITATextEncoderConfig()
        assert config.num_strokes == 20

    def test_default_num_radicals(self):
        """Test default number of radicals (Kangxi)."""
        config = HiGITATextEncoderConfig()
        assert config.num_radicals == 214

    def test_default_stroke_dim(self):
        """Test default stroke embedding dimension."""
        config = HiGITATextEncoderConfig()
        assert config.stroke_dim == 128

    def test_default_radical_dim(self):
        """Test default radical embedding dimension."""
        config = HiGITATextEncoderConfig()
        assert config.radical_dim == 256

    def test_all_defaults(self):
        """Test all default values together."""
        config = HiGITATextEncoderConfig()
        assert config.num_strokes == 20
        assert config.num_radicals == 214
        assert config.stroke_dim == 128
        assert config.radical_dim == 256


class TestHiGITATextEncoderCustomization:
    """Tests for HiGITA text encoder customization."""

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = HiGITATextEncoderConfig(
            num_strokes=30, num_radicals=300, stroke_dim=256, radical_dim=512
        )

        assert config.num_strokes == 30
        assert config.num_radicals == 300
        assert config.stroke_dim == 256
        assert config.radical_dim == 512


class TestHiGITAContrastiveLossConfigDefaults:
    """Tests for HiGITA contrastive loss configuration defaults."""

    def test_default_temperature(self):
        """Test default temperature parameter."""
        config = HiGITAContrastiveLossConfig()
        assert config.temperature == 0.07

    def test_default_weights(self):
        """Test default loss weights."""
        config = HiGITAContrastiveLossConfig()
        assert config.weight_stroke == 0.3
        assert config.weight_radical == 0.5
        assert config.weight_character == 0.2
        assert config.total_weight == 0.5

    def test_weights_sum_to_one(self):
        """Test that granularity weights sum to 1.0."""
        config = HiGITAContrastiveLossConfig()
        granularity_sum = config.weight_stroke + config.weight_radical + config.weight_character
        assert abs(granularity_sum - 1.0) < 0.01


class TestHiGITATrainingConfigDefaults:
    """Tests for HiGITA training configuration defaults."""

    def test_default_num_classes(self):
        """Test default number of classes."""
        config = HiGITATrainingConfig()
        assert config.num_classes == 43427

    def test_default_use_higita(self):
        """Test that Hi-GITA enhancement is enabled by default."""
        config = HiGITATrainingConfig()
        assert config.use_higita_enhancement is True

    def test_default_batch_size(self):
        """Test default batch size."""
        config = HiGITATrainingConfig()
        assert config.batch_size == 32

    def test_default_epochs(self):
        """Test default number of epochs."""
        config = HiGITATrainingConfig()
        assert config.epochs == 30

    def test_has_encoder_configs(self):
        """Test that training config includes encoder configurations."""
        config = HiGITATrainingConfig()
        assert isinstance(config.image_encoder, HiGITAImageEncoderConfig)
        assert isinstance(config.text_encoder, HiGITATextEncoderConfig)


class TestDimensionalityConsistency:
    """Tests for dimensional consistency across configs."""

    def test_image_encoder_dimensions(self):
        """Test that image encoder dimensions are consistent."""
        config = HiGITAImageEncoderConfig()
        # Stroke dim should be smaller than radical dim
        assert config.stroke_dim < config.radical_dim
        # Radical dim should be smaller than character dim
        assert config.radical_dim < config.character_dim

    def test_text_encoder_dimensions(self):
        """Test that text encoder dimensions are consistent."""
        config = HiGITATextEncoderConfig()
        # Stroke dim should be smaller than radical dim
        assert config.stroke_dim < config.radical_dim

    def test_image_text_encoder_stroke_dims_match(self):
        """Test that stroke dims match between image and text encoders."""
        image_config = HiGITAImageEncoderConfig()
        text_config = HiGITATextEncoderConfig()
        # Can have same stroke_dim for consistency
        assert image_config.stroke_dim == text_config.stroke_dim


class TestPatchCalculations:
    """Tests for patch-related calculations."""

    def test_patches_match_image_size(self):
        """Test that number of patches is reasonable for image size."""
        image_size = 64
        num_patches = 64
        # 64x64 image with 64 patches = 8x8 grid of patches
        patches_per_side = int(num_patches**0.5)
        patch_size = image_size // patches_per_side
        assert patch_size == 8  # 64 / 8 = 8

    def test_patches_are_square(self):
        """Test that patches form a square grid."""
        config = HiGITAImageEncoderConfig()
        patches_per_side = int(config.num_patches**0.5)
        assert patches_per_side * patches_per_side == config.num_patches


class TestRadicalCounts:
    """Tests for radical-related counts."""

    def test_text_encoder_kangxi_radicals(self):
        """Test that text encoder uses correct Kangxi radical count."""
        config = HiGITATextEncoderConfig()
        # Kangxi radical set has 214 radicals
        assert config.num_radicals == 214

    def test_image_encoder_fewer_radicals_than_text(self):
        """Test that image encoder uses fewer radicals than text encoder."""
        image_config = HiGITAImageEncoderConfig()
        text_config = HiGITATextEncoderConfig()
        # Image encoder extracts fewer synthetic radicals
        assert image_config.num_radicals < text_config.num_radicals


class TestStrokeTypes:
    """Tests for stroke-related configurations."""

    def test_text_encoder_has_stroke_types(self):
        """Test that text encoder defines stroke types."""
        config = HiGITATextEncoderConfig()
        assert config.num_strokes > 0
        # Common stroke types in CJK: dot, horizontal, vertical, etc.
        assert config.num_strokes >= 5


class TestConfigFieldTypes:
    """Tests for configuration field types."""

    def test_image_encoder_field_types(self):
        """Test field types in image encoder config."""
        config = HiGITAImageEncoderConfig()
        assert isinstance(config.stroke_dim, int)
        assert isinstance(config.radical_dim, int)
        assert isinstance(config.character_dim, int)
        assert isinstance(config.num_radicals, int)
        assert isinstance(config.num_patches, int)

    def test_text_encoder_field_types(self):
        """Test field types in text encoder config."""
        config = HiGITATextEncoderConfig()
        assert isinstance(config.num_strokes, int)
        assert isinstance(config.num_radicals, int)
        assert isinstance(config.stroke_dim, int)
        assert isinstance(config.radical_dim, int)


class TestConfigPresets:
    """Tests for configuration factory functions."""

    def test_full_config_preset(self):
        """Test full Hi-GITA configuration preset."""
        config = get_higita_full_config()
        assert config.use_higita_enhancement is True
        assert config.image_encoder.stroke_dim == 256
        assert config.image_encoder.radical_dim == 512
        assert config.image_encoder.character_dim == 1024
        assert config.batch_size == 16
        assert config.epochs == 50

    def test_balanced_config_preset(self):
        """Test balanced Hi-GITA configuration preset."""
        config = get_higita_balanced_config()
        assert config.use_higita_enhancement is True
        assert config.batch_size == 32
        assert config.epochs == 30

    def test_lite_config_preset(self):
        """Test lightweight Hi-GITA configuration preset."""
        config = get_higita_lite_config()
        assert config.use_higita_enhancement is True
        assert config.image_encoder.stroke_dim == 64
        assert config.image_encoder.radical_dim == 128
        assert config.image_encoder.character_dim == 256
        assert config.batch_size == 64
        assert config.epochs == 20

    def test_standard_hiercode_preset(self):
        """Test standard HierCode configuration (no Hi-GITA)."""
        config = get_standard_hiercode_config()
        assert config.use_higita_enhancement is False
        assert config.batch_size == 128
        assert config.epochs == 30


class TestConfigConversions:
    """Tests for configuration conversion methods."""

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = HiGITATrainingConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "use_higita" in config_dict
        assert config_dict["use_higita"] is True
        assert "num_classes" in config_dict
        assert config_dict["num_classes"] == 43427

    def test_to_dict_includes_encoders(self):
        """Test that to_dict includes encoder configurations."""
        config = HiGITATrainingConfig()
        config_dict = config.to_dict()

        assert "image_encoder" in config_dict
        assert "text_encoder" in config_dict
        assert config_dict["image_encoder"]["stroke_dim"] == 128
        assert config_dict["text_encoder"]["num_radicals"] == 214

    def test_to_dict_includes_contrastive_loss(self):
        """Test that to_dict includes contrastive loss configuration."""
        config = HiGITATrainingConfig()
        config_dict = config.to_dict()

        assert "contrastive_loss" in config_dict
        assert config_dict["contrastive_loss"]["temperature"] == 0.07

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "use_higita_enhancement": False,
            "num_classes": 2000,
            "batch_size": 64,
            "epochs": 20,
        }
        config = HiGITATrainingConfig.from_dict(config_dict)

        assert config.use_higita_enhancement is False
        assert config.num_classes == 2000
        assert config.batch_size == 64
        assert config.epochs == 20

    def test_round_trip_conversion(self):
        """Test converting to dict and back."""
        original = HiGITATrainingConfig(
            batch_size=48,
            epochs=25,
            learning_rate=0.0005,
        )
        config_dict = original.to_dict()
        # Extract just the top-level fields that from_dict expects
        config_dict_filtered = {
            "use_higita_enhancement": config_dict["use_higita"],
            "num_classes": config_dict["num_classes"],
            "batch_size": config_dict["training"]["batch_size"],
            "epochs": config_dict["training"]["epochs"],
        }
        restored = HiGITATrainingConfig.from_dict(config_dict_filtered)

        assert restored.batch_size == original.batch_size
        assert restored.epochs == original.epochs


class TestHyperparameterValidation:
    """Tests for hyperparameter validation."""

    def test_temperature_is_positive(self):
        """Test that temperature is a positive value."""
        config = HiGITAContrastiveLossConfig()
        assert config.temperature > 0

    def test_batch_size_is_positive(self):
        """Test that batch size is positive."""
        config = HiGITATrainingConfig()
        assert config.batch_size > 0

    def test_learning_rate_is_positive(self):
        """Test that learning rate is positive."""
        config = HiGITATrainingConfig()
        assert config.learning_rate > 0

    def test_epochs_is_positive(self):
        """Test that number of epochs is positive."""
        config = HiGITATrainingConfig()
        assert config.epochs > 0

    def test_weight_decay_is_non_negative(self):
        """Test that weight decay is non-negative."""
        config = HiGITATrainingConfig()
        assert config.weight_decay >= 0

    def test_train_val_split_is_valid_proportion(self):
        """Test that train/val split is between 0 and 1."""
        config = HiGITATrainingConfig()
        assert 0 < config.train_val_split < 1


class TestDeviceConfiguration:
    """Tests for device configuration."""

    def test_device_is_string(self):
        """Test that device is a string."""
        config = HiGITATrainingConfig()
        assert isinstance(config.device, str)

    def test_device_is_valid_torch_device(self):
        """Test that device is either cuda or cpu."""
        config = HiGITATrainingConfig()
        assert config.device in ["cuda", "cpu"]

    def test_custom_device(self):
        """Test setting custom device."""
        config = HiGITATrainingConfig(device="cpu")
        assert config.device == "cpu"


class TestCheckpointConfiguration:
    """Tests for checkpoint-related configuration."""

    def test_checkpoint_dir_exists(self):
        """Test that checkpoint directory is configured."""
        config = HiGITATrainingConfig()
        assert config.checkpoint_dir is not None

    def test_save_interval_is_positive(self):
        """Test that save interval is positive."""
        config = HiGITATrainingConfig()
        assert config.save_interval > 0

    def test_checkpoint_dir_customization(self):
        """Test customizing checkpoint directory."""
        custom_dir = "custom/checkpoint/path"
        config = HiGITATrainingConfig(checkpoint_dir=custom_dir)
        assert config.checkpoint_dir == custom_dir


class TestGradientConfiguration:
    """Tests for gradient-related configuration."""

    def test_gradient_clip_is_positive(self):
        """Test that gradient clip value is positive."""
        config = HiGITATrainingConfig()
        assert config.gradient_clip > 0

    def test_warmup_epochs_less_than_total_epochs(self):
        """Test that warmup epochs is less than total epochs."""
        config = HiGITATrainingConfig()
        assert config.warmup_epochs < config.epochs

    def test_warmup_configuration(self):
        """Test warmup epoch configuration."""
        config = HiGITATrainingConfig(epochs=50, warmup_epochs=5)
        assert config.warmup_epochs == 5
        assert config.epochs == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
