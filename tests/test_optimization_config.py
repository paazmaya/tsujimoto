#!/usr/bin/env python3
"""Unit tests for optimization_config module."""

from dataclasses import fields

import pytest

from src.lib.config import OptimizationConfig


class TestOptimizationConfigDefaults:
    """Tests for default configuration values."""

    def test_default_dataset_parameters(self):
        """Test default dataset parameters."""
        config = OptimizationConfig()
        assert config.data_dir == "dataset"
        assert config.image_size == 64
        assert config.num_classes == 43427

    def test_default_training_parameters(self):
        """Test default training hyperparameters."""
        config = OptimizationConfig()
        assert config.epochs == 30
        assert config.batch_size == 64
        assert config.learning_rate == 0.001
        assert config.weight_decay == 1e-5

    def test_default_split_parameters(self):
        """Test default train/val/test split parameters."""
        config = OptimizationConfig()
        assert config.val_split == 0.1
        assert config.test_split == 0.1
        assert config.random_seed == 42

    def test_default_augmentation_parameters(self):
        """Test default data augmentation parameters."""
        config = OptimizationConfig()
        assert config.augment_enabled is True
        assert config.augment_probability == 0.3
        assert config.augment_noise_level == 0.05

    def test_default_optimization_parameters(self):
        """Test default optimization algorithm parameters."""
        config = OptimizationConfig()
        assert config.optimizer == "adamw"
        assert config.scheduler == "cosine"


class TestOptimizationConfigCustomization:
    """Tests for configuration customization."""

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = OptimizationConfig(
            data_dir="custom_dataset",
            image_size=128,
            num_classes=5000,
            epochs=50,
            batch_size=32,
            learning_rate=0.01,
        )

        assert config.data_dir == "custom_dataset"
        assert config.image_size == 128
        assert config.num_classes == 5000
        assert config.epochs == 50
        assert config.batch_size == 32
        assert config.learning_rate == 0.01

    def test_partial_custom_values(self):
        """Test creating config with partial custom values."""
        config = OptimizationConfig(epochs=100, batch_size=128)

        # Custom values
        assert config.epochs == 100
        assert config.batch_size == 128

        # Default values
        assert config.image_size == 64
        assert config.learning_rate == 0.001


class TestOptimizationConfigValidation:
    """Tests for configuration validation."""

    def test_split_parameters_sum_not_exceeding_one(self):
        """Test that train/val/test splits don't exceed 1.0."""
        config = OptimizationConfig(val_split=0.1, test_split=0.1)
        train_split = 1.0 - config.val_split - config.test_split
        assert train_split > 0
        assert train_split + config.val_split + config.test_split == pytest.approx(1.0)

    def test_positive_batch_size(self):
        """Test that batch size is positive."""
        config = OptimizationConfig()
        assert config.batch_size > 0

    def test_positive_epochs(self):
        """Test that epochs is positive."""
        config = OptimizationConfig()
        assert config.epochs > 0

    def test_positive_learning_rate(self):
        """Test that learning rate is positive."""
        config = OptimizationConfig()
        assert config.learning_rate > 0

    def test_augmentation_probability_in_range(self):
        """Test that augmentation probability is in [0, 1]."""
        config = OptimizationConfig()
        assert 0 <= config.augment_probability <= 1

    def test_augmentation_noise_level_positive(self):
        """Test that augmentation noise level is positive."""
        config = OptimizationConfig()
        assert config.augment_noise_level > 0


class TestOptimizationConfigFields:
    """Tests for configuration fields."""

    def test_all_fields_have_defaults(self):
        """Test that all dataclass fields have defaults."""
        config = OptimizationConfig()
        for field in fields(OptimizationConfig):
            # Verify field exists and has a value
            assert hasattr(config, field.name)
            assert getattr(config, field.name) is not None

    def test_field_types(self):
        """Test field types."""
        config = OptimizationConfig()

        # String fields
        assert isinstance(config.data_dir, str)
        assert isinstance(config.optimizer, str)
        assert isinstance(config.scheduler, str)

        # Integer fields
        assert isinstance(config.image_size, int)
        assert isinstance(config.num_classes, int)
        assert isinstance(config.epochs, int)
        assert isinstance(config.batch_size, int)
        assert isinstance(config.random_seed, int)

        # Float fields
        assert isinstance(config.learning_rate, float)
        assert isinstance(config.weight_decay, float)
        assert isinstance(config.val_split, float)
        assert isinstance(config.test_split, float)
        assert isinstance(config.augment_probability, float)
        assert isinstance(config.augment_noise_level, float)

        # Boolean fields
        assert isinstance(config.augment_enabled, bool)


class TestOptimizationConfigReproducibility:
    """Tests for configuration reproducibility."""

    def test_two_configs_with_same_params_are_equal(self):
        """Test that two configs with same parameters are equal."""
        config1 = OptimizationConfig(epochs=50, batch_size=32)
        config2 = OptimizationConfig(epochs=50, batch_size=32)

        assert config1.epochs == config2.epochs
        assert config1.batch_size == config2.batch_size

    def test_random_seed_preserved(self):
        """Test that random seed is preserved."""
        config = OptimizationConfig(random_seed=123)
        assert config.random_seed == 123

        config2 = OptimizationConfig(random_seed=456)
        assert config2.random_seed == 456
        assert config.random_seed != config2.random_seed


class TestOptimizationConfigComputedValues:
    """Tests for computed values from configuration."""

    def test_compute_train_split(self):
        """Test computing training split from config."""
        config = OptimizationConfig(val_split=0.15, test_split=0.1)
        train_split = 1.0 - config.val_split - config.test_split
        assert train_split == pytest.approx(0.75)

    def test_compute_num_augmented_samples(self):
        """Test computing number of augmented samples."""
        config = OptimizationConfig(batch_size=64, augment_probability=0.3)
        augmented_per_batch = int(config.batch_size * config.augment_probability)
        assert augmented_per_batch == int(64 * 0.3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
