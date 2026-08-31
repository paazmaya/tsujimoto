"""
Unified configuration classes using Pydantic v2.

Provides base configuration and specialized configs for each training methodology
with built-in validation, type safety, and schema generation.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

logger: logging.Logger = logging.getLogger(__name__)

# Default configuration constants
DEFAULT_DATA_DIR = "dataset"
DEFAULT_IMAGE_SIZE = 64
DEFAULT_NUM_CLASSES = 43427  # Combined all ETL dataset
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 30
DEFAULT_RANDOM_SEED = 42


class OptimizationConfig(BaseModel):
    """
    Unified configuration for all optimization approaches.
    Each approach inherits from this and adds approach-specific parameters.
    """

    model_config = {"validate_assignment": True}

    # ========== DATASET PARAMETERS ==========
    data_dir: str = Field(
        default=DEFAULT_DATA_DIR,
        description="Path to dataset directory",
    )
    image_size: int = Field(
        default=DEFAULT_IMAGE_SIZE,
        ge=32,
        le=512,
        description="Size of input images (pixels)",
    )
    num_classes: int = Field(
        default=DEFAULT_NUM_CLASSES,
        ge=1,
        description="Number of character classes",
    )

    # ========== TRAINING HYPERPARAMETERS ==========
    epochs: int = Field(
        default=DEFAULT_EPOCHS,
        ge=1,
        le=1000,
        description="Number of training epochs",
    )
    batch_size: int = Field(
        default=DEFAULT_BATCH_SIZE,
        ge=1,
        le=1024,
        description="Batch size for training",
    )
    learning_rate: float = Field(
        default=DEFAULT_LEARNING_RATE,
        gt=0.0,
        description="Initial learning rate",
    )
    weight_decay: float = Field(
        default=1e-5,
        ge=0.0,
        description="L2 regularization coefficient",
    )

    # ========== TRAIN/VAL/TEST SPLIT ==========
    val_split: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Validation split ratio",
    )
    test_split: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Test split ratio",
    )
    random_seed: int = Field(
        default=DEFAULT_RANDOM_SEED,
        description="Random seed for reproducibility",
    )

    # ========== DATA AUGMENTATION PARAMETERS ==========
    augment_enabled: bool = Field(
        default=True,
        description="Enable data augmentation",
    )
    augment_probability: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Probability of applying augmentation",
    )
    augment_noise_level: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Gaussian noise level for augmentation",
    )

    # ========== OPTIMIZATION ALGORITHM PARAMETERS ==========
    optimizer: str = Field(
        default="adamw",
        pattern="^(adamw|sgd)$",
        description="Optimizer type: adamw or sgd",
    )
    scheduler: str = Field(
        default="cosine",
        pattern="^(cosine|step)$",
        description="Learning rate scheduler type",
    )
    scheduler_t_max: int = Field(
        default=30,
        ge=1,
        description="Max iterations for cosine annealing",
    )

    # ========== DEVICE & LOGGING ==========
    device: str = Field(
        default="cuda",
        pattern="^(cuda|cpu)$",
        description="Training device: cuda or cpu",
    )
    log_interval: int = Field(
        default=100,
        ge=1,
        description="Batches between logs",
    )

    # ========== OUTPUT PATHS ==========
    model_dir: str = Field(
        default="training",
        description="Directory for saving models",
    )
    results_dir: str = Field(
        default="results",
        description="Directory for results",
    )

    @field_validator("data_dir", mode="before")
    @classmethod
    def validate_data_dir(cls, v):
        """Validate that data directory exists or create it."""
        path = Path(v)
        # Don't enforce strict existence check - allow creation during training
        return str(path)

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging/saving (backward compatibility)"""
        return {
            "data_dir": self.data_dir,
            "image_size": self.image_size,
            "num_classes": self.num_classes,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
        }

    def save(self, path: str) -> None:
        """Save configuration to JSON file"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            # Use Pydantic's model_dump_json for better serialization
            f.write(self.model_dump_json(indent=2))
        logger.info(f"✓ Configuration saved to {path}")

    @classmethod
    def load(cls, path: str) -> "OptimizationConfig":
        """Load configuration from JSON file"""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class CNNConfig(OptimizationConfig):
    """Configuration for Lightweight CNN approach."""

    pass


class RNNConfig(OptimizationConfig):
    """
    Configuration for RNN-based approaches.

    Supports 5 model variants with configurable RNN and CNN parameters.
    """

    # ========== MODEL VARIANT SELECTION ==========
    model_variant: str = Field(
        default="hybrid_cnn_rnn",
        pattern="^(basic_rnn|stroke_rnn|simple_radical_rnn|hybrid_cnn_rnn|linguistic_radical_rnn)$",
        description="RNN model variant",
    )

    # ========== RNN PARAMETERS ==========
    rnn_type: str = Field(
        default="lstm",
        pattern="^(lstm|gru)$",
        description="RNN cell type: lstm or gru",
    )
    hidden_size: int = Field(
        default=256,
        ge=32,
        le=2048,
        description="RNN hidden state dimension",
    )
    num_layers: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Number of stacked RNN layers",
    )
    bidirectional: bool = Field(
        default=True,
        description="Use bidirectional RNN",
    )
    dropout: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Dropout rate in RNN layers",
    )

    # ========== RADICAL DECOMPOSITION PARAMETERS ==========
    radical_vocab_size: int = Field(
        default=2000,
        ge=100,
        le=5000,
        description="Radical vocabulary size",
    )
    radical_embedding_dim: int = Field(
        default=128,
        ge=32,
        le=512,
        description="Dimension of radical embeddings",
    )
    radical_encoding_type: str = Field(
        default="binary_tree",
        pattern="^(binary_tree|one_hot|learned)$",
        description="Radical encoding type",
    )

    # ========== CNN BACKBONE PARAMETERS ==========
    cnn_channels: Tuple[int, ...] = Field(
        default=(32, 64, 128),
        description="Channel progression for CNN components",
    )

    def to_dict(self) -> dict:
        """Convert config to dictionary including RNN-specific fields"""
        config = super().to_dict()
        config.update({
            "model_variant": self.model_variant,
            "rnn_type": self.rnn_type,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout,
            "radical_vocab_size": self.radical_vocab_size,
            "radical_embedding_dim": self.radical_embedding_dim,
            "radical_encoding_type": self.radical_encoding_type,
            "cnn_channels": self.cnn_channels,
        })
        return config


class QATConfig(OptimizationConfig):
    """
    Configuration for Quantization-Aware Training (QAT).

    Extends base config with QAT-specific quantization parameters.
    """

    # ========== QAT SPECIFIC PARAMETERS ==========
    qat_backend: str = Field(
        default="fbgemm",
        pattern="^(fbgemm|qnnpack|x86)$",
        description="Quantization backend",
    )
    qat_bits: int = Field(
        default=8,
        ge=4,
        le=16,
        description="Bit width for quantization",
    )
    qat_calibration_batches: int = Field(
        default=32,
        ge=1,
        description="Batches for calibration phase",
    )
    qat_freeze_bn: bool = Field(
        default=True,
        description="Freeze batch norm statistics",
    )
    qat_start_epoch: int = Field(
        default=5,
        ge=1,
        description="Epoch to start QAT phase",
    )
    qat_fine_tune_lr: float = Field(
        default=0.00001,
        gt=0.0,
        description="Learning rate for QAT fine-tuning",
    )

    def to_dict(self) -> dict:
        """Convert config to dictionary including QAT-specific fields"""
        config = super().to_dict()
        config.update({
            "qat_backend": self.qat_backend,
            "qat_bits": self.qat_bits,
            "qat_calibration_batches": self.qat_calibration_batches,
            "qat_freeze_bn": self.qat_freeze_bn,
            "qat_start_epoch": self.qat_start_epoch,
            "qat_fine_tune_lr": self.qat_fine_tune_lr,
        })
        return config


class RadicalRNNConfig(OptimizationConfig):
    """
    Configuration for Radical RNN / Radical Decomposition approach.

    Specializes radical and RNN parameters for decomposition-based training.
    """

    # ========== RADICAL DECOMPOSITION PARAMETERS ==========
    radical_vocab_size: int = Field(
        default=2000,
        ge=100,
        le=5000,
        description="Number of unique radicals",
    )
    radical_embedding_dim: int = Field(
        default=128,
        ge=32,
        le=512,
        description="Dimension for radical embeddings",
    )
    radical_encoding_type: str = Field(
        default="binary_tree",
        pattern="^(binary_tree|one_hot|learned)$",
        description="Radical encoding type",
    )

    # ========== RNN PARAMETERS ==========
    rnn_type: str = Field(
        default="lstm",
        pattern="^(lstm|gru)$",
        description="RNN cell type",
    )
    rnn_hidden_size: int = Field(
        default=256,
        ge=32,
        le=2048,
        description="RNN hidden dimension",
    )
    rnn_num_layers: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Number of RNN layers",
    )
    rnn_dropout: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Dropout in RNN",
    )

    # ========== CNN BACKBONE FOR RADICAL EXTRACTION ==========
    cnn_channels: Tuple[int, ...] = Field(
        default=(32, 64, 128),
        description="Channel progression for CNN backbone",
    )

    def to_dict(self) -> dict:
        """Convert config to dictionary including radical RNN fields"""
        config = super().to_dict()
        config.update({
            "radical_vocab_size": self.radical_vocab_size,
            "radical_embedding_dim": self.radical_embedding_dim,
            "radical_encoding_type": self.radical_encoding_type,
            "rnn_type": self.rnn_type,
            "rnn_hidden_size": self.rnn_hidden_size,
            "rnn_num_layers": self.rnn_num_layers,
            "rnn_dropout": self.rnn_dropout,
            "cnn_channels": self.cnn_channels,
        })
        return config


class HierCodeConfig(OptimizationConfig):
    """
    Configuration for HierCode: Hierarchical Codebook approach.

    Implements hierarchical vector quantization with multi-hot encoding
    and optional prototype learning.
    """

    # ========== HIERCODE SPECIFIC PARAMETERS ==========
    codebook_total_size: int = Field(
        default=1024,
        ge=256,
        le=8192,
        description="Total codebook entries (2^depth)",
    )
    codebook_dim: int = Field(
        default=128,
        ge=32,
        le=512,
        description="Dimension of codebook vectors",
    )
    hierarch_depth: int = Field(
        default=10,
        ge=2,
        le=16,
        description="Depth of binary tree hierarchy",
    )

    # Multi-hot encoding parameters
    multi_hot_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of active codewords per character",
    )
    temperature: float = Field(
        default=0.1,
        gt=0.0,
        description="Gumbel-softmax temperature",
    )

    # Feature extraction backbone
    backbone_type: str = Field(
        default="lightweight_cnn",
        pattern="^(lightweight_cnn|vit_small)$",
        description="Backbone architecture type",
    )
    backbone_output_dim: int = Field(
        default=256,
        ge=64,
        le=1024,
        description="Output dimension from backbone",
    )

    # Prototype learning
    enable_prototype_learning: bool = Field(
        default=True,
        description="Enable prototype learning",
    )
    prototype_learning_weight: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Loss weight for prototype learning",
    )

    # Zero-shot learning parameters
    enable_zero_shot: bool = Field(
        default=True,
        description="Enable zero-shot learning",
    )
    zero_shot_radical_aware: bool = Field(
        default=True,
        description="Use radical decomposition for zero-shot",
    )

    def to_dict(self) -> dict:
        """Convert config to dictionary including HierCode fields"""
        config = super().to_dict()
        config.update({
            "codebook_total_size": self.codebook_total_size,
            "codebook_dim": self.codebook_dim,
            "hierarch_depth": self.hierarch_depth,
            "multi_hot_k": self.multi_hot_k,
            "temperature": self.temperature,
            "backbone_type": self.backbone_type,
            "backbone_output_dim": self.backbone_output_dim,
            "enable_prototype_learning": self.enable_prototype_learning,
            "prototype_learning_weight": self.prototype_learning_weight,
            "enable_zero_shot": self.enable_zero_shot,
            "zero_shot_radical_aware": self.zero_shot_radical_aware,
        })
        return config


class ViTConfig(OptimizationConfig):
    """
    Configuration for Vision Transformer (ViT) approach.

    Uses T2T-ViT concepts for efficient token generation and transformer
    block stacking.
    """

    # ========== VISION TRANSFORMER PARAMETERS ==========
    patch_size: int = Field(
        default=8,
        ge=4,
        le=32,
        description="Patch size for tokenization",
    )
    embedding_dim: int = Field(
        default=256,
        ge=64,
        le=1024,
        description="Transformer embedding dimension",
    )
    num_heads: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Number of attention heads",
    )
    num_transformer_layers: int = Field(
        default=12,
        ge=1,
        le=48,
        description="Number of transformer blocks",
    )
    mlp_dim: int = Field(
        default=1024,
        ge=256,
        le=4096,
        description="MLP hidden dimension",
    )

    # T2T progressive tokenization
    use_tokens_to_tokens: bool = Field(
        default=True,
        description="Use T2T progressive tokenization",
    )
    t2t_kernel_sizes: Tuple[int, ...] = Field(
        default=(3, 3, 3),
        description="Progressive kernel sizes for T2T",
    )

    # Efficiency parameters
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Dropout rate",
    )
    attention_dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Attention dropout rate",
    )

    def to_dict(self) -> dict:
        """Convert config to dictionary including ViT fields"""
        config = super().to_dict()
        config.update({
            "patch_size": self.patch_size,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_transformer_layers": self.num_transformer_layers,
            "mlp_dim": self.mlp_dim,
            "use_tokens_to_tokens": self.use_tokens_to_tokens,
            "t2t_kernel_sizes": self.t2t_kernel_sizes,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
        })
        return config


# Backward compatibility: export dataclass names
__all__ = [
    "OptimizationConfig",
    "CNNConfig",
    "RNNConfig",
    "QATConfig",
    "RadicalRNNConfig",
    "HierCodeConfig",
    "ViTConfig",
]
