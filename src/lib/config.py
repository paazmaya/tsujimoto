"""
Unified configuration classes for all training approaches.

Provides base configuration and specialized configs for each training methodology.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

logger: logging.Logger = logging.getLogger(__name__)

# Default configuration constants
DEFAULT_DATA_DIR = "dataset"
DEFAULT_IMAGE_SIZE = 64
DEFAULT_NUM_CLASSES = 43427  # Combined all ETL dataset
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 30
DEFAULT_RANDOM_SEED = 42


@dataclass
class OptimizationConfig:
    """
    Unified configuration for all optimization approaches.
    Each approach inherits from this and adds approach-specific parameters.
    """

    # ========== DATASET PARAMETERS ==========
    data_dir: str = DEFAULT_DATA_DIR
    image_size: int = DEFAULT_IMAGE_SIZE
    num_classes: int = (
        DEFAULT_NUM_CLASSES  # Kanji classes (combined_all_etl dataset - from metadata.json)
    )

    # ========== TRAINING HYPERPARAMETERS ==========
    epochs: int = DEFAULT_EPOCHS  # Complete passes through dataset
    batch_size: int = DEFAULT_BATCH_SIZE  # Samples per batch
    learning_rate: float = DEFAULT_LEARNING_RATE  # Initial learning rate
    weight_decay: float = 1e-5  # L2 regularization coefficient

    # ========== TRAIN/VAL/TEST SPLIT ==========
    val_split: float = 0.1  # 10% validation
    test_split: float = 0.1  # 10% test
    random_seed: int = 42  # For reproducibility

    # ========== DATA AUGMENTATION PARAMETERS ==========
    augment_enabled: bool = True  # Enable/disable augmentation
    augment_probability: float = 0.3  # 30% of samples augmented
    augment_noise_level: float = 0.05  # 5% Gaussian noise

    # ========== OPTIMIZATION ALGORITHM PARAMETERS ==========
    optimizer: str = "adamw"  # adamw or sgd
    scheduler: str = "cosine"  # cosine or step
    scheduler_t_max: int = 30  # For cosine annealing (usually = epochs)

    # ========== DEVICE & LOGGING ==========
    device: str = "cuda"  # GPU required (verified at startup)
    log_interval: int = 100  # Batches between logs

    # ========== OUTPUT PATHS ==========
    model_dir: str = "training"
    results_dir: str = "results"

    def to_dict(self) -> Dict:
        """Convert config to dictionary for logging/saving"""
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
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"✓ Configuration saved to {path}")


@dataclass
class CNNConfig(OptimizationConfig):
    """Configuration for Lightweight CNN approach."""


@dataclass
class RNNConfig(OptimizationConfig):
    """Configuration for RNN-based approaches."""


@dataclass
class QATConfig(OptimizationConfig):
    """
    Configuration for Quantization-Aware Training (QAT).

    Key parameters:
    - qat_backend: Which quantization backend to use
    - qat_bits: Bit width (8 for INT8 standard)
    - qat_epochs: Usually fewer than full training
    """

    # ========== QAT SPECIFIC PARAMETERS ==========
    qat_backend: str = "fbgemm"  # fbgemm (Intel CPU), qnnpack (mobile), x86 (server)
    qat_bits: int = 8  # INT8 quantization
    qat_calibration_batches: int = 32  # Batches for calibration phase
    qat_freeze_bn: bool = True  # Freeze batch norm statistics
    qat_start_epoch: int = 5  # Start QAT after warming up model
    qat_fine_tune_lr: float = 0.00001  # Reduced learning rate for QAT phase

    def to_dict(self) -> Dict:
        config = super().to_dict()
        config.update(
            {
                "qat_backend": self.qat_backend,
                "qat_bits": self.qat_bits,
                "qat_calibration_batches": self.qat_calibration_batches,
                "qat_freeze_bn": self.qat_freeze_bn,
                "qat_start_epoch": self.qat_start_epoch,
            }
        )
        return config


@dataclass
class RadicalRNNConfig(OptimizationConfig):
    """
    Configuration for Radical RNN / Radical Decomposition approach.

    Key parameters:
    - radical_vocab_size: Number of unique radicals
    - radical_embedding_dim: Dimension of radical embeddings
    - rnn_hidden_size: RNN hidden state dimension
    - rnn_num_layers: Number of RNN layers
    """

    # ========== RADICAL DECOMPOSITION PARAMETERS ==========
    radical_vocab_size: int = 2000  # Increased from 500 for better character discrimination
    radical_embedding_dim: int = 128  # Dimension for radical embeddings
    radical_encoding_type: str = "binary_tree"  # binary_tree, one_hot, or learned

    # ========== RNN PARAMETERS ==========
    rnn_type: str = "lstm"  # lstm or gru
    rnn_hidden_size: int = 256  # RNN hidden dimension
    rnn_num_layers: int = 2  # Number of stacked RNN layers
    rnn_dropout: float = 0.3  # Dropout in RNN

    # ========== CNN BACKBONE FOR RADICAL EXTRACTION ==========
    cnn_channels: Tuple[int, ...] = (32, 64, 128)  # Channel progression

    def to_dict(self) -> Dict:
        config = super().to_dict()
        config.update(
            {
                "radical_vocab_size": self.radical_vocab_size,
                "radical_embedding_dim": self.radical_embedding_dim,
                "rnn_type": self.rnn_type,
                "rnn_hidden_size": self.rnn_hidden_size,
                "rnn_num_layers": self.rnn_num_layers,
            }
        )
        return config


@dataclass
class HierCodeConfig(OptimizationConfig):
    """
    Configuration for HierCode: Hierarchical Codebook approach.
    Based on arXiv:2403.13761

    Key parameters:
    - codebook_size: Total number of codewords
    - codebook_dim: Dimension of each codeword
    - hierarch_depth: Depth of binary tree hierarchy
    - prototype_learning: Enable prototype learning
    """

    # ========== HIERCODE SPECIFIC PARAMETERS ==========
    codebook_total_size: int = 1024  # Total codebook entries
    codebook_dim: int = 128  # Dimension of codebook vectors
    hierarch_depth: int = 10  # Depth of binary tree (2^10 = 1024 leaf nodes)

    # Multi-hot encoding parameters
    multi_hot_k: int = 5  # Number of active codewords per character (multi-hot)
    temperature: float = 0.1  # Gumbel-softmax temperature

    # Feature extraction backbone
    backbone_type: str = "lightweight_cnn"  # lightweight_cnn or vit_small
    backbone_output_dim: int = 256  # Output dimension from backbone

    # Prototype learning
    enable_prototype_learning: bool = True
    prototype_learning_weight: float = 0.1  # Loss weight for prototype learning

    # Zero-shot learning parameters
    enable_zero_shot: bool = True
    zero_shot_radical_aware: bool = True  # Use radical decomposition for zero-shot

    def to_dict(self) -> Dict:
        config = super().to_dict()
        config.update(
            {
                "codebook_total_size": self.codebook_total_size,
                "codebook_dim": self.codebook_dim,
                "hierarch_depth": self.hierarch_depth,
                "multi_hot_k": self.multi_hot_k,
                "enable_prototype_learning": self.enable_prototype_learning,
                "enable_zero_shot": self.enable_zero_shot,
            }
        )
        return config


@dataclass
class ViTConfig(OptimizationConfig):
    """
    Configuration for Vision Transformer (ViT) approach.
    Uses T2T-ViT concepts for efficiency.

    Key parameters:
    - patch_size: Size of image patches
    - embedding_dim: Transformer embedding dimension
    - num_heads: Number of attention heads
    - num_layers: Number of transformer blocks
    """

    # ========== VISION TRANSFORMER PARAMETERS ==========
    patch_size: int = 8  # Divide 64x64 image into 8x8 patches = 64 tokens
    embedding_dim: int = 256  # Transformer embedding dimension
    num_heads: int = 8  # Multi-head attention heads
    num_transformer_layers: int = 12  # Number of transformer blocks
    mlp_dim: int = 1024  # MLP hidden dimension

    # T2T progressive tokenization
    use_tokens_to_tokens: bool = True  # Progressive tokenization for efficiency
    t2t_kernel_sizes: Tuple[int, ...] = (3, 3, 3)  # Progressive kernel sizes

    # Efficiency parameters
    dropout: float = 0.1
    attention_dropout: float = 0.0

    def to_dict(self) -> Dict:
        config = super().to_dict()
        config.update(
            {
                "patch_size": self.patch_size,
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "num_transformer_layers": self.num_transformer_layers,
                "use_tokens_to_tokens": self.use_tokens_to_tokens,
            }
        )
        return config
