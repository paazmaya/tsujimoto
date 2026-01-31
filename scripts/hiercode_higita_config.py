#!/usr/bin/env python3
"""
Configuration module for HierCode + Hi-GITA training

Extends optimization_config.py with Hi-GITA specific settings.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import torch

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import setup_logger

logger = setup_logger(__name__)


@dataclass
class HiGITAImageEncoderConfig:
    """Configuration for Hi-GITA multi-granularity image encoder"""

    stroke_dim: int = 128
    """Dimensionality of stroke-level embeddings"""

    radical_dim: int = 256
    """Dimensionality of radical-level embeddings"""

    character_dim: int = 512
    """Dimensionality of character-level embeddings"""

    num_radicals: int = 16
    """Number of radical components to extract"""

    num_patches: int = 64
    """Number of patches for stroke extraction (8x8 patches from 64x64 image)"""


@dataclass
class HiGITATextEncoderConfig:
    """Configuration for Hi-GITA multi-granularity text encoder"""

    num_strokes: int = 20
    """Number of different stroke types"""

    num_radicals: int = 214
    """Number of CJK radicals (Kangxi radicals)"""

    stroke_dim: int = 128
    """Dimensionality of stroke embeddings"""

    radical_dim: int = 256
    """Dimensionality of radical embeddings"""


@dataclass
class HiGITAContrastiveLossConfig:
    """Configuration for Hi-GITA fine-grained contrastive loss"""

    temperature: float = 0.07
    """Temperature parameter for softmax in contrastive loss"""

    weight_stroke: float = 0.3
    """Weight for stroke-level contrastive loss"""

    weight_radical: float = 0.5
    """Weight for radical-level contrastive loss"""

    weight_character: float = 0.2
    """Weight for character-level contrastive loss"""

    total_weight: float = 0.5
    """Weight of contrastive loss in total training objective"""


@dataclass
class HiGITATrainingConfig:
    """Configuration for Hi-GITA enhanced HierCode training"""

    # Model configuration
    use_higita_enhancement: bool = True
    """Enable Hi-GITA enhancement (set to False for standard HierCode)"""

    num_classes: int = 43427
    """Number of character classes (combined ETL6-9 dataset)"""

    image_encoder: HiGITAImageEncoderConfig = field(default_factory=HiGITAImageEncoderConfig)
    """Image encoder configuration"""

    text_encoder: HiGITATextEncoderConfig = field(default_factory=HiGITATextEncoderConfig)
    """Text encoder configuration"""

    contrastive_loss: HiGITAContrastiveLossConfig = field(
        default_factory=HiGITAContrastiveLossConfig
    )
    """Contrastive loss configuration"""

    # Training configuration
    batch_size: int = 32
    """Batch size for training"""

    learning_rate: float = 0.001
    """Initial learning rate"""

    weight_decay: float = 1e-5
    """L2 regularization weight"""

    epochs: int = 30
    """Total number of training epochs"""

    warmup_epochs: int = 2
    """Number of warmup epochs with lower learning rate"""

    gradient_clip: float = 1.0
    """Maximum gradient norm for clipping"""

    # Data configuration
    train_val_split: float = 0.9
    """Fraction of data to use for training (rest for validation)"""

    random_seed: int = 42
    """Random seed for reproducibility"""

    # Device configuration
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    """Torch device (cuda or cpu)"""

    # Checkpoint configuration
    checkpoint_dir: str = "training/higita/checkpoints"
    """Directory for saving checkpoints"""

    save_interval: int = 1
    """Save checkpoint every N epochs"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "use_higita": self.use_higita_enhancement,
            "num_classes": self.num_classes,
            "image_encoder": {
                "stroke_dim": self.image_encoder.stroke_dim,
                "radical_dim": self.image_encoder.radical_dim,
                "character_dim": self.image_encoder.character_dim,
                "num_radicals": self.image_encoder.num_radicals,
            },
            "text_encoder": {
                "num_strokes": self.text_encoder.num_strokes,
                "num_radicals": self.text_encoder.num_radicals,
            },
            "contrastive_loss": {
                "temperature": self.contrastive_loss.temperature,
                "weight_stroke": self.contrastive_loss.weight_stroke,
                "weight_radical": self.contrastive_loss.weight_radical,
                "weight_character": self.contrastive_loss.weight_character,
                "total_weight": self.contrastive_loss.total_weight,
            },
            "training": {
                "batch_size": self.batch_size,
                "lr": self.learning_rate,
                "epochs": self.epochs,
                "device": self.device,
            },
        }

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "HiGITATrainingConfig":
        """Create config from dictionary"""
        return HiGITATrainingConfig(**config_dict)


# Default configurations for different scenarios


def get_higita_full_config() -> HiGITATrainingConfig:
    """Full Hi-GITA enhancement (highest quality)"""
    return HiGITATrainingConfig(
        use_higita_enhancement=True,
        image_encoder=HiGITAImageEncoderConfig(
            stroke_dim=256,
            radical_dim=512,
            character_dim=1024,
        ),
        batch_size=16,
        epochs=50,
    )


def get_higita_balanced_config() -> HiGITATrainingConfig:
    """Balanced Hi-GITA (good quality, reasonable speed)"""
    return HiGITATrainingConfig(
        use_higita_enhancement=True,
        batch_size=32,
        epochs=30,
    )


def get_higita_lite_config() -> HiGITATrainingConfig:
    """Lightweight Hi-GITA (speed optimized)"""
    return HiGITATrainingConfig(
        use_higita_enhancement=True,
        image_encoder=HiGITAImageEncoderConfig(
            stroke_dim=64,
            radical_dim=128,
            character_dim=256,
        ),
        batch_size=64,
        epochs=20,
    )


def get_standard_hiercode_config() -> HiGITATrainingConfig:
    """Standard HierCode without Hi-GITA enhancement"""
    return HiGITATrainingConfig(
        use_higita_enhancement=False,
        batch_size=128,
        epochs=30,
    )


if __name__ == "__main__":
    # Show available configs

    configs = {
        "full": get_higita_full_config(),
        "balanced": get_higita_balanced_config(),
        "lite": get_higita_lite_config(),
        "standard": get_standard_hiercode_config(),
    }

    logger.info("Available configurations:")
    for name, config in configs.items():
        logger.info("  - %s: %s parameters", name, len(config.__dict__))
