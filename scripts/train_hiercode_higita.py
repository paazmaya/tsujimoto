#!/usr/bin/env python3
"""
HierCode Training with Optional Hi-GITA Enhancement
=====================================================

Trains HierCode model with optional Hi-GITA improvements:
- Multi-granularity image encoding (strokes → radicals → character)
- Contrastive image-text alignment
- Fine-grained fusion modules

Features:
- Automatic checkpoint management with resume from latest checkpoint
- Dataset auto-detection (combined_all_etl, etl9g, etl8g, etl7, etl6, etl1)
- NVIDIA GPU required with CUDA optimizations enabled

Usage:
    python scripts/train_hiercode_higita.py --data-dir dataset

Author: Enhancement for tsujimoto
Date: November 17, 2025
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import (
    CheckpointManager,
    get_dataset_directory,
    get_optimizer,
    get_scheduler,
    prepare_dataset_and_loaders,
    save_best_model,
    setup_checkpoint_arguments,
    setup_logger,
    verify_and_setup_gpu,
)

# Add scripts to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from hiercode_higita_enhancement import (
    FineGrainedContrastiveLoss,
    HierCodeWithHiGITA,
    MultiGranularityTextEncoder,
)

logger = setup_logger(__name__)


class HiGITAConfig:
    """Hi-GITA enhancement configuration"""

    def __init__(self):
        self.stroke_dim = 128
        self.radical_dim = 256
        self.character_dim = 512
        self.num_radicals = 16

        # Contrastive learning
        self.contrastive_weight = 0.1  # Auxiliary loss weight (reduced from 0.5 for better accuracy)
        self.temperature = 0.07

        # Text encoder
        self.num_strokes = 20
        self.num_hierarcical_radicals = 214

        # Training
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 30
        self.warmup_epochs = 2
        self.weight_decay = 1e-5

        # Checkpoint
        self.checkpoint_dir = "training/hiercode_higita/checkpoints"

        # Dataset (for character mapping generation)
        self.data_dir = "dataset"

    def to_dict(self) -> dict:
        return dict(self.__dict__.items())


def create_synthetic_text_data(
    labels: np.ndarray, num_samples: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic text representations (stroke and radical codes)

    For real application, this would come from CJK radical decomposition database.
    Currently generates random representations for proof-of-concept.
    """
    if num_samples is None:
        num_samples = len(labels)

    # Synthetic stroke codes (each character has ~5-15 strokes)
    stroke_lengths = np.random.randint(5, 16, num_samples)
    max_strokes = max(stroke_lengths)
    stroke_codes = np.zeros((num_samples, max_strokes), dtype=np.int64)
    for i, length in enumerate(stroke_lengths):
        stroke_codes[i, :length] = np.random.randint(0, 20, length)

    # Synthetic radical codes (each character has ~1-6 radicals)
    radical_lengths = np.random.randint(1, 7, num_samples)
    max_radicals = max(radical_lengths)
    radical_codes = np.zeros((num_samples, max_radicals), dtype=np.int64)
    for i, length in enumerate(radical_lengths):
        radical_codes[i, :length] = np.random.randint(0, 214, length)

    return torch.from_numpy(stroke_codes), torch.from_numpy(radical_codes)


def train_epoch(
    model: HierCodeWithHiGITA,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    args: argparse.Namespace,
    config: HiGITAConfig,
    text_encoder: Optional[MultiGranularityTextEncoder] = None,
    contrastive_loss_fn: Optional[FineGrainedContrastiveLoss] = None,
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()

    total_loss = 0
    classification_loss = 0
    contrastive_loss = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        output = model(images)
        logits = output["logits"]

        # Classification loss
        ce_loss = criterion(logits, labels)

        # Optionally add contrastive loss
        total_batch_loss = ce_loss
        if text_encoder and contrastive_loss_fn:
            # Generate synthetic text for this batch
            stroke_codes, radical_codes = create_synthetic_text_data(
                labels.cpu().numpy(), len(labels)
            )
            stroke_codes = stroke_codes.to(device)
            radical_codes = radical_codes.to(device)

            # Get text encodings
            text_output = text_encoder(stroke_codes, radical_codes)

            # Compute contrastive loss
            losses_dict = contrastive_loss_fn(output["features"], text_output)
            con_loss = losses_dict["total_loss"]

            # Combined loss
            total_batch_loss = ce_loss + config.contrastive_weight * con_loss
            contrastive_loss += con_loss.item()

        # Backward pass
        optimizer.zero_grad()
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        total_loss += total_batch_loss.item()
        classification_loss += ce_loss.item()

        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Progress
        if (batch_idx + 1) % 10 == 0:
            batch_acc = 100 * correct / total
            avg_loss = total_loss / (batch_idx + 1)
            logger.info(
                f"  Epoch {epoch} [{batch_idx + 1:3d}/{len(train_loader):3d}] "
                f"Loss: {avg_loss:.4f} | Acc: {batch_acc:.2f}%"
            )

    return {
        "total_loss": total_loss / len(train_loader),
        "ce_loss": classification_loss / len(train_loader),
        "contrastive_loss": contrastive_loss / len(train_loader),
        "accuracy": 100 * correct / total,
    }


def validate(
    model: HierCodeWithHiGITA,
    val_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Validate model"""
    model.eval()

    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            logits = output["logits"]

            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return {
        "loss": total_loss / len(val_loader),
        "accuracy": 100 * correct / total,
    }


def main():
    parser = argparse.ArgumentParser(description="Train HierCode with optional Hi-GITA enhancement")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--sample-limit", type=int, default=None, help="Limit number of samples (default: None)"
    )

    # Add checkpoint management arguments
    setup_checkpoint_arguments(parser, "hiercode_higita")

    args = parser.parse_args()

    # ========== VERIFY GPU ==========
    verify_and_setup_gpu()

    # Auto-detect dataset directory
    data_dir = str(get_dataset_directory())
    logger.info(f"Using dataset from: {data_dir}")

    # Setup
    config = HiGITAConfig()
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.epochs = args.epochs
    config.checkpoint_dir = args.checkpoint_dir
    config.data_dir = data_dir

    device = "cuda"
    logger.info(f"🔧 Device: {device} (auto-detected)")

    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset using unified helper
    def create_tensor_dataset(x: np.ndarray, y: np.ndarray):
        """Factory for TensorDataset to work with prepare_dataset_and_loaders helper.
        Handles image reshaping: [4096] -> [64, 64] -> [1, 64, 64]
        """
        # Reshape if flattened (4096,) -> (64, 64)
        if x.ndim == 2 and x.shape[1] == 4096:
            x = x.reshape(-1, 64, 64)

        # Add channel dimension if needed: (N, 64, 64) -> (N, 1, 64, 64)
        if x.ndim == 3:
            x = x[:, np.newaxis, :, :]

        # Convert to torch tensors and wrap in TensorDataset
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(y).long()
        return TensorDataset(x_tensor, y_tensor)

    (images, labels), num_classes, train_loader, val_loader = prepare_dataset_and_loaders(
        data_dir=data_dir,
        dataset_fn=create_tensor_dataset,
        batch_size=config.batch_size,
        sample_limit=args.sample_limit,
        logger=logger,
    )

    # Get dataset sizes
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    train_size = len(train_dataset) if isinstance(train_dataset, TensorDataset) else len(images)  # type: ignore
    val_size = len(val_dataset) if isinstance(val_dataset, TensorDataset) else 0  # type: ignore
    logger.info(f"✅ Train: {train_size} | Val: {val_size}")
    logger.info(f"✅ Num classes: {num_classes} (from label range [0, {int(np.max(labels))}])")

    # Create model (use num_classes from prepare_dataset_and_loaders, not unique count)
    model = HierCodeWithHiGITA(
        num_classes=num_classes,
        stroke_dim=config.stroke_dim,
        radical_dim=config.radical_dim,
        character_dim=config.character_dim,
    )
    model = model.to(device)

    # Load text encoder and contrastive loss for Hi-GITA training
    text_encoder = MultiGranularityTextEncoder().to(device)
    contrastive_loss_fn = FineGrainedContrastiveLoss()

    # Optimizer and scheduler using factory functions
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # ========== INITIALIZE CHECKPOINT MANAGER ==========
    checkpoint_manager = CheckpointManager(config.checkpoint_dir, "hiercode_higita")

    # Training loop
    best_val_acc = 0
    best_path = Path(config.checkpoint_dir) / "best_hiercode_higita.pth"
    history = []

    # Resume from checkpoint using unified DRY method
    start_epoch, best_metrics = checkpoint_manager.load_checkpoint_for_training(
        model,
        optimizer,
        scheduler,
        device,
        resume_from=args.resume_from,
        args_no_checkpoint=args.no_checkpoint,
    )
    best_val_acc = best_metrics.get("val_accuracy", 0.0)
    start_epoch = max(start_epoch, 1)  # Epoch numbering starts at 1

    for epoch in range(start_epoch, config.epochs + 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Epoch {epoch}/{config.epochs}")
        logger.info(f"{'=' * 60}")

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            args,
            config,
            text_encoder,
            contrastive_loss_fn,
        )

        val_metrics = validate(model, val_loader, device)

        scheduler.step()

        logger.info(
            f"\n📊 Train - Loss: {train_metrics['total_loss']:.4f} | Acc: {train_metrics['accuracy']:.2f}%"
        )
        logger.info(
            f"📋 Val   - Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.2f}%"
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        # Save checkpoint after each epoch for resuming later
        checkpoint_manager.save_checkpoint(
            epoch, model, optimizer, scheduler, {"val_accuracy": val_metrics["accuracy"]}
        )

        # Save best model
        if save_best_model(model, val_metrics["accuracy"], best_val_acc, best_path):
            best_val_acc = val_metrics["accuracy"]

    # Save final history
    history_path = Path(config.checkpoint_dir) / "training_history_higita.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    logger.info(f"\n{'=' * 60}")
    logger.info("✅ Training complete!")
    logger.info(f"   Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"   Model saved to: {config.checkpoint_dir}")
    logger.info(f"   History saved to: {history_path}")

    # ========== CREATE CHARACTER MAPPING ==========
    logger.info("\n📊 Creating character mapping for inference...")
    try:
        from subprocess import run

        result = run(  # noqa: S603
            [
                sys.executable,
                "scripts/create_class_mapping.py",
                "--metadata-path",
                str(Path(config.data_dir) / "metadata.json"),
                "--output-dir",
                str(Path(config.checkpoint_dir).parent),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("✓ Character mapping created successfully")
        else:
            logger.debug("Character mapping creation skipped (metadata incomplete)")
    except Exception as e:
        logger.debug(f"Character mapping creation skipped: {e}")


if __name__ == "__main__":
    main()
