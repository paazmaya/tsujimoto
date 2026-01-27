"""
Checkpoint management utilities for training scripts.

Provides unified checkpoint handling with automatic resume from latest checkpoint.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

logger: logging.Logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages training checkpoints with automatic resume capabilities."""

    def __init__(self, checkpoint_dir: str, approach_name: str) -> None:
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Checkpoint directory (e.g., "training/cnn/checkpoints")
            approach_name: Training approach name (e.g., "cnn", "rnn", "hiercode", "vit", "qat")
        """
        self.approach_dir = Path(checkpoint_dir)
        self.approach_name: str = approach_name
        self.approach_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        epoch: int,
        model,
        optimizer,
        scheduler,
        metrics: Optional[Dict] = None,
        is_best: bool = False,
    ) -> Path:
        """
        Save training checkpoint.

        Args:
            epoch: Current epoch number
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            metrics: Optional dictionary of metrics (accuracy, loss, etc.)
            is_best: Whether this is the best checkpoint so far

        Returns:
            Path to saved checkpoint
        """
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics or {},
        }

        # Regular checkpoint
        checkpoint_path: Path = self.approach_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"  ✓ Checkpoint saved: {checkpoint_path}")

        # Best checkpoint (optional)
        if is_best:
            best_path: Path = self.approach_dir / "checkpoint_best.pt"
            torch.save(checkpoint_data, best_path)
            logger.info(f"  ✓ Best checkpoint saved: {best_path}")

        return checkpoint_path

    def load_checkpoint(
        self, checkpoint_path: Path, model, optimizer, scheduler
    ) -> Tuple[int, Dict]:
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into

        Returns:
            Tuple of (epoch, metrics_dict)
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch = checkpoint["epoch"]
        metrics = checkpoint.get("metrics", {})

        logger.info(f"✓ Loaded checkpoint: {checkpoint_path} (epoch {epoch})")
        if metrics:
            logger.debug(f"  Metrics: {metrics}")

        return epoch, metrics

    def find_latest_checkpoint(self) -> Optional[Path]:
        """
        Find the latest completed checkpoint.

        Returns:
            Path to latest checkpoint file, or None if no checkpoints exist
        """
        if not self.approach_dir.exists():
            return None

        def extract_epoch_num(path) -> int:
            match: re.Match[str] | None = re.search(r"_(\d+)", path.name)
            return int(match.group(1)) if match else -1

        checkpoint_files: list[Path] = sorted(
            self.approach_dir.glob("checkpoint_epoch_*.pt"),
            key=extract_epoch_num,
            reverse=True,
        )

        if checkpoint_files:
            return checkpoint_files[0]

        return None

    def load_checkpoint_for_training(
        self,
        model,
        optimizer,
        scheduler,
        device,
        resume_from: Optional[str] = None,
        args_no_checkpoint: bool = False,
    ) -> Tuple[int, Dict]:
        """
        Unified checkpoint loading for training scripts (DRY pattern).

        Handles both auto-detection and explicit resume paths with consistent
        error handling, printing, and return values.

        Args:
            model: Model to load into
            optimizer: Optimizer to load into
            scheduler: Scheduler to load into
            device: Device to load checkpoint on
            resume_from: Explicit checkpoint path (optional, overrides auto-detection)
            args_no_checkpoint: Skip checkpoint loading/saving if True

        Returns:
            Tuple of (start_epoch, best_metrics_dict)
            - start_epoch: Epoch to resume from (0 if starting fresh)
            - best_metrics_dict: Best metrics found in checkpoint, or empty dict
        """
        start_epoch = 0
        best_metrics = {}

        # Skip checkpoint loading if explicitly disabled
        if args_no_checkpoint:
            logger.info("📌 Checkpoint loading disabled (--no-checkpoint flag)")
            return start_epoch, best_metrics

        # Case 1: Explicit checkpoint path provided
        if resume_from:
            checkpoint_path = Path(resume_from)
            if checkpoint_path.exists():
                logger.info(f"📂 Loading checkpoint from: {resume_from}")
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    if scheduler and checkpoint.get("scheduler_state_dict"):
                        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

                    start_epoch = checkpoint["epoch"] + 1
                    best_metrics = checkpoint.get("metrics", {})

                    logger.info(f"✓ Checkpoint loaded (epoch {checkpoint['epoch']})")
                    if best_metrics:
                        logger.debug(f"  Metrics: {best_metrics}")
                    logger.info(f"📍 Will continue from epoch {start_epoch}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load checkpoint: {e}")
                    logger.info("   Starting fresh training...")
            else:
                logger.warning(f"❌ Checkpoint not found: {checkpoint_path}")
                logger.info("   Starting fresh training...")
            return start_epoch, best_metrics

        # Case 2: Auto-detect latest checkpoint
        latest_checkpoint: Path | None = self.find_latest_checkpoint()
        if latest_checkpoint:
            logger.info(f"🔄 Auto-detected checkpoint: {latest_checkpoint.name}")
            try:
                checkpoint = torch.load(latest_checkpoint, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if scheduler and checkpoint.get("scheduler_state_dict"):
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

                start_epoch = checkpoint["epoch"] + 1
                best_metrics = checkpoint.get("metrics", {})

                logger.info(f"✓ Checkpoint loaded (epoch {checkpoint['epoch']})")
                if best_metrics:
                    logger.debug(f"  Metrics: {best_metrics}")
                logger.info(f"📍 Will continue from epoch {start_epoch}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load checkpoint: {e}")
                logger.info("   Starting fresh training...")
        else:
            logger.info(f"ℹ️  No checkpoint found for '{self.approach_name}' in {self.approach_dir}")
            logger.info("   Starting fresh training...")

        return start_epoch, best_metrics

    def find_and_load_latest_checkpoint(
        self, model, optimizer, scheduler=None
    ) -> Tuple[Optional[Dict], int]:
        """
        Automatically find and load the latest checkpoint.

        Args:
            model: Model to load into
            optimizer: Optimizer to load into
            scheduler: Optional scheduler to load into

        Returns:
            Tuple of (checkpoint_data or None, start_epoch)
        """
        latest_checkpoint: Path | None = self.find_latest_checkpoint()

        if latest_checkpoint is None:
            logger.info(f"ℹ️  No checkpoint found for '{self.approach_name}' in {self.approach_dir}")
            logger.info("   Starting fresh training...")
            return None, 0

        logger.info(f"🔄 Auto-detected checkpoint: {latest_checkpoint.name}")

        try:
            epoch, metrics = self.load_checkpoint(latest_checkpoint, model, optimizer, scheduler)
            start_epoch: int = epoch + 1
            logger.info(f"📍 Will continue from epoch {start_epoch}")
            return {"epoch": epoch, "metrics": metrics}, start_epoch
        except Exception as e:
            logger.warning(f"⚠️  Failed to load checkpoint: {e}")
            logger.info("   Starting fresh training...")
            return None, 0

    def list_all_checkpoints(self) -> list:
        """
        List all available checkpoints for this approach.

        Returns:
            Sorted list of checkpoint paths (most recent first)
        """
        if not self.approach_dir.exists():
            return []

        def extract_epoch_num(path) -> int:
            match: re.Match[str] | None = re.search(r"_(\d+)", path.name)
            return int(match.group(1)) if match else -1

        checkpoints: list[Path] = sorted(
            self.approach_dir.glob("checkpoint_epoch_*.pt"),
            key=extract_epoch_num,
            reverse=True,
        )

        return checkpoints

    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> None:
        """
        Remove old checkpoints, keeping only the last N.

        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoints = self.list_all_checkpoints()

        if len(checkpoints) > keep_last_n:
            for checkpoint_path in checkpoints[keep_last_n:]:
                checkpoint_path.unlink()
                logger.debug(f"  Removed old checkpoint: {checkpoint_path.name}")

    def get_checkpoint_info(self) -> Dict:
        """
        Get summary information about available checkpoints.

        Returns:
            Dictionary with checkpoint information
        """
        checkpoints = self.list_all_checkpoints()
        best_checkpoint: Path = self.approach_dir / "checkpoint_best.pt"

        info = {
            "approach_name": self.approach_name,
            "checkpoint_dir": str(self.approach_dir),
            "total_checkpoints": len(checkpoints),
            "latest_checkpoint": str(checkpoints[0]) if checkpoints else None,
            "best_checkpoint": str(best_checkpoint) if best_checkpoint.exists() else None,
            "all_checkpoints": [str(c) for c in checkpoints],
        }

        return info


def setup_checkpoint_arguments(parser, approach_name: str) -> None:
    """
    Add checkpoint-related arguments to argument parser.

    Args:
        parser: argparse ArgumentParser instance
        approach_name: Training approach name (for default checkpoint dir)
    """
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=f"training/{approach_name}/checkpoints",
        help=f"Checkpoint directory (default: training/{approach_name}/checkpoints)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Specific checkpoint file to resume from (overrides auto-detection)",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Start fresh training without loading any checkpoint",
    )
    parser.add_argument(
        "--keep-last-n",
        type=int,
        default=5,
        help="Keep only last N checkpoints, delete older ones (default: 5)",
    )
