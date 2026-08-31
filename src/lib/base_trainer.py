"""Base trainer class and architecture-specific subclasses for model training.

This module consolidates training loop logic from multiple training scripts into
a reusable base class pattern. The BaseModelTrainer class provides common
functionality for training different model architectures.

Key Features:
- Unified training loop interface across all architectures
- Automatic GPU/CPU setup and device management
- Checkpoint management and best model tracking
- Training metrics collection and logging
- Extensible design for architecture-specific behaviors via mixins

Common Patterns Extracted From:
- train_cnn_model.py
- train_rnn.py
- train_vit.py
- train_hiercode.py
- train_qat.py
- train_radical_rnn.py
- train_hiercode_higita.py

Example Usage:
    >>> from src.lib.base_trainer import CNNTrainer
    >>> trainer = CNNTrainer(model, train_loader, val_loader, device="cuda")
    >>> history = trainer.train(num_epochs=100)
    >>> best_model = trainer.load_best_model()

Classes:
    BaseModelTrainer: Abstract base class with common training logic
    CNNTrainer: Trainer for CNN models (lightweight Kanji Net)
    RNNTrainer: Trainer for RNN models (KanjiRNN, RadicalRNN)
    ViTTrainer: Trainer for Vision Transformer models
    HierCodeTrainer: Trainer for HierCode models
    QATTrainer: Trainer for Quantization-Aware Training models
    HierCodeHiGITATrainer: Trainer for HierCode-HiGITA models

Functions:
    setup_trainer_for_model: Factory function to create appropriate trainer

"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from .checkpoint import CheckpointManager
from .conversion import quantize_model
from .logging_utils import setup_logger
from .system import verify_and_setup_gpu


logger = setup_logger(__name__)


class BaseModelTrainer(ABC):
    """Abstract base trainer class for model training across different architectures.

    This class provides common training functionality including:
    - Training loop (epoch-based with validation)
    - Checkpoint management
    - Metrics tracking
    - Device management (GPU/CPU)

    Subclasses should override:
    - build_model(): Instantiate the model architecture
    - get_loss_fn(): Return appropriate loss function
    - forward_pass(): Custom forward pass logic if needed
    - process_batch(): Custom batch processing if needed

    Args:
        model: PyTorch model instance to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: PyTorch optimizer instance
        device: torch device (cuda or cpu)
        checkpoint_dir: Directory for saving checkpoints
        model_type: Type of model being trained (cnn, rnn, vit, etc)
        num_classes: Number of output classes
        image_size: Input image size (default: 64)

    Attributes:
        model: The model being trained
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler (optional)
        device: torch device
        checkpoint_dir: Path to checkpoint directory
        checkpoint_manager: CheckpointManager instance
        loss_fn: Loss function
        history: Training history dictionary
        best_val_accuracy: Best validation accuracy achieved
        patience: Early stopping patience
        patience_counter: Early stopping counter

    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: str = "cpu",
        checkpoint_dir: Optional[str] = None,
        model_type: str = "generic",
        num_classes: int = 10,
        image_size: int = 64,
    ):
        """Initialize the trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer instance
            device: Device to use (cuda or cpu)
            checkpoint_dir: Directory for checkpoints
            model_type: Type of model being trained
            num_classes: Number of output classes
            image_size: Input image size

        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.model_type = model_type
        self.num_classes = num_classes
        self.image_size = image_size

        # Checkpoint management
        if checkpoint_dir is None:
            checkpoint_dir = f"checkpoints/{model_type}"
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            approach_name=model_type
        )

        # Loss function
        self.loss_fn = self.get_loss_fn()

        # Training tracking
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.current_epoch = 0

        # Early stopping
        self.patience = 10
        self.patience_counter = 0

        # Learning rate scheduler (optional, set by subclasses)
        self.scheduler: Optional[LRScheduler] = None

    @abstractmethod
    def get_loss_fn(self) -> nn.Module:
        """Return the loss function for this trainer.

        Subclasses should override to return appropriate loss function.

        Returns:
            Loss function module

        """
        pass

    def set_scheduler(self, scheduler: LRScheduler) -> None:
        """Set learning rate scheduler.

        Args:
            scheduler: Learning rate scheduler instance

        """
        self.scheduler = scheduler
        logger.info(f"Scheduler set: {type(scheduler).__name__}")

    def forward_pass(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute forward pass through model.

        Can be overridden by subclasses for custom forward logic.

        Args:
            batch: Tuple of (images, labels)

        Returns:
            Tuple of (outputs, loss)

        """
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(images)
        loss = self.loss_fn(outputs, labels)

        return outputs, loss

    def process_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        """Process a single batch (for training loop).

        Can be overridden by subclasses for custom batch processing.

        Args:
            batch: Tuple of (images, labels)

        Returns:
            Dictionary with batch metrics

        """
        outputs, loss = self.forward_pass(batch)
        _, labels = batch
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels.to(self.device)).sum().item()
        total = labels.size(0)

        return {
            "loss": loss.item(),
            "correct": correct,
            "total": total,
        }

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch.

        Returns:
            Tuple of (average loss, accuracy)

        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            # Forward pass
            batch_metrics = self.process_batch(batch)
            loss = torch.tensor(batch_metrics["loss"], requires_grad=True).to(self.device)

            # For actual loss computation, get fresh forward pass
            outputs, loss = self.forward_pass(batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            total_correct += batch_metrics["correct"]
            total_samples += batch_metrics["total"]

            if (batch_idx + 1) % 100 == 0:
                logger.debug(
                    f"Epoch progress: {batch_idx + 1}/{len(self.train_loader)} "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate model on validation set.

        Returns:
            Tuple of (average loss, accuracy)

        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in self.val_loader:
            outputs, loss = self.forward_pass(batch)

            _, labels = batch
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels.to(self.device)).sum().item()
            total = labels.size(0)

            total_loss += loss.item()
            total_correct += correct
            total_samples += total

        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def train(
        self,
        num_epochs: int = 100,
        early_stopping: bool = True,
        save_best_model: bool = True,
    ) -> Dict[str, List[float]]:
        """Main training loop.

        Args:
            num_epochs: Number of epochs to train
            early_stopping: Whether to use early stopping
            save_best_model: Whether to save best model

        Returns:
            Training history dictionary

        """
        logger.info(
            f"Starting training for {num_epochs} epochs on {self.device} "
            f"(model: {self.model_type})"
        )

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            # Training epoch
            train_loss, train_acc = self.train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)

            # Validation epoch
            val_loss, val_acc = self.validate()
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)

            # Logging
            logger.info(
                f"Epoch {epoch + 1:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Best model tracking
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0

                if save_best_model:
                    self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1

            # Early stopping
            if early_stopping and self.patience_counter >= self.patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch + 1} "
                    f"(best: {self.best_epoch + 1})"
                )
                break

        logger.info(
            f"Training completed. Best accuracy: {self.best_val_accuracy:.4f} "
            f"at epoch {self.best_epoch + 1}"
        )

        return self.history

    def save_checkpoint(self, is_best: bool = False) -> str:
        """Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far

        Returns:
            Path to saved checkpoint

        """
        metrics = {
            "train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else 0.0,
            "train_accuracy": self.history["train_accuracy"][-1] if self.history["train_accuracy"] else 0.0,
            "val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else 0.0,
            "val_accuracy": self.history["val_accuracy"][-1] if self.history["val_accuracy"] else 0.0,
        }

        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            epoch=self.current_epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metrics=metrics,
            is_best=is_best
        )

        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        """
        epoch, metrics = self.checkpoint_manager.load_checkpoint(
            Path(checkpoint_path),
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler
        )
        
        self.current_epoch = epoch
        logger.info(f"Checkpoint loaded: {checkpoint_path}")

    def load_best_model(self) -> nn.Module:
        """Load and return the best model found during training.

        Returns:
            Model with best weights

        """
        best_model_path = self.checkpoint_manager.approach_dir / "checkpoint_best.pt"
        if best_model_path.exists():
            self.load_checkpoint(str(best_model_path))
            logger.info(f"Best model loaded from {best_model_path}")
        else:
            logger.warning("No best model checkpoint found")

        return self.model

    def save_training_history(self, output_path: Optional[str] = None) -> str:
        """Save training history to JSON file.

        Args:
            output_path: Path to save history (optional)

        Returns:
            Path to saved history file

        """
        if output_path is None:
            output_path = str(self.checkpoint_dir / "training_history.json")

        with open(output_path, "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Training history saved: {output_path}")
        return output_path


# ============================================================================
# Architecture-Specific Trainers
# ============================================================================


class CNNTrainer(BaseModelTrainer):
    """Trainer for CNN models (LightweightKanjiNet).

    Uses CrossEntropyLoss for classification.

    Example:
        >>> from models import LightweightKanjiNet
        >>> model = LightweightKanjiNet(num_classes=43427, input_channels=1)
        >>> trainer = CNNTrainer(model, train_loader, val_loader, optimizer)
        >>> history = trainer.train(num_epochs=100)

    """

    def get_loss_fn(self) -> nn.Module:
        """Return CrossEntropyLoss for CNN training.

        Returns:
            CrossEntropyLoss module

        """
        return nn.CrossEntropyLoss()


class RNNTrainer(BaseModelTrainer):
    """Trainer for RNN models (KanjiRNN, RadicalRNN).

    Uses CrossEntropyLoss for classification. Subclasses can override
    forward_pass() to handle variable-length sequences.

    Example:
        >>> from models import KanjiRNN
        >>> model = KanjiRNN(num_classes=43427, hidden_size=256)
        >>> trainer = RNNTrainer(model, train_loader, val_loader, optimizer)
        >>> history = trainer.train(num_epochs=100)

    """

    def get_loss_fn(self) -> nn.Module:
        """Return CrossEntropyLoss for RNN training.

        Returns:
            CrossEntropyLoss module

        """
        return nn.CrossEntropyLoss()


class ViTTrainer(BaseModelTrainer):
    """Trainer for Vision Transformer models.

    Supports automatic mixed precision (autocast) for efficient training.

    Example:
        >>> from models import VisionTransformer
        >>> model = VisionTransformer(num_classes=43427, patch_size=8)
        >>> trainer = ViTTrainer(model, train_loader, val_loader, optimizer)
        >>> trainer.enable_mixed_precision()
        >>> history = trainer.train(num_epochs=100)

    """

    def __init__(self, *args, **kwargs):
        """Initialize ViT trainer with optional mixed precision."""
        super().__init__(*args, **kwargs)
        self.use_mixed_precision = False
        self.scaler = None

    def get_loss_fn(self) -> nn.Module:
        """Return CrossEntropyLoss for ViT training.

        Returns:
            CrossEntropyLoss module

        """
        return nn.CrossEntropyLoss()

    def enable_mixed_precision(self) -> None:
        """Enable automatic mixed precision training."""
        self.use_mixed_precision = True
        self.scaler = torch.cuda.amp.GradScaler()
        logger.info("Mixed precision training enabled")

    def train_epoch(self) -> Tuple[float, float]:
        """Train one epoch with optional mixed precision.

        Returns:
            Tuple of (average loss, accuracy)

        """
        if not self.use_mixed_precision:
            return super().train_epoch()

        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs, loss = self.forward_pass(batch)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            _, labels = batch
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels.to(self.device)).sum().item()
            total = labels.size(0)

            total_loss += loss.item()
            total_correct += correct
            total_samples += total

        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy


class HierCodeTrainer(BaseModelTrainer):
    """Trainer for HierCode models.

    Standard CNN trainer with hierarchical encoding.

    Example:
        >>> from models import HierCodeClassifier
        >>> model = HierCodeClassifier(num_classes=43427)
        >>> trainer = HierCodeTrainer(model, train_loader, val_loader, optimizer)
        >>> history = trainer.train(num_epochs=100)

    """

    def get_loss_fn(self) -> nn.Module:
        """Return CrossEntropyLoss for HierCode training.

        Returns:
            CrossEntropyLoss module

        """
        return nn.CrossEntropyLoss()


class QATTrainer(BaseModelTrainer):
    """Trainer for Quantization-Aware Training models.

    Enables quantization operations during training for post-training INT8
    quantization compatibility.

    Example:
        >>> from models import QuantizableLightweightKanjiNet
        >>> model = QuantizableLightweightKanjiNet(num_classes=43427)
        >>> trainer = QATTrainer(model, train_loader, val_loader, optimizer)
        >>> history = trainer.train(num_epochs=50, qat_enabled=True)

    """

    def get_loss_fn(self) -> nn.Module:
        """Return CrossEntropyLoss for QAT training.

        Returns:
            CrossEntropyLoss module

        """
        return nn.CrossEntropyLoss()

    def train(
        self,
        num_epochs: int = 100,
        early_stopping: bool = True,
        save_best_model: bool = True,
        qat_enabled: bool = False,
    ) -> Dict[str, List[float]]:
        """Train with optional quantization-aware training.

        Args:
            num_epochs: Number of epochs to train
            early_stopping: Whether to use early stopping
            save_best_model: Whether to save best model
            qat_enabled: Whether to enable fake quantization during training

        Returns:
            Training history dictionary

        """
        if qat_enabled:
            self._enable_qat()

        history = super().train(
            num_epochs=num_epochs,
            early_stopping=early_stopping,
            save_best_model=save_best_model,
        )

        if qat_enabled:
            self._convert_to_quantized()

        return history

    def _enable_qat(self) -> None:
        """Enable quantization-aware training (fake quantization)."""
        self.model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        torch.quantization.prepare_qat(self.model, inplace=True)
        logger.info("QAT (fake quantization) enabled")

    def _convert_to_quantized(self) -> None:
        """Convert model to quantized after training."""
        torch.quantization.convert(self.model, inplace=True)
        logger.info("Model converted to quantized state")


class HierCodeHiGITATrainer(BaseModelTrainer):
    """Trainer for HierCode-HiGITA models with auxiliary loss.

    Supports contrastive or auxiliary loss terms during training.

    Example:
        >>> from models import HierCodeHiGITA
        >>> model = HierCodeHiGITA(num_classes=43427)
        >>> trainer = HierCodeHiGITATrainer(model, train_loader, val_loader, optimizer)
        >>> history = trainer.train(num_epochs=100)

    """

    def __init__(self, *args, auxiliary_weight: float = 0.1, **kwargs):
        """Initialize HiGITA trainer.

        Args:
            auxiliary_weight: Weight for auxiliary loss term (default: 0.1)

        """
        super().__init__(*args, **kwargs)
        self.auxiliary_weight = auxiliary_weight

    def get_loss_fn(self) -> nn.Module:
        """Return CrossEntropyLoss for main task.

        Returns:
            CrossEntropyLoss module

        """
        return nn.CrossEntropyLoss()

    def forward_pass(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with auxiliary loss support.

        Expects model to return (main_output, auxiliary_output) tuple.

        Args:
            batch: Tuple of (images, labels)

        Returns:
            Tuple of (outputs, total_loss)

        """
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Model should return (main_output, auxiliary_output)
        outputs = self.model(images)

        if isinstance(outputs, tuple) and len(outputs) == 2:
            main_output, aux_output = outputs
            main_loss = self.loss_fn(main_output, labels)
            aux_loss = self.loss_fn(aux_output, labels)
            total_loss = main_loss + self.auxiliary_weight * aux_loss
            return main_output, total_loss
        else:
            # Fallback for models that don't return auxiliary output
            loss = self.loss_fn(outputs, labels)
            return outputs, loss


# ============================================================================
# Factory Function
# ============================================================================


def setup_trainer_for_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str = "cpu",
    model_type: str = "cnn",
    checkpoint_dir: Optional[str] = None,
    num_classes: int = 10,
    image_size: int = 64,
) -> BaseModelTrainer:
    """Factory function to create appropriate trainer for model type.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer instance
        device: Device to use
        model_type: Type of model (cnn, rnn, vit, hiercode, qat, hiercode_higita)
        checkpoint_dir: Directory for checkpoints
        num_classes: Number of output classes
        image_size: Input image size

    Returns:
        Appropriate trainer instance

    Raises:
        ValueError: If model_type is not recognized

    """
    trainer_map = {
        "cnn": CNNTrainer,
        "rnn": RNNTrainer,
        "vit": ViTTrainer,
        "hiercode": HierCodeTrainer,
        "qat": QATTrainer,
        "hiercode_higita": HierCodeHiGITATrainer,
    }

    if model_type not in trainer_map:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: {list(trainer_map.keys())}"
        )

    trainer_class = trainer_map[model_type]
    trainer = trainer_class(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        model_type=model_type,
        num_classes=num_classes,
        image_size=image_size,
    )

    logger.info(f"Created {trainer_class.__name__} for {model_type} model")
    return trainer
