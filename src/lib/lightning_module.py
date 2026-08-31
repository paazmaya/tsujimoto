"""
PyTorch Lightning module for unified training across all model architectures.

This module replaces the manual training loop from base_trainer.py with
Lightning's automatic handling of:
- GPU/CPU device management
- Distributed training (DDP, multi-GPU)
- Checkpointing and model restoration
- Learning rate scheduling
- Metrics logging to Tensorboard/WandB
- Early stopping
- Mixed precision training

The KanjiRecognitionLightningModule wraps any model architecture and provides
a consistent training interface via Lightning's Trainer API.

Example Usage:
    >>> from src.lib.lightning_module import KanjiRecognitionLightningModule
    >>> from src.lib.config import CNNConfig
    >>> from models import LightweightCNNKanji
    >>> import pytorch_lightning as pl
    >>>
    >>> config = CNNConfig()
    >>> model = LightweightCNNKanji(num_classes=config.num_classes)
    >>> module = KanjiRecognitionLightningModule(model, config)
    >>>
    >>> trainer = pl.Trainer(max_epochs=config.epochs, accelerator='gpu')
    >>> trainer.fit(module, train_dataloaders, val_dataloaders)
    >>> trainer.validate(module, dataloaders)
    >>> trainer.test(module, dataloaders)
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from .config import OptimizationConfig
from .logging_utils import setup_logger

logger = setup_logger(__name__)


class KanjiRecognitionLightningModule(pl.LightningModule):
    """
    Lightning module for Kanji character recognition training.

    Wraps a PyTorch model and provides Lightning training interface with:
    - Automatic device handling
    - Metrics tracking and logging
    - Checkpointing
    - Learning rate scheduling
    - Distributed training support

    Args:
        model: PyTorch model to train (CNN, RNN, ViT, HierCode, etc.)
        config: OptimizationConfig instance with training hyperparameters
        model_name: Descriptive name for the model (for logging)
        freeze_backbone: If True, freeze model backbone (for transfer learning)

    Attributes:
        model: The wrapped PyTorch model
        config: Configuration object
        model_name: Name of the model being trained
        loss_fn: Loss function (CrossEntropyLoss for classification)
        train_accuracy: Metric for tracking training accuracy
        val_accuracy: Metric for tracking validation accuracy
    """

    def __init__(
        self,
        model: nn.Module,
        config: OptimizationConfig,
        model_name: str = "KanjiRecognition",
        freeze_backbone: bool = False,
    ):
        """Initialize the Lightning module.

        Args:
            model: PyTorch model instance
            config: Training configuration
            model_name: Descriptive name for logging
            freeze_backbone: Whether to freeze model parameters (transfer learning)
        """
        super().__init__()

        self.model = model
        self.config = config
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone

        # Loss function for multi-class classification
        self.loss_fn = nn.CrossEntropyLoss()

        # Save hyperparameters for checkpointing
        self.save_hyperparameters(
            {
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "optimizer": config.optimizer,
                "scheduler": config.scheduler,
                "model_name": model_name,
            },
            ignore=["model"]  # Don't save model in hyperparams
        )

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("Model backbone frozen for transfer learning")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Lightning training step.

        Called automatically for each batch during training.

        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index

        Returns:
            Training loss (scalar tensor)
        """
        images, labels = batch
        logits = self.forward(images)
        loss = self.loss_fn(logits, labels)

        # Log training loss
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Lightning validation step.

        Called automatically for each batch during validation.

        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index
        """
        images, labels = batch
        logits = self.forward(images)
        loss = self.loss_fn(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """Lightning test step.

        Called automatically for each batch during testing.

        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index
        """
        images, labels = batch
        logits = self.forward(images)
        loss = self.loss_fn(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()

        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary with 'optimizer' and 'lr_scheduler' keys
        """
        # Select optimizer based on config
        if self.config.optimizer.lower() == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            logger.debug("Using AdamW optimizer")
        elif self.config.optimizer.lower() == "sgd":
            optimizer = SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
            )
            logger.debug("Using SGD optimizer with momentum=0.9")
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        # Select learning rate scheduler based on config
        if self.config.scheduler.lower() == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config.scheduler_t_max,
                eta_min=1e-6,
            )
            logger.debug(f"Using CosineAnnealingLR with T_max={self.config.scheduler_t_max}")
        elif self.config.scheduler.lower() == "step":
            scheduler = StepLR(
                optimizer,
                step_size=max(1, self.config.epochs // 5),
                gamma=0.1,
            )
            logger.debug("Using StepLR scheduler")
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_start(self) -> None:
        """Called at the start of training.

        Logs training configuration and model info.
        """
        logger.info(f"Starting training: {self.model_name}")
        logger.info(f"  Epochs: {self.config.epochs}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Optimizer: {self.config.optimizer}")
        logger.info(f"  Scheduler: {self.config.scheduler}")
        logger.info(f"  Device: {self.device}")

    def on_train_end(self) -> None:
        """Called at the end of training.

        Logs final training statistics.
        """
        logger.info(f"Training complete: {self.model_name}")

    def on_validation_epoch_end(self) -> None:
        """Called at the end of each validation epoch.

        Can be used for custom logic (e.g., early stopping decisions).
        """
        # Lightning handles early stopping via callbacks, so we don't need manual logic here
        pass
