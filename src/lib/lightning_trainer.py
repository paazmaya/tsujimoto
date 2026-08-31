"""
Simplified trainer wrappers using PyTorch Lightning.

This module provides backward-compatible trainer classes that use Lightning
internally for training. It simplifies the training API while leveraging
Lightning's powerful features.

The trainers work with existing model architectures and configuration classes,
making them drop-in replacements for the old base_trainer.BaseModelTrainer.

Example Usage:
    >>> from src.lib.lightning_trainer import LightningTrainer
    >>> from src.lib.config import CNNConfig
    >>> from models import LightweightCNNKanji
    >>> 
    >>> config = CNNConfig(epochs=50)
    >>> model = LightweightCNNKanji(num_classes=config.num_classes)
    >>> trainer = LightningTrainer(model, config, checkpoint_dir='checkpoints/cnn')
    >>> history = trainer.train(train_loader, val_loader)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader

from .config import OptimizationConfig
from .lightning_module import KanjiRecognitionLightningModule
from .logging_utils import setup_logger

logger = setup_logger(__name__)


class LightningTrainer:
    """
    Simplified trainer wrapper using PyTorch Lightning.

    Provides a clean interface for training models with Lightning's advantages:
    - Automatic GPU/CPU handling
    - Built-in checkpointing and early stopping
    - Metrics logging
    - Distributed training support

    Args:
        model: PyTorch model to train
        config: OptimizationConfig instance
        checkpoint_dir: Directory for saving checkpoints
        model_type: Name of model type (for logging)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: OptimizationConfig,
        checkpoint_dir: Optional[str] = None,
        model_type: str = "generic",
    ):
        """Initialize the Lightning trainer wrapper.

        Args:
            model: PyTorch model instance
            config: Training configuration
            checkpoint_dir: Where to save checkpoints
            model_type: Type of model (cnn, rnn, vit, etc.)
        """
        self.model = model
        self.config = config
        self.model_type = model_type

        # Setup checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = f"checkpoints/{model_type}"
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create Lightning module wrapper
        self.lightning_module = KanjiRecognitionLightningModule(
            model=model,
            config=config,
            model_name=model_type,
        )

        # Training history (for backward compatibility)
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.best_model_path: Optional[str] = None

        logger.info(f"LightningTrainer initialized for {model_type}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
        early_stopping_patience: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Train the model using Lightning Trainer.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of epochs (if None, uses config.epochs)
            early_stopping_patience: Patience for early stopping

        Returns:
            Dictionary with training history
        """
        num_epochs = num_epochs or self.config.epochs

        logger.info(f"Starting training with Lightning Trainer")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")

        # Setup callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.checkpoint_dir),
            filename="epoch_{epoch:03d}-val_acc_{val_accuracy:.3f}",
            monitor="val_accuracy",
            mode="max",
            save_top_k=3,
            save_last=True,
            verbose=True,
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_accuracy",
            patience=early_stopping_patience,
            mode="max",
            verbose=True,
            check_on_train_epoch_end=False,
        )

        # Setup logger
        tb_logger = TensorBoardLogger(
            save_dir=str(self.checkpoint_dir.parent),
            name=self.model_type,
            version=0,
        )

        # Create Lightning Trainer
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator="auto",  # Auto-detect GPU/CPU
            devices="auto",  # Use all available devices
            callbacks=[checkpoint_callback, early_stopping_callback],
            logger=tb_logger,
            enable_progress_bar=True,
            log_every_n_steps=10,
        )

        # Train
        trainer.fit(
            self.lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        # Extract best model path
        if checkpoint_callback.best_model_path:
            self.best_model_path = checkpoint_callback.best_model_path
            self.best_val_accuracy = checkpoint_callback.best_model_score.item()
            self.best_epoch = checkpoint_callback.best_epoch

            logger.info(
                f"Best model at epoch {self.best_epoch}: "
                f"val_accuracy={self.best_val_accuracy:.4f}"
            )

        return self.history

    def load_best_model(self) -> torch.nn.Module:
        """
        Load the best model from checkpoint.

        Returns:
            The best model instance
        """
        if not self.best_model_path:
            logger.warning("No best model checkpoint found")
            return self.model

        logger.info(f"Loading best model from {self.best_model_path}")
        checkpoint = torch.load(self.best_model_path)

        # Handle Lightning checkpoint format
        if "state_dict" in checkpoint:
            # Load from Lightning checkpoint
            state_dict = checkpoint["state_dict"]
            # Remove 'model.' prefix added by Lightning
            state_dict = {
                k.replace("model.", ""): v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }
            self.model.load_state_dict(state_dict)
        else:
            # Try loading directly
            self.model.load_state_dict(checkpoint)

        return self.model

    def validate(
        self,
        val_loader: DataLoader,
    ) -> Tuple[float, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation DataLoader

        Returns:
            Tuple of (loss, accuracy)
        """
        trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",
            enable_progress_bar=True,
        )

        results = trainer.validate(self.lightning_module, val_loader)

        if results:
            metrics = results[0]
            loss = metrics.get("val_loss", 0.0)
            accuracy = metrics.get("val_accuracy", 0.0)
            return loss, accuracy

        return 0.0, 0.0


# Factory function for backward compatibility
def create_trainer(
    model: torch.nn.Module,
    config: OptimizationConfig,
    model_type: str = "generic",
    checkpoint_dir: Optional[str] = None,
) -> LightningTrainer:
    """
    Factory function to create a Lightning trainer.

    Provides backward compatibility with old trainer creation patterns.

    Args:
        model: PyTorch model
        config: Training configuration
        model_type: Type of model (cnn, rnn, vit, etc.)
        checkpoint_dir: Checkpoint directory

    Returns:
        LightningTrainer instance
    """
    return LightningTrainer(
        model=model,
        config=config,
        checkpoint_dir=checkpoint_dir,
        model_type=model_type,
    )
