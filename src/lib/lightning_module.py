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

from typing import TYPE_CHECKING, Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from .config import OptimizationConfig
from .logging_utils import setup_logger

if TYPE_CHECKING:
    from .config import (
        DegradationAwareConfig,
        DTRNetConfig,
        GLHPNConfig,
        MultiGranularConfig,
        RestorationPipelineConfig,
        TrajectoryConfig,
    )

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
            ignore=["model"],  # Don't save model in hyperparams
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


# ============================================================================
# PHASE 1: GL-HPN Lightning Module
# ============================================================================


class GLHPNLightningModule(pl.LightningModule):
    """Lightning module for GL-HPN (Global-Local Hierarchical Retrieval)."""

    def __init__(
        self,
        backbone: nn.Module,
        retriever: nn.Module,
        config: "GLHPNConfig",
    ):
        """Initialize GL-HPN module.

        Args:
            backbone: Feature extraction backbone (CNN/ViT)
            retriever: CoarseToFineRetriever instance
            config: GLHPNConfig configuration
        """
        super().__init__()
        self.backbone = backbone
        self.retriever = retriever
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=["backbone", "retriever"])

    def forward(self, query_image: torch.Tensor) -> torch.Tensor:
        """Forward pass: extract features from query image."""
        return self.backbone(query_image)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step with contrastive loss."""
        images, labels = batch
        query_features = self.forward(images)
        logits = query_features  # For now, direct classification
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Validation step."""
        images, labels = batch
        logits = self.forward(images)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = AdamW(
            list(self.backbone.parameters()) + list(self.retriever.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ============================================================================
# PHASE 2: DTRNet Lightning Module
# ============================================================================


class DTRNetLightningModule(pl.LightningModule):
    """Lightning module for DTRNet (Dual Text-Radical Decoding)."""

    def __init__(
        self,
        dtrnet: nn.Module,
        config: "DTRNetConfig",
    ):
        """Initialize DTRNet module.

        Args:
            dtrnet: DTRNetModule instance
            config: DTRNetConfig configuration
        """
        super().__init__()
        self.dtrnet = dtrnet
        self.config = config
        self.text_loss_fn = nn.CrossEntropyLoss()
        self.ids_loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=["dtrnet"])

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through DTRNet."""
        return self.dtrnet(features)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step with dual loss."""
        images, labels = batch
        # Assume we have a feature extractor that produces (batch, seq_len, dim) features
        outputs = self.forward(images)

        text_loss = self.text_loss_fn(
            outputs["text_logits"].view(-1, outputs["text_logits"].size(-1)),
            labels.repeat_interleave(outputs["text_logits"].size(1)),
        )
        loss = text_loss * self.config.structure_agreement_weight

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Validation step."""
        images, labels = batch
        outputs = self.forward(images)
        text_loss = self.text_loss_fn(
            outputs["text_logits"].view(-1, outputs["text_logits"].size(-1)),
            labels.repeat_interleave(outputs["text_logits"].size(1)),
        )

        self.log("val_loss", text_loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = AdamW(
            self.dtrnet.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ============================================================================
# PHASE 3: Degradation-Aware Lightning Module
# ============================================================================


class DegradationAwareLightningModule(pl.LightningModule):
    """Lightning module for Degradation-Aware training."""

    def __init__(
        self,
        backbone: nn.Module,
        degradation_pipeline: nn.Module,
        restoration_preprocessor: nn.Module,
        config: "DegradationAwareConfig",
    ):
        """Initialize Degradation-Aware module.

        Args:
            backbone: Classification backbone
            degradation_pipeline: Degradation pipeline for training data
            restoration_preprocessor: Restoration for degraded images
            config: DegradationAwareConfig
        """
        super().__init__()
        self.backbone = backbone
        self.degradation = degradation_pipeline
        self.restoration = restoration_preprocessor
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=["backbone", "degradation", "restoration"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone."""
        return self.backbone(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training with synthetic degradation augmentation."""
        images, labels = batch

        # Apply random degradation
        degraded = self.degradation(images, severity=torch.rand(1).item())

        # Optional restoration
        if self.config.restoration_enabled:
            restored = self.restoration(degraded)["restored"]
        else:
            restored = degraded

        logits = self.forward(restored)
        loss = self.loss_fn(logits, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Validation step (no degradation)."""
        images, labels = batch
        logits = self.forward(images)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = AdamW(
            self.backbone.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ============================================================================
# PHASE 4: Trajectory Lightning Module
# ============================================================================


class TrajectoryLightningModule(pl.LightningModule):
    """Lightning module for Online Handwriting Trajectory training."""

    def __init__(
        self,
        hybrid_model: nn.Module,
        config: "TrajectoryConfig",
    ):
        """Initialize Trajectory module.

        Args:
            hybrid_model: HybridTrajectoryVisionModel instance
            config: TrajectoryConfig
        """
        super().__init__()
        self.hybrid_model = hybrid_model
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=["hybrid_model"])

    def forward(self, image_feat: torch.Tensor, traj_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass through hybrid model."""
        outputs = self.hybrid_model(image_feat, traj_emb)
        return outputs["logits"]

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step with trajectory data."""
        # batch should be (images, trajectories, labels)
        images, trajectories, labels = batch

        # Extract features from images (assumed from backbone)
        # In practice, would use a pretrained backbone
        image_feat = images.view(images.size(0), -1)[:, :512]  # Placeholder

        logits = self.forward(image_feat, trajectories)
        loss = self.loss_fn(logits, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Validation step."""
        images, trajectories, labels = batch
        image_feat = images.view(images.size(0), -1)[:, :512]
        logits = self.forward(image_feat, trajectories)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = AdamW(
            self.hybrid_model.parameters(),
            lr=self.config.learning_rate if hasattr(self.config, "learning_rate") else 1e-3,
            weight_decay=self.config.weight_decay if hasattr(self.config, "weight_decay") else 1e-4,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.max_strokes if hasattr(self.config, "max_strokes") else 30,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ============================================================================
# PHASE 5: Multi-Granular Lightning Module
# ============================================================================


class MultiGranularLightningModule(pl.LightningModule):
    """Lightning module for Multi-Granular Contrastive Learning."""

    def __init__(
        self,
        encoders: Dict[str, nn.Module],
        loss_fn: nn.Module,
        config: "MultiGranularConfig",
    ):
        """Initialize Multi-Granular module.

        Args:
            encoders: Dictionary of per-level encoders
            loss_fn: Contrastive loss function
            config: MultiGranularConfig
        """
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.loss_fn = loss_fn
        self.config = config
        self.save_hyperparameters(ignore=["encoders", "loss_fn"])

    def forward(self, stroke_feat, radical_feat, char_feat, text_ids):
        """Forward pass through multi-granular encoders."""
        stroke_emb = self.encoders["stroke"](stroke_feat)["embedding"]
        radical_emb = self.encoders["radical"](radical_feat)["embedding"]
        char_emb = self.encoders["character"](char_feat)["embedding"]
        text_emb = self.encoders["text"](text_ids)

        return {
            "stroke": stroke_emb,
            "radical": radical_emb,
            "character": char_emb,
            "text": text_emb,
        }

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training with multi-level contrastive loss."""
        stroke_feat, radical_feat, char_feat, text_ids, _ = batch

        embeddings = self.forward(stroke_feat, radical_feat, char_feat, text_ids)

        # Compute contrastive loss at each level
        loss = (
            self.config.stroke_loss_weight * self.loss_fn(embeddings["stroke"], embeddings["text"])
            + self.config.radical_loss_weight
            * self.loss_fn(embeddings["radical"], embeddings["text"])
            + self.config.character_loss_weight
            * self.loss_fn(embeddings["character"], embeddings["text"])
        )

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Validation step."""
        stroke_feat, radical_feat, char_feat, text_ids, labels = batch
        embeddings = self.forward(stroke_feat, radical_feat, char_feat, text_ids)

        loss = (
            self.config.stroke_loss_weight * self.loss_fn(embeddings["stroke"], embeddings["text"])
            + self.config.radical_loss_weight
            * self.loss_fn(embeddings["radical"], embeddings["text"])
            + self.config.character_loss_weight
            * self.loss_fn(embeddings["character"], embeddings["text"])
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        all_params = list(self.encoders.parameters())
        optimizer = AdamW(
            all_params,
            lr=self.config.learning_rate if hasattr(self.config, "learning_rate") else 1e-3,
            weight_decay=self.config.weight_decay if hasattr(self.config, "weight_decay") else 1e-4,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs if hasattr(self.config, "epochs") else 30,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ============================================================================
# PHASE 6: Restoration Pipeline Lightning Module
# ============================================================================


class RestorationPipelineLightningModule(pl.LightningModule):
    """Lightning module for End-to-End Restoration Pipeline."""

    def __init__(
        self,
        pipeline: nn.Module,
        trainer: nn.Module,
        config: "RestorationPipelineConfig",
    ):
        """Initialize Restoration Pipeline module.

        Args:
            pipeline: DetectionRestorationClassificationPipeline instance
            trainer: PipelineTrainer instance
            config: RestorationPipelineConfig
        """
        super().__init__()
        self.pipeline = pipeline
        self.trainer = trainer
        self.config = config
        self.save_hyperparameters(ignore=["pipeline", "trainer"])

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through pipeline."""
        return self.pipeline(images, return_intermediate=True)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step with multi-task loss."""
        images, detections, clean_images, labels = batch

        outputs = self.trainer(
            image=images,
            detection_targets=detections,
            clean_image=clean_images,
            classification_targets=labels,
        )

        loss = outputs["total_loss"]
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if "detection_loss" in outputs:
            self.log(
                "train_detection_loss", outputs["detection_loss"], on_step=False, on_epoch=True
            )
        if "restoration_loss" in outputs:
            self.log(
                "train_restoration_loss", outputs["restoration_loss"], on_step=False, on_epoch=True
            )
        if "classification_loss" in outputs:
            self.log(
                "train_classification_loss",
                outputs["classification_loss"],
                on_step=False,
                on_epoch=True,
            )

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Validation step (classification only)."""
        images, detections, clean_images, labels = batch

        outputs = self.pipeline(images)
        logits = outputs["logits"]

        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = AdamW(
            self.pipeline.parameters(),
            lr=self.config.learning_rate if hasattr(self.config, "learning_rate") else 1e-3,
            weight_decay=self.config.weight_decay if hasattr(self.config, "weight_decay") else 1e-4,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs if hasattr(self.config, "epochs") else 30,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
