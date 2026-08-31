"""
Phase 6: Restoration-Guided Multi-Stage Training Pipeline

Orchestrates end-to-end pipeline: Detection → Restoration → Classification
Jointly trains all stages or can freeze individual stages for staged training.

Based on: Restoration-Guided Kuzushiji Character Recognition Framework under Seal
Interference (Ju, Yamashita, Kameko, & Mori, February 2026)
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for restoration pipeline."""

    # Stage training strategy
    training_strategy: str = "end_to_end"  # "staged" | "end_to_end" | "alternating"

    # Loss weights
    detection_loss_weight: float = 0.2
    restoration_loss_weight: float = 0.3
    classification_loss_weight: float = 0.5

    # Stage freezing (for staged training)
    freeze_detector: bool = False
    freeze_restorer: bool = False
    freeze_classifier: bool = False

    # Inference settings
    detection_threshold: float = 0.5
    nms_iou_threshold: float = 0.5
    top_k_detections: int = 100


class SimpleYOLODetector(nn.Module):
    """
    Simplified YOLO-style detector for character and seal detection.

    In production, would use ultralytics.YOLO, but this is a learnable wrapper.
    """

    def __init__(
        self,
        num_classes: int = 2,  # 0=character, 1=seal
        num_anchors: int = 3,
    ):
        """
        Initialize detector.

        Args:
            num_classes: Number of detection classes
            num_anchors: Number of anchor boxes per grid cell
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Backbone (simplified)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )

        # Detection head (predicts: x, y, w, h, objectness, class_probs)
        output_channels = num_anchors * (5 + num_classes)
        self.detection_head = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, output_channels, 1),
        )

    def forward(self, image: Tensor) -> Tensor:
        """
        Detect objects in image.

        Args:
            image: (B, C, H, W) image tensor

        Returns:
            detections: (B, num_detections, 5+num_classes) detections
        """
        # Backbone
        features = self.backbone(image)  # (B, 128, H/4, W/4)

        # Detection head
        predictions = self.detection_head(features)  # (B, num_anchors*(5+num_classes), H/4, W/4)

        b, _, h, w = predictions.shape

        # Reshape to (B, num_detections, 5+num_classes)
        predictions = predictions.view(b, self.num_anchors, 5 + self.num_classes, h, w)
        predictions = predictions.permute(0, 3, 4, 1, 2).contiguous()
        predictions = predictions.view(b, -1, 5 + self.num_classes)

        return predictions


class RestorationHead(nn.Module):
    """
    Restoration module for removing degradation/seals.

    Simplifi ed GAN-inspired restoration network.
    """

    def __init__(self, input_channels: int = 3):
        """
        Initialize restoration head.

        Args:
            input_channels: Number of input channels
        """
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )

        # Decoder (restore image)
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, input_channels, 3, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, image: Tensor) -> Tensor:
        """
        Restore degraded image.

        Args:
            image: (B, C, H, W) degraded image

        Returns:
            restored: (B, C, H, W) restored image
        """
        encoded = self.encoder(image)
        restored = self.decoder(encoded)
        return restored


class ClassificationHead(nn.Module):
    """Character classification head."""

    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 3036,
    ):
        """
        Initialize classifier.

        Args:
            feature_dim: Feature dimension
            num_classes: Number of classes
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, features: Tensor) -> Tensor:
        """
        Classify characters.

        Args:
            features: (B, feature_dim) features

        Returns:
            logits: (B, num_classes) classification logits
        """
        return self.classifier(features)


class DetectionRestorationClassificationPipeline(nn.Module):
    """
    Complete three-stage pipeline orchestrator.
    """

    def __init__(
        self,
        backbone: nn.Module,  # Existing CNN/ViT backbone
        detector: SimpleYOLODetector,
        restorer: RestorationHead,
        classifier: ClassificationHead,
        config: PipelineConfig,
    ):
        """
        Initialize pipeline.

        Args:
            backbone: Pre-trained feature extractor
            detector: Detection module
            restorer: Restoration module
            classifier: Classification head
            config: Pipeline configuration
        """
        super().__init__()

        self.backbone = backbone
        self.detector = detector
        self.restorer = restorer
        self.classifier = classifier
        self.config = config

        # Apply freezing if specified
        if config.freeze_detector:
            for p in self.detector.parameters():
                p.requires_grad = False
        if config.freeze_restorer:
            for p in self.restorer.parameters():
                p.requires_grad = False
        if config.freeze_classifier:
            for p in self.classifier.parameters():
                p.requires_grad = False

    def forward(
        self,
        image: Tensor,
        return_intermediate: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Forward pass through complete pipeline.

        Args:
            image: (B, C, H, W) input image
            return_intermediate: Whether to return intermediate stages

        Returns:
            Dictionary with pipeline outputs
        """
        outputs = {"original_image": image}

        # Stage 1: Detection
        detections = self.detector(image)  # (B, num_detections, 5+num_classes)
        outputs["detections"] = detections

        if return_intermediate:
            outputs["detection_features"] = detections

        # Stage 2: Restoration
        restored_image = self.restorer(image)  # (B, C, H, W)
        outputs["restored_image"] = restored_image

        if return_intermediate:
            outputs["restoration_features"] = restored_image

        # Stage 3: Classification
        # Extract features from backbone
        features = self.backbone(restored_image)  # (B, feature_dim, H, W) or (B, feature_dim)

        # Global average pooling if needed
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.shape[0], -1)

        # Classify
        logits = self.classifier(features)
        outputs["logits"] = logits

        if return_intermediate:
            outputs["classification_features"] = features

        return outputs


class PipelineTrainer(nn.Module):
    """
    Trainer module for the complete pipeline with loss computation.
    """

    def __init__(
        self,
        pipeline: DetectionRestorationClassificationPipeline,
        config: PipelineConfig,
    ):
        """
        Initialize trainer.

        Args:
            pipeline: The pipeline module
            config: Training configuration
        """
        super().__init__()

        self.pipeline = pipeline
        self.config = config

        # Loss functions
        self.detection_loss = nn.BCEWithLogitsLoss()
        self.restoration_loss = nn.L1Loss()
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        image: Tensor,
        detection_targets: Optional[Tensor] = None,
        clean_image: Optional[Tensor] = None,
        classification_targets: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass with loss computation.

        Args:
            image: (B, C, H, W) input (degraded) image
            detection_targets: (B, num_detections, 5+num_classes) detection targets
            clean_image: (B, C, H, W) clean reference image for restoration
            classification_targets: (B,) character class targets

        Returns:
            Dictionary with outputs and losses
        """
        # Forward through pipeline
        outputs = self.pipeline(image, return_intermediate=True)

        losses = {}
        total_loss = 0

        # Detection loss
        if detection_targets is not None:
            detection_loss = self.detection_loss(outputs["detections"], detection_targets)
            losses["detection_loss"] = detection_loss
            total_loss += self.config.detection_loss_weight * detection_loss

        # Restoration loss (L1 distance to clean image)
        if clean_image is not None:
            restoration_loss = self.restoration_loss(outputs["restored_image"], clean_image)
            losses["restoration_loss"] = restoration_loss
            total_loss += self.config.restoration_loss_weight * restoration_loss

        # Classification loss
        if classification_targets is not None:
            classification_loss = self.classification_loss(
                outputs["logits"], classification_targets
            )
            losses["classification_loss"] = classification_loss
            total_loss += self.config.classification_loss_weight * classification_loss

        losses["total_loss"] = total_loss

        return {**outputs, **losses}


# ==================== Integration ====================


def create_pipeline(
    backbone: nn.Module,
    config: PipelineConfig = None,
) -> Tuple[DetectionRestorationClassificationPipeline, PipelineTrainer]:
    """
    Factory function to create complete pipeline.

    Args:
        backbone: Pre-trained feature extraction backbone
        config: Pipeline configuration

    Returns:
        Tuple of (pipeline, trainer)
    """
    if config is None:
        config = PipelineConfig()

    detector = SimpleYOLODetector(num_classes=2)  # Character vs Seal
    restorer = RestorationHead(input_channels=3)
    classifier = ClassificationHead(feature_dim=512, num_classes=3036)

    pipeline = DetectionRestorationClassificationPipeline(
        backbone=backbone,
        detector=detector,
        restorer=restorer,
        classifier=classifier,
        config=config,
    )

    trainer = PipelineTrainer(pipeline, config)

    return pipeline, trainer
