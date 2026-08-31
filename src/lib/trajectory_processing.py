"""
Phase 4: Online Handwriting Trajectory Training

Processes pen coordinate sequences (x, y, pressure, timing) for stroke-aware learning.
Supports training on digital ink data where trajectory information is available.

Based on: A Stroke-Level Large-Scale Database of Chinese Character Handwriting
(Xu et al., September 2025)

Key contributions:
- Stroke extraction from coordinate sequences
- Trajectory normalization and augmentation
- RNN encoding of stroke sequences
- Hybrid models combining image + trajectory
- Writer variation modeling
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory processing."""

    # Input types
    input_type: str = "hybrid"  # "coordinates" | "images" | "hybrid"
    max_strokes: int = 20
    max_points_per_stroke: int = 100

    # Normalization
    normalize_scale: bool = True
    normalize_speed: bool = True

    # Augmentation
    augmentation_enabled: bool = True
    rotation_range: float = 0.2  # Radians
    scale_range: float = 0.15  # Relative scale
    jitter_std: float = 1.0  # Pixel-level jitter

    # Model
    embedding_dim: int = 128
    rnn_hidden_dim: int = 256
    dropout: float = 0.2


class TrajectoryNormalizer(nn.Module):
    """
    Normalizes trajectory data for consistent model input.

    Handles: scale normalization, speed normalization, coordinate alignment.
    """

    def __init__(self, config: TrajectoryConfig):
        """
        Initialize normalizer.

        Args:
            config: TrajectoryConfig
        """
        super().__init__()
        self.config = config

    def forward(self, coordinates: Tensor) -> Tensor:
        """
        Normalize trajectory coordinates.

        Args:
            coordinates: (num_points, 2) or (num_points, 4) coordinate array
                        Columns: x, y [, pressure, timestamp]

        Returns:
            normalized: (num_points, features) normalized coordinates
        """
        # Extract x, y coordinates
        xy = coordinates[:, :2]

        # Scale normalization (center and scale to [-1, 1])
        if self.config.normalize_scale:
            x_range = xy[:, 0].max() - xy[:, 0].min()
            y_range = xy[:, 1].max() - xy[:, 1].min()
            range_max = max(x_range, y_range, 1.0)  # Avoid division by zero

            xy_center = xy.mean(dim=0)
            xy_norm = (xy - xy_center) / (range_max / 2 + 1e-8)
        else:
            xy_norm = xy

        # Compute velocity/speed
        deltas = torch.diff(xy_norm, dim=0, prepend=xy_norm[:1])
        speed = torch.norm(deltas, dim=1, keepdim=True)

        # Speed normalization
        if self.config.normalize_speed and speed.max() > 0:
            speed = speed / (speed.max() + 1e-8)

        # Concatenate features
        if coordinates.shape[1] >= 4:
            # Include pressure and timestamp
            pressure = coordinates[:, 2:3]
            timestamp = coordinates[:, 3:4]

            # Normalize pressure [0, 1]
            pressure = (pressure - pressure.min()) / (pressure.max() - pressure.min() + 1e-8)

            # Normalize timestamp (relative differences)
            timestamp_diff = torch.diff(timestamp, dim=0, prepend=timestamp[:1])

            features = torch.cat([xy_norm, speed, pressure, timestamp_diff], dim=1)
        else:
            features = torch.cat([xy_norm, speed], dim=1)

        return features


class TrajectoryAugmentation(nn.Module):
    """
    Augments trajectory data for better generalization.

    Applies: rotation, scaling, jittering, time warping.
    """

    def __init__(self, config: TrajectoryConfig):
        """
        Initialize augmentation.

        Args:
            config: TrajectoryConfig
        """
        super().__init__()
        self.config = config

    def forward(self, coordinates: Tensor, training: bool = True) -> Tensor:
        """
        Augment trajectory.

        Args:
            coordinates: (num_points, features) normalized coordinates
            training: Whether to apply augmentation (True) or not

        Returns:
            augmented: (num_points, features) augmented coordinates
        """
        if not training or not self.config.augmentation_enabled:
            return coordinates

        augmented = coordinates.clone()

        # Random rotation
        if self.config.rotation_range > 0:
            angle = np.random.uniform(-self.config.rotation_range, self.config.rotation_range)
            augmented = self._rotate(augmented, angle)

        # Random scaling
        if self.config.scale_range > 0:
            scale = np.random.uniform(1 - self.config.scale_range, 1 + self.config.scale_range)
            augmented[:, :2] *= scale

        # Random jittering
        if self.config.jitter_std > 0:
            jitter = torch.randn_like(augmented[:, :2]) * self.config.jitter_std
            augmented[:, :2] += jitter

        return augmented

    @staticmethod
    def _rotate(coordinates: Tensor, angle: float) -> Tensor:
        """Rotate 2D coordinates."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        rotation_matrix = torch.tensor(
            [[cos_a, -sin_a], [sin_a, cos_a]], dtype=coordinates.dtype, device=coordinates.device
        )

        xy = coordinates[:, :2]
        rotated = torch.matmul(xy, rotation_matrix.T)

        result = coordinates.clone()
        result[:, :2] = rotated

        return result


class StrokeExtractor(nn.Module):
    """
    Extracts individual strokes from continuous coordinate sequence.

    Detects pen-up/pen-down events and segments into strokes.
    """

    def __init__(self):
        """Initialize extractor."""
        super().__init__()

    def forward(
        self,
        coordinates: Tensor,
        pen_state: Optional[Tensor] = None,
    ) -> Tuple[List[Tensor], Tensor]:
        """
        Extract strokes from coordinates.

        Args:
            coordinates: (num_points, 2+) coordinate array
            pen_state: Optional (num_points,) binary array (1=down, 0=up)

        Returns:
            strokes: List of (num_points_i, features) stroke tensors
            stroke_mask: (num_points,) stroke assignment (which stroke each point belongs to)
        """
        num_points = coordinates.shape[0]

        if pen_state is None:
            # Infer pen state from coordinate jumps
            deltas = torch.diff(coordinates[:, :2], dim=0)
            distances = torch.norm(deltas, dim=1)
            threshold = distances.median() * 3  # Large jump = pen-up

            pen_state = torch.ones(num_points, dtype=torch.long, device=coordinates.device)
            pen_state[1:][distances > threshold] = 0

        # Segment into strokes
        stroke_ids = torch.cumsum(1 - pen_state, dim=0)

        # Extract individual strokes
        strokes = []
        for stroke_id in range(stroke_ids.max() + 1):
            mask = stroke_ids == stroke_id
            stroke = coordinates[mask]
            if len(stroke) > 1:  # Only include strokes with at least 2 points
                strokes.append(stroke)

        return strokes, stroke_ids


class StrokeRNNEncoder(nn.Module):
    """
    RNN-based encoder for stroke sequences.

    Processes sequence of strokes with LSTM/GRU to capture writing patterns.
    """

    def __init__(
        self,
        input_dim: int = 4,  # x, y, speed, pressure
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Initialize encoder.

        Args:
            input_dim: Feature dimension per point
            embedding_dim: Embedding dimension
            hidden_dim: RNN hidden dimension
            num_layers: Number of RNN layers
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Point-level embedding
        self.point_embed = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Stroke-level RNN
        self.stroke_rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)

    def forward(
        self,
        strokes: List[Tensor],
        max_strokes: int = 20,
        max_points_per_stroke: int = 100,
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode stroke sequence.

        Args:
            strokes: List of (num_points_i, input_dim) stroke tensors
            max_strokes: Maximum number of strokes to process
            max_points_per_stroke: Maximum points per stroke

        Returns:
            stroke_embeddings: (num_strokes, embedding_dim) stroke representations
            character_embedding: (embedding_dim,) aggregated character embedding
        """
        device = next(self.parameters()).device

        # Limit number of strokes
        strokes = strokes[:max_strokes]
        num_strokes = len(strokes)

        if num_strokes == 0:
            # Return zero embeddings if no strokes
            return (
                torch.zeros(1, self.embedding_dim, device=device),
                torch.zeros(self.embedding_dim, device=device),
            )

        stroke_embeddings_list = []

        for stroke in strokes:
            # Limit points per stroke
            if len(stroke) > max_points_per_stroke:
                stroke = stroke[:max_points_per_stroke]

            # Embed points
            point_emb = self.point_embed(stroke)  # (num_points, embedding_dim)

            # Pad to max_points_per_stroke
            if len(stroke) < max_points_per_stroke:
                padding = torch.zeros(
                    max_points_per_stroke - len(stroke), self.embedding_dim, device=device
                )
                point_emb = torch.cat([point_emb, padding], dim=0)

            # Process with RNN
            point_emb_batch = point_emb.unsqueeze(0)  # (1, num_points, embedding_dim)
            _, (h_n, _) = self.stroke_rnn(point_emb_batch)

            # Use final hidden state as stroke embedding
            stroke_emb = h_n[-1].squeeze(0)  # (hidden_dim,) - squeeze out batch dimension
            stroke_emb = self.output_proj(stroke_emb)  # (embedding_dim,)
            stroke_embeddings_list.append(stroke_emb)

        # Stack stroke embeddings
        stroke_embeddings = torch.stack(
            stroke_embeddings_list, dim=0
        )  # (num_strokes, embedding_dim)

        # Aggregate to character embedding
        character_embedding = stroke_embeddings.mean(dim=0)  # (embedding_dim,)

        return stroke_embeddings, character_embedding


class StrokeAttentionModule(nn.Module):
    """
    Attention mechanism over strokes.

    Learns to weight importance of different strokes for character recognition.
    """

    def __init__(self, embedding_dim: int = 128):
        """
        Initialize attention module.

        Args:
            embedding_dim: Embedding dimension
        """
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads=4,
            batch_first=True,
        )

        self.attention_weights = None

    def forward(self, stroke_embeddings: Tensor) -> Tensor:
        """
        Apply attention over strokes.

        Args:
            stroke_embeddings: (num_strokes, embedding_dim) or (batch, num_strokes, embedding_dim)

        Returns:
            attended_embedding: (embedding_dim,) or (batch, embedding_dim) weighted embedding
        """
        # Add batch dimension if needed
        if stroke_embeddings.dim() == 2:
            stroke_embeddings = stroke_embeddings.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Apply self-attention
        attn_out, attn_weights = self.attention(
            stroke_embeddings,
            stroke_embeddings,
            stroke_embeddings,
        )

        # Store attention weights for visualization
        self.attention_weights = attn_weights

        # Aggregate weighted strokes
        attended = attn_out.mean(dim=1)  # (batch, embedding_dim)

        if squeeze:
            attended = attended.squeeze(0)

        return attended


class HybridTrajectoryVisionModel(nn.Module):
    """
    Combines image and trajectory branches for character recognition.

    Fuses visual features from image with writing trajectory features.
    """

    def __init__(
        self,
        image_feature_dim: int = 512,
        trajectory_embedding_dim: int = 128,
        fusion_dim: int = 256,
        num_classes: int = 3036,
    ):
        """
        Initialize hybrid model.

        Args:
            image_feature_dim: Dimension of image features
            trajectory_embedding_dim: Dimension of trajectory embeddings
            fusion_dim: Dimension after fusion
            num_classes: Number of character classes
        """
        super().__init__()

        # Image branch projection
        self.image_proj = nn.Sequential(
            nn.Linear(image_feature_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Trajectory branch projection
        self.trajectory_proj = nn.Sequential(
            nn.Linear(trajectory_embedding_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim),
        )

        # Classification head
        self.classifier = nn.Linear(fusion_dim, num_classes)

        # Auxiliary: pen pressure prediction (for realism loss)
        self.pressure_predictor = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        image_features: Optional[Tensor] = None,
        trajectory_embedding: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass combining both modalities.

        Args:
            image_features: (batch, image_feature_dim) or None
            trajectory_embedding: (batch, trajectory_embedding_dim) or None

        Returns:
            Dictionary with:
                - logits: (batch, num_classes) classification logits
                - pressure_pred: (batch, 1) predicted pen pressure
                - fused_features: (batch, fusion_dim) fused representation
        """
        if image_features is None and trajectory_embedding is None:
            raise ValueError("At least one modality must be provided")

        outputs = {}

        if image_features is not None and trajectory_embedding is not None:
            # Fuse both modalities
            image_proj = self.image_proj(image_features)
            trajectory_proj = self.trajectory_proj(trajectory_embedding)

            fused = torch.cat([image_proj, trajectory_proj], dim=1)
            fused = self.fusion(fused)
        elif image_features is not None:
            # Image only
            fused = self.image_proj(image_features)
        else:
            # Trajectory only
            fused = self.trajectory_proj(trajectory_embedding)

        outputs["fused_features"] = fused
        outputs["logits"] = self.classifier(fused)
        outputs["pressure_pred"] = self.pressure_predictor(fused)

        return outputs


# ==================== Integration ====================


def create_trajectory_encoder(
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    max_strokes: int = 20,
) -> Tuple[TrajectoryNormalizer, StrokeExtractor, StrokeRNNEncoder]:
    """
    Factory function to create trajectory encoding pipeline.

    Args:
        embedding_dim: Embedding dimension
        hidden_dim: RNN hidden dimension
        max_strokes: Maximum strokes per character

    Returns:
        Tuple of (normalizer, extractor, encoder)
    """
    config = TrajectoryConfig(
        embedding_dim=embedding_dim,
        rnn_hidden_dim=hidden_dim,
        max_strokes=max_strokes,
    )

    normalizer = TrajectoryNormalizer(config)
    extractor = StrokeExtractor()
    encoder = StrokeRNNEncoder(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )

    return normalizer, extractor, encoder
