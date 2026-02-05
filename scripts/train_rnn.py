#!/usr/bin/env python3
"""
RNN-based Kanji Recognition Training

Unified training script for multiple RNN architectures:
- Basic RNN: Spatial sequence processing with bidirectional LSTM
- Stroke RNN: Stroke-order aware temporal processing
- Radical RNN: Radical decomposition with structural awareness
- Hybrid CNN-RNN: Combined spatial-temporal feature learning

Features:
- Automatic checkpoint management with resume capability
- Dataset auto-detection (combined_all_etl, etl9g, etl8g, etl7, etl6)
- Multiple model architectures with configurable hyperparameters
- Training history visualization and metrics tracking
- NVIDIA GPU required with CUDA optimizations enabled

Reference: Multiple RNN variants integrated into unified training script
Configuration parameters are documented inline.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import (
    RNNConfig,
    get_dataset_directory,
    get_optimizer,
    get_scheduler,
    setup_logger,
    verify_and_setup_gpu,
)

logger = setup_logger(__name__)


# ============================================================================
# RNN MODEL ARCHITECTURES
# ============================================================================


class KanjiRNN(nn.Module):
    """Base RNN model for kanji recognition using spatial sequences."""

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 43427,
        rnn_type: str = "LSTM",
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # RNN layer
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        # Output dimensions after RNN
        rnn_output_size = hidden_size * (2 if bidirectional else 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch_size, seq_length, input_size) -> (batch_size, num_classes)"""
        rnn_out, _ = self.rnn(x)

        # Pooling strategy
        if self.bidirectional:
            pooled = torch.mean(rnn_out, dim=1)
        else:
            pooled = rnn_out[:, -1, :]

        output = self.classifier(pooled)
        return output


class StrokeBasedRNN(nn.Module):
    """RNN processing kanji as stroke sequences with temporal awareness."""

    def __init__(
        self,
        stroke_features: int = 8,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 43427,
        max_strokes: int = 30,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.stroke_features = stroke_features
        self.hidden_size = hidden_size
        self.max_strokes = max_strokes

        # Stroke embedding
        self.stroke_embedding = nn.Linear(stroke_features, hidden_size // 2)

        # Bi-directional LSTM for stroke sequences
        self.stroke_lstm = nn.LSTM(
            input_size=hidden_size // 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )

        # Attention mechanism for stroke importance
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, num_heads=8, dropout=dropout, batch_first=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, stroke_sequences: torch.Tensor, stroke_lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass with stroke sequences and lengths."""
        # Embed stroke features
        embedded_strokes = self.stroke_embedding(stroke_sequences)
        embedded_strokes = torch.nn.functional.relu(embedded_strokes)

        # Pack padded sequences for efficient RNN processing
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embedded_strokes, stroke_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM forward pass
        packed_output, _ = self.stroke_lstm(packed_input)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Self-attention over stroke sequence
        attended_output, _ = self.attention(lstm_output, lstm_output, lstm_output)

        # Global max pooling over stroke dimension
        pooled_output = torch.max(attended_output, dim=1)[0]

        # Classification
        output = self.classifier(pooled_output)
        return output


class SimpleRadicalRNN(nn.Module):
    """RNN processing simple radical decomposition sequences (500 vocab)."""

    def __init__(
        self,
        radical_vocab_size: int = 500,
        radical_embed_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 43427,
        max_radicals: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.radical_vocab_size = radical_vocab_size
        self.radical_embed_dim = radical_embed_dim
        self.hidden_size = hidden_size
        self.max_radicals = max_radicals

        # Radical embedding layer
        self.radical_embedding = nn.Embedding(
            num_embeddings=radical_vocab_size, embedding_dim=radical_embed_dim, padding_idx=0
        )

        # Bi-directional LSTM for radical sequences
        self.radical_lstm = nn.LSTM(
            input_size=radical_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(
        self, radical_sequences: torch.Tensor, radical_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with radical sequences."""
        # Embed radical sequences
        embedded_radicals = self.radical_embedding(radical_sequences)

        # Pack padded sequences
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embedded_radicals, radical_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM forward pass
        packed_output, _ = self.radical_lstm(packed_input)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Use mean pooling over radical sequence
        batch_size, max_len, hidden_dim = lstm_output.size()

        # Create mask for padding positions
        mask = torch.arange(max_len, device=radical_sequences.device).expand(batch_size, max_len)
        mask = mask < radical_lengths.unsqueeze(1)

        # Apply mask and compute mean
        masked_output = lstm_output * mask.unsqueeze(-1).float()
        pooled_output = masked_output.sum(dim=1) / radical_lengths.unsqueeze(-1).float()

        # Classification
        output = self.classifier(pooled_output)
        return output


class RadicalCNNExtractor(nn.Module):
    """
    CNN backbone for extracting visual features from images.
    Used by LinguisticRadicalRNN for visual-to-radical mapping.
    """

    def __init__(self, channels: Tuple[int, ...] = (32, 64, 128)):
        super().__init__()

        layers = []
        in_channels = 1

        # Build convolutional layers with depthwise separable convolutions
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        3,
                        stride=2,
                        padding=1,
                        groups=in_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # Calculate feature map size after all stride-2 convolutions
        # 64 -> 32 -> 16 -> 8 (for 3 layers)
        self.feature_dim = in_channels * (64 // (2 ** len(channels))) ** 2

    def forward(self, x):
        """Extract visual features from image."""
        x = self.features(x)  # (batch, channels[-1], 8, 8)
        x = x.view(x.size(0), -1)  # Flatten: (batch, feature_dim)
        return x


class LinguisticRadicalRNN(nn.Module):
    """
    Advanced Radical Decomposition + RNN Classifier with linguistic features.

    This model uses a larger radical vocabulary (2000) and learns to map
    visual features to radical presence through a CNN-to-radical pipeline.

    ==================== ARCHITECTURE ====================

    Stage 1: Visual Feature Extraction
    - CNN processes 64x64 image → Visual feature vector

    Stage 2: Visual-to-Radical Mapping
    - Dense layers: Visual features → Radical presence scores
    - Top-k selection: Which radicals are present?

    Stage 3: Radical Embedding
    - Embed selected radical IDs to dense vectors

    Stage 4: Radical Sequence Processing
    - RNN processes ordered radical sequence
    - Captures compositional structure

    Stage 5: Classification
    - Final character prediction from RNN output

    ==================== WHY LINGUISTIC APPROACH? ====================

    1. Larger vocabulary (2000): More fine-grained radical discrimination
    2. Visual-to-radical learning: Learns radical detection from data
    3. Hierarchical encoding: Captures radical composition patterns
    4. Parameter efficient: 70-80% fewer params than pure CNN
    5. Interpretable: Can visualize which radicals activate
    """

    def __init__(
        self,
        num_classes: int,
        radical_vocab_size: int = 2000,
        radical_embedding_dim: int = 128,
        rnn_type: str = "lstm",
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 2,
        rnn_dropout: float = 0.3,
        cnn_channels: Tuple[int, ...] = (32, 64, 128),
    ):
        super().__init__()

        self.num_classes = num_classes
        self.radical_vocab_size = radical_vocab_size

        # ==================== STAGE 1: VISUAL FEATURE EXTRACTION ====================
        self.cnn = RadicalCNNExtractor(channels=cnn_channels)

        # ==================== STAGE 2: VISUAL-TO-RADICAL MAPPING ====================
        # Map visual features to radical presence
        # (batch, visual_features) -> (batch, 4, radical_vocab_size)
        self.visual_to_radical = nn.Sequential(
            nn.Linear(self.cnn.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(rnn_dropout),
            nn.Linear(512, 4 * radical_vocab_size),  # 4 radicals max per character
        )

        # ==================== STAGE 3: RADICAL EMBEDDING ====================
        self.radical_embedding = nn.Embedding(
            radical_vocab_size, radical_embedding_dim, padding_idx=0
        )

        # ==================== STAGE 4: RADICAL SEQUENCE PROCESSING ====================
        rnn_class = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_class(
            input_size=radical_embedding_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout if rnn_num_layers > 1 else 0,
            batch_first=True,
        )

        # ==================== STAGE 5: CLASSIFICATION HEAD ====================
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(rnn_hidden_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        """Forward pass through the full pipeline."""
        batch_size = x.size(0)

        # Reshape to image format if needed
        if len(x.shape) == 2:
            x = x.view(batch_size, 1, 64, 64)

        # ========== STAGE 1: VISUAL EXTRACTION ==========
        visual_features = self.cnn(x)  # (batch, feature_dim)

        # ========== STAGE 2: VISUAL TO RADICAL MAPPING ==========
        radical_logits = self.visual_to_radical(visual_features)
        radical_logits = radical_logits.view(batch_size, 4, self.radical_vocab_size)

        # Select top radical for each position
        radical_ids = torch.topk(radical_logits, k=1, dim=2)[1].squeeze(2)
        # (batch, 4) - indices of selected radicals

        # ========== STAGE 3: RADICAL EMBEDDING ==========
        embedded_radicals = self.radical_embedding(radical_ids)
        # (batch, 4, embedding_dim)

        # ========== STAGE 4: RADICAL SEQUENCE PROCESSING ==========
        if isinstance(self.rnn, nn.LSTM):
            rnn_out, (h_n, c_n) = self.rnn(embedded_radicals)
            final_hidden = h_n[-1]  # Last layer's hidden state
        else:  # GRU
            rnn_out, h_n = self.rnn(embedded_radicals)
            final_hidden = h_n[-1]

        # ========== STAGE 5: CLASSIFICATION ==========
        logits = self.classifier(final_hidden)  # (batch, num_classes)

        return logits


class HybridCNNRNN(nn.Module):
    """Hybrid model combining CNN spatial features with RNN temporal processing."""

    def __init__(
        self,
        image_size: int = 64,
        cnn_features: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 43427,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.image_size = image_size
        self.cnn_features = cnn_features
        self.hidden_size = hidden_size

        # CNN feature extractor (simplified ResNet-style)
        self.cnn_backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> 4x4
        )

        # Feature projection
        self.feature_projection = nn.Linear(256 * 4 * 4, cnn_features)

        # RNN for processing spatial features as sequences
        self.spatial_rnn = nn.LSTM(
            input_size=256,  # Features per spatial location
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )

        # Classification head combining both CNN and RNN features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(cnn_features + hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for hybrid CNN-RNN."""
        batch_size = x.size(0)

        # CNN feature extraction
        cnn_features = self.cnn_backbone(x)  # (batch_size, 256, 4, 4)

        # Global CNN features
        global_features = cnn_features.view(batch_size, -1)  # (batch_size, 256*4*4)
        global_features = self.feature_projection(global_features)  # (batch_size, cnn_features)

        # Prepare spatial features for RNN
        # Treat each spatial location as a sequence element
        spatial_features = cnn_features.view(batch_size, 256, -1)  # (batch_size, 256, 16)
        spatial_features = spatial_features.transpose(1, 2)  # (batch_size, 16, 256)

        # RNN processing of spatial sequence
        rnn_output, _ = self.spatial_rnn(spatial_features)

        # Pool RNN output
        rnn_pooled = torch.mean(rnn_output, dim=1)  # (batch_size, hidden_size*2)

        # Combine CNN and RNN features
        combined_features = torch.cat([global_features, rnn_pooled], dim=1)

        # Classification
        output = self.classifier(combined_features)
        return output


def create_rnn_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create RNN models.

    Supported variants:
    - basic_rnn: Spatial grid scanning (KanjiRNN)
    - stroke_rnn: Stroke sequence processing (StrokeBasedRNN)
    - simple_radical_rnn: Simple radical decomposition with 500 vocab (SimpleRadicalRNN)
    - hybrid_cnn_rnn: Combined CNN-RNN architecture (HybridCNNRNN)
    - linguistic_radical_rnn: Advanced radical decomposition with 2000 vocab (LinguisticRadicalRNN)
    """
    models = {
        "basic_rnn": KanjiRNN,
        "stroke_rnn": StrokeBasedRNN,
        "simple_radical_rnn": SimpleRadicalRNN,
        "hybrid_cnn_rnn": HybridCNNRNN,
        "linguistic_radical_rnn": LinguisticRadicalRNN,
    }

    if model_type not in models:
        available = ", ".join(models.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")

    return models[model_type](**kwargs)


# ============================================================================
# DATASET CLASSES
# ============================================================================


class RNNKanjiDataset(Dataset):
    """Dataset for RNN-based kanji recognition."""

    def __init__(self, x, y, model_type: str = "basic_rnn", image_size: int = 64):
        self.X = x
        self.y = y
        self.model_type = model_type
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image = self.X[idx].reshape(self.image_size, self.image_size)
        label = self.y[idx]

        if self.model_type == "basic_rnn":
            # Convert image to spatial sequence
            sequence = self._image_to_spatial_sequence(image)
            return {
                "sequences": torch.tensor(sequence, dtype=torch.float32),
                "labels": torch.tensor(label, dtype=torch.long),
            }

        elif self.model_type == "stroke_rnn":
            stroke_sequence = self._extract_stroke_sequence(image)
            # Ensure it's a numpy array before converting to tensor
            if not isinstance(stroke_sequence, np.ndarray):
                stroke_sequence = np.array(stroke_sequence, dtype=np.float32)
            return {
                "stroke_sequences": torch.tensor(stroke_sequence, dtype=torch.float32),
                "stroke_lengths": torch.tensor(len(stroke_sequence), dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.long),
            }

        elif self.model_type == "simple_radical_rnn":
            radical_sequence = self._extract_radical_sequence(image, vocab_size=500)
            # Ensure it's a numpy array before converting to tensor
            if not isinstance(radical_sequence, np.ndarray):
                radical_sequence = np.array(radical_sequence, dtype=np.int64)
            return {
                "radical_sequences": torch.tensor(radical_sequence, dtype=torch.long),
                "radical_lengths": torch.tensor(len(radical_sequence), dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.long),
            }

        elif self.model_type == "linguistic_radical_rnn":
            # Linguistic radical RNN processes raw images (CNN does feature extraction)
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            return {
                "images": image_tensor,
                "labels": torch.tensor(label, dtype=torch.long),
            }

        elif self.model_type == "hybrid_cnn_rnn":
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            return {
                "images": image_tensor,
                "labels": torch.tensor(label, dtype=torch.long),
            }

        raise ValueError(f"Unknown model type: {self.model_type}")

    def _image_to_spatial_sequence(self, image: np.ndarray) -> np.ndarray:
        """Convert image to spatial sequence using grid-based scanning."""
        h, w = image.shape
        grid_size = 8
        cell_h, cell_w = h // grid_size, w // grid_size

        # Flatten into a long sequence of features
        # 64 cells × 4 features = 256 timesteps with 4 features each
        # OR: reshape to 64 timesteps with 16 features each
        sequence = []
        for i in range(grid_size):
            for j in range(grid_size):
                cell = image[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
                # Use mean intensity and variance as features
                cell_features = [
                    np.mean(cell),
                    np.std(cell),
                    np.max(cell),
                    np.min(cell),
                ]
                sequence.append(cell_features)

        # Convert to numpy array (64, 4) - 64 timesteps, 4 features each
        return np.array(sequence, dtype=np.float32)

    def _extract_stroke_sequence(self, image: np.ndarray) -> np.ndarray:
        """
        Extract stroke-like features from image using contour analysis.

        This is a simplified approach that extracts visual features from the image
        rather than true stroke order. For production with true stroke order data,
        integrate KanjiVG or similar stroke databases.

        Note: ETL images are white strokes on black background (low values = background,
        high values = strokes).
        """
        # For ETL images: high values are strokes, low values are background
        # Normalize to 0-1 if not already
        if image.max() > 1.0:
            image = image / 255.0

        # Threshold: values above mean are likely strokes
        threshold = np.mean(image) + 0.1  # Slightly above mean to focus on strokes
        binary = (image > threshold).astype(np.uint8)

        # Use a coarser grid to get meaningful stroke segments (8x8 = 64 regions)
        h, w = image.shape
        grid_size = 8
        cell_h, cell_w = h // grid_size, w // grid_size

        sequence = []

        # Scan in reading order (top to bottom, left to right)
        for i in range(grid_size):
            for j in range(grid_size):
                cell = image[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
                binary_cell = binary[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]

                # Skip nearly empty cells
                if np.sum(binary_cell) < 2:
                    continue

                # Extract 8 features per cell
                features = [
                    np.mean(cell),  # Average intensity
                    np.std(cell),  # Intensity variation
                    np.sum(binary_cell) / (cell_h * cell_w),  # Stroke density
                    i / grid_size,  # Vertical position (normalized)
                    j / grid_size,  # Horizontal position (normalized)
                    np.max(cell),  # Peak intensity
                    np.mean(cell[cell > threshold])
                    if np.any(cell > threshold)
                    else 0,  # Mean of stroke pixels
                    np.sum(binary_cell > 0) / (cell_h * cell_w),  # Non-zero pixel ratio
                ]
                sequence.append(features)

        # If we got too few features, add some from the whole image
        if len(sequence) < 5:
            # Add global features
            global_features = [
                np.mean(image),
                np.std(image),
                np.sum(binary) / image.size,
                0.5,
                0.5,  # Center position
                np.max(image),
                np.mean(image[image > threshold]) if np.any(image > threshold) else 0,
                np.sum(image > 0) / image.size,
            ]
            sequence.append(global_features)

        stroke_features = np.array(sequence, dtype=np.float32)

        # Limit to max_strokes (30)
        if len(stroke_features) > 30:
            # Keep evenly spaced samples
            indices = np.linspace(0, len(stroke_features) - 1, 30, dtype=int)
            stroke_features = stroke_features[indices]

        return stroke_features

    def _extract_radical_sequence(self, image: np.ndarray, vocab_size: int = 500) -> np.ndarray:
        """
        Extract radical-like features from image using visual decomposition.

        This creates a pseudo-radical representation based on visual regions.
        For production, integrate actual radical databases like KRADFILE or Unihan.

        Args:
            image: Input image as numpy array (ETL: white on black, high values = strokes)
            vocab_size: Radical vocabulary size (500 for simple, 2000 for linguistic)
        """
        # Normalize if needed
        if image.max() > 1.0:
            image = image / 255.0

        # For ETL: high values are strokes
        threshold = np.mean(image) + 0.1

        # Divide image into 4 quadrants and analyze each
        h, w = image.shape
        h_mid, w_mid = h // 2, w // 2

        quadrants = [
            image[:h_mid, :w_mid],  # Top-left
            image[:h_mid, w_mid:],  # Top-right
            image[h_mid:, :w_mid],  # Bottom-left
            image[h_mid:, w_mid:],  # Bottom-right
        ]

        radical_indices = []
        for idx, quad in enumerate(quadrants):
            # Extract features from each quadrant
            stroke_pixels = np.sum(quad > threshold)
            if stroke_pixels < 5:  # Skip nearly empty quadrants
                continue

            density = stroke_pixels / quad.size
            mean_intensity = np.mean(quad[quad > threshold]) if stroke_pixels > 0 else 0
            max_intensity = np.max(quad)

            # Map to pseudo-radical index based on visual features + position
            # This creates consistent mappings for similar visual patterns
            feature_hash = (
                int(
                    (density * 100 + mean_intensity * 50 + max_intensity * 30 + idx * 10)
                    % (vocab_size - 1)
                )
                + 1
            )  # Avoid 0 index

            radical_indices.append(feature_hash)

        # Add center region as potential radical
        center = image[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        center_strokes = np.sum(center > threshold)
        if center_strokes >= 5:
            center_density = center_strokes / center.size
            center_mean = np.mean(center[center > threshold]) if center_strokes > 0 else 0
            center_idx = int((center_density * 150 + center_mean * 80) % (vocab_size - 1)) + 1
            radical_indices.append(center_idx)

        # Ensure we have at least 2 radicals
        if len(radical_indices) < 2:
            # Use global features as fallback
            global_density = np.sum(image > threshold) / image.size
            global_mean = np.mean(image[image > threshold]) if np.any(image > threshold) else 0.1
            radical_indices = [
                int(global_density * (vocab_size - 1)) + 1,
                int(global_mean * (vocab_size - 1)) + 1,
            ]

        return np.array(radical_indices, dtype=np.int64)


# ============================================================================
# COLLATE FUNCTIONS
# ============================================================================


def collate_fn_factory(model_type: str):
    """Factory function to create collate functions for different model types."""

    def basic_rnn_collate(batch):
        sequences = torch.stack(
            [
                item["sequences"]
                if isinstance(item["sequences"], torch.Tensor)
                else torch.tensor(item["sequences"])
                for item in batch
            ]
        )
        labels = torch.stack([item["labels"] for item in batch])
        return {"sequences": sequences, "labels": labels}

    def stroke_rnn_collate(batch):
        # Handle variable-length stroke sequences with padding
        stroke_sequences_list = [
            item["stroke_sequences"]
            if isinstance(item["stroke_sequences"], torch.Tensor)
            else torch.tensor(item["stroke_sequences"], dtype=torch.float32)
            for item in batch
        ]
        stroke_lengths = torch.stack([item["stroke_lengths"] for item in batch])

        # Pad to max length in batch
        max_length = max(seq.size(0) for seq in stroke_sequences_list)
        padded_sequences = []
        for seq in stroke_sequences_list:
            if seq.size(0) < max_length:
                padding = torch.zeros(max_length - seq.size(0), seq.size(1), dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding], dim=0)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)

        stroke_sequences = torch.stack(padded_sequences)
        labels = torch.stack([item["labels"] for item in batch])
        return {
            "stroke_sequences": stroke_sequences,
            "stroke_lengths": stroke_lengths,
            "labels": labels,
        }

    def radical_rnn_collate(batch):
        # Handle variable-length radical sequences with padding
        radical_sequences_list = [
            item["radical_sequences"]
            if isinstance(item["radical_sequences"], torch.Tensor)
            else torch.tensor(item["radical_sequences"], dtype=torch.long)
            for item in batch
        ]
        radical_lengths = torch.stack([item["radical_lengths"] for item in batch])

        # Pad to max length in batch
        max_length = max(seq.size(0) for seq in radical_sequences_list)
        padded_sequences = []
        for seq in radical_sequences_list:
            if seq.size(0) < max_length:
                padding = torch.zeros(max_length - seq.size(0), dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding], dim=0)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)

        radical_sequences = torch.stack(padded_sequences)
        labels = torch.stack([item["labels"] for item in batch])
        return {
            "radical_sequences": radical_sequences,
            "radical_lengths": radical_lengths,
            "labels": labels,
        }

    def hybrid_cnn_rnn_collate(batch):
        # Hybrid CNN-RNN uses raw images
        images = torch.stack([item["images"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"images": images, "labels": labels}

    def linguistic_radical_rnn_collate(batch):
        # Linguistic radical RNN uses raw images (same as hybrid_cnn_rnn)
        images = torch.stack([item["images"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"images": images, "labels": labels}

    collate_functions = {
        "basic_rnn": basic_rnn_collate,
        "stroke_rnn": stroke_rnn_collate,
        "simple_radical_rnn": radical_rnn_collate,
        "hybrid_cnn_rnn": hybrid_cnn_rnn_collate,
        "linguistic_radical_rnn": linguistic_radical_rnn_collate,
    }

    return collate_functions[model_type]


# ============================================================================
# TRAINER
# ============================================================================


class RNNTrainer:
    """Trainer class for RNN models."""

    def __init__(
        self,
        model: nn.Module,
        device: str,
        model_type: str,
        checkpoint_dir: Path,
    ):
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0

    def train_epoch(
        self, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc="Training")

        for batch in progress_bar:
            optimizer.zero_grad()

            # Forward pass based on model type
            if self.model_type == "basic_rnn":
                outputs = self.model(batch["sequences"].to(self.device))
                labels = batch["labels"].to(self.device)

            elif self.model_type == "stroke_rnn":
                outputs = self.model(
                    batch["stroke_sequences"].to(self.device),
                    batch["stroke_lengths"].to(self.device),
                )
                labels = batch["labels"].to(self.device)

            elif self.model_type == "simple_radical_rnn":
                outputs = self.model(
                    batch["radical_sequences"].to(self.device),
                    batch["radical_lengths"].to(self.device),
                )
                labels = batch["labels"].to(self.device)

            elif self.model_type in ("hybrid_cnn_rnn", "linguistic_radical_rnn"):
                outputs = self.model(batch["images"].to(self.device))
                labels = batch["labels"].to(self.device)

            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            # Compute loss and backpropagate
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / num_batches

    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Forward pass based on model type
                if self.model_type == "basic_rnn":
                    outputs = self.model(batch["sequences"].to(self.device))
                    labels = batch["labels"].to(self.device)

                elif self.model_type == "stroke_rnn":
                    outputs = self.model(
                        batch["stroke_sequences"].to(self.device),
                        batch["stroke_lengths"].to(self.device),
                    )
                    labels = batch["labels"].to(self.device)

                elif self.model_type == "simple_radical_rnn":
                    outputs = self.model(
                        batch["radical_sequences"].to(self.device),
                        batch["radical_lengths"].to(self.device),
                    )
                    labels = batch["labels"].to(self.device)

                elif self.model_type in ("hybrid_cnn_rnn", "linguistic_radical_rnn"):
                    outputs = self.model(batch["images"].to(self.device))
                    labels = batch["labels"].to(self.device)

                else:
                    raise ValueError(f"Unknown model type: {self.model_type}")

                # Compute loss and accuracy
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        config=None,
    ):
        """Full training loop."""
        # Use provided config or create default
        if config is None:
            # Fallback for backward compatibility
            from optimization_config import RNNConfig

            config = RNNConfig(epochs=epochs)

        # Setup optimizer and scheduler using unified functions
        optimizer = get_optimizer(self.model, config)
        scheduler = get_scheduler(optimizer, config)

        criterion = nn.CrossEntropyLoss()

        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)

            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)

            # Update learning rate - handle different scheduler types
            if config.scheduler == "cosine" or config.scheduler == "step":
                scheduler.step()
            else:
                # ReduceLROnPlateau uses metric-based stepping
                scheduler.step(val_acc)

            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(f"best_{self.model_type}_model.pth", epoch, val_acc)

            epoch_time = time.time() - start_time

            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"Time: {epoch_time:.1f}s"
            )

        # Save final model and training history
        self.save_model(f"final_{self.model_type}_model.pth", epochs, self.best_val_acc)
        self.save_training_history()
        self.plot_training_curves()

    def save_model(self, filename: str, epoch: int, val_acc: float):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_accuracy": val_acc,
            "model_type": self.model_type,
        }

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")

    def save_training_history(self):
        """Save training history."""
        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "best_val_acc": self.best_val_acc,
            "model_type": self.model_type,
        }

        results_dir = Path("training") / self.model_type / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        history_path = results_dir / f"{self.model_type}_training_history.json"

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Training history saved to {history_path}")

    def plot_training_curves(self):
        """Plot and save training curves."""
        try:
            import matplotlib

            matplotlib.use("Agg")  # Use non-interactive backend for headless environments
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Loss curves
            ax1.plot(self.train_losses, label="Train Loss")
            ax1.plot(self.val_losses, label="Val Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title(f"{self.model_type.upper()} - Loss Curves")
            ax1.legend()
            ax1.grid(True)

            # Accuracy curve
            ax2.plot(self.val_accuracies, label="Val Accuracy", color="green")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy (%)")
            ax2.set_title(f"{self.model_type.upper()} - Validation Accuracy")
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()

            results_dir = Path("training") / self.model_type / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            # Add date to filename
            date_str = datetime.now().strftime("%Y-%m-%d")
            plot_path = results_dir / f"{self.model_type}_training_curves_{date_str}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Training curves saved to {plot_path}")
        except Exception as e:
            logger.warning(f"Could not save training curves: {e}")


# ============================================================================
# MAIN
# ============================================================================


def train_rnn(args):
    """
    Core RNN training function callable from unified entry point.

    Args:
        args: Namespace or dict-like object with training parameters
    """
    # Setup device
    device = verify_and_setup_gpu()

    # Get parameters with safe defaults
    model_type = getattr(args, "model_type", "hybrid_cnn_rnn")
    batch_size = getattr(args, "batch_size", 32)
    sample_limit = getattr(args, "sample_limit", None)
    hidden_size = getattr(args, "hidden_size", 256)
    num_layers = getattr(args, "num_layers", 2)
    dropout = getattr(args, "dropout", 0.3)
    learning_rate = getattr(args, "learning_rate", 0.001)
    weight_decay = getattr(args, "weight_decay", 1e-4)
    epochs = getattr(args, "epochs", 30)
    optimizer_name = getattr(args, "optimizer", "adamw")
    scheduler_name = getattr(args, "scheduler", "cosine")

    # Prepare dataset and loaders using unified helper
    collate_fn = collate_fn_factory(model_type)

    # Get data_dir from arguments or use default
    data_dir_arg = getattr(args, "data_dir", "dataset")

    # Use specified data_dir or auto-detect if using default
    if data_dir_arg == "dataset":
        data_path = get_dataset_directory()  # Auto-detect
    else:
        data_path = Path(data_dir_arg)  # Use specified

    data_dir = str(data_path)
    logger.info(f"Using dataset from: {data_dir}")

    # Load the raw data
    from sklearn.model_selection import train_test_split

    from src.lib.dataset import load_dataset

    x, y = load_dataset(
        dataset_path=data_dir,
        dataset_name="combined_all_etl",
    )

    # Apply sample limit if specified
    if sample_limit:
        logger.info(f"Limiting dataset to {sample_limit} samples")
        # Just take the first n samples to avoid stratification issues
        x = x[:sample_limit]
        y = y[:sample_limit]

    # Remap labels to be contiguous (0 to num_classes-1)
    unique_labels = np.unique(y)
    label_mapping = {old_label: int(new_label) for new_label, old_label in enumerate(unique_labels)}
    y = np.array([label_mapping[label] for label in y], dtype=np.int64)

    num_classes = len(unique_labels)
    logger.info(f"Number of unique classes: {num_classes}")

    # Split into train/validation
    val_split = 0.1
    try:
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=val_split, random_state=42, stratify=y
        )
    except ValueError:
        logger.warning("Some classes have <2 samples, using non-stratified split")
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=val_split, random_state=42
        )

    # Create Dataset instances
    train_dataset = RNNKanjiDataset(x_train, y_train, model_type=model_type)
    val_dataset = RNNKanjiDataset(x_val, y_val, model_type=model_type)

    # Create DataLoaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    logger.info(
        f"Created loaders: train={len(train_loader)} batches, val={len(val_loader)} batches"
    )

    # Create model
    model_kwargs = {
        "num_classes": num_classes,
    }

    # Add model-specific parameters
    if model_type == "simple_radical_rnn":
        model_kwargs["hidden_size"] = hidden_size
        model_kwargs["num_layers"] = num_layers
        model_kwargs["dropout"] = dropout
        model_kwargs["radical_vocab_size"] = 500
        model_kwargs["radical_embed_dim"] = 128
        model_kwargs["max_radicals"] = 10
    elif model_type == "linguistic_radical_rnn":
        model_kwargs["radical_vocab_size"] = 2000
        model_kwargs["radical_embedding_dim"] = 128
        model_kwargs["rnn_type"] = getattr(args, "rnn_type", "lstm")
        model_kwargs["rnn_hidden_size"] = hidden_size
        model_kwargs["rnn_num_layers"] = num_layers
        model_kwargs["rnn_dropout"] = dropout
        model_kwargs["cnn_channels"] = getattr(args, "cnn_channels", (32, 64, 128))
    elif model_type == "stroke_rnn":
        model_kwargs["hidden_size"] = hidden_size
        model_kwargs["num_layers"] = num_layers
        model_kwargs["dropout"] = dropout
        model_kwargs["stroke_features"] = getattr(args, "stroke_features", 8)
        model_kwargs["max_strokes"] = getattr(args, "max_strokes", 30)
    elif model_type == "hybrid_cnn_rnn":
        model_kwargs["hidden_size"] = hidden_size
        model_kwargs["num_layers"] = num_layers
        model_kwargs["dropout"] = dropout
        model_kwargs["image_size"] = 64
        model_kwargs["cnn_features"] = 512
    else:  # basic_rnn
        model_kwargs["hidden_size"] = hidden_size
        model_kwargs["num_layers"] = num_layers
        model_kwargs["dropout"] = dropout

    model = create_rnn_model(model_type, **model_kwargs)
    logger.info(
        f"Created {model_type} model with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Create trainer with variant-specific checkpoint directory
    checkpoint_dir = Path("training") / model_type / "checkpoints"
    trainer = RNNTrainer(model, device, model_type, checkpoint_dir)

    # Create config for optimizer/scheduler
    config = RNNConfig(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer_name,
        scheduler=scheduler_name,
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        config=config,
    )

    logger.info(f"Training completed! Best validation accuracy: {trainer.best_val_acc:.2f}%")


def main():
    from scripts.training_args import add_variant_args_to_parser

    parser = argparse.ArgumentParser(description="Train RNN-based Kanji Recognition Models")

    # Add all arguments for RNN variant from centralized config
    # checkpoint_dir is set dynamically based on model_type
    add_variant_args_to_parser(
        parser, "rnn", checkpoint_dir_default="training/<model_type>/checkpoints"
    )

    args = parser.parse_args()
    train_rnn(args)


if __name__ == "__main__":
    main()
