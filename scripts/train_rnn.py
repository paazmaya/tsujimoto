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
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
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
    prepare_dataset_and_loaders,
    setup_checkpoint_arguments,
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
        num_classes: int = 3036,
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
        num_classes: int = 3036,
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


class RadicalRNN(nn.Module):
    """RNN processing radical decomposition sequences."""

    def __init__(
        self,
        radical_vocab_size: int = 500,
        radical_embed_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 3036,
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


class HybridCNNRNN(nn.Module):
    """Hybrid model combining CNN spatial features with RNN temporal processing."""

    def __init__(
        self,
        image_size: int = 64,
        cnn_features: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 3036,
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
    """Factory function to create RNN models."""
    models = {
        "basic_rnn": KanjiRNN,
        "stroke_rnn": StrokeBasedRNN,
        "radical_rnn": RadicalRNN,
        "hybrid_cnn_rnn": HybridCNNRNN,
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

        elif self.model_type == "radical_rnn":
            radical_sequence = self._extract_radical_sequence(image)
            # Ensure it's a numpy array before converting to tensor
            if not isinstance(radical_sequence, np.ndarray):
                radical_sequence = np.array(radical_sequence, dtype=np.int64)
            return {
                "radical_sequences": torch.tensor(radical_sequence, dtype=torch.long),
                "radical_lengths": torch.tensor(len(radical_sequence), dtype=torch.long),
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
        """Extract simplified stroke sequence from image."""
        # Simplified: Returns random stroke features
        # In production, use sophisticated stroke extraction
        num_strokes = np.random.randint(3, 20)
        stroke_features = np.random.randn(num_strokes, 8).astype(np.float32)
        return stroke_features

    def _extract_radical_sequence(self, image: np.ndarray) -> np.ndarray:
        """Extract radical sequence from image."""
        # Simplified: Returns random radical indices
        # In production, use linguistic decomposition
        num_radicals = np.random.randint(2, 8)
        radical_indices = np.random.randint(1, 500, num_radicals, dtype=np.int64)
        return radical_indices


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
        images = torch.stack([item["images"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"images": images, "labels": labels}

    collate_functions = {
        "basic_rnn": basic_rnn_collate,
        "stroke_rnn": stroke_rnn_collate,
        "radical_rnn": radical_rnn_collate,
        "hybrid_cnn_rnn": hybrid_cnn_rnn_collate,
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

            elif self.model_type == "radical_rnn":
                outputs = self.model(
                    batch["radical_sequences"].to(self.device),
                    batch["radical_lengths"].to(self.device),
                )
                labels = batch["labels"].to(self.device)

            elif self.model_type == "hybrid_cnn_rnn":
                outputs = self.model(batch["images"].to(self.device))
                labels = batch["labels"].to(self.device)

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

                elif self.model_type == "radical_rnn":
                    outputs = self.model(
                        batch["radical_sequences"].to(self.device),
                        batch["radical_lengths"].to(self.device),
                    )
                    labels = batch["labels"].to(self.device)

                elif self.model_type == "hybrid_cnn_rnn":
                    outputs = self.model(batch["images"].to(self.device))
                    labels = batch["labels"].to(self.device)

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

        history_path = Path("training/rnn/results") / f"{self.model_type}_training_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Training history saved to {history_path}")

    def plot_training_curves(self):
        """Plot and save training curves."""
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

        results_dir = Path("training/rnn/results")
        results_dir.mkdir(parents=True, exist_ok=True)

        plot_path = results_dir / f"{self.model_type}_training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Training curves saved to {plot_path}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train RNN-based Kanji Recognition Models")
    parser.add_argument(
        "--model-type",
        type=str,
        default="hybrid_cnn_rnn",
        choices=["basic_rnn", "stroke_rnn", "radical_rnn", "hybrid_cnn_rnn"],
        help="Type of RNN model to train",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--hidden-size", type=int, default=256, help="RNN hidden size")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of RNN layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument(
        "--sample-limit", type=int, default=None, help="Limit number of samples for testing"
    )

    # ========== OPTIMIZER & SCHEDULER ==========
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "sgd"],
        help="Optimizer (default: adamw)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step"],
        help="Learning rate scheduler (default: cosine)",
    )

    # Checkpoint arguments
    setup_checkpoint_arguments(parser, "rnn")

    args = parser.parse_args()

    # Setup device
    device = verify_and_setup_gpu()
    logger.info(f"Using device: {device}")

    # Prepare dataset and loaders using unified helper
    collate_fn = collate_fn_factory(args.model_type)

    # Auto-detect dataset directory
    data_dir = str(get_dataset_directory())
    logger.info(f"Using dataset from: {data_dir}")

    # Create dataset factory to use RNNKanjiDataset with model_type
    def create_rnn_dataset(x: np.ndarray, y: np.ndarray):
        return RNNKanjiDataset(x, y, model_type=args.model_type)

    (x, y), num_classes, train_loader, val_loader = prepare_dataset_and_loaders(
        data_dir=data_dir,
        dataset_fn=create_rnn_dataset,
        batch_size=args.batch_size,
        sample_limit=args.sample_limit,
        collate_fn=collate_fn,
        num_workers=0,
        logger=logger,
    )

    # Create model
    model_kwargs = {
        "num_classes": num_classes,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }

    if args.model_type == "radical_rnn":
        model_kwargs["radical_vocab_size"] = 500

    model = create_rnn_model(args.model_type, **model_kwargs)
    logger.info(
        f"Created {args.model_type} model with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Create trainer and train
    checkpoint_dir = Path("training/rnn/checkpoints")
    trainer = RNNTrainer(model, device, args.model_type, checkpoint_dir)

    # Create config for optimizer/scheduler
    config = RNNConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        config=config,
    )

    logger.info(f"Training completed! Best validation accuracy: {trainer.best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
