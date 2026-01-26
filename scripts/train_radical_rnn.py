#!/usr/bin/env python3
"""
Radical RNN Training for Kanji Recognition
Leverages radical (component) decomposition + RNN for efficient character recognition.
Target: 3-5 MB model, 96-99% accuracy

Features:
- Automatic checkpoint management with resume from latest checkpoint
- Dataset auto-detection (combined_all_etl, etl9g, etl8g, etl7, etl6, etl1)
- NVIDIA GPU required with CUDA optimizations enabled

Configuration parameters are documented inline.
For more info: See GITHUB_IMPLEMENTATION_REFERENCES.md Section 2
Reference papers: RAN 2017, DenseRAN 2018, STAR 2022, RSST 2022, MegaHan97K 2025
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.lib import (
    CheckpointManager,
    RadicalRNNConfig,
    get_dataset_directory,
    get_optimizer,
    get_scheduler,
    prepare_dataset_and_loaders,
    save_best_model,
    save_config,
    save_training_results,
    setup_checkpoint_arguments,
    setup_logger,
    verify_and_setup_gpu,
)

logger = setup_logger(__name__)

# ============================================================================
# RADICAL EXTRACTION
# ============================================================================


class RadicalExtractor:
    """
    Simple radical (component) extractor for Kanji.

    ==================== RADICAL DECOMPOSITION THEORY ====================

    Kanji characters are composed of smaller components called radicals.
    Example: 明 (brightness) = 日 (sun) + 月 (moon)

    Benefits:
    1. Reduces complexity: 3000+ characters → 500 radicals
    2. Improves generalization: Recognizing 500 radicals easier than 3000 chars
    3. Enables zero-shot learning: New chars = combinations of known radicals
    4. More efficient: RNN on 500 elements << CNN on 3000 classes

    ==================== ENCODING TYPES ====================

    1. one_hot: Binary vector (500 dims, k ones) for k radicals
       - Simple, interpretable
       - May lose positional information

    2. binary_tree: Hierarchical tree encoding
       - Uses log(500) ≈ 9 bits per radical
       - Preserves hierarchical structure
       - Good for RNN processing

    3. learned: Embedding vectors from training
       - Task-specific learned representations
       - Can be more expressive than fixed encodings
    """

    def __init__(self, vocab_size: int = 500, encoding_type: str = "binary_tree"):
        self.vocab_size = vocab_size
        self.encoding_type = encoding_type

        # Simple radical mapping: character_id -> list of radical_ids
        # In production, this would come from a linguistic database (Unihan, etc.)
        self._build_radical_map()

    def _build_radical_map(self):
        """Build character to radical mapping"""
        np.random.seed(42)

        # For each character, randomly assign 2-4 radicals
        # (simplified; real databases have linguistically accurate mappings)
        self.char_to_radicals = {}
        for char_id in range(3036):
            num_radicals = np.random.randint(2, 5)
            radicals = np.random.choice(self.vocab_size, num_radicals, replace=False)
            self.char_to_radicals[char_id] = list(radicals)

    def encode_radical(self, radical_id: int) -> np.ndarray:
        """
        Encode a single radical ID according to encoding_type.

        Returns: 1D numpy array representing the radical
        """
        if self.encoding_type == "one_hot":
            # One-hot: All zeros except position radical_id
            encoding = np.zeros(self.vocab_size, dtype=np.float32)
            encoding[radical_id] = 1.0
            return encoding

        elif self.encoding_type == "binary_tree":
            # Binary tree: Use log2(vocab_size) bits
            num_bits = int(np.ceil(np.log2(self.vocab_size)))
            encoding = np.array([(radical_id >> i) & 1 for i in range(num_bits)], dtype=np.float32)
            return encoding

        elif self.encoding_type == "learned":
            # Placeholder: In training, these will be learned embeddings
            # Return index for embedding lookup
            return np.array([radical_id], dtype=np.float32)

        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")

    def get_character_encoding(self, char_id: int) -> Tuple[np.ndarray, int]:
        """
        Get encoding for a character (sequence of radical encodings).

        Returns:
        - encoding: 2D array (num_radicals, encoding_dim)
        - num_radicals: Actual number of radicals for this character
        """
        radicals = self.char_to_radicals.get(char_id, [0])

        # Pad to fixed sequence length (max radicals per character)
        max_radicals = 4  # Most kanji have 2-4 radicals
        padded_radicals = radicals + [0] * (max_radicals - len(radicals))

        encodings = [self.encode_radical(r) for r in padded_radicals[:max_radicals]]
        encoding = np.stack(encodings, axis=0)  # Shape: (max_radicals, encoding_dim)

        return encoding, len(radicals)


# ============================================================================
# RADICAL RNN MODELS
# ============================================================================


class RadicalCNNExtractor(nn.Module):
    """
    CNN backbone for extracting visual features from images.

    ==================== CNN FEATURE EXTRACTION ====================

    - Input: 64x64 grayscale image
    - Output: Feature vector encoding visual patterns
    - Purpose: Convert image pixels → abstract features
    - Architecture: Standard depthwise separable convolutions
    """

    def __init__(self, channels: Tuple[int, ...] = (32, 64, 128)):
        super().__init__()

        # Input channels: 1 (grayscale)
        layers = []
        in_channels = 1

        # Build convolutional layers
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
        # Input: (batch, 1, 64, 64)
        x = self.features(x)
        # Output: (batch, channels[-1], 8, 8)
        x = x.view(x.size(0), -1)  # Flatten: (batch, feature_dim)
        return x


class RadicalRNNClassifier(nn.Module):
    """
    Radical Decomposition + RNN Classifier.

    ==================== ARCHITECTURE ====================

    Stage 1: Visual Feature Extraction
    - CNN processes 64x64 image
    - Output: Visual feature vector (512 dims)

    Stage 2: Visual-to-Radical Mapping
    - Dense layer: Visual features → Radical presence scores
    - Softmax: Which radicals are in this character?

    Stage 3: Radical Sequence Processing
    - RNN processes ordered radical sequence
    - LSTM/GRU: Captures radical compositional structure
    - Output: Character classification

    ==================== WHY RNN FOR RADICALS? ====================

    1. Radicals have structure: Some orders more common than others
    2. Compositional: Character = Ordered sequence of radicals
    3. Efficient: 500 radical classes << 3000 character classes
    4. Interpretable: Can visualize which radicals activate
    """

    def __init__(self, num_classes: int, config: RadicalRNNConfig):
        super().__init__()

        self.num_classes = num_classes
        self.config = config

        # ==================== STAGE 1: VISUAL FEATURE EXTRACTION ====================
        self.cnn = RadicalCNNExtractor(channels=config.cnn_channels)

        # ==================== STAGE 2: VISUAL-TO-RADICAL MAPPING ====================
        # Map visual features to radical presence
        # (batch, visual_features) -> (batch, 4, radical_vocab_size)
        # where 4 is max radicals per character
        self.visual_to_radical = nn.Sequential(
            nn.Linear(self.cnn.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.rnn_dropout),
            nn.Linear(512, 4 * config.radical_vocab_size),
        )

        # ==================== STAGE 3: RADICAL EMBEDDING ====================
        # Embed radical indices to vectors
        # (4, radical_vocab_size) -> (4, embedding_dim)
        self.radical_embedding = nn.Embedding(
            config.radical_vocab_size, config.radical_embedding_dim, padding_idx=0
        )

        # ==================== STAGE 4: RADICAL SEQUENCE PROCESSING ====================
        # Process ordered radical sequence with RNN
        rnn_class = nn.LSTM if config.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_class(
            input_size=config.radical_embedding_dim,
            hidden_size=config.rnn_hidden_size,
            num_layers=config.rnn_num_layers,
            dropout=config.rnn_dropout if config.rnn_num_layers > 1 else 0,
            batch_first=True,
        )

        # ==================== STAGE 5: CLASSIFICATION HEAD ====================
        # RNN output -> Character class
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(config.rnn_hidden_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Reshape to image format if needed
        if len(x.shape) == 2:
            x = x.view(batch_size, 1, 64, 64)

        # ========== STAGE 1: VISUAL EXTRACTION ==========
        visual_features = self.cnn(x)  # (batch, feature_dim)

        # ========== STAGE 2: VISUAL TO RADICAL MAPPING ==========
        # Predict which radicals are in this character
        radical_logits = self.visual_to_radical(visual_features)
        # (batch, 4 * radical_vocab_size)
        radical_logits = radical_logits.view(batch_size, 4, self.config.radical_vocab_size)
        # (batch, 4, radical_vocab_size)

        # Select top-k radicals for each character
        radical_ids = torch.topk(radical_logits, k=1, dim=2)[1].squeeze(2)
        # (batch, 4) - indices of selected radicals

        # ========== STAGE 3: RADICAL EMBEDDING ==========
        embedded_radicals = self.radical_embedding(radical_ids)
        # (batch, 4, embedding_dim)

        # ========== STAGE 4: RADICAL SEQUENCE PROCESSING ==========
        rnn_out, (h_n, c_n) = self.rnn(embedded_radicals)
        # rnn_out: (batch, 4, rnn_hidden_size)
        # Use last hidden state for classification
        final_hidden = h_n[-1] if isinstance(h_n, tuple) else h_n[-1]
        # (batch, rnn_hidden_size)

        # ========== STAGE 5: CLASSIFICATION ==========
        logits = self.classifier(final_hidden)  # (batch, num_classes)

        return logits


# ============================================================================
# TRAINING UTILITIES
# ============================================================================


class RadicalRNNTrainer:
    """Helper class for Radical RNN training"""

    def __init__(self, model: nn.Module, config: RadicalRNNConfig, device: str = "cuda"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def train_epoch(self, train_loader: DataLoader, optimizer, criterion, epoch: int):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        with tqdm(train_loader, desc=f"Epoch {epoch} Train") as pbar:
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.1f}%"}
                )

        avg_loss = total_loss / len(train_loader)
        avg_acc = 100.0 * correct / total

        return avg_loss, avg_acc

    def validate(self, val_loader: DataLoader, criterion):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        avg_acc = 100.0 * correct / total

        return avg_loss, avg_acc


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Radical RNN Training for Kanji Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_radical_rnn.py
  python train_radical_rnn.py --rnn-type lstm --rnn-hidden-size 256
  python train_radical_rnn.py --radical-vocab-size 500 --batch-size 32 --checkpoint-dir training/rnn/checkpoints
  python train_radical_rnn.py --no-checkpoint --epochs 50
        """,
    )

    # Dataset (auto-detected from common location)
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit samples for testing")

    # Model
    parser.add_argument("--image-size", type=int, default=64, help="Input image size (default: 64)")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=43528,
        help="Number of character classes (default: 43,528 for combined ETL6-9 dataset)",
    )

    # Radical parameters
    parser.add_argument(
        "--radical-vocab-size",
        type=int,
        default=500,
        help="Number of unique radicals (default: 500)",
    )
    parser.add_argument(
        "--radical-embedding-dim",
        type=int,
        default=128,
        help="Radical embedding dimension (default: 128)",
    )
    parser.add_argument(
        "--radical-encoding-type",
        type=str,
        default="binary_tree",
        choices=["one_hot", "binary_tree", "learned"],
        help="How to encode radicals (default: binary_tree)",
    )

    # RNN parameters
    parser.add_argument(
        "--rnn-type",
        type=str,
        default="lstm",
        choices=["lstm", "gru"],
        help="RNN type (default: lstm)",
    )
    parser.add_argument(
        "--rnn-hidden-size", type=int, default=256, help="RNN hidden size (default: 256)"
    )
    parser.add_argument(
        "--rnn-num-layers", type=int, default=2, help="Number of RNN layers (default: 2)"
    )
    parser.add_argument("--rnn-dropout", type=float, default=0.3, help="RNN dropout (default: 0.3)")

    # CNN parameters
    parser.add_argument(
        "--cnn-channels",
        type=str,
        default="32,64,128",
        help="CNN channels per layer (default: 32,64,128)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=30, help="Total training epochs (default: 30)"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Initial learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="L2 regularization (default: 1e-4)"
    )

    # Optimizer & scheduler
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
        help="LR scheduler (default: cosine)",
    )

    # Output
    parser.add_argument(
        "--model-dir",
        type=str,
        default="training/radical_rnn/config",
        help="Directory to save model config (default: training/radical_rnn/config)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="training/radical_rnn/results",
        help="Directory to save results (default: training/radical_rnn/results)",
    )

    # Add checkpoint management arguments
    setup_checkpoint_arguments(parser, "rnn")

    args = parser.parse_args()

    # Auto-detect dataset directory
    data_dir = str(get_dataset_directory())
    logger.info(f"Using dataset from: {data_dir}")

    # Parse CNN channels
    cnn_channels = tuple(map(int, args.cnn_channels.split(",")))

    # ========== CREATE CONFIG ==========
    config = RadicalRNNConfig(
        data_dir=data_dir,
        image_size=args.image_size,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        radical_vocab_size=args.radical_vocab_size,
        radical_embedding_dim=args.radical_embedding_dim,
        radical_encoding_type=args.radical_encoding_type,
        rnn_type=args.rnn_type,
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_num_layers=args.rnn_num_layers,
        rnn_dropout=args.rnn_dropout,
        cnn_channels=cnn_channels,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        model_dir=args.model_dir,
        results_dir=args.results_dir,
    )

    # ========== VERIFY GPU ==========
    verify_and_setup_gpu()

    logger.info("=" * 70)
    logger.info("RADICAL RNN TRAINING FOR KANJI RECOGNITION")
    logger.info("=" * 70)
    logger.info("📋 CONFIGURATION:")
    logger.info(f"  Data: {config.data_dir}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Radical vocab: {config.radical_vocab_size}")
    logger.info(
        f"  RNN: {config.rnn_type} (hidden: {config.rnn_hidden_size}, layers: {config.rnn_num_layers})"
    )
    logger.info(f"  CNN channels: {config.cnn_channels}")
    logger.info(f"  Optimizer: {config.optimizer}, Scheduler: {config.scheduler}")

    # ========== LOAD DATA ==========
    logger.info("📂 LOADING DATASET (auto-detecting best available)...")

    # Define dataset factory for radical RNN processing
    def create_radical_dataset(x: np.ndarray, y: np.ndarray):
        """Factory for creating datasets compatible with RadicalRNN.
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

    # Load and prepare datasets using unified helper
    (X, y), num_classes, train_loader, val_loader = prepare_dataset_and_loaders(
        data_dir=data_dir,
        dataset_fn=create_radical_dataset,
        batch_size=config.batch_size,
        sample_limit=args.sample_limit,
        logger=logger,
    )

    # ========== CREATE MODEL ==========
    logger.info("🧠 CREATING MODEL...")
    device = torch.device(config.device)
    model = RadicalRNNClassifier(num_classes=config.num_classes, config=config)

    # ========== TRAINING SETUP ==========
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    trainer = RadicalRNNTrainer(model, config, device=str(device))

    # ========== SAVE CONFIG ==========
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)
    save_config(config, config.model_dir, "radical_rnn_config.json")

    # ========== INITIALIZE CHECKPOINT MANAGER ==========
    checkpoint_manager = CheckpointManager(args.checkpoint_dir, "rnn")

    # ========== TRAINING LOOP ==========
    logger.info("🚀 TRAINING...")
    best_val_acc = 0.0
    best_model_path = Path(config.model_dir) / "radical_rnn_model_best.pth"

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
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion, epoch)
        val_loss, val_acc = trainer.validate(val_loader, criterion)

        trainer.history["train_loss"].append(train_loss)
        trainer.history["train_acc"].append(train_acc)
        trainer.history["val_loss"].append(val_loss)
        trainer.history["val_acc"].append(val_acc)

        scheduler.step()

        logger.info(
            f"Epoch {epoch}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
        )

        if save_best_model(model, val_acc, best_val_acc, best_model_path):
            best_val_acc = val_acc

        # Save checkpoint after each epoch for resuming later
        checkpoint_manager.save_checkpoint(
            epoch - 1,  # Convert to 0-indexed
            model,
            optimizer,
            scheduler,
            metrics={
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            },
            is_best=(val_acc > best_val_acc),
        )
        # Clean up old checkpoints
        checkpoint_manager.cleanup_old_checkpoints(keep_last_n=5)

    # ========== VALIDATION SUMMARY ==========
    logger.info("✅ TRAINING COMPLETE")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")

    # ========== RESULTS ==========
    save_training_results(
        config,
        best_val_acc,
        trainer.history.get("final_test_acc", 0.0),
        trainer.history.get("final_test_loss", float("inf")),
        trainer.history,
        Path(config.results_dir) / "radical_rnn_results.json",
    )

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
                config.results_dir,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("✓ Character mapping created successfully")
        else:
            logger.warning(f"⚠ Character mapping creation failed: {result.stderr}")
    except Exception as e:
        logger.warning(f"⚠ Could not create character mapping: {e}")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
