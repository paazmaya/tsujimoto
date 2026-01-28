#!/usr/bin/env python3
"""
Vision Transformer (ViT) Training with Tokens-to-Token (T2T) for Kanji Recognition
Uses efficient transformer-based architecture for high accuracy recognition.
Target: 8-15 MB model, 97-99% accuracy

Features:
- Automatic checkpoint management with resume from latest checkpoint
- Dataset auto-detection (combined_all_etl, etl9g, etl8g, etl7, etl6, etl1)
- NVIDIA GPU required with CUDA optimizations enabled (TF32, cuDNN benchmarking)

Configuration parameters are documented inline.
For more info: See GITHUB_IMPLEMENTATION_REFERENCES.md Section 4
Reference: T2T-ViT 2021 (arXiv:2101.11986v3)
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import (
    CheckpointManager,
    ViTConfig,
    create_data_loaders,
    get_dataset_directory,
    get_optimizer,
    get_scheduler,
    load_chunked_dataset,
    save_best_model,
    save_config,
    save_training_results,
    setup_logger,
    verify_and_setup_gpu,
)

logger = setup_logger(__name__)

# ============================================================================
# VISION TRANSFORMER COMPONENTS
# ============================================================================


class TokensToTokens(nn.Module):
    """
    Tokens-to-Token (T2T) module - simplified for speed.
    """

    def __init__(self, kernel_sizes: Tuple[int, ...] = (3,)):
        super().__init__()

        self.kernel_sizes = kernel_sizes
        self.num_stages = len(kernel_sizes)

        layers = []
        in_channels = 1
        out_channels = 16

        # Single-stage progression
        for _i, kernel_size in enumerate(kernel_sizes):
            padding = kernel_size // 2

            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )

            in_channels = out_channels

        self.t2t_layers = nn.Sequential(*layers)
        self.output_channels = in_channels

    def forward(self, x):
        x = self.t2t_layers(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention - simplified for speed"""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embedding_dim)

        # Final linear projection
        output = self.proj(attn_output)

        return output


class TransformerBlock(nn.Module):
    """
    Single Transformer block: Multi-head attention + Feed-forward network.

    ==================== TRANSFORMER BLOCK ====================

    1. Layer Normalization
    2. Multi-head attention
    3. Residual connection
    4. Layer Normalization
    5. Feed-forward MLP (2 linear layers with activation)
    6. Residual connection
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, num_heads, dropout=attention_dropout)

        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Attention block with residual connection
        x = x + self.attention(self.norm1(x))

        # MLP block with residual connection
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image classification.

    ==================== VISION TRANSFORMER ARCHITECTURE ====================

    1. Image Tokenization:
       - Split image into patches (typically 16x16 for ImageNet)
       - For us (64x64 image, patch_size=8): 64/8 = 8 patches per side = 64 patches
       - Plus 1 class token = 65 tokens total

    2. Positional Embedding:
       - Add learnable positional embeddings to patches
       - Helps model understand spatial layout

    3. Transformer Encoder:
       - Stack of transformer blocks
       - Each block: Multi-head attention + Feed-forward MLP
       - Residual connections and layer normalization

    4. Classification:
       - Extract class token representation
       - Pass through MLP head for classification

    ==================== WHY ViT FOR KANJI? ====================

    - Long-range dependencies: ViT can capture global structure
    - Compositional: Can learn part-whole relationships
    - Scalable: Efficient for modern hardware (GPUs)
    - State-of-the-art: Strong results on classification tasks
    """

    def __init__(self, num_classes: int, config: ViTConfig):
        super().__init__()

        self.num_classes = num_classes
        self.config = config

        # ==================== T2T TOKENIZATION (Optional) ====================
        if config.use_tokens_to_tokens:
            self.t2t = TokensToTokens(kernel_sizes=config.t2t_kernel_sizes)
            t2t_output_channels = self.t2t.output_channels
            image_size_after_t2t = 64  # Remains same due to padding
            num_patches = (image_size_after_t2t // config.patch_size) ** 2
            patch_embed_dim = t2t_output_channels * (config.patch_size**2)
        else:
            self.t2t = None
            num_patches = (64 // config.patch_size) ** 2
            patch_embed_dim = 1 * (config.patch_size**2)  # 1 channel for grayscale kanji

        self.num_patches = num_patches

        # ==================== PATCH EMBEDDING ====================
        # Convert patches to embedding vectors
        self.patch_embed = nn.Linear(patch_embed_dim, config.embedding_dim)

        # ==================== CLASS TOKEN ====================
        # Learnable class token prepended to sequence
        # Similar to BERT: Use its representation for classification
        self.class_token = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))

        # ==================== POSITIONAL EMBEDDING ====================
        # Learnable positional embeddings for each token (including class token)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embedding_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ==================== TRANSFORMER ENCODER ====================
        self.transformer = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_dim=config.embedding_dim,
                    num_heads=config.num_heads,
                    mlp_dim=config.mlp_dim,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                )
                for _ in range(config.num_transformer_layers)
            ]
        )

        # ==================== CLASSIFICATION HEAD ====================
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.classifier = nn.Linear(config.embedding_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Ensure image format: (batch, 1, 64, 64)
        if x.dim() == 2:
            x = x.view(batch_size, 1, 64, 64)

        # ==================== T2T TOKENIZATION ====================
        if self.t2t is not None:
            x = self.t2t(x)

        # ==================== PATCH EMBEDDING ====================
        patches = self._get_patches(x)
        x = self.patch_embed(patches)

        # ==================== PREPEND CLASS TOKEN ====================
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)

        # ==================== ADD POSITIONAL EMBEDDINGS ====================
        x = x + self.pos_embed

        # ==================== TRANSFORMER ENCODER ====================
        x = self.transformer(x)

        # ==================== EXTRACT CLASS TOKEN & CLASSIFY ====================
        class_token_output = x[:, 0]
        x = self.norm(class_token_output)
        logits = self.classifier(x)

        return logits

    def _get_patches(self, x):
        """
        Extract non-overlapping patches from image efficiently.

        Parameters:
        - x: (batch, channels, height, width)

        Returns:
        - patches: (batch, num_patches, patch_dim)
        """
        batch_size, channels, height, width = x.shape
        patch_size = self.config.patch_size

        # Use unfold for efficient patch extraction
        # unfold(dimension, size, step) -> (batch, channels*patch_size*patch_size, num_patches)
        patches = torch.nn.functional.unfold(x, kernel_size=patch_size, stride=patch_size)
        # (batch, channels*patch_size*patch_size, num_patches)

        # Transpose to (batch, num_patches, channels*patch_size*patch_size)
        patches = patches.transpose(1, 2).contiguous()

        return patches


# ============================================================================
# TRAINING UTILITIES
# ============================================================================


class ViTTrainer:
    """Helper class for ViT training"""

    def __init__(self, model: nn.Module, config: ViTConfig, device: str = "cuda"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        # GradScaler for mixed precision training stability
        self.scaler: GradScaler = GradScaler(device)

    def train_epoch(self, train_loader: DataLoader, optimizer, criterion, epoch: int):
        """Train one epoch with mixed precision"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        with tqdm(train_loader, desc=f"Epoch {epoch} Train") as pbar:
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                # Use mixed precision for faster computation
                with autocast(self.device, dtype=torch.float16):
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                # Scale loss for mixed precision
                self.scaler.scale(loss).backward()

                # Gradient clipping to prevent exploding gradients
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer step with scaled gradients
                self.scaler.step(optimizer)
                self.scaler.update()

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
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Mixed precision for validation too
                with autocast(self.device, dtype=torch.float16):
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
        description="Vision Transformer (ViT) Training for Kanji Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_vit.py
  python train_vit.py --embedding-dim 192 --num-heads 6
  python train_vit.py --num-transformer-layers 8 --patch-size 4
  python train_vit.py --use-tokens-to-tokens --t2t-kernel-sizes 3,3,3 --resume-from training/vit/checkpoints/checkpoint_epoch_010.pt
        """,
    )

    from scripts.training_args import add_variant_args_to_parser

    add_variant_args_to_parser(parser, "vit", checkpoint_dir_default="training/vit/checkpoints")

    args = parser.parse_args()
    train_vit(args)


def train_vit(args):
    """
    Core ViT training function callable from unified entry point.

    Args:
        args: Namespace or dict-like object with training parameters
    """
    # Get parameters with safe defaults
    sample_limit = getattr(args, "sample_limit", None)
    image_size = getattr(args, "image_size", 64)
    num_classes = getattr(args, "num_classes", 43528)
    patch_size = getattr(args, "patch_size", 8)
    embedding_dim = getattr(args, "embedding_dim", 64)
    num_heads = getattr(args, "num_heads", 2)
    num_transformer_layers = getattr(args, "num_transformer_layers", 2)
    mlp_dim = getattr(args, "mlp_dim", 256)
    use_tokens_to_tokens = getattr(args, "use_tokens_to_tokens", False)
    t2t_kernel_sizes_str = getattr(args, "t2t_kernel_sizes", "3,3,3")
    dropout = getattr(args, "dropout", 0.05)
    attention_dropout = getattr(args, "attention_dropout", 0.0)
    epochs = getattr(args, "epochs", 30)
    batch_size = getattr(args, "batch_size", 256)
    learning_rate = getattr(args, "learning_rate", 0.0005)
    weight_decay = getattr(args, "weight_decay", 1e-4)
    optimizer_name = getattr(args, "optimizer", "adamw")
    scheduler_name = getattr(args, "scheduler", "cosine")
    model_dir = getattr(args, "model_dir", "training/vit/config")
    results_dir = getattr(args, "results_dir", "training/vit/results")
    checkpoint_dir = getattr(args, "checkpoint_dir", "training/vit/checkpoints")
    resume_from = getattr(args, "resume_from", None)
    no_checkpoint = getattr(args, "no_checkpoint", False)

    # Parse T2T kernel sizes
    t2t_kernel_sizes = tuple(map(int, t2t_kernel_sizes_str.split(",")))

    # Get data_dir from arguments or use default
    data_dir_arg = getattr(args, "data_dir", "dataset")

    # Use specified data_dir or auto-detect if using default
    if data_dir_arg == "dataset":
        data_path = get_dataset_directory()  # Auto-detect
    else:
        data_path = Path(data_dir_arg)  # Use specified

    data_dir = str(data_path)
    logger.info(f"Using dataset from: {data_dir}")

    # ========== CREATE CONFIG ==========
    config = ViTConfig(
        data_dir=data_dir,
        image_size=image_size,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patch_size=patch_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_transformer_layers=num_transformer_layers,
        mlp_dim=mlp_dim,
        use_tokens_to_tokens=use_tokens_to_tokens,
        t2t_kernel_sizes=t2t_kernel_sizes,
        dropout=dropout,
        attention_dropout=attention_dropout,
        optimizer=optimizer_name,
        scheduler=scheduler_name,
        model_dir=model_dir,
        results_dir=results_dir,
    )

    # ========== VERIFY GPU ==========
    verify_and_setup_gpu()

    logger.info("=" * 70)
    logger.info("VISION TRANSFORMER (ViT) TRAINING FOR KANJI RECOGNITION")
    logger.info("=" * 70)
    logger.info("📋 CONFIGURATION:")
    logger.info(f"  Data: {config.data_dir}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Patch size: {config.patch_size} (→ {(64 // config.patch_size) ** 2} patches)")
    logger.info(f"  Embedding dim: {config.embedding_dim}")
    logger.info(f"  Attention heads: {config.num_heads}")
    logger.info(f"  Transformer layers: {config.num_transformer_layers}")
    logger.info(f"  MLP dim: {config.mlp_dim}")
    logger.info(f"  T2T enabled: {config.use_tokens_to_tokens}")
    if config.use_tokens_to_tokens:
        logger.info(f"  T2T kernel sizes: {config.t2t_kernel_sizes}")
    logger.info(f"  Optimizer: {config.optimizer}, Scheduler: {config.scheduler}")

    # ========== LOAD DATA ==========
    logger.info("📂 LOADING DATASET...")
    X, y = load_chunked_dataset(config.data_dir)
    train_loader, val_loader, test_loader = create_data_loaders(
        X, y, config, sample_limit=sample_limit
    )

    # ========== CREATE MODEL ==========
    logger.info("🧠 CREATING MODEL...")
    device = torch.device(config.device)
    model = VisionTransformer(num_classes=config.num_classes, config=config)

    # ========== TRAINING SETUP ==========
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    trainer = ViTTrainer(model, config, device=str(device))

    # ========== SAVE CONFIG ==========
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)
    save_config(config, config.model_dir, "vit_config.json")

    # ========== INITIALIZE CHECKPOINT MANAGER ==========
    checkpoint_manager = CheckpointManager(checkpoint_dir, "vit")

    # ========== TRAINING LOOP ==========
    logger.info("🚀 TRAINING...")
    best_val_acc = 0.0
    best_model_path = Path(config.model_dir) / "vit_model_best.pth"

    # Resume from checkpoint using unified DRY method
    start_epoch, best_metrics = checkpoint_manager.load_checkpoint_for_training(
        model,
        optimizer,
        scheduler,
        device,
        resume_from=resume_from,
        args_no_checkpoint=no_checkpoint,
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
            epoch, model, optimizer, scheduler, {"val_accuracy": val_acc}
        )

    # ========== TESTING ==========
    logger.info("🧪 TESTING...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc = trainer.validate(test_loader, criterion)
    logger.info(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

    # ========== RESULTS ==========
    save_training_results(
        config,
        best_val_acc,
        test_acc,
        test_loss,
        trainer.history,
        Path(config.results_dir) / "vit_results.json",
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
