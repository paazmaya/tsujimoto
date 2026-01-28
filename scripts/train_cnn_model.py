#!/usr/bin/env python3
"""
Lightweight Kanji Recognition Model for Combined ETL Dataset
Optimized for ONNX deployment with variable character classes (3,036-4,154 from ETL datasets)

Features:
- Automatic checkpoint management with resume from latest checkpoint
- Dataset auto-detection with combined_all_etl priority (43,427 classes)
- Scalable classifier head for variable number of classes
- NVIDIA GPU required with CUDA optimizations enabled
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import (
    CheckpointManager,
    CNNConfig,
    get_dataset_directory,
    get_optimizer,
    get_scheduler,
    prepare_dataset_and_loaders,
    setup_logger,
    verify_and_setup_gpu,
)

logger = setup_logger(__name__)


class ETL9GDataset(Dataset):
    """Efficient dataset for ETL9G with memory management"""

    def __init__(self, x, y, augment=False):  # noqa: N803
        self.X = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        if self.augment and torch.rand(1) < 0.3:
            # =========================
            # DATA AUGMENTATION ALGORITHMS - ADJUSTABLE
            # =========================
            # Current: Simple Gaussian noise augmentation
            # Alternatives: rotation, elastic deformation, stroke width variation, shearing
            # Probability: 0.3 (30% of training samples)
            # Noise level: 0.05 (5% of pixel intensity range)
            noise = torch.randn_like(image) * 0.05
            image = torch.clamp(image + noise, 0, 1)

            # Other augmentation options (commented out):
            # image = transforms.RandomRotation(degrees=15)(image.reshape(64, 64)).flatten()
            # image = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))(image.reshape(64, 64)).flatten()

        return image, label


class ChannelAttention(nn.Module):
    """SENet-style Channel Attention Module for feature recalibration"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # =========================
        # CHANNEL ATTENTION ALGORITHMS - SENet STYLE
        # =========================
        # Purpose: Adaptively recalibrate channel-wise feature responses
        # Process: Global Pool -> FC -> ReLU -> FC -> Sigmoid -> Scale
        # Reduction ratio: 16 (standard SENet configuration)

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze: H×W -> 1×1
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),  # Reduction layer
            nn.ReLU(inplace=True),  # Non-linearity
            nn.Linear(in_channels // reduction, in_channels, bias=False),  # Excitation layer
            nn.Sigmoid(),  # Gating function (0-1 scale per channel)
        )

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        b, c, _, _ = x.size()

        # Squeeze: Global spatial information into channel descriptor
        y = self.global_pool(x).view(b, c)  # (batch, channels)

        # Excitation: Generate channel-wise attention weights
        y = self.fc(y).view(b, c, 1, 1)  # (batch, channels, 1, 1)

        # Scale: Apply attention weights to input features
        return x * y.expand_as(x)  # Element-wise multiplication


class LightweightKanjiNet(nn.Module):
    """Lightweight CNN optimized for web deployment with Channel Attention"""

    def __init__(self, num_classes: int, image_size: int = 64):
        super().__init__()

        self.image_size = image_size
        self.num_classes = num_classes

        # =========================
        # CNN ARCHITECTURE WITH CHANNEL ATTENTION
        # =========================
        # Current: Depthwise separable convolutions + SENet-style channel attention
        # Attention placement: After conv3, conv4, conv5 (deeper layers benefit most)
        # Channel progression: 1 -> 32 -> 64 -> 128 -> 256 -> 512 (increased capacity)
        # Stride pattern: 2 for downsampling layers, 1 for feature refinement
        self.conv1 = self._depthwise_separable_conv(1, 32, stride=2)  # 64x64 -> 32x32
        self.conv2 = self._depthwise_separable_conv(32, 64, stride=2)  # 32x32 -> 16x16
        self.conv3 = self._depthwise_separable_conv(64, 128, stride=2)  # 16x16 -> 8x8
        self.conv4 = self._depthwise_separable_conv(128, 256, stride=2)  # 8x8 -> 4x4
        self.conv5 = self._depthwise_separable_conv(
            256, 512, stride=1
        )  # 4x4 -> 4x4 (feature refinement)

        # =========================
        # CHANNEL ATTENTION MODULES - SENet INTEGRATION
        # =========================
        # Apply attention after deeper layers for maximum impact
        # Reduction ratio: 16 (balances effectiveness vs efficiency)
        self.attention3 = ChannelAttention(
            128, reduction=8
        )  # After conv3 (smaller reduction for fewer channels)
        self.attention4 = ChannelAttention(256, reduction=16)  # After conv4 (standard reduction)
        self.attention5 = ChannelAttention(512, reduction=16)  # After conv5 (standard reduction)

        # Alternative architectures (commented out):
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # Regular convolution
        # self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)  # Attention mechanism

        # =========================
        # POOLING ALGORITHM - ADJUSTABLE
        # =========================
        # Current: Global Average Pooling (reduces parameters vs large FC layers)
        # Alternatives: AdaptiveMaxPool2d, regular pooling + flatten
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # =========================
        # CLASSIFIER HEAD ALGORITHMS - ADJUSTABLE
        # =========================
        # Scalable two-layer MLP with capacity scaled for number of classes
        # Hidden layer size: 2048 for large number of classes, scales down for fewer classes
        # Dropout rates: 0.3 and 0.2 (prevent overfitting with larger model)
        # Activation: ReLU (could use GELU, Swish, etc.)
        hidden_dim = max(1024, min(2048, num_classes // 20))  # Scale hidden layer with class count
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # First dropout layer
            nn.Linear(512, hidden_dim),  # Hidden layer (512->hidden_dim)
            nn.ReLU(inplace=True),  # Activation function
            nn.Dropout(0.2),  # Second dropout layer
            nn.Linear(hidden_dim, num_classes),  # Output layer
        )

        # Alternative classifier options (commented out):
        # Single layer: nn.Linear(512, num_classes)
        # Original smaller: nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, num_classes)
        # Larger hidden: nn.Linear(256, 1024), nn.ReLU(), nn.Linear(1024, num_classes)
        # Different activation: nn.GELU(), nn.Swish()

        # Initialize weights
        self._initialize_weights()

    def _depthwise_separable_conv(self, in_channels, out_channels, stride=1):
        """Efficient depthwise separable convolution block"""
        # =========================
        # CONVOLUTION BLOCK ALGORITHMS - ADJUSTABLE
        # =========================
        # Current: Depthwise Separable Convolution (MobileNet-style)
        # Benefits: Fewer parameters, faster inference for mobile/web deployment
        # Components: Depthwise conv -> BatchNorm -> ReLU -> Pointwise conv -> BatchNorm -> ReLU
        # Kernel size: 3x3 for spatial features, 1x1 for channel mixing
        return nn.Sequential(
            # Depthwise convolution (spatial filtering)
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),  # groups=in_channels makes it depthwise
            nn.BatchNorm2d(in_channels),  # Normalization for training stability
            nn.ReLU(inplace=True),  # Activation function
            # Pointwise convolution (channel mixing)
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            ),  # 1x1 conv for channels
            nn.BatchNorm2d(out_channels),  # Normalization
            nn.ReLU(inplace=True),  # Activation function
        )

        # Alternative convolution blocks (commented out):
        # Regular convolution: nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # ResNet block: with skip connections
        # SENet block: with squeeze-and-excitation attention

    def _initialize_weights(self):
        """Initialize network weights"""
        # =========================
        # WEIGHT INITIALIZATION ALGORITHMS - ADJUSTABLE
        # =========================
        # Current: Kaiming (He) Normal initialization for conv layers
        # Purpose: Maintains variance in deeper networks with ReLU activations
        # Formula: N(0, √(2/fan_in)) for ReLU networks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )  # He initialization for ReLU
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # BatchNorm scale parameter
                nn.init.constant_(m.bias, 0)  # BatchNorm shift parameter
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # Small normal distribution for linear layers
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # Zero bias initialization

        # Alternative initialization methods (commented out):
        # Xavier/Glorot uniform: nn.init.xavier_uniform_(m.weight)
        # Xavier/Glorot normal: nn.init.xavier_normal_(m.weight)
        # Kaiming uniform: nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        # Orthogonal: nn.init.orthogonal_(m.weight)

    def forward(self, x):
        # Reshape flattened input to 2D image
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.image_size, self.image_size)

        # Feature extraction with Channel Attention
        x = self.conv1(x)  # 64x64 -> 32x32, channels: 1->32
        x = self.conv2(x)  # 32x32 -> 16x16, channels: 32->64

        x = self.conv3(x)  # 16x16 -> 8x8, channels: 64->128
        x = self.attention3(x)  # Apply channel attention to refine 128 features

        x = self.conv4(x)  # 8x8 -> 4x4, channels: 128->256
        x = self.attention4(x)  # Apply channel attention to refine 256 features

        x = self.conv5(x)  # 4x4 -> 4x4, channels: 256->512 (feature refinement)
        x = self.attention5(x)  # Apply channel attention to refine 512 features

        # Global pooling and classification
        x = self.global_pool(x)  # 4x4 -> 1x1
        x = x.view(batch_size, -1)  # Flatten to 512 features
        x = self.classifier(x)

        return x


class ProgressiveTrainer:
    """Progressive training strategy for large character sets"""

    def __init__(self, model, device, num_classes, results_dir="training/cnn/results"):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def train_epoch(self, dataloader, optimizer, criterion, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            # Update progress bar
            if batch_idx % 100 == 0:
                pbar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Acc": f"{100.0 * correct / total:.1f}%",
                    }
                )

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def validate(self, dataloader, criterion):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(dataloader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def train(
        self,
        train_loader,
        val_loader,
        epochs,
        learning_rate=0.001,
        checkpoint_manager=None,
        start_epoch=0,
    ):
        """
        Progressive training with learning rate scheduling and checkpoint management.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Total number of epochs to train
            learning_rate: Initial learning rate
            checkpoint_manager: Optional CheckpointManager for saving/resuming checkpoints
            start_epoch: Epoch to start training from (default: 0, use >0 when resuming)
        """

        # =========================
        # TRAINING LOSS FUNCTION - ADJUSTABLE
        # =========================
        # Current: CrossEntropyLoss with label smoothing = 0.1
        # Purpose: Label smoothing reduces overconfidence on large datasets (3,036 classes)
        # Effect: Smooths target distribution, prevents overfitting to training labels
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 0.1 = 10% smoothing

        # Alternative loss functions (commented out):
        # Standard cross-entropy: nn.CrossEntropyLoss()
        # Focal loss: FocalLoss(alpha=1, gamma=2) for class imbalance
        # Label smoothing values: 0.05 (light), 0.1 (moderate), 0.2 (heavy)

        # =========================
        # OPTIMIZER ALGORITHM - ADJUSTABLE
        # =========================
        # Uses factory function from src.lib
        # Creates AdamW optimizer with learning_rate and weight_decay
        optimizer = get_optimizer(
            self.model,
            type(
                "Config",
                (),
                {"learning_rate": learning_rate, "optimizer": "adamw", "weight_decay": 1e-4},
            )(),
        )

        # =========================
        # LEARNING RATE SCHEDULER - ADJUSTABLE
        # =========================
        # Uses factory function from src.lib
        # Creates CosineAnnealingLR scheduler with T_max=epochs, eta_min=1e-6
        scheduler = get_scheduler(
            optimizer,
            type(
                "Config",
                (),
                {
                    "epochs": epochs,
                    "scheduler": "cosine",
                    "scheduler_t_max": epochs,
                    "scheduler_eta_min": 1e-6,
                },
            )(),
        )

        # =========================
        # EARLY STOPPING PARAMETERS - ADJUSTABLE
        # =========================
        best_val_acc = 0
        patience_counter = 0
        max_patience = 15  # Stop training if no improvement for 15 epochs

        # Alternative early stopping strategies:
        # Shorter patience: max_patience = 10 (faster stopping)
        # Longer patience: max_patience = 20 (more training)
        # Loss-based stopping: Track validation loss instead of accuracy

        # Create progress log
        progress_log = {
            "epochs": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }

        for epoch in range(start_epoch, epochs):
            logger.info(f"一Epoch {epoch + 1}/{epochs}")
            logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Update learning rate
            scheduler.step()

            # Log progress
            progress_log["epochs"].append(epoch + 1)
            progress_log["train_loss"].append(train_loss)
            progress_log["train_acc"].append(train_acc)
            progress_log["val_loss"].append(val_loss)
            progress_log["val_acc"].append(val_acc)
            progress_log["learning_rate"].append(optimizer.param_groups[0]["lr"])

            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save progress periodically
            if (epoch + 1) % 5 == 0 or epoch == 0:
                with open(self.results_dir / "training_progress.json", "w") as f:
                    json.dump(progress_log, f, indent=2)

            # Save checkpoint after each epoch if checkpoint manager is provided
            if checkpoint_manager:
                is_best = val_acc > best_val_acc
                checkpoint_manager.save_checkpoint(
                    epoch,
                    self.model,
                    optimizer,
                    scheduler,
                    metrics={
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    is_best=is_best,
                )
                # Clean up old checkpoints
                checkpoint_manager.cleanup_old_checkpoints(keep_last_n=5)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.results_dir / "best_kanji_model.pth")
                patience_counter = 0
                logger.info(f"New best model saved! Accuracy: {best_val_acc:.2f}%")

                # Save model info
                model_info = {
                    "epoch": epoch + 1,
                    "val_accuracy": best_val_acc,
                    "val_loss": val_loss,
                    "train_accuracy": train_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                with open(self.results_dir / "best_model_info.json", "w") as f:
                    json.dump(model_info, f, indent=2)
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        logger.info(f"一Training completed! Best validation accuracy: {best_val_acc:.2f}%")

        # Save final progress
        with open(self.results_dir / "training_progress.json", "w") as f:
            json.dump(progress_log, f, indent=2)

        return best_val_acc


def create_balanced_loaders(x, y, batch_size, test_size=0.15, val_size=0.15):  # noqa: N803
    """Create balanced data loaders with stratification"""

    logger.debug("Creating stratified splits...")

    # Check if we have enough samples per class for stratified splitting
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_samples = np.min(class_counts)

    logger.debug(f"Classes: {len(unique_classes)}, Min samples per class: {min_samples}")

    if min_samples < 2:
        logger.warning(
            "⚠️Warning: Some classes have only 1 sample. Using non-stratified splitting for small datasets."
        )
        # Use simple random splitting when stratification isn't possible

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=42,
        )

        # Second split: train vs val
        val_size_adj = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adj, random_state=42
        )
    else:
        # Use stratified splitting when we have enough samples
        logger.debug("✅ Using stratified splitting (recommended for balanced training)")

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            stratify=y,
            random_state=42,
        )

        # Second split: train vs val
        val_size_adj = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adj, stratify=y_temp, random_state=42
        )

    logger.debug(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Create datasets with augmentation for training
    train_dataset = ETL9GDataset(X_train, y_train, augment=True)
    val_dataset = ETL9GDataset(X_val, y_val, augment=False)
    test_dataset = ETL9GDataset(X_test, y_test, augment=False)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# Note: load_chunked_dataset is imported from optimization_config (line 29), not redefined below
# The original function implementation has been removed to avoid duplication


def train_cnn(args):
    """
    Core CNN training function callable from unified entry point.

    Args:
        args: Namespace or dict-like object with training parameters
    """
    # Get parameters with safe defaults (supports both argparse Namespace and Click dict)
    batch_size = getattr(args, "batch_size", 64)
    sample_limit = getattr(args, "sample_limit", None)
    learning_rate = getattr(args, "learning_rate", 0.001)
    epochs = getattr(args, "epochs", 30)
    image_size = getattr(args, "image_size", 64)
    optimizer_name = getattr(args, "optimizer", "adamw")
    scheduler_name = getattr(args, "scheduler", "cosine")
    checkpoint_dir = getattr(args, "checkpoint_dir", "training/cnn/checkpoints")
    resume_from = getattr(args, "resume_from", None)
    no_checkpoint = getattr(args, "no_checkpoint", False)

    # Get data_dir from arguments or use default
    data_dir_arg = getattr(args, "data_dir", "dataset")

    # Use specified data_dir or auto-detect if using default
    if data_dir_arg == "dataset":
        data_path = get_dataset_directory()  # Auto-detect
    else:
        data_path = Path(data_dir_arg)  # Use specified

    logger.info(f"Using dataset from: {data_path}")

    # Read metadata for num_classes
    with open(data_path / "metadata.json") as f:
        metadata = json.load(f)
    num_classes = metadata.get("num_classes", 3036)

    # Create dataset factory for ETL9GDataset
    def create_etl_dataset(x: np.ndarray, y: np.ndarray):
        return ETL9GDataset(x, y, augment=False)

    # Prepare dataset using unified helper
    (x, y), num_classes_calc, _, _ = prepare_dataset_and_loaders(
        data_dir=str(data_path),
        dataset_fn=create_etl_dataset,
        batch_size=batch_size,
        sample_limit=sample_limit,
        logger=logger,
    )

    # Use metadata num_classes if available, fallback to calculated
    num_classes = max(num_classes, num_classes_calc)
    logger.info(f"Memory usage: {x.nbytes / (1024**3):.1f} GB")

    # Create balanced loaders with stratification
    train_loader, val_loader, test_loader = create_balanced_loaders(x, y, batch_size)

    # Initialize GPU and enable CUDA optimizations
    device = verify_and_setup_gpu()
    device = torch.device(device)
    logger.info(f"Using device: {device}")

    model = LightweightKanjiNet(num_classes, image_size)
    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create training directory structure if it doesn't exist
    Path("training/cnn").mkdir(parents=True, exist_ok=True)

    # Create config for optimizer/scheduler
    config = CNNConfig(
        learning_rate=learning_rate,
        weight_decay=1e-4,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer_name,
        scheduler=scheduler_name,
    )

    # Initialize optimizer and scheduler using unified functions
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir, "cnn")

    # Initialize trainer with results directory
    results_dir = Path(checkpoint_dir).parent / "results"
    trainer = ProgressiveTrainer(model, device, num_classes, results_dir=str(results_dir))

    # Check for existing checkpoint and resume if available
    start_epoch = 0

    # Use unified checkpoint loading (DRY pattern)
    start_epoch, best_metrics = checkpoint_manager.load_checkpoint_for_training(
        model,
        optimizer,
        scheduler,
        device,
        resume_from=resume_from,
        args_no_checkpoint=no_checkpoint,
    )
    best_metrics.get("val_accuracy", 0.0)

    # Train model
    logger.info("Starting training...")
    trainer.train(
        train_loader,
        val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        checkpoint_manager=checkpoint_manager,
        start_epoch=start_epoch,
    )

    # Test final model
    test_loss, test_acc = trainer.validate(test_loader, nn.CrossEntropyLoss())
    logger.info(f"Final test accuracy: {test_acc:.2f}%")

    # ========== CREATE CHARACTER MAPPING ==========
    logger.info("\n📊 Creating character mapping for inference...")
    try:
        from subprocess import run

        result = run(  # noqa: S603
            [
                sys.executable,
                "scripts/create_class_mapping.py",
                "--metadata-path",
                str(data_path / "metadata.json"),
                "--output-dir",
                str(results_dir),
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


def main():
    """Legacy main function for direct script execution."""
    from scripts.training_args import add_variant_args_to_parser

    parser = argparse.ArgumentParser(description="Train Lightweight Kanji Model for ETL9G")

    # Add all arguments for CNN variant from centralized config
    add_variant_args_to_parser(parser, "cnn", checkpoint_dir_default="training/cnn/checkpoints")

    args = parser.parse_args()
    train_cnn(args)


if __name__ == "__main__":
    main()
