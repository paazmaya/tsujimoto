#!/usr/bin/env python3
"""
Create comprehensive training visualizations from JSON training logs.

Supports any training history JSON with metrics like:
- epochs: list of epoch numbers
- train_loss: list of training losses
- val_loss: list of validation losses
- val_acc: list of validation accuracies
- train_acc: list of training accuracies (optional)

Usage:
    uv run python create_training_visualizations.py training/hiercode_higita/checkpoints/training_history_higita.json
    python create_training_visualizations.py training/cnn/checkpoints/training_progress.json
    python create_training_visualizations.py training/rnn/results/training_metrics.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Soft, muted colors with high contrast (WCAG AA compliant)
# These maintain a "pastel-like" aesthetic while ensuring readability
PASTEL_COLORS = {
    "train_loss": "#C73A3A",  # Soft red (5.13:1 vs white)
    "val_loss": "#2563BE",  # Soft blue (5.82:1 vs white)
    "train_acc": "#936D25",  # Soft brown-gold (4.53:1 vs white)
    "val_acc": "#2D5016",  # Soft dark green (9.25:1 vs white)
    "accent": "#8B3B9B",  # Soft purple
}

# Darker accent colors for contrast in annotations
ACCENT_COLORS = {
    "train_loss": "#7B1C1C",  # Deep red
    "val_loss": "#0C2859",  # Deep blue
    "train_acc": "#5C4A0A",  # Deep goldenrod
    "val_acc": "#1B3A0D",  # Deep green
}


def load_training_log(log_path: str) -> dict:
    """Load training log from JSON file.

    Supports two formats:
    1. Flat format: {"epochs": [...], "train_loss": [...], "val_loss": [...]}
    2. Nested format: [{"epoch": 1, "train": {...}, "val": {...}}, ...]
    """
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Training log not found: {log_path}")

    with open(path) as f:
        raw_data = json.load(f)

    # Convert nested format to flat format
    if isinstance(raw_data, list):
        data = {
            "epochs": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for entry in raw_data:
            if "epoch" in entry:
                data["epochs"].append(entry["epoch"])

            # Extract training metrics
            if "train" in entry:
                train = entry["train"]
                if "total_loss" in train:
                    data["train_loss"].append(train["total_loss"])
                if "accuracy" in train:
                    data["train_acc"].append(train["accuracy"])

            # Extract validation metrics
            if "val" in entry:
                val = entry["val"]
                if "loss" in val:
                    data["val_loss"].append(val["loss"])
                if "accuracy" in val:
                    data["val_acc"].append(val["accuracy"])

        return data

    # Already in flat format
    return raw_data


def validate_log_data(data: dict) -> dict:
    """Validate that log contains required fields."""
    required_fields = ["epochs"]
    metric_fields = ["train_loss", "val_loss", "val_acc", "train_acc"]

    # Check required fields
    for field in required_fields:
        if field not in data or len(data[field]) == 0:
            raise ValueError(f"Training log missing or empty required field: {field}")

    # Check that at least some metrics exist
    metrics_found = [f for f in metric_fields if f in data and len(data[f]) > 0]
    if not metrics_found:
        raise ValueError(f"Training log must contain at least one metric from: {metric_fields}")

    return data


def get_output_path(log_path: str) -> str:
    """Generate output path based on input log path."""
    input_path = Path(log_path)
    output_name = f"{input_path.stem}_visualization.png"
    output_path = input_path.parent / output_name
    return str(output_path)


def create_single_log_visualization(log_path: str, output_path: Optional[str] = None):
    """Create comprehensive visualization from a single training log."""
    # Load and validate data
    logger.info("Loading training log: %s", log_path)
    data = load_training_log(log_path)
    data = validate_log_data(data)

    if output_path is None:
        output_path = get_output_path(log_path)

    epochs = data["epochs"]
    num_epochs = len(epochs)
    logger.info("Loaded %d epochs of training data", num_epochs)

    # Determine which metrics are available
    has_train_loss = "train_loss" in data
    has_val_loss = "val_loss" in data
    has_train_acc = "train_acc" in data
    has_val_acc = "val_acc" in data

    # Determine subplot layout based on available metrics
    metrics_count = sum([has_train_loss or has_val_loss, has_train_acc or has_val_acc])
    if metrics_count == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        axes = [ax1, ax2]
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        axes = [ax1]

    # Extract log filename for title
    log_filename = Path(log_path).stem.replace("_", " ").title()
    fig.suptitle(f"Training Progress: {log_filename}", fontsize=16, fontweight="bold")

    axis_idx = 0

    # Plot loss curves
    if has_train_loss or has_val_loss:
        ax = axes[axis_idx]
        axis_idx += 1

        if has_train_loss:
            ax.plot(
                epochs,
                data["train_loss"],
                color=PASTEL_COLORS["train_loss"],
                label="Training Loss",
                linewidth=2.5,
                marker="o",
                markersize=3,
            )

        if has_val_loss:
            ax.plot(
                epochs,
                data["val_loss"],
                color=PASTEL_COLORS["val_loss"],
                label="Validation Loss",
                linewidth=2.5,
                marker="s",
                markersize=3,
            )

        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=11, fontweight="bold")
        ax.set_title("Training Loss Curves", fontsize=12, fontweight="bold")
        ax.legend(loc="best", framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_yscale("log")

        # Add best loss annotation
        if has_val_loss:
            best_val_loss = min(data["val_loss"])
            best_epoch_idx = data["val_loss"].index(best_val_loss)
            ax.annotate(
                f"Best: {best_val_loss:.4f}\nEpoch {epochs[best_epoch_idx]}",
                xy=(epochs[best_epoch_idx], best_val_loss),
                xytext=(epochs[best_epoch_idx] + num_epochs * 0.05, best_val_loss * 1.5),
                bbox={
                    "boxstyle": "round,pad=0.5",
                    "facecolor": PASTEL_COLORS["val_loss"],
                    "alpha": 0.7,
                    "edgecolor": ACCENT_COLORS["val_loss"],
                    "linewidth": 2,
                },
                arrowprops={
                    "arrowstyle": "->",
                    "color": ACCENT_COLORS["val_loss"],
                    "lw": 2,
                },
            )

    # Plot accuracy curves
    if has_train_acc or has_val_acc:
        ax = axes[axis_idx]

        if has_train_acc:
            ax.plot(
                epochs,
                data["train_acc"],
                color=PASTEL_COLORS["train_acc"],
                label="Training Accuracy",
                linewidth=2.5,
                marker="o",
                markersize=3,
            )

        if has_val_acc:
            ax.plot(
                epochs,
                data["val_acc"],
                color=PASTEL_COLORS["val_acc"],
                label="Validation Accuracy",
                linewidth=2.5,
                marker="s",
                markersize=3,
            )

        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_title("Training Accuracy Curves", fontsize=12, fontweight="bold")
        ax.legend(loc="best", framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_ylim(bottom=0)

        # Add peak accuracy annotation
        if has_val_acc:
            max_val_acc = max(data["val_acc"])
            best_epoch_idx = data["val_acc"].index(max_val_acc)
            ax.annotate(
                f"Peak: {max_val_acc:.2f}%\nEpoch {epochs[best_epoch_idx]}",
                xy=(epochs[best_epoch_idx], max_val_acc),
                xytext=(epochs[best_epoch_idx] + num_epochs * 0.05, max_val_acc - 2),
                bbox={
                    "boxstyle": "round,pad=0.5",
                    "facecolor": PASTEL_COLORS["val_acc"],
                    "alpha": 0.7,
                    "edgecolor": ACCENT_COLORS["val_acc"],
                    "linewidth": 2,
                },
                arrowprops={
                    "arrowstyle": "->",
                    "color": ACCENT_COLORS["val_acc"],
                    "lw": 2,
                },
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    logger.info("✓ Visualization saved: %s", output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create training visualizations from JSON logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_training_visualizations.py training/cnn/results/training_progress.json
  python create_training_visualizations.py training/rnn/results/training_metrics.json
  python create_training_visualizations.py training/hiercode_higita/results/metrics.json --output output.png

JSON Format:
  {
    "epochs": [1, 2, 3, ...],
    "train_loss": [0.5, 0.4, ...],
    "val_loss": [0.6, 0.5, ...],
    "train_acc": [85.0, 87.0, ...],
    "val_acc": [84.0, 86.0, ...]
  }
        """,
    )

    parser.add_argument(
        "log_file",
        type=str,
        help="Path to training log JSON file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path for visualization PNG (default: auto-generated from input path)",
    )

    args = parser.parse_args()

    try:
        logger.info("Creating visualization...")
        output = create_single_log_visualization(args.log_file, args.output)
        logger.info("✓ Success! Output: %s", output)
    except FileNotFoundError as e:
        logger.error("✗ File not found: %s", str(e))
        exit(1)
    except ValueError as e:
        logger.error("✗ Invalid data: %s", str(e))
        exit(1)
    except Exception as e:
        logger.error("✗ Error: %s", str(e))
        exit(1)


if __name__ == "__main__":
    main()
