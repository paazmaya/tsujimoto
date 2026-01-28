#!/usr/bin/env python3
"""
Plot training history for HierCode Hi-GITA model.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_training_history(history_file: str) -> list:
    """Load training history from JSON file."""
    with open(history_file) as f:
        return json.load(f)


def plot_training_history(history_file: str, output_dir: str = "results") -> None:
    """
    Plot training and validation metrics.

    Args:
        history_file: Path to training history JSON file
        output_dir: Directory to save plots
    """
    # Load history
    history = load_training_history(history_file)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract metrics
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train"]["total_loss"] for h in history]
    train_accs = [h["train"]["accuracy"] for h in history]
    val_losses = [h["val"]["loss"] for h in history]
    val_accs = [h["val"]["accuracy"] for h in history]
    ce_losses = [h["train"]["ce_loss"] for h in history]
    contrastive_losses = [h["train"]["contrastive_loss"] for h in history]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("HierCode Hi-GITA Training History", fontsize=16, fontweight="bold")

    # Plot 1: Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_losses, "b-o", label="Train Loss", linewidth=2, markersize=4)
    ax.plot(epochs, val_losses, "r-s", label="Val Loss", linewidth=2, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss Over Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, train_accs, "b-o", label="Train Accuracy", linewidth=2, markersize=4)
    ax.plot(epochs, val_accs, "r-s", label="Val Accuracy", linewidth=2, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Over Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Loss Components
    ax = axes[1, 0]
    ax.plot(epochs, ce_losses, "g-o", label="Cross-Entropy Loss", linewidth=2, markersize=4)
    ax.plot(epochs, contrastive_losses, "m-s", label="Contrastive Loss", linewidth=2, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Components (Training)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary Statistics
    ax = axes[1, 1]
    ax.axis("off")

    # Calculate summary stats
    best_val_acc_idx = np.argmax(val_accs)
    best_val_loss_idx = np.argmin(val_losses)
    final_train_acc = train_accs[-1]
    final_val_acc = val_accs[-1]

    summary_text = f"""
    Training Summary
    ================
    Total Epochs: {len(epochs)}

    Final Training Accuracy: {final_train_acc:.2f}%
    Final Validation Accuracy: {final_val_acc:.2f}%

    Best Validation Accuracy: {val_accs[best_val_acc_idx]:.2f}% (Epoch {epochs[best_val_acc_idx]})
    Best Validation Loss: {val_losses[best_val_loss_idx]:.4f} (Epoch {epochs[best_val_loss_idx]})

    Final Train Loss: {train_losses[-1]:.4f}
    Final Val Loss: {val_losses[-1]:.4f}

    Initial Train Loss: {train_losses[0]:.4f}
    Initial Val Loss: {val_losses[0]:.4f}

    Loss Reduction: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%
    """

    ax.text(
        0.1,
        0.5,
        summary_text,
        fontsize=11,
        family="monospace",
        verticalalignment="center",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()

    # Save figure
    output_file = Path(output_dir) / "training_history.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")

    # Also create individual plots for better detail
    create_individual_plots(
        epochs,
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        ce_losses,
        contrastive_losses,
        output_dir,
    )


def create_individual_plots(
    epochs,
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    ce_losses,
    contrastive_losses,
    output_dir,
):
    """Create individual high-resolution plots."""

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, "b-o", label="Train Loss", linewidth=2.5, markersize=5)
    plt.plot(epochs, val_losses, "r-s", label="Val Loss", linewidth=2.5, markersize=5)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "loss_only.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, "b-o", label="Train Accuracy", linewidth=2.5, markersize=5)
    plt.plot(epochs, val_accs, "r-s", label="Val Accuracy", linewidth=2.5, markersize=5)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "accuracy_only.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Loss components plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, ce_losses, "g-o", label="Cross-Entropy Loss", linewidth=2.5, markersize=5)
    plt.plot(
        epochs, contrastive_losses, "m-s", label="Contrastive Loss", linewidth=2.5, markersize=5
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Loss Components (Training)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "loss_components.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Plot training history
    history_file = "training/hiercode_higita/checkpoints/training_history_higita.json"
    plot_training_history(history_file, "results")
