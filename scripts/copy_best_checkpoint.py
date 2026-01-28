#!/usr/bin/env python3
"""
Copy the best checkpoint based on validation accuracy from training history.

Reads the training history JSON to find the epoch with highest validation accuracy,
then copies that checkpoint to the standard model output location.

Usage:
    python scripts/copy_best_checkpoint.py --model-type hiercode
    python scripts/copy_best_checkpoint.py --model-type cnn --checkpoint-dir training/cnn/checkpoints
"""

import argparse
import json
import shutil
from pathlib import Path


def find_best_epoch(history_file: str) -> int:
    """Find the epoch with the best validation accuracy."""
    with open(history_file) as f:
        history = json.load(f)

    best_epoch = 1
    best_acc = -1

    for entry in history:
        val_acc = entry["val"]["accuracy"]
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = entry["epoch"]

    return best_epoch, best_acc


def copy_best_checkpoint(model_type: str, checkpoint_dir: str = None) -> None:
    """
    Copy the best checkpoint to the standard model location.

    Args:
        model_type: Type of model (cnn, rnn, hiercode, qat, vit, hiercode-higita, radical-rnn)
        checkpoint_dir: Optional custom checkpoint directory
    """
    # Map model types to directories and output files
    model_config = {
        "cnn": {
            "default_dir": "training/cnn/checkpoints",
            "history_file": "training/cnn/checkpoints/training_history_cnn.json",
            "output_file": "training/cnn/best_cnn_model.pth",
        },
        "rnn": {
            "default_dir": "training/rnn/checkpoints",
            "history_file": "training/rnn/checkpoints/training_history_rnn.json",
            "output_file": "training/rnn/best_rnn_model.pth",
        },
        "radical-rnn": {
            "default_dir": "training/radical_rnn/checkpoints",
            "history_file": "training/radical_rnn/checkpoints/training_history_radical_rnn.json",
            "output_file": "training/radical_rnn/best_radical_rnn_model.pth",
        },
        "hiercode": {
            "default_dir": "training/hiercode/checkpoints",
            "history_file": "training/hiercode/checkpoints/training_history_hiercode.json",
            "output_file": "training/hiercode/hiercode_model_best.pth",
        },
        "hiercode-higita": {
            "default_dir": "training/hiercode_higita/checkpoints",
            "history_file": "training/hiercode_higita/checkpoints/training_history_higita.json",
            "output_file": "training/hiercode_higita/best_hiercode_higita.pth",
        },
        "qat": {
            "default_dir": "training/qat/checkpoints",
            "history_file": "training/qat/checkpoints/training_history_qat.json",
            "output_file": "training/qat/best_qat_model.pth",
        },
        "vit": {
            "default_dir": "training/vit/checkpoints",
            "history_file": "training/vit/checkpoints/training_history_vit.json",
            "output_file": "training/vit/best_vit_model.pth",
        },
    }

    if model_type not in model_config:
        raise ValueError(
            f"Unknown model type: {model_type}. Supported: {', '.join(model_config.keys())}"
        )

    config = model_config[model_type]
    chk_dir = checkpoint_dir or config["default_dir"]
    history_file = config["history_file"]
    output_file = config["output_file"]

    # Check if history file exists
    if not Path(history_file).exists():
        return

    # Find best epoch
    try:
        best_epoch, best_acc = find_best_epoch(history_file)
    except (json.JSONDecodeError, KeyError):
        return

    # Find checkpoint file
    checkpoint_file = Path(chk_dir) / f"checkpoint_epoch_{best_epoch:03d}.pt"

    if not checkpoint_file.exists():
        # If best checkpoint was cleaned up, find the latest available checkpoint
        available_checkpoints = sorted(Path(chk_dir).glob("checkpoint_epoch_*.pt"))
        if available_checkpoints:
            checkpoint_file = available_checkpoints[-1]
        else:
            return

    # Copy checkpoint
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(str(checkpoint_file), str(output_path))


def main():
    parser = argparse.ArgumentParser(
        description="Copy best checkpoint based on validation accuracy"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="Model type (cnn, rnn, radical-rnn, hiercode, hiercode-higita, qat, vit)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Custom checkpoint directory (optional)",
    )

    args = parser.parse_args()
    copy_best_checkpoint(args.model_type, args.checkpoint_dir)


if __name__ == "__main__":
    main()
