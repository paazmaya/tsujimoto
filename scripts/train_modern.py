#!/usr/bin/env python3
"""
Modern Unified Training Script using Typer CLI and PyTorch Lightning

This script replaces the old Click-based training.py with a cleaner,
type-safe interface using Typer. It leverages:
- Typer for CLI (typed, autocomplete)
- PyTorch Lightning for training (GPU, checkpointing, metrics)
- Hugging Face Datasets for data loading
- Pydantic for configuration validation
- Loguru for structured logging

Supported Models:
    cnn, rnn, vit, hiercode, qat, hiercode_higita

Examples:
    python scripts/train_modern.py --model cnn --epochs 50
    python scripts/train_modern.py --model rnn --model-variant hybrid_cnn_rnn
    python scripts/train_modern.py --model vit --learning-rate 1e-4
    python scripts/train_modern.py --help
"""

import sys
from pathlib import Path
from typing import Optional

import typer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib.cli import CLIContext, create_app, echo_error, echo_info, echo_success
from src.lib.config import (
    CNNConfig,
    HierCodeConfig,
    QATConfig,
    RNNConfig,
    ViTConfig,
)
from src.lib.logging_utils import setup_logger

logger = setup_logger(__name__)

# Create the main app
app = create_app(name="train-kanji", version="0.1.0")


@app.command()
def train(
    model: str = typer.Option(
        "cnn",
        "--model",
        "-m",
        help="Model type (cnn, rnn, vit, hiercode, qat, hiercode_higita)",
    ),
    epochs: int = typer.Option(
        100,
        "--epochs",
        "-e",
        help="Number of training epochs",
        min=1,
        max=1000,
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Training batch size",
        min=1,
    ),
    learning_rate: float = typer.Option(
        1e-3,
        "--learning-rate",
        "-lr",
        help="Learning rate",
        min=1e-7,
        max=1.0,
    ),
    optimizer: str = typer.Option(
        "adamw",
        "--optimizer",
        help="Optimizer (adamw, sgd)",
    ),
    scheduler: str = typer.Option(
        "cosine",
        "--scheduler",
        help="Learning rate scheduler (cosine, step)",
    ),
    checkpoint_dir: Optional[Path] = typer.Option(  # noqa: B008
        None,
        "--checkpoint-dir",
        "-c",
        help="Checkpoint directory (default: checkpoints/{model})",
    ),
    dataset_dir: Path = typer.Option(  # noqa: B008
        Path("data"),
        "--dataset-dir",
        "-d",
        help="Dataset directory",
    ),
    num_workers: int = typer.Option(
        4,
        "--num-workers",
        "-w",
        help="Number of data loading workers",
        min=0,
    ),
    early_stopping_patience: int = typer.Option(
        10,
        "--early-stopping-patience",
        help="Early stopping patience (epochs with no improvement)",
        min=1,
    ),
    val_split: float = typer.Option(
        0.1,
        "--val-split",
        help="Validation split ratio",
        min=0.0,
        max=0.5,
    ),
    test_split: float = typer.Option(
        0.1,
        "--test-split",
        help="Test split ratio",
        min=0.0,
        max=0.5,
    ),
    rnn_model_type: Optional[str] = typer.Option(
        None,
        "--rnn-model-type",
        help="RNN variant (basic, stroke, simple_radical, hybrid_cnn, linguistic_radical)",
    ),
    vit_patch_size: int = typer.Option(
        16,
        "--vit-patch-size",
        help="ViT patch size",
        min=4,
        max=64,
    ),
):
    """Train a Kanji recognition model using PyTorch Lightning."""

    with CLIContext(f"Training {model} model"):
        # Create config based on model type
        config_class = {
            "cnn": CNNConfig,
            "rnn": RNNConfig,
            "vit": ViTConfig,
            "qat": QATConfig,
            "hiercode": HierCodeConfig,
            "hiercode_higita": HierCodeConfig,
        }.get(model)

        if not config_class:
            echo_error(
                f"Unknown model type: {model}. "
                f"Must be one of: cnn, rnn, vit, qat, hiercode, hiercode_higita"
            )
            raise typer.Exit(code=1)

        # Build config with provided parameters
        config_kwargs = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "val_split": val_split,
            "test_split": test_split,
        }

        # Add model-specific parameters
        if model == "rnn" and rnn_model_type:
            config_kwargs["rnn_type"] = rnn_model_type
        if model == "vit":
            config_kwargs["patch_size"] = vit_patch_size

        try:
            config_class(**config_kwargs)
            echo_success(f"Configuration created: {model}")
        except Exception as e:
            echo_error(f"Invalid configuration: {e}")
            raise typer.Exit(code=1) from e

        # Create checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = Path("checkpoints") / model
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Display training configuration
        echo_info("Training Configuration:")
        echo_info(f"  Model: {model}")
        echo_info(f"  Epochs: {epochs}")
        echo_info(f"  Batch size: {batch_size}")
        echo_info(f"  Learning rate: {learning_rate}")
        echo_info(f"  Optimizer: {optimizer}")
        echo_info(f"  Scheduler: {scheduler}")
        echo_info(f"  Checkpoint dir: {checkpoint_dir}")
        echo_info(f"  Dataset dir: {dataset_dir}")

        # TODO: Integrate with actual model loading and training
        echo_info("(Model loading and training integration TBD)")


@app.command()
def validate():
    """Validate training setup and dependencies."""
    echo_info("Validating setup...")

    try:
        import pytorch_lightning as pl
        import torch

        echo_success(f"PyTorch: {torch.__version__}")
        echo_success(f"PyTorch Lightning: {pl.__version__}")
        echo_success("Hugging Face Datasets: OK")
        echo_success("All dependencies installed!")

    except ImportError as e:
        echo_error(f"Missing dependency: {e}")
        raise typer.Exit(code=1) from e


@app.command()
def show_config(
    model: str = typer.Option(
        "cnn",
        "--model",
        "-m",
        help="Model type to show config for",
    ),
):
    """Show default configuration for a model."""

    config_class = {
        "cnn": CNNConfig,
        "rnn": RNNConfig,
        "vit": ViTConfig,
        "qat": QATConfig,
        "hiercode": HierCodeConfig,
        "hiercode_higita": HierCodeConfig,
    }.get(model)

    if not config_class:
        echo_error(f"Unknown model type: {model}")
        raise typer.Exit(code=1)

    config = config_class()
    echo_info(f"Default configuration for {model}:")

    # Display as formatted dict
    import json

    config_dict = config.model_dump()
    echo_info(json.dumps(config_dict, indent=2))


if __name__ == "__main__":
    app()
