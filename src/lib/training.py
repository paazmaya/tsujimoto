"""
Training utilities for common training loop patterns and result handling.

Provides DRY utilities for:
- Saving best models
- Saving training results to JSON
- Training result collection
- Common training patterns
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


def save_best_model(
    model,
    current_accuracy: float,
    best_accuracy: float,
    model_path: Path,
    device: str = "cpu",
) -> bool:
    """
    Save model if current accuracy is better than best.

    Args:
        model: Model to save
        current_accuracy: Current epoch accuracy
        best_accuracy: Best accuracy so far
        model_path: Path to save model to
        device: Device model is on (for logging)

    Returns:
        True if model was saved (new best), False otherwise

    Example:
        >>> was_saved = save_best_model(model, val_acc, best_acc, Path("best.pth"))
        >>> if was_saved:
        ...     print("New best model saved!")
    """
    if current_accuracy > best_accuracy:
        torch.save(model.state_dict(), model_path)
        logger.info(f"  ✓ Saved best model (acc: {current_accuracy:.2f}%)")
        return True
    return False


def save_training_results(
    config: Any,
    best_val_acc: float,
    test_acc: float,
    test_loss: float,
    history: Dict[str, Any],
    results_path: Path,
    create_dir: bool = True,
) -> None:
    """
    Save training results to JSON file.

    Args:
        config: Training configuration object (must have to_dict() method)
        best_val_acc: Best validation accuracy
        test_acc: Test set accuracy
        test_loss: Test set loss
        history: Training history (dict or trainer.history)
        results_path: Path to save results to
        create_dir: Create results directory if True

    Example:
        >>> save_training_results(config, 0.97, 0.96, 0.15, history, Path("results.json"))
    """
    if create_dir:
        results_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "history": history,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"✓ Results saved to {results_path}")


def save_training_history(
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    history_dict: Dict[str, list],
) -> None:
    """
    Append epoch metrics to training history.

    Args:
        epoch: Epoch number (for logging)
        train_loss: Training loss
        train_acc: Training accuracy
        val_loss: Validation loss
        val_acc: Validation accuracy
        history_dict: History dictionary with 'train_loss', 'train_acc', 'val_loss', 'val_acc' keys

    Example:
        >>> history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        >>> save_training_history(1, 0.5, 0.8, 0.6, 0.75, history)
    """
    history_dict["train_loss"].append(train_loss)
    history_dict["train_acc"].append(train_acc)
    history_dict["val_loss"].append(val_loss)
    history_dict["val_acc"].append(val_acc)


def load_best_model_for_testing(
    model, best_model_path: Path, device: str = "cpu"
) -> None:
    """
    Load best saved model for testing.

    Args:
        model: Model to load weights into
        best_model_path: Path to best model checkpoint
        device: Device to load on

    Example:
        >>> load_best_model_for_testing(model, Path("best_model.pth"), device="cuda")
    """
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    logger.info(f"✓ Loaded best model from {best_model_path}")


def create_results_directory(results_dir: Path) -> None:
    """
    Create results directory if it doesn't exist.

    Args:
        results_dir: Path to results directory

    Example:
        >>> create_results_directory(Path("results"))
    """
    results_dir.mkdir(parents=True, exist_ok=True)


def collect_training_metrics(
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
) -> Dict[str, Any]:
    """
    Collect metrics for a single epoch into a dictionary.

    Args:
        epoch: Epoch number
        train_loss: Training loss
        train_acc: Training accuracy
        val_loss: Validation loss
        val_acc: Validation accuracy

    Returns:
        Dictionary with all metrics

    Example:
        >>> metrics = collect_training_metrics(1, 0.5, 0.8, 0.6, 0.75)
        >>> print(metrics["train_acc"])
        0.8
    """
    return {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
    }
