"""
Optimization utilities for model training.

Provides factory functions for creating optimizers, schedulers, and related utilities.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch.optim as optim

logger = logging.getLogger(__name__)


def get_optimizer(model, config, learning_rate: Optional[float] = None) -> optim.Optimizer:
    """
    Create an optimizer based on config.

    Args:
        model: PyTorch model to optimize
        config: Configuration object with optimizer settings
        learning_rate: Override learning rate (default: use config.learning_rate)

    Returns:
        optim.Optimizer: Optimizer instance
    """
    lr = learning_rate or config.learning_rate
    optimizer_name = getattr(config, "optimizer", "adamw").lower()

    if optimizer_name == "adamw":
        weight_decay = getattr(config, "weight_decay", 1e-5)
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adam":
        weight_decay = getattr(config, "weight_decay", 1e-5)
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        momentum = getattr(config, "momentum", 0.9)
        weight_decay = getattr(config, "weight_decay", 1e-5)
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        logger.warning(f"Unknown optimizer: {optimizer_name}, defaulting to AdamW")
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)


def get_scheduler(
    optimizer: optim.Optimizer,
    config,
    num_epochs: Optional[int] = None,
) -> optim.lr_scheduler.LRScheduler:
    """
    Create a learning rate scheduler based on config.

    Args:
        optimizer: Optimizer to schedule
        config: Configuration object with scheduler settings
        num_epochs: Number of training epochs (default: use config.epochs)

    Returns:
        optim.lr_scheduler.LRScheduler: Scheduler instance
    """
    epochs = num_epochs or getattr(config, "epochs", 30)
    scheduler_name = getattr(config, "scheduler", "cosine").lower()
    t_max = getattr(config, "scheduler_t_max", epochs)

    if scheduler_name == "cosine":
        eta_min = getattr(config, "scheduler_eta_min", 1e-6)
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )
    elif scheduler_name == "step":
        step_size = getattr(config, "scheduler_step_size", max(1, epochs // 3))
        gamma = getattr(config, "scheduler_gamma", 0.1)
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    elif scheduler_name == "exponential":
        gamma = getattr(config, "scheduler_gamma", 0.95)
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        logger.warning(f"Unknown scheduler: {scheduler_name}, defaulting to CosineAnnealingLR")
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


def save_config(config, output_dir: str, filename: str = "config.json") -> None:
    """
    Save configuration to JSON file.

    Args:
        config: Configuration object (should have to_dict() method or be dict-like)
        output_dir: Directory to save config to
        filename: Name of config file (default: config.json)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_file = output_path / filename

    # Convert config to dict if it has to_dict method
    if hasattr(config, "to_dict"):
        config_dict = config.to_dict()
    elif isinstance(config, dict):
        config_dict = config
    else:
        # Fall back to __dict__
        config_dict = vars(config)

    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"✓ Configuration saved to {config_file}")


def load_config_from_json(config_path: str) -> dict:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to config JSON file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path) as f:
        config_dict = json.load(f)

    logger.info(f"✓ Configuration loaded from {config_path}")
    return config_dict
