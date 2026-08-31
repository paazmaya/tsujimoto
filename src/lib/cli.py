"""
CLI utilities using Typer for command-line interfaces.

This module provides a clean, professional CLI framework for training scripts
and utilities, replacing manual argparse setup.

Features:
- Automatic argument parsing with type hints
- Built-in help messages
- Command groups and subcommands
- Automatic shell completion
- Rich output formatting

Example Usage:
    >>> import typer
    >>> from src.lib.cli import create_app, add_training_arguments
    >>> 
    >>> app = create_app(name="train-kanji")
    >>> 
    >>> @app.command()
    >>> def train_cnn(
    ...     epochs: int = typer.Option(100, help="Number of epochs"),
    ...     batch_size: int = typer.Option(32, help="Batch size"),
    ...     model_type: str = typer.Option("cnn", help="Model type"),
    ... ):
    ...     '''Train a CNN model.'''
    ...     typer.echo(f"Training {model_type} for {epochs} epochs")
    >>> 
    >>> if __name__ == "__main__":
    ...     app()
"""

from pathlib import Path
from typing import Optional

import typer
from typer.colors import BRIGHT_CYAN, GREEN

from .logging_utils import setup_logger

logger = setup_logger(__name__)


def create_app(
    name: str = "kanji-cli",
    help: Optional[str] = None,
    version: str = "0.1.0",
) -> typer.Typer:
    """
    Create a new Typer CLI application with standard configuration.

    Args:
        name: Application name
        help: Application help text
        version: Application version

    Returns:
        Configured Typer application
    """
    app = typer.Typer(
        name=name,
        help=help or f"{name} - Kanji Recognition Training Tools",
        pretty_exceptions_enable=True,
    )

    @app.callback()
    def callback(
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-v",
            help="Enable verbose logging",
        ),
    ):
        """Kanji recognition training toolkit."""
        if verbose:
            logger.enable(__name__)

    return app


def echo_success(message: str):
    """Echo a success message with green color."""
    typer.echo(typer.style(f"✓ {message}", fg=GREEN, bold=True))


def echo_info(message: str):
    """Echo an info message with cyan color."""
    typer.echo(typer.style(f"ℹ {message}", fg=BRIGHT_CYAN))


def echo_error(message: str):
    """Echo an error message."""
    typer.echo(typer.style(f"✗ {message}", fg=typer.colors.RED, bold=True), err=True)


def add_training_command(
    app: typer.Typer,
    model_type: str = "cnn",
):
    """
    Add a training command template to a Typer app.

    Args:
        app: Typer application
        model_type: Type of model (cnn, rnn, vit, etc.)
    """

    @app.command()
    def train(
        epochs: int = typer.Option(
            100,
            "--epochs",
            "-e",
            help="Number of training epochs",
            min=1,
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
            min=0.0,
        ),
        checkpoint_dir: Path = typer.Option(
            f"checkpoints/{model_type}",
            "--checkpoint-dir",
            "-c",
            help="Checkpoint directory",
        ),
        dataset_dir: Path = typer.Option(
            "data",
            "--dataset-dir",
            "-d",
            help="Dataset directory",
        ),
        num_workers: int = typer.Option(
            4,
            "--num-workers",
            "-w",
            help="Number of data workers",
            min=0,
        ),
        early_stopping_patience: int = typer.Option(
            10,
            "--early-stopping-patience",
            help="Early stopping patience",
            min=1,
        ),
    ):
        """Train a Kanji recognition model."""
        echo_info(f"Training {model_type} model")
        echo_info(f"  Epochs: {epochs}")
        echo_info(f"  Batch size: {batch_size}")
        echo_info(f"  Learning rate: {learning_rate}")
        echo_info(f"  Checkpoint dir: {checkpoint_dir}")
        echo_info(f"  Dataset dir: {dataset_dir}")

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # This is a placeholder - actual training logic goes in the app
        echo_success(f"Training configured (implementation goes in calling code)")

    return train


def validate_path(path: Path, must_exist: bool = True) -> bool:
    """
    Validate that a path exists or is creatable.

    Args:
        path: Path to validate
        must_exist: If True, path must exist. If False, parent must exist.

    Returns:
        True if valid
    """
    if must_exist:
        if not path.exists():
            echo_error(f"Path does not exist: {path}")
            raise typer.Exit(code=1)
    else:
        if not path.parent.exists():
            echo_error(f"Parent directory does not exist: {path.parent}")
            raise typer.Exit(code=1)

    return True


def confirm(message: str, abort: bool = True) -> bool:
    """
    Ask for confirmation.

    Args:
        message: Confirmation message
        abort: If True, raise Exit on cancellation

    Returns:
        True if confirmed
    """
    if typer.confirm(message):
        return True

    if abort:
        echo_info("Cancelled")
        raise typer.Exit()

    return False


class CLIContext:
    """Context manager for CLI operations with logging."""

    def __init__(self, operation: str):
        """
        Initialize CLI context.

        Args:
            operation: Name of operation
        """
        self.operation = operation

    def __enter__(self):
        """Enter context."""
        echo_info(f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if exc_type is None:
            echo_success(f"Completed: {self.operation}")
        else:
            echo_error(f"Failed: {self.operation}")
            logger.exception(f"Error in {self.operation}")
        return False


# CLI app instance for reuse
main_app = create_app(
    name="tsujimoto",
    help="Kanji recognition training and evaluation tools",
    version="0.1.0",
)
