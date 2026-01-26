"""
Unified logging configuration for all scripts.

Provides consistent logging setup across the project with optional file output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "__main__",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)
        log_file: Optional file path for log output

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only add handlers if none exist to avoid duplicates
    if not logger.handlers:
        logger.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(level)
            detailed_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)

    return logger


def suppress_warnings():
    """Suppress non-critical warnings from dependencies."""
    # Suppress PyTorch's TypedStorage deprecation warning (internal, not in user code)
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
