"""
Unified logging configuration using Loguru.

Provides consistent logging setup across the project with optional file output.
Uses loguru for better performance, automatic rotation, and structured logging.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger as loguru_logger

# Configure loguru default logging to console
loguru_logger.remove()  # Remove default handler
loguru_logger.add(
    sys.stderr,
    format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)


def setup_logger(
    name: str = "__main__",
    level: str = "INFO",
    log_file: Optional[str] = None,
):
    """
    Configure and return a logger with consistent formatting.

    Uses loguru for superior performance and features. This function provides
    backward compatibility with the previous logging API.

    Args:
        name: Logger name (typically __name__)
        level: Logging level as string (default: "INFO")
        log_file: Optional file path for log output

    Returns:
        loguru logger instance
    """
    # Configure file logging if requested
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        loguru_logger.add(
            str(log_path),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
            level=level,
            rotation="100 MB",  # Auto-rotate when file reaches 100MB
            retention="7 days",  # Keep logs for 7 days
            compression="zip",  # Compress rotated logs
        )

    return loguru_logger


def suppress_warnings():
    """Suppress non-critical warnings from dependencies."""
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
