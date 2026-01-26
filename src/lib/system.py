"""
System and GPU utilities for model training.

Provides functions for GPU verification, system info gathering, and environment setup.
"""

import logging
import platform
import sys
from typing import Dict

import psutil
import torch

logger = logging.getLogger(__name__)


def verify_and_setup_gpu() -> str:
    """
    Verify NVIDIA GPU availability and setup CUDA optimizations.

    Returns:
        str: "cuda" device string

    Raises:
        SystemExit: If GPU is not available
    """
    if not torch.cuda.is_available():
        logger.error("✗ GPU not available")
        sys.exit(1)

    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Log GPU info
    device_name = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"✓ GPU: {device_name} ({memory_gb:.1f}GB)")

    return "cuda"


def check_virtual_environment() -> bool:
    """
    Check if running in a virtual environment.

    Returns:
        bool: True if in virtual environment, False otherwise
    """
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )

    if in_venv:
        logger.info("✓ Virtual environment detected")
    else:
        logger.warning("✗ Not in a virtual environment (recommended)")

    return in_venv


def check_gpu_availability() -> bool:
    """
    Check NVIDIA GPU and CUDA availability with detailed info.

    Returns:
        bool: True if CUDA is available, False otherwise
    """
    try:
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            gpu_count = torch.cuda.device_count()
            logger.info("✓ CUDA available with %d GPU(s)", gpu_count)

            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info("  GPU %d: %s (%.2f GB)", i, gpu_name, gpu_memory_gb)

            # Test GPU memory allocation
            try:
                _ = torch.randn(1, device="cuda")
                logger.info("✓ GPU memory allocation working")
            except RuntimeError as e:
                logger.warning("✗ GPU memory allocation issue: %s", e)
                return False

            return True
        else:
            logger.warning("✗ CUDA not available")
            return False
    except Exception as e:
        logger.error("✗ Error checking GPU availability: %s", e)
        return False


def get_system_info() -> Dict:
    """
    Get detailed system information for logging and CO2 calculation.

    Returns:
        dict: System information including CPU, memory, and GPU details
    """
    from datetime import datetime

    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }

    # GPU information
    if torch.cuda.is_available():
        info["gpu_available"] = True
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )
        info["cuda_version"] = torch.version.cuda
    else:
        info["gpu_available"] = False

    return info
