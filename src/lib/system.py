"""
System and GPU utilities for model training.

Provides functions for GPU verification, system info gathering, and environment setup.
Uses py-cpuinfo for better CPU detection and gpustat for enhanced GPU monitoring.
"""

import logging
import platform
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict

import psutil
import torch
from cpuinfo import get_cpu_info

logger = logging.getLogger(__name__)

DEFAULT_MAX_EXTRACT_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB default limit


def _is_within_directory(base_dir: Path, target_path: Path) -> bool:
    """Return True if target_path resolves inside base_dir."""
    try:
        base_resolved = base_dir.resolve()
        target_resolved = target_path.resolve()
        return target_resolved == base_resolved or base_resolved in target_resolved.parents
    except Exception:
        return False


def safe_extract_tar(
    archive_path: Path,
    extract_dir: Path,
    mode: str = "r:*",
    max_total_bytes: int = DEFAULT_MAX_EXTRACT_BYTES,
) -> Path:
    """
    Safely extract a TAR archive while preventing path traversal, link attacks, and zip bombs.

    Rejects:
    - Absolute paths or ``..`` components escaping ``extract_dir``
    - Symbolic links, hard links, FIFOs, character/block devices
    - Archives whose uncompressed size exceeds ``max_total_bytes``

    Args:
        archive_path: Path to the tar archive
        extract_dir: Destination directory
        mode: Tarfile open mode (default: ``"r:*"``)
        max_total_bytes: Maximum allowed total extracted bytes

    Returns:
        Path to ``extract_dir``

    Raises:
        ValueError: If an unsafe archive member or size violation is detected
    """
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    total_extracted = 0

    with tarfile.open(archive_path, mode) as tf:
        for member in tf.getmembers():
            # Reject symlinks, hardlinks, and device nodes
            if (
                member.issym()
                or member.islnk()
                or member.isfifo()
                or member.ischr()
                or member.isblk()
                or member.isdev()
            ):
                raise ValueError(f"Unsafe tar member type (link or device) rejected: {member.name}")

            if member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ValueError(f"Unsafe tar member path rejected: {member.name}")

            target_path = extract_dir / member.name
            if not _is_within_directory(extract_dir, target_path):
                raise ValueError(
                    f"Tar path traversal detected: {member.name} escapes {extract_dir}"
                )

            total_extracted += max(0, member.size)
            if total_extracted > max_total_bytes:
                raise ValueError(
                    f"Tar extraction exceeded maximum size limit ({max_total_bytes} bytes)"
                )

            try:
                tf.extract(member, extract_dir, filter="data")
            except TypeError:
                tf.extract(member, extract_dir)  # noqa: S202

    return extract_dir


def safe_extract_zip(
    archive_path: Path,
    extract_dir: Path,
    max_total_bytes: int = DEFAULT_MAX_EXTRACT_BYTES,
) -> Path:
    """
    Safely extract a ZIP archive while preventing Zip Slip traversal, symlinks, and zip bombs.

    Args:
        archive_path: Path to the zip archive
        extract_dir: Destination directory
        max_total_bytes: Maximum allowed total extracted bytes

    Returns:
        Path to ``extract_dir``

    Raises:
        ValueError: If an unsafe archive member or size violation is detected
    """
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    total_extracted = 0

    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            member_name = info.filename
            if member_name.startswith("/") or ".." in Path(member_name).parts:
                raise ValueError(f"Unsafe zip member path rejected: {member_name}")

            # Check Unix symlink mode in external_attr (0o120000)
            unix_mode = info.external_attr >> 16
            if (unix_mode & 0o170000) == 0o120000:
                raise ValueError(f"Unsafe zip symlink rejected: {member_name}")

            target_path = extract_dir / member_name
            if not _is_within_directory(extract_dir, target_path):
                raise ValueError(
                    f"Zip path traversal detected: {member_name} escapes {extract_dir}"
                )

            total_extracted += max(0, info.file_size)
            if total_extracted > max_total_bytes:
                raise ValueError(
                    f"Zip extraction exceeded maximum size limit ({max_total_bytes} bytes)"
                )

            zf.extract(info, extract_dir)  # noqa: S202

    return extract_dir


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

    Uses py-cpuinfo for comprehensive CPU details and enhanced GPU detection.

    Returns:
        dict: System information including CPU, memory, and GPU details
    """
    from datetime import datetime

    # Get detailed CPU info using py-cpuinfo
    cpu_info = get_cpu_info()

    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "processor": cpu_info.get("brand_raw", platform.processor()),
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
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
        info["cudnn_version"] = torch.backends.cudnn.version()
    else:
        info["gpu_available"] = False

    return info
