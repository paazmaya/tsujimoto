#!/usr/bin/env python3
"""
Pre-flight check for ETL9G training setup
Run this before starting actual training
"""

import importlib.util
import sys
from pathlib import Path

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import setup_logger

logger = setup_logger(__name__)


def check_virtual_environment():
    """Check if virtual environment is active"""

    # Check if we're in a virtual environment
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )

    if in_venv:
        logger.info("✓ Virtual environment detected")
        return True
    else:
        logger.warning("✗ Not in a virtual environment (recommended)")
        return False


def check_gpu_availability():
    """Check NVIDIA GPU and CUDA availability"""

    try:
        import torch

        # Check CUDA availability
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
                test_tensor = torch.randn(1000, 1000).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                logger.info("✓ GPU memory allocation test passed")
            except Exception as e:  # noqa: S110
                logger.warning("GPU memory test failed: %s", str(e))

            return True
        else:
            logger.warning("✗ CUDA not available (CPU mode only)")
            return False

    except ImportError:
        logger.error("PyTorch not installed")
        return False


def check_requirements():
    """Check if all required packages are available"""

    required_packages = [
        "torch",
        "torchvision",
        "numpy",
        "sklearn",
        "matplotlib",
        "tqdm",
        "cv2",
        "PIL",
        "onnx",
        "onnxruntime",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == "cv2":
                importlib.util.find_spec("cv2")
            elif package == "PIL":
                importlib.util.find_spec("PIL")
            elif package == "sklearn":
                importlib.util.find_spec("sklearn")
            else:
                __import__(package)
            logger.debug("  ✓ %s", package)
        except ImportError:
            logger.warning("  ✗ %s not found", package)
            missing_packages.append(package)

    if missing_packages:
        logger.error("Missing packages: %s", ", ".join(missing_packages))
        return False

    logger.info("✓ All required packages installed")
    return True


def check_data_files():
    """Check if ETL9G data files are available"""

    etl9g_dir = Path("ETL9G")
    if not etl9g_dir.exists():
        logger.error("✗ ETL9G directory not found")
        return False

    # Check for ETL9G files
    etl_files = list(etl9g_dir.glob("ETL9G_*"))
    etl_files = [f for f in etl_files if f.is_file() and "INFO" not in f.name]

    if len(etl_files) < 50:
        logger.warning("✗ Expected ~50 ETL9G files, found %d", len(etl_files))
        return False

    logger.info("✓ Found %d ETL9G data files", len(etl_files))

    # Check file sizes (should be around 99MB each)
    sample_file = etl_files[0]
    file_size_mb = sample_file.stat().st_size / (1024 * 1024)

    if file_size_mb < 90 or file_size_mb > 110:
        logger.warning("✗ Sample file size %.1f MB (expected ~99 MB)", file_size_mb)
        return False

    logger.debug("Sample file size: %.1f MB", file_size_mb)
    return True


def check_system_resources():
    """Check system memory and disk space"""

    try:
        import psutil

        # Memory check
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)

        if available_gb < 8:
            logger.warning("⚠ Low available memory: %.1f GB (recommended: 8+ GB)", available_gb)
        else:
            logger.info("✓ System memory: %.1f GB total, %.1f GB available", total_gb, available_gb)

        # Disk space check
        disk = psutil.disk_usage(".")
        free_gb = disk.free / (1024**3)

        if free_gb < 10:
            logger.warning("⚠ Low disk space: %.1f GB free (recommended: 10+ GB)", free_gb)
        else:
            logger.info("✓ Disk space available: %.1f GB", free_gb)

    except ImportError:
        logger.debug("psutil not available, skipping system resource check")


def check_training_scripts():
    """Check if all training scripts are present"""

    required_files = [
        "scripts/prepare_etl9g_dataset.py",
        "scripts/train_cnn_model.py",
        "scripts/test_etl9g_setup.py",
    ]

    all_present = True

    for filename in required_files:
        if Path(filename).exists():
            logger.debug("  ✓ %s", filename)
        else:
            logger.warning("  ✗ %s not found", filename)
            all_present = False

    if all_present:
        logger.info("✓ All training scripts present")
    else:
        logger.error("✗ Some training scripts missing")

    return all_present


def estimate_training_time():
    """Estimate training time based on system"""

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("Training on: %s", gpu_name)
            logger.info("Estimated time for 30 epochs: 2-4 hours")
        else:
            logger.warning("Training on CPU (very slow, >10 hours for 30 epochs)")
            logger.info("Recommended: Use GPU for practical training")

    except ImportError:
        logger.error("PyTorch not available for time estimation")


def main():
    """Run all checks"""

    logger.info("\n=== ETL9G Training Pre-Flight Check ===")
    checks_passed = 0
    total_checks = 0

    # Virtual environment check
    total_checks += 1
    if check_virtual_environment():
        checks_passed += 1

    # GPU and CUDA check
    total_checks += 1
    if check_gpu_availability():
        checks_passed += 1

    # Requirements check
    total_checks += 1
    logger.info("Checking required packages...")
    if check_requirements():
        checks_passed += 1

    # Data files check
    total_checks += 1
    if check_data_files():
        checks_passed += 1

    # System resources
    logger.info("Checking system resources...")
    check_system_resources()

    # Training scripts check
    total_checks += 1
    logger.info("Checking training scripts...")
    if check_training_scripts():
        checks_passed += 1

    # Training time estimate
    logger.info("\nTraining time estimate:")
    estimate_training_time()

    # Summary
    logger.info("\n=== Summary ===")
    logger.info("Checks passed: %d/%d", checks_passed, total_checks)
    if checks_passed == total_checks:
        logger.info("✓ All checks passed! Ready to train.")
    else:
        logger.warning("⚠ Some checks failed. Review above for details.")

    return checks_passed == total_checks


if __name__ == "__main__":
    main()
