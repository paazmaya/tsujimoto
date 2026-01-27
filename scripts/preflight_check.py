#!/usr/bin/env python3
"""
Pre-flight check for Tsujimoto kanji recognition training setup.
Verifies CUDA, dependencies, dataset availability, and system resources.
Run this before starting actual training.
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
    """Check if ETL dataset files are available (combined or individual)"""

    dataset_dir = Path("dataset")

    # Check priority: combined_all_etl → etl9g → etl8g → etl7 → etl6
    dataset_priority = [
        "combined_all_etl",
        "etl9g",
        "etl8g",
        "etl7",
        "etl6",
    ]

    found_dataset = None
    found_path = None

    for dataset_name in dataset_priority:
        dataset_path = dataset_dir / dataset_name
        if dataset_path.exists():
            found_dataset = dataset_name
            found_path = dataset_path
            break

    if found_dataset is None:
        logger.error("✗ No ETL dataset found (checked: %s)", ", ".join(dataset_priority))
        return False

    logger.info("✓ Found dataset: %s", found_dataset)

    # Check metadata file
    metadata_file = found_path / "chunk_info.json"
    if not metadata_file.exists():
        logger.warning("✗ chunk_info.json not found in %s", found_dataset)
        return False

    logger.debug("✓ Found metadata: %s", metadata_file.name)
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
    """Check if main training scripts are present"""

    required_scripts = [
        "scripts/train_cnn_model.py",
        "scripts/train_hiercode.py",
        "scripts/train_qat.py",
        "scripts/train_radical_rnn.py",
        "scripts/train_vit.py",
        "scripts/prepare_dataset.py",
        "scripts/quantize_model.py",
    ]

    all_present = True

    for filename in required_scripts:
        if Path(filename).exists():
            logger.debug("  ✓ %s", filename)
        else:
            logger.warning("  ✗ %s not found", filename)
            all_present = False

    if all_present:
        logger.info("✓ All main training scripts present")
    else:
        logger.error("✗ Some training scripts missing")

    return all_present


def estimate_training_time():
    """Estimate training time based on system and dataset"""

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("Training on: %s", gpu_name)

            # Dataset-aware estimates
            dataset_dir = Path("dataset")
            if (dataset_dir / "combined_all_etl").exists():
                logger.info("With combined_all_etl (934K samples):")
                logger.info("  - CNN: ~2-3 hours for 30 epochs")
                logger.info("  - RNN: ~4-6 hours for 30 epochs")
                logger.info("  - HierCode: ~3-4 hours for 30 epochs")
            else:
                logger.info("With ETL9G (607K samples):")
                logger.info("  - CNN: ~1-2 hours for 30 epochs")
                logger.info("  - RNN: ~2-3 hours for 30 epochs")
                logger.info("  - HierCode: ~1.5-2.5 hours for 30 epochs")
        else:
            logger.warning("Training on CPU (very slow, >20 hours per approach)")
            logger.info("Recommendation: Use GPU for practical training")

    except ImportError:
        logger.error("PyTorch not available for time estimation")


def main():
    """Run all checks"""

    logger.info("\n=== Tsujimoto Kanji Recognition Pre-Flight Check ===")
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
    logger.info("\nTraining time estimates:")
    estimate_training_time()

    # Summary
    logger.info("\n=== Summary ===")
    logger.info("Checks passed: %d/%d", checks_passed, total_checks)
    if checks_passed == total_checks:
        logger.info("✓ All checks passed! Ready to train.")
        logger.info("\nNext steps:")
        logger.info("  1. Prepare dataset: uv run python scripts/prepare_dataset.py")
        logger.info("  2. Train model: uv run python scripts/train_cnn_model.py --data-dir dataset")
    else:
        logger.warning("⚠ Some checks failed. Review above for details.")

    return checks_passed == total_checks


if __name__ == "__main__":
    main()
