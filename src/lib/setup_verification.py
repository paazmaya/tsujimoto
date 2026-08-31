"""Unified setup verification module for Tsujimoto training environment.

This module consolidates setup verification logic from preflight_check.py and
verify_etl9g_setup.py into a single, reusable module.

Provides verification functions for:
- Python version and environment
- Required dependencies
- GPU/CUDA setup
- System resources
- Dataset structure and completeness
- ETL9G-specific dataset validation
- Training script availability

Example Usage:
    >>> from src.lib.setup_verification import SetupVerifier
    >>> verifier = SetupVerifier()
    >>> results = verifier.run_all_checks()
    >>> if results["all_ok"]:
    ...     print("Environment ready for training!")

Classes:
    SetupVerifier: Main verification class with check methods

Functions:
    verify_python_version: Check Python version compatibility
    verify_virtual_environment: Check if running in venv
    verify_dependencies: Check required packages
    verify_gpu_setup: Check GPU/CUDA availability
    verify_system_resources: Check RAM and disk space
    verify_dataset_structure: Check for ETL datasets
    verify_etl9g_datasets: ETL9G-specific validation

"""

import importlib.util
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from sys import version_info
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .logging_utils import setup_logger

logger = setup_logger(__name__)


class SetupVerifier:
    """Unified setup verification for Tsujimoto training environment.

    Provides methods for verifying:
    - Python version and virtual environment
    - Required dependencies (PyTorch, CUDA, etc.)
    - GPU/CUDA setup and memory
    - System resources (RAM, disk)
    - Dataset structure and metadata
    - ETL9G-specific dataset completeness

    Attributes:
        data_dir: Path to data directory
        check_results: Dictionary of check results
        verbose: Whether to log detailed information
    """

    def __init__(self, data_dir: str = "dataset", verbose: bool = True):
        """Initialize setup verifier.

        Args:
            data_dir: Path to data directory (default: "dataset")
            verbose: Whether to log detailed information

        """
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        self.check_results: Dict[str, Any] = {}

    def verify_python_version(self, min_version: Tuple[int, int] = (3, 8)) -> bool:
        """Verify Python version meets minimum requirement.

        Args:
            min_version: Minimum Python version required (default: 3.8)

        Returns:
            True if version meets requirement, False otherwise

        """
        if version_info >= min_version:
            msg = f"✓ Python version: {version_info.major}.{version_info.minor}.{version_info.micro}"
            logger.info(msg) if self.verbose else None
            return True
        else:
            msg = f"✗ Python version {version_info.major}.{version_info.minor} < required {min_version[0]}.{min_version[1]}"
            logger.error(msg)
            return False

    def verify_virtual_environment(self) -> bool:
        """Verify running in virtual environment.

        Returns:
            True if in venv, False otherwise

        """
        in_venv = hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        )

        if in_venv:
            logger.info(f"✓ Running in virtual environment: {os.environ.get('VIRTUAL_ENV', 'unknown')}")
            return True
        else:
            logger.warning("⚠ Not running in virtual environment (recommended)")
            return False

    def verify_dependencies(
        self, packages: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """Verify required packages are installed.

        Args:
            packages: List of package names to check (default: standard packages)

        Returns:
            Dictionary mapping package names to availability (True/False)

        """
        if packages is None:
            packages = [
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
                "safetensors",
                "bitsandbytes",
            ]

        results = {}
        missing = []
        found = []

        for package in packages:
            spec = importlib.util.find_spec(package)
            available = spec is not None
            results[package] = available

            if available:
                found.append(package)
            else:
                missing.append(package)

        logger.info(f"✓ Found {len(found)} dependencies: {', '.join(found)}")
        if missing:
            logger.warning(f"⚠ Missing {len(missing)} dependencies: {', '.join(missing)}")

        return results

    def verify_gpu_setup(self) -> Dict[str, Any]:
        """Verify GPU/CUDA setup and availability.

        Returns:
            Dictionary with GPU information:
            - cuda_available: Whether CUDA is available
            - gpu_count: Number of GPUs
            - gpu_models: List of GPU model names
            - gpu_memory: Total GPU memory in GB
            - gpu_memory_per_device: Memory per GPU
            - allocation_test_passed: Whether GPU memory allocation test passed

        """
        results: Dict[str, Any] = {}

        # CUDA availability
        results["cuda_available"] = torch.cuda.is_available()
        if not results["cuda_available"]:
            logger.warning("⚠ CUDA not available, using CPU (training will be slower)")
            return results

        # GPU count
        results["gpu_count"] = torch.cuda.device_count()
        logger.info(f"✓ GPU count: {results['gpu_count']}")

        # GPU models and memory
        gpu_models = []
        gpu_memory = []
        for i in range(results["gpu_count"]):
            name = torch.cuda.get_device_name(i)
            memory_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            gpu_models.append(name)
            gpu_memory.append(memory_gb)
            logger.info(f"  GPU {i}: {name} ({memory_gb:.1f} GB)")

        results["gpu_models"] = gpu_models
        results["gpu_memory"] = sum(gpu_memory)
        results["gpu_memory_per_device"] = gpu_memory

        # GPU memory allocation test
        try:
            test_tensor = torch.zeros((256, 256, 256, 256), device="cuda")
            del test_tensor
            torch.cuda.empty_cache()
            results["allocation_test_passed"] = True
            logger.info("✓ GPU memory allocation test passed")
        except RuntimeError:
            results["allocation_test_passed"] = False
            logger.warning("⚠ GPU memory allocation test failed (insufficient GPU memory)")

        return results

    def verify_system_resources(self) -> Dict[str, float]:
        """Verify system RAM and disk space.

        Returns:
            Dictionary with resource information:
            - ram_available_gb: Available RAM in GB
            - ram_total_gb: Total RAM in GB
            - disk_available_gb: Available disk space in GB
            - disk_total_gb: Total disk space in GB

        """
        results: Dict[str, float] = {}

        # RAM check
        ram_total_bytes = os.popen("vm_stat | grep 'Pages free' | awk '{print $3}'").read()
        ram_available_gb = float(ram_total_bytes.strip()) / 1024 / 1024 if ram_total_bytes else 0
        ram_total_gb = shutil.disk_usage("/").total / 1e9 / 1000  # Approximation

        results["ram_available_gb"] = ram_available_gb
        results["ram_total_gb"] = ram_total_gb

        if ram_available_gb < 8:
            logger.warning(f"⚠ Limited RAM: {ram_available_gb:.1f} GB (recommend ≥8 GB)")
        else:
            logger.info(f"✓ RAM: {ram_available_gb:.1f} GB available")

        # Disk space check
        total, used, free = shutil.disk_usage("/")
        disk_free_gb = free / 1e9
        disk_total_gb = total / 1e9

        results["disk_available_gb"] = disk_free_gb
        results["disk_total_gb"] = disk_total_gb

        if disk_free_gb < 10:
            logger.warning(f"⚠ Low disk space: {disk_free_gb:.1f} GB (recommend ≥10 GB)")
        else:
            logger.info(f"✓ Disk space: {disk_free_gb:.1f} GB available")

        return results

    def verify_dataset_structure(self, dataset_dir: Optional[str] = None) -> Dict[str, Any]:
        """Verify dataset structure for any ETL dataset.

        Checks for dataset priority order:
        1. combined_all_etl
        2. etl9g
        3. etl8g
        4. etl7
        5. etl6
        6. etl1

        Args:
            dataset_dir: Path to data directory (uses self.data_dir if not provided)

        Returns:
            Dictionary with dataset information:
            - dataset_found: Whether a dataset exists
            - primary_dataset: Name of primary dataset (if found)
            - metadata_exists: Whether metadata.json exists
            - available_datasets: List of datasets found

        """
        if dataset_dir is None:
            dataset_dir = self.data_dir
        else:
            dataset_dir = Path(dataset_dir)

        results: Dict[str, Any] = {
            "dataset_found": False,
            "primary_dataset": None,
            "metadata_exists": False,
            "available_datasets": [],
        }

        if not dataset_dir.exists():
            logger.warning(f"⚠ Dataset directory not found: {dataset_dir}")
            return results

        # Check for datasets in priority order
        priority_datasets = ["combined_all_etl", "etl9g", "etl8g", "etl7", "etl6", "etl1"]

        for dataset_name in priority_datasets:
            dataset_path = dataset_dir / dataset_name
            if dataset_path.exists():
                results["available_datasets"].append(dataset_name)
                if results["primary_dataset"] is None:
                    results["primary_dataset"] = dataset_name
                    results["dataset_found"] = True

        # Check for metadata
        metadata_path = dataset_dir / "metadata.json"
        results["metadata_exists"] = metadata_path.exists()

        if results["dataset_found"]:
            logger.info(f"✓ Dataset found: {results['primary_dataset']}")
            if results["metadata_exists"]:
                logger.info("✓ Metadata file exists")
            else:
                logger.warning("⚠ Metadata file missing")
        else:
            logger.warning("⚠ No datasets found")

        return results

    def verify_etl9g_datasets(self, data_dir: Optional[str] = None) -> Dict[str, Any]:
        """Verify ETL9G dataset structure and completeness.

        Checks:
        - Metadata.json presence and validity
        - chunk_info.json presence and validity
        - All expected chunk files exist
        - Sample data can be loaded
        - Character mapping file exists

        Args:
            data_dir: Path to data directory (uses self.data_dir if not provided)

        Returns:
            Dictionary with ETL9G verification results

        """
        if data_dir is None:
            data_dir = self.data_dir
        else:
            data_dir = Path(data_dir)

        results: Dict[str, Any] = {
            "etl9g_found": False,
            "metadata_valid": False,
            "chunks_valid": False,
            "sample_data_loadable": False,
            "character_mapping_exists": False,
            "errors": [],
        }

        etl9g_dir = data_dir / "etl9g"
        if not etl9g_dir.exists():
            results["errors"].append("ETL9G directory not found")
            return results

        results["etl9g_found"] = True
        logger.info("✓ ETL9G directory found")

        # Check metadata.json
        metadata_path = data_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                results["metadata_valid"] = True
                logger.info(f"✓ Metadata valid: {len(metadata)} entries")
            except (json.JSONDecodeError, IOError) as e:
                results["errors"].append(f"Metadata invalid: {str(e)}")
        else:
            results["errors"].append("metadata.json not found")

        # Check chunk_info.json
        chunk_info_path = etl9g_dir / "chunk_info.json"
        if chunk_info_path.exists():
            try:
                with open(chunk_info_path) as f:
                    chunk_info = json.load(f)

                # Verify chunks exist
                missing_chunks = []
                for chunk_name in chunk_info.get("chunks", []):
                    chunk_path = etl9g_dir / f"{chunk_name}.npz"
                    if not chunk_path.exists():
                        missing_chunks.append(chunk_name)

                if not missing_chunks:
                    results["chunks_valid"] = True
                    logger.info(f"✓ All {len(chunk_info.get('chunks', []))} chunks present")
                else:
                    results["errors"].append(f"Missing chunks: {missing_chunks}")
            except (json.JSONDecodeError, IOError) as e:
                results["errors"].append(f"chunk_info.json invalid: {str(e)}")
        else:
            results["errors"].append("chunk_info.json not found")

        # Try loading sample data
        try:
            sample_chunk_path = next(etl9g_dir.glob("*_chunk_0.npz"), None)
            if sample_chunk_path:
                data = np.load(sample_chunk_path)
                results["sample_data_loadable"] = True
                logger.info(f"✓ Sample data loadable: {list(data.files)}")
        except Exception as e:
            results["errors"].append(f"Cannot load sample data: {str(e)}")

        # Check character mapping
        char_mapping_path = data_dir / "character_mapping.json"
        results["character_mapping_exists"] = char_mapping_path.exists()
        if results["character_mapping_exists"]:
            logger.info("✓ Character mapping file found")

        # Log any errors
        if results["errors"]:
            for error in results["errors"]:
                logger.warning(f"⚠ {error}")

        return results

    def verify_training_scripts(self) -> Dict[str, bool]:
        """Verify presence of required training scripts.

        Returns:
            Dictionary mapping script names to existence (True/False)

        """
        scripts = [
            "train_cnn_model.py",
            "train_rnn.py",
            "train_vit.py",
            "train_hiercode.py",
            "train_qat.py",
            "train_radical_rnn.py",
            "train_hiercode_higita.py",
        ]

        results = {}
        scripts_dir = Path("scripts")

        for script in scripts:
            script_path = scripts_dir / script
            exists = script_path.exists()
            results[script] = exists

        found_count = sum(1 for v in results.values() if v)
        logger.info(f"✓ Found {found_count}/{len(scripts)} training scripts")

        return results

    def estimate_training_time(
        self, num_samples: int = 3700000, batch_size: int = 128
    ) -> Dict[str, str]:
        """Estimate training time based on hardware.

        Args:
            num_samples: Number of training samples
            batch_size: Training batch size

        Returns:
            Dictionary with training time estimates

        """
        estimates: Dict[str, str] = {}

        # Get GPU info
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

            if gpu_count >= 1 and gpu_memory_gb >= 16:
                # Assume ~5000 samples/sec on high-end GPU
                samples_per_sec = 5000 * gpu_count
                hours_per_epoch = (num_samples / batch_size) / (samples_per_sec / 60 / 60)
                estimates["per_epoch_gpu"] = f"{hours_per_epoch:.1f} hours"
                estimates["100_epochs_gpu"] = f"{hours_per_epoch * 100:.0f} hours ({hours_per_epoch * 100 / 24:.1f} days)"
            else:
                # Assume ~2000 samples/sec on modest GPU
                samples_per_sec = 2000
                hours_per_epoch = (num_samples / batch_size) / (samples_per_sec / 60 / 60)
                estimates["per_epoch_gpu"] = f"{hours_per_epoch:.1f} hours"
                estimates["100_epochs_gpu"] = f"{hours_per_epoch * 100:.0f} hours ({hours_per_epoch * 100 / 24:.1f} days)"
        else:
            # Assume ~100 samples/sec on CPU
            samples_per_sec = 100
            hours_per_epoch = (num_samples / batch_size) / (samples_per_sec / 60 / 60)
            estimates["per_epoch_cpu"] = f"{hours_per_epoch:.1f} hours"
            estimates["100_epochs_cpu"] = f"{hours_per_epoch * 100:.0f} hours ({hours_per_epoch * 100 / 24:.1f} days)"

        return estimates

    def run_all_checks(
        self, check_training_scripts: bool = True, check_model: bool = False
    ) -> Dict[str, Any]:
        """Run all verification checks and return unified results.

        Args:
            check_training_scripts: Whether to check for training scripts
            check_model: Whether to verify model architecture (future)

        Returns:
            Dictionary with all check results

        """
        logger.info("=" * 70)
        logger.info("TSUJIMOTO SETUP VERIFICATION")
        logger.info("=" * 70)

        all_results: Dict[str, Any] = {}

        # Python and environment
        logger.info("\n[Python & Environment]")
        all_results["python_version_ok"] = self.verify_python_version()
        all_results["venv_ok"] = self.verify_virtual_environment()

        # Dependencies
        logger.info("\n[Dependencies]")
        all_results["dependencies"] = self.verify_dependencies()
        all_results["dependencies_ok"] = all(all_results["dependencies"].values())

        # GPU/CUDA
        logger.info("\n[GPU/CUDA]")
        all_results["gpu"] = self.verify_gpu_setup()

        # System resources
        logger.info("\n[System Resources]")
        all_results["system"] = self.verify_system_resources()

        # Dataset
        logger.info("\n[Dataset]")
        all_results["dataset"] = self.verify_dataset_structure()
        all_results["dataset_ok"] = all_results["dataset"]["dataset_found"]

        # ETL9G specific
        if all_results["dataset"].get("primary_dataset") == "etl9g":
            logger.info("\n[ETL9G Validation]")
            all_results["etl9g"] = self.verify_etl9g_datasets()

        # Training scripts
        if check_training_scripts:
            logger.info("\n[Training Scripts]")
            all_results["training_scripts"] = self.verify_training_scripts()

        # Training time estimate
        logger.info("\n[Training Time Estimates]")
        all_results["training_time_estimates"] = self.estimate_training_time()
        for key, value in all_results["training_time_estimates"].items():
            logger.info(f"  {key}: {value}")

        # Summary
        all_results["all_ok"] = all([
            all_results["python_version_ok"],
            all_results["dependencies_ok"],
            all_results.get("dataset_ok", False),
        ])

        logger.info("\n" + "=" * 70)
        if all_results["all_ok"]:
            logger.info("✓ All checks passed! Ready for training.")
        else:
            logger.warning("⚠ Some checks failed. Please review above.")
        logger.info("=" * 70)

        return all_results


# ============================================================================
# Convenience Functions
# ============================================================================


def verify_python_version(min_version: Tuple[int, int] = (3, 8)) -> bool:
    """Verify Python version (convenience function).

    Args:
        min_version: Minimum Python version required

    Returns:
        True if version meets requirement

    """
    verifier = SetupVerifier(verbose=False)
    return verifier.verify_python_version(min_version)


def verify_dependencies(packages: Optional[List[str]] = None) -> Dict[str, bool]:
    """Verify required packages (convenience function).

    Args:
        packages: List of package names to check

    Returns:
        Dictionary mapping package names to availability

    """
    verifier = SetupVerifier(verbose=False)
    return verifier.verify_dependencies(packages)


def verify_gpu_setup() -> Dict[str, Any]:
    """Verify GPU setup (convenience function).

    Returns:
        Dictionary with GPU information

    """
    verifier = SetupVerifier(verbose=False)
    return verifier.verify_gpu_setup()


def verify_system_resources() -> Dict[str, float]:
    """Verify system resources (convenience function).

    Returns:
        Dictionary with resource information

    """
    verifier = SetupVerifier(verbose=False)
    return verifier.verify_system_resources()


def verify_etl9g_datasets(data_dir: str = "dataset") -> Dict[str, Any]:
    """Verify ETL9G datasets (convenience function).

    Args:
        data_dir: Path to data directory

    Returns:
        Dictionary with ETL9G verification results

    """
    verifier = SetupVerifier(data_dir=data_dir, verbose=True)
    return verifier.verify_etl9g_datasets()
