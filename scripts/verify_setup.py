"""Unified setup verification CLI tool.

Consolidates environment verification from:
- preflight_check.py
- verify_etl9g_setup.py

Provides comprehensive environment and dataset validation for training.

Usage:
    python verify_setup.py                           # Run all checks
    python verify_setup.py --check environment      # Environment only
    python verify_setup.py --check dependencies     # Dependencies only
    python verify_setup.py --check gpu              # GPU setup only
    python verify_setup.py --check dataset          # Dataset verification
    python verify_setup.py --check etl9g            # ETL9G specific checks
    python verify_setup.py --data-dir /path/to/data # Custom data directory

"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib.logging_utils import setup_logger
from src.lib.setup_verification import SetupVerifier

logger = setup_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for setup verification CLI.

    Returns:
        ArgumentParser instance

    """
    parser = argparse.ArgumentParser(
        description="Verify Tsujimoto training environment and dataset setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run all checks:
    python verify_setup.py

  Check only environment:
    python verify_setup.py --check environment

  Check dataset with custom data directory:
    python verify_setup.py --check dataset --data-dir ./data

  Check ETL9G specific validation:
    python verify_setup.py --check etl9g --data-dir ./data
        """,
    )

    # Checks to run
    parser.add_argument(
        "--check",
        type=str,
        choices=[
            "all",
            "environment",
            "dependencies",
            "gpu",
            "system",
            "dataset",
            "etl9g",
            "training-scripts",
            "training-time",
        ],
        default="all",
        help="Which checks to run (default: all)",
    )

    # Data directory
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset",
        help="Path to data directory (default: dataset)",
    )

    # Flags
    parser.add_argument(
        "--include-training-scripts",
        action="store_true",
        default=False,
        help="Include training scripts verification",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    return parser


def run_checks(args) -> bool:
    """Run specified verification checks.

    Args:
        args: Parsed command-line arguments

    Returns:
        True if all checks passed, False otherwise

    """
    verifier = SetupVerifier(data_dir=args.data_dir, verbose=args.verbose)

    # Determine which checks to run
    check = args.check.lower()

    if check == "all":
        logger.info("Running all verification checks...")
        results = verifier.run_all_checks(check_training_scripts=args.include_training_scripts)
        all_passed = results.get("all_ok", False)

    elif check == "environment":
        logger.info("Checking Python and environment...")
        python_ok = verifier.verify_python_version()
        venv_ok = verifier.verify_virtual_environment()
        all_passed = python_ok and venv_ok

    elif check == "dependencies":
        logger.info("Checking dependencies...")
        deps = verifier.verify_dependencies()
        all_passed = all(deps.values())

    elif check == "gpu":
        logger.info("Checking GPU/CUDA setup...")
        gpu_info = verifier.verify_gpu_setup()
        all_passed = gpu_info.get("cuda_available", False)

    elif check == "system":
        logger.info("Checking system resources...")
        system_info = verifier.verify_system_resources()
        all_passed = system_info.get("disk_available_gb", 0) > 0

    elif check == "dataset":
        logger.info("Checking dataset structure...")
        dataset_info = verifier.verify_dataset_structure()
        all_passed = dataset_info.get("dataset_found", False)

    elif check == "etl9g":
        logger.info("Checking ETL9G datasets...")
        etl9g_info = verifier.verify_etl9g_datasets()
        all_passed = etl9g_info.get("etl9g_found", False) and not etl9g_info.get("errors", [])

    elif check == "training-scripts":
        logger.info("Checking training scripts...")
        scripts = verifier.verify_training_scripts()
        all_passed = any(scripts.values())

    elif check == "training-time":
        logger.info("Estimating training time...")
        estimates = verifier.estimate_training_time()
        logger.info(f"Training time estimates: {estimates}")
        all_passed = len(estimates) > 0

    else:
        all_passed = False

    return all_passed


def main():
    """Main CLI entry point for setup verification."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("=" * 70)
    logger.info("TSUJIMOTO SETUP VERIFICATION")
    logger.info("=" * 70)

    try:
        all_passed = run_checks(args)

        logger.info("=" * 70)
        if all_passed:
            logger.info("✓ Verification completed successfully!")
            logger.info("Ready for training.")
        else:
            logger.warning("⚠ Some checks failed. Please review above.")
        logger.info("=" * 70)

        return 0 if all_passed else 1

    except Exception as e:
        logger.error(f"✗ Error during verification: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
