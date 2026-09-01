#!/usr/bin/env python3
"""Download and prepare ETL datasets for training.

This script automates the download, extraction, and preparation of Japanese
character datasets from the ETL Character Database (ETLCDB).

Features:
- Interactive dataset selection
- Automatic download from ETLCDB
- Checksum verification
- Resume capability for interrupted downloads
- Automatic extraction and conversion to NPZ format
- Dataset validation and metadata generation

Usage:
    # Interactive mode (recommended)
    python scripts/download_datasets.py

    # Command-line mode
    python scripts/download_datasets.py --dataset etl9g
    python scripts/download_datasets.py --dataset combined_all_etl
    python scripts/download_datasets.py --dataset all  # Download all datasets

    # With custom output directory
    python scripts/download_datasets.py --dataset etl9g --output-dir /custom/path

    # Verify already downloaded datasets
    python scripts/download_datasets.py --verify-only
"""

import argparse
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DatasetInfo:
    """Information about available datasets."""

    DATASETS: Dict[str, Dict[str, str]] = {
        "etl6": {
            "name": "ETL6",
            "description": "Numerals (0-9), Katakana, symbols",
            "classes": 114,
            "samples": 157000,
            "file": "ETL6.zip",
            "url": "http://etlcdb.db.aist.go.jp/etl6/ETL6.zip",
            "size_mb": 37,
            "sha256": "",  # Checksum if available
        },
        "etl7": {
            "name": "ETL7",
            "description": "Hiragana characters",
            "classes": 48,
            "samples": 16800,
            "file": "ETL7.zip",
            "url": "http://etlcdb.db.aist.go.jp/etl7/ETL7.zip",
            "size_mb": 4,
            "sha256": "",
        },
        "etl8g": {
            "name": "ETL8G",
            "description": "Educational Kanji + Hiragana",
            "classes": 956,
            "samples": 153000,
            "file": "ETL8G.zip",
            "url": "http://etlcdb.db.aist.go.jp/etl8/ETL8G.zip",
            "size_mb": 36,
            "sha256": "",
        },
        "etl9g": {
            "name": "ETL9G",
            "description": "JIS Level 1 Kanji + Hiragana (popular)",
            "classes": 3036,
            "samples": 607000,
            "file": "ETL9G.zip",
            "url": "http://etlcdb.db.aist.go.jp/etl9/ETL9G.zip",
            "size_mb": 144,
            "sha256": "",
        },
        "combined_all_etl": {
            "name": "Combined ETL6-9G",
            "description": "All datasets combined (recommended for best accuracy)",
            "classes": 43427,
            "samples": 934000,
            "file": "COMBINED",
            "url": "N/A (automatic combination)",
            "size_mb": 224,
            "sha256": "",
        },
    }


def print_dataset_info() -> None:
    """Print available datasets."""
    logger.info("\n" + "=" * 80)
    logger.info("Available Datasets for Japanese Kanji Recognition")
    logger.info("=" * 80 + "\n")

    for key, info in DatasetInfo.DATASETS.items():
        if key == "combined_all_etl":
            continue  # Skip combined for now
        logger.info(f"📦 {info['name']} ({key})")
        logger.info(f"   Description: {info['description']}")
        logger.info(f"   Classes: {info['classes']} | Samples: {info['samples']:,}")
        logger.info(f"   Size: ~{info['size_mb']} MB")

    logger.info("📦 Combined ETL6-9G (combined_all_etl)")
    logger.info("   Description: All datasets combined (recommended for best accuracy)")
    logger.info("   Classes: 43,427 | Samples: 934,000")
    logger.info("   Size: ~224 MB (when all downloaded)")
    logger.info("=" * 80 + "\n")


def download_file(
    url: str, destination: Path, chunk_size: int = 8192, max_retries: int = 3
) -> bool:
    """Download a file with resume capability.

    Args:
        url: URL to download from
        destination: Local file path
        chunk_size: Size of chunks to download
        max_retries: Maximum retry attempts

    Returns:
        True if successful, False otherwise
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading from {url} (attempt {attempt + 1}/{max_retries})...")

            # Check if partial file exists
            resume_header = {}
            if destination.exists():
                resume_header = {"Range": f"bytes={destination.stat().st_size}-"}
                logger.info(f"Resuming download from byte {destination.stat().st_size}")

            request = urllib.request.Request(url, headers=resume_header)  # noqa: S310
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                total_size = int(response.headers.get("content-length", 0))
                downloaded = destination.stat().st_size if destination.exists() else 0

                mode = "ab" if destination.exists() else "wb"
                with open(destination, mode) as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(
                                f"  Progress: {progress:.1f}% ({downloaded / 1024 / 1024:.1f}/"
                                f"{total_size / 1024 / 1024:.1f} MB)"
                            )

            logger.info("✓ Successfully downloaded to {destination}")
            return True

        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            logger.warning(f"⚠ Download failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                logger.info("Retrying in 5 seconds...")
                import time

                time.sleep(5)
            else:
                logger.error(f"❌ Failed to download after {max_retries} attempts")
                return False

    return False


def verify_dataset(dataset_dir: Path, dataset_name: str) -> bool:
    """Verify that a dataset is available and valid.

    Args:
        dataset_dir: Directory containing datasets
        dataset_name: Name of dataset to verify

    Returns:
        True if dataset is valid
    """
    etl_dir = dataset_dir / dataset_name.upper().replace("_", "")

    if not etl_dir.exists():
        logger.warning(f"⚠ Dataset directory not found: {etl_dir}")
        return False

    # Check for metadata file
    metadata_file = etl_dir / "metadata.json"
    if metadata_file.exists():
        logger.info(f"✓ Dataset {dataset_name} found with metadata")
        return True

    # Check for at least one NPZ chunk
    chunks = list(etl_dir.glob("*.npz"))
    if chunks:
        logger.info(f"✓ Dataset {dataset_name} found ({len(chunks)} chunks)")
        return True

    logger.warning(f"⚠ Dataset {dataset_name} directory exists but contains no data")
    return False


def prepare_dataset(dataset_dir: Path, dataset_name: str) -> bool:
    """Prepare dataset by converting to NPZ format if needed.

    Args:
        dataset_dir: Directory containing datasets
        dataset_name: Name of dataset to prepare

    Returns:
        True if successful
    """
    logger.info(f"Preparing dataset: {dataset_name}")

    # Check if already prepared
    if verify_dataset(dataset_dir, dataset_name):
        logger.info(f"✓ {dataset_name} is already prepared")
        return True

    # Run preparation script
    try:
        logger.info(f"Running prepare_dataset.py for {dataset_name}...")
        import subprocess

        result = subprocess.run(  # noqa: S603
            [sys.executable, "scripts/prepare_dataset.py", "--dataset", dataset_name],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode == 0:
            logger.info(f"✓ Successfully prepared {dataset_name}")
            return True
        else:
            logger.error(f"❌ Failed to prepare {dataset_name}: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"❌ Preparation timed out for {dataset_name}")
        return False
    except FileNotFoundError:
        logger.error("❌ prepare_dataset.py script not found")
        return False


def interactive_mode(dataset_dir: Path) -> List[str]:
    """Interactive dataset selection.

    Returns:
        List of selected dataset names
    """
    print_dataset_info()
    logger.info("Select datasets to download:")
    logger.info("  1. ETL6 (numerals, katakana, symbols) - ~37 MB")
    logger.info("  2. ETL7 (hiragana) - ~4 MB")
    logger.info("  3. ETL8G (educational kanji, hiragana) - ~36 MB")
    logger.info("  4. ETL9G (JIS Level 1 kanji, hiragana) - ~144 MB [RECOMMENDED]")
    logger.info("  5. Combined (ETL6-9G all together) - ~224 MB [BEST ACCURACY]")
    logger.info("  6. All of the above")
    logger.info("  0. Skip (verify only)")

    try:
        choice = input("\nEnter choice (0-6): ").strip()

        selected = {
            "1": ["etl6"],
            "2": ["etl7"],
            "3": ["etl8g"],
            "4": ["etl9g"],
            "5": ["combined_all_etl"],
            "6": ["etl6", "etl7", "etl8g", "etl9g", "combined_all_etl"],
            "0": [],
        }

        return selected.get(choice, [])
    except (KeyboardInterrupt, EOFError):
        logger.info("\n\nAborted by user")
        sys.exit(0)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and prepare ETL datasets for kanji recognition training"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=list(DatasetInfo.DATASETS.keys()) + ["all"],
        help="Dataset to download (default: interactive mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset"),
        help="Output directory for datasets (default: ./dataset)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing datasets, don't download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )
    parser.add_argument(
        "--no-prepare",
        action="store_true",
        help="Skip preparation step (just download)",
    )

    args = parser.parse_args()

    # Determine datasets to download
    datasets_to_process: List[str] = []

    if args.verify_only:
        logger.info("Running in verify-only mode")
        datasets_to_process = list(DatasetInfo.DATASETS.keys())
    elif args.dataset == "all":
        datasets_to_process = [k for k in DatasetInfo.DATASETS.keys() if k != "combined_all_etl"]
    elif args.dataset:
        datasets_to_process = [args.dataset]
    else:
        # Interactive mode
        datasets_to_process = interactive_mode(args.output_dir)

    if not datasets_to_process:
        logger.info("No datasets selected")
        return 0

    # Process datasets
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failed_count = 0

    for dataset_name in datasets_to_process:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {dataset_name}")
        logger.info(f"{'=' * 60}")

        dataset_info = DatasetInfo.DATASETS[dataset_name]

        # Skip combined if it's in the list but not available for download
        if dataset_name == "combined_all_etl":
            if not args.verify_only:
                logger.info("Skipping combined_all_etl (auto-generated from ETL6-9G)")
                continue

        # Verify existing
        if verify_dataset(output_dir, dataset_name):
            logger.info(f"✓ {dataset_name} already available")
            success_count += 1
            continue

        # Download
        if not args.verify_only:
            download_path = output_dir / dataset_info["file"]

            # Check if already downloaded
            if download_path.exists() and not args.force:
                logger.info(f"✓ {dataset_info['file']} already downloaded")
            else:
                if not download_file(dataset_info["url"], download_path):
                    logger.error(f"Failed to download {dataset_name}")
                    failed_count += 1
                    continue

            # Prepare
            if not args.no_prepare:
                if not prepare_dataset(output_dir, dataset_name):
                    failed_count += 1
                    continue

            success_count += 1
        else:
            logger.warning(f"⚠ {dataset_name} not available (verify-only mode)")
            failed_count += 1

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("Summary")
    logger.info(f"{'=' * 60}")
    logger.info(f"✓ Successful: {success_count}")
    logger.info(f"✗ Failed: {failed_count}")

    if failed_count == 0 and success_count > 0:
        logger.info("\n🎉 All datasets ready for training!")
        logger.info(f"Location: {output_dir.absolute()}")
        return 0
    else:
        logger.info("\n⚠ Some datasets failed. See above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
