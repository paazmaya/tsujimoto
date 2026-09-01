#!/usr/bin/env python3
"""
Research dataset downloader - extends existing ETL download infrastructure.

Handles downloading and verifying research datasets:
- MegaHan97K (97k Chinese character classes)
- DKDS (degraded Kuzushiji documents)
- Chronicles-OCR (historical character evolution)
- JaWildText (Japanese scene text)
- MCCD (multi-attribute calligraphy)
- Stroke-level handwriting database

Usage:
    python scripts/download_research_datasets.py --dataset megahan97k
    python scripts/download_research_datasets.py --all
    python scripts/download_research_datasets.py --dataset dkds --force
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib.dataset_handlers import DatasetRegistry, DownloadManager
from src.lib.logging_utils import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# RESEARCH DATASET DEFINITIONS
# ============================================================================

RESEARCH_DATASETS: Dict[str, Dict] = {
    "megahan97k": {
        "name": "MegaHan97K",
        "url": "https://github.com/SCUT-DLVCLab/MegaHan97K/releases/download/v1.0/megahan97k_v1.tar.gz",
        "size_mb": 5000,
        "description": "Large-scale Chinese character dataset with 97,455 classes",
        "format": "tar.gz",
        "checksum": None,  # Will be verified after download
        "tags": ["zero-shot", "chinese", "large-scale"],
        "notes": "Download requires ~5GB space. Consider bandwidth before downloading.",
    },
    "dkds": {
        "name": "DKDS",
        "url": "https://github.com/RuiyangJu/DKDS/releases/download/v1.0/dkds_dataset.zip",
        "size_mb": 500,
        "description": "Degraded Kuzushiji documents benchmark with seal interference",
        "format": "zip",
        "checksum": None,
        "tags": ["japanese", "historical", "degraded", "seals"],
        "notes": "Japanese historical handwriting dataset for testing robustness.",
    },
    "chronicles_ocr": {
        "name": "Chronicles-OCR",
        "url": "https://github.com/VT-NLP/Chronicles/releases/download/v1.0/chronicles_ocr.zip",
        "size_mb": 200,
        "description": "Cross-temporal Chinese character recognition benchmark (historical evolution)",
        "format": "zip",
        "checksum": None,
        "tags": ["historical", "cross-temporal", "evolution", "benchmark"],
        "notes": "Tests visual perception across 7 historical Chinese scripts.",
    },
    "jawildtext": {
        "name": "JaWildText",
        "url": "https://github.com/maeda-ltl/jawildtext/releases/download/v1.0/jawildtext.zip",
        "size_mb": 800,
        "description": "Japanese scene text understanding benchmark (in-the-wild)",
        "format": "zip",
        "checksum": None,
        "tags": ["japanese", "scene-text", "vqa", "in-the-wild"],
        "notes": "Real-world Japanese text from images with diverse layouts.",
    },
    "mccd": {
        "name": "MCCD",
        "url": "https://github.com/PRIS-CV/MCCD/releases/download/v1.0/mccd_dataset.zip",
        "size_mb": 300,
        "description": "Multi-attribute Chinese calligraphy character dataset",
        "format": "zip",
        "checksum": None,
        "tags": ["calligraphy", "style", "period", "attributes"],
        "notes": "Calligraphic variations with style and period metadata.",
    },
    "stroke_database": {
        "name": "Stroke-level Handwriting Database",
        "url": "https://github.com/shirooo39/stroke-dataset/releases/download/v1.0/stroke_database.zip",
        "size_mb": 400,
        "description": "Stroke-level handwriting data from 42 writers, 1200 characters each",
        "format": "zip",
        "checksum": None,
        "tags": ["trajectory", "online-handwriting", "stroke-level"],
        "notes": "Pen trajectory data for sequence models. ~50MB per writer.",
    },
}


# ============================================================================
# DOWNLOAD MANAGER
# ============================================================================


class ResearchDatasetDownloader:
    """High-level manager for downloading and verifying research datasets."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize downloader.

        Args:
            output_dir: Directory to store downloads
        """
        self.output_dir = Path(output_dir or "data/research_datasets")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.download_manager = DownloadManager(cache_dir=self.output_dir)
        self.registry = DatasetRegistry()

    def download_dataset(
        self,
        dataset_name: str,
        force: bool = False,
        extract: bool = True,
    ) -> bool:
        """
        Download a research dataset.

        Args:
            dataset_name: Dataset identifier (from RESEARCH_DATASETS keys)
            force: If True, re-download even if file exists
            extract: If True, automatically extract archive

        Returns:
            True if successful, False otherwise
        """
        if dataset_name not in RESEARCH_DATASETS:
            logger.error(f"Unknown dataset: {dataset_name}")
            logger.info(f"Available datasets: {list(RESEARCH_DATASETS.keys())}")
            return False

        config = RESEARCH_DATASETS[dataset_name]

        logger.info("=" * 70)
        logger.info(f"DOWNLOADING: {config['name']}")
        logger.info("=" * 70)
        logger.info(f"Description: {config['description']}")
        logger.info(f"Size: ~{config['size_mb']}MB")
        logger.info(f"URL: {config['url']}")
        if config.get("notes"):
            logger.warning(f"⚠️  {config['notes']}")

        # Download file
        filename = Path(config["url"]).name
        output_path = self.output_dir / dataset_name / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = self.download_manager.download_file(
            url=config["url"],
            output_path=output_path,
            force=force,
            checksum=config.get("checksum"),
            max_retries=3,
        )

        if not success:
            logger.error(f"Failed to download {dataset_name}")
            return False

        logger.info(f"✓ Downloaded: {output_path}")

        # Extract if requested
        if extract and config["format"] in ["zip", "tar.gz"]:
            if self._extract_dataset(output_path, config["format"]):
                logger.info(f"✓ Extracted: {output_path}")
            else:
                logger.warning(f"⚠️  Failed to extract {output_path}")

        return True

    def download_all(self, force: bool = False, extract: bool = True) -> Dict[str, bool]:
        """
        Download all research datasets.

        Args:
            force: If True, re-download even if files exist
            extract: If True, automatically extract archives

        Returns:
            Dict mapping dataset names to success status
        """
        results = {}

        logger.info("=" * 70)
        logger.info("DOWNLOADING ALL RESEARCH DATASETS")
        logger.info("=" * 70)

        total_size_mb = sum(config["size_mb"] for config in RESEARCH_DATASETS.values())
        logger.warning(f"⚠️  Total estimated size: ~{total_size_mb}MB")

        for dataset_name in RESEARCH_DATASETS.keys():
            logger.info(f"\n[{dataset_name}]")
            results[dataset_name] = self.download_dataset(
                dataset_name, force=force, extract=extract
            )

        # Summary
        logger.info("\n" + "=" * 70)
        successful = sum(1 for v in results.values() if v)
        logger.info(f"Download Summary: {successful}/{len(results)} successful")

        for dataset_name, success in results.items():
            status = "✓" if success else "✗"
            logger.info(f"  {status} {dataset_name}")

        return results

    def _extract_dataset(self, archive_path: Path, format_type: str) -> bool:
        """
        Extract dataset archive.

        Args:
            archive_path: Path to archive file
            format_type: Archive format (zip, tar.gz, etc.)

        Returns:
            True if successful
        """
        try:
            extract_dir = archive_path.parent / archive_path.stem
            extract_dir.mkdir(parents=True, exist_ok=True)

            if format_type == "zip":
                import zipfile

                with zipfile.ZipFile(archive_path, "r") as zf:
                    zf.extractall(extract_dir)
            elif format_type == "tar.gz":
                import tarfile

                with tarfile.open(archive_path, "r:gz") as tf:
                    tf.extractall(extract_dir)
            else:
                logger.warning(f"Unsupported archive format: {format_type}")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to extract {archive_path}: {e}")
            return False

    def list_datasets(self) -> None:
        """Print list of available research datasets."""
        logger.info("=" * 70)
        logger.info("AVAILABLE RESEARCH DATASETS")
        logger.info("=" * 70)

        for dataset_id, config in RESEARCH_DATASETS.items():
            logger.info(f"\n{dataset_id.upper()}")
            logger.info(f"  Name: {config['name']}")
            logger.info(f"  Description: {config['description']}")
            logger.info(f"  Size: ~{config['size_mb']}MB")
            logger.info(f"  Tags: {', '.join(config['tags'])}")
            if config.get("notes"):
                logger.info(f"  Notes: {config['notes']}")

    def verify_downloads(self, dataset_name: Optional[str] = None) -> Dict[str, bool]:
        """
        Verify downloaded datasets.

        Args:
            dataset_name: Specific dataset to verify, or None for all

        Returns:
            Dict mapping dataset names to verification status
        """
        datasets_to_check = [dataset_name] if dataset_name else RESEARCH_DATASETS.keys()

        results = {}
        for ds_name in datasets_to_check:
            dataset_dir = self.output_dir / ds_name
            results[ds_name] = dataset_dir.exists()

            status = "✓" if results[ds_name] else "✗"
            logger.info(f"{status} {ds_name}: {dataset_dir}")

        return results


# ============================================================================
# CLI INTERFACE
# ============================================================================


def main():
    """Command-line interface for research dataset downloader."""

    parser = argparse.ArgumentParser(
        description="Download research datasets for Kanji recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download specific dataset
  python scripts/download_research_datasets.py --dataset megahan97k
  
  # Download all research datasets
  python scripts/download_research_datasets.py --all
  
  # Force re-download
  python scripts/download_research_datasets.py --dataset dkds --force
  
  # List available datasets
  python scripts/download_research_datasets.py --list
  
  # Verify downloaded datasets
  python scripts/download_research_datasets.py --verify
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Download specific dataset (see --list for options)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all research datasets",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded datasets",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Don't extract archives after download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/research_datasets"),
        help="Output directory for downloads (default: data/research_datasets)",
    )

    args = parser.parse_args()

    downloader = ResearchDatasetDownloader(output_dir=args.output_dir)

    # Handle --list
    if args.list:
        downloader.list_datasets()
        return 0

    # Handle --verify
    if args.verify:
        downloader.verify_downloads()
        return 0

    # Handle --all
    if args.all:
        results = downloader.download_all(
            force=args.force,
            extract=not args.no_extract,
        )
        return 0 if all(results.values()) else 1

    # Handle --dataset
    if args.dataset:
        success = downloader.download_dataset(
            args.dataset,
            force=args.force,
            extract=not args.no_extract,
        )
        return 0 if success else 1

    # No action specified
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
