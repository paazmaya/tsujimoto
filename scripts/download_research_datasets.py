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
        "url": None,  # Manual download required - no GitHub releases provided
        "size_mb": 5000,
        "description": "Large-scale Chinese character dataset with 97,455 classes",
        "format": "tar.gz or zip",
        "checksum": None,
        "tags": ["zero-shot", "chinese", "large-scale"],
        "manual_download": True,
        "application_url": "http://121.41.49.212:9000/",
        "sources": {
            "general_ccr": "Baiduyun: k4ch / OneDrive",
            "zero_shot_ccr": "Baiduyun: bxde / OneDrive",
        },
        "notes": (
            "⚠️  MANUAL DOWNLOAD REQUIRED:\n"
            "  1. Apply at: http://121.41.49.212:9000/\n"
            "  2. Download from Baiduyun (k4ch) or OneDrive after approval\n"
            "  3. Extract with password provided in approval email\n"
            "  4. Place extracted data in: data/research_datasets/megahan97k/\n"
            "  See: https://github.com/SCUT-DLVCLab/MegaHan97K#download"
        ),
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
        "url": "huggingface://VirtualLUO/Chronicles-OCR",
        "size_mb": 200,
        "description": "Cross-temporal Chinese character recognition benchmark (historical evolution)",
        "format": "huggingface",
        "checksum": None,
        "tags": ["historical", "cross-temporal", "evolution", "benchmark"],
        "notes": "Tests visual perception across 7 historical Chinese scripts. Downloaded from Hugging Face.",
        "hf_repo": "VirtualLUO/Chronicles-OCR",
    },
    "jawildtext": {
        "name": "JaWildText",
        "url": "huggingface://llm-jp/jawildtext",
        "size_mb": 800,
        "description": "Japanese scene text understanding benchmark (in-the-wild)",
        "format": "huggingface",
        "checksum": None,
        "tags": ["japanese", "scene-text", "vqa", "in-the-wild"],
        "notes": "Real-world Japanese text from images with diverse layouts. Downloaded from Hugging Face.",
        "hf_repo": "llm-jp/jawildtext",
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
    "kanji_full": {
        "name": "Kanji Full",
        "url": "huggingface://epts/kanji-full",
        "size_mb": 150,
        "description": "Comprehensive kanji character dataset with full coverage",
        "format": "huggingface",
        "checksum": None,
        "tags": ["kanji", "japanese", "character-recognition"],
        "notes": "Full kanji character dataset from EPTS. Downloaded from Hugging Face.",
        "hf_repo": "epts/kanji-full",
    },
    "kanji_dataset_v3": {
        "name": "Kanji Dataset v3",
        "url": "huggingface://Ayphoss/kanji-dataset-v3",
        "size_mb": 200,
        "description": "Kanji character dataset version 3 with expanded coverage",
        "format": "huggingface",
        "checksum": None,
        "tags": ["kanji", "japanese", "character-recognition", "v3"],
        "notes": "Version 3 of kanji dataset from Ayphoss with improved coverage. Downloaded from Hugging Face.",
        "hf_repo": "Ayphoss/kanji-dataset-v3",
    },
    "kanji": {
        "name": "Kanji",
        "url": "huggingface://jmonas/kanji",
        "size_mb": 100,
        "description": "Kanji character dataset with diverse writing styles",
        "format": "huggingface",
        "checksum": None,
        "tags": ["kanji", "japanese", "character-recognition", "styles"],
        "notes": "Kanji dataset from jmonas with diverse writing styles. Downloaded from Hugging Face.",
        "hf_repo": "jmonas/kanji",
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

        # Handle Hugging Face datasets
        if config.get("format") == "huggingface":
            return self._download_huggingface_dataset(dataset_name, config, force)

        # Handle manual downloads
        if config.get("manual_download"):
            logger.warning("⚠️  MANUAL DOWNLOAD REQUIRED")
            logger.info(f"Application URL: {config.get('application_url')}")
            if config.get("sources"):
                logger.info("Download sources (after approval):")
                for source_name, source_url in config["sources"].items():
                    logger.info(f"  - {source_name}: {source_url}")
            if config.get("notes"):
                logger.warning(config["notes"])

            # Check if data already exists locally
            dataset_dir = self.output_dir / dataset_name
            if dataset_dir.exists() and list(dataset_dir.glob("*")):
                logger.info(f"✓ Dataset found at {dataset_dir}")
                return True
            else:
                logger.error(f"Dataset not found at {dataset_dir}")
                logger.info(f"Please download manually and extract to: {dataset_dir}")
                return False

        if config["url"] is None:
            logger.error("No download URL provided for this dataset")
            return False

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

    def _download_huggingface_dataset(
        self,
        dataset_name: str,
        config: Dict,
        force: bool = False,
    ) -> bool:
        """
        Download dataset from Hugging Face.

        Args:
            dataset_name: Dataset identifier
            config: Dataset configuration
            force: If True, re-download even if exists

        Returns:
            True if successful, False otherwise
        """
        try:
            from datasets import load_dataset

            dataset_dir = self.output_dir / dataset_name

            # Check if already downloaded
            if dataset_dir.exists() and list(dataset_dir.glob("*")) and not force:
                logger.info(f"✓ Dataset already exists at {dataset_dir}")
                return True

            dataset_dir.mkdir(parents=True, exist_ok=True)

            hf_repo = config.get("hf_repo")
            if not hf_repo:
                logger.error("No Hugging Face repository specified in config")
                return False

            logger.info(f"Loading dataset from Hugging Face: {hf_repo}")

            # Load and cache dataset
            load_dataset(hf_repo, cache_dir=str(dataset_dir), trust_remote_code=True)

            # Save dataset info
            info_file = dataset_dir / "dataset_info.json"
            import json

            with open(info_file, "w") as f:
                json.dump(
                    {
                        "name": config["name"],
                        "description": config["description"],
                        "source": hf_repo,
                        "tags": config.get("tags", []),
                    },
                    f,
                    indent=2,
                )

            logger.info("✓ Successfully loaded dataset from Hugging Face")
            logger.info(f"✓ Cached at: {dataset_dir}")

            return True

        except ImportError:
            logger.error("The 'datasets' library is not installed")
            logger.info("Install it with: pip install datasets")
            return False
        except Exception as e:
            logger.error(f"Failed to download Hugging Face dataset: {e}")
            return False

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
                    # Safely extract with path filtering
                    for member in zf.namelist():
                        if member.startswith("/") or ".." in member:
                            logger.warning(f"Skipping potentially unsafe path: {member}")
                            continue
                        zf.extract(member, extract_dir)
            elif format_type == "tar.gz":
                import tarfile

                with tarfile.open(archive_path, "r:gz") as tf:
                    # Safely extract with path filtering
                    for member in tf.getmembers():
                        if member.name.startswith("/") or ".." in member.name:
                            logger.warning(f"Skipping potentially unsafe path: {member.name}")
                            continue
                        tf.extract(member, extract_dir)
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
            if config.get("manual_download"):
                logger.warning("  [MANUAL DOWNLOAD REQUIRED]")
            logger.info(f"  Name: {config['name']}")
            logger.info(f"  Description: {config['description']}")
            logger.info(f"  Size: ~{config['size_mb']}MB")
            logger.info(f"  Tags: {', '.join(config['tags'])}")
            if config.get("application_url"):
                logger.info(f"  Apply: {config['application_url']}")
            if config.get("sources"):
                logger.info("  Sources:")
                for source_name, source_url in config["sources"].items():
                    logger.info(f"    - {source_name}: {source_url}")
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

    def setup_local_dataset(self, dataset_name: str, local_path: Path) -> bool:
        """
        Set up a locally-available dataset (for manual downloads).

        Supports:
        - Direct dataset directories
        - Compressed archives (tar.gz, zip)

        Args:
            dataset_name: Dataset identifier
            local_path: Path to local file or directory

        Returns:
            True if successful
        """
        if dataset_name not in RESEARCH_DATASETS:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False

        RESEARCH_DATASETS[dataset_name]
        local_path = Path(local_path)

        if not local_path.exists():
            logger.error(f"Local path does not exist: {local_path}")
            return False

        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # If local path is a directory, assume it's already extracted
        if local_path.is_dir():
            logger.info(f"Using directory: {local_path}")
            # Copy contents if needed, or just verify structure
            logger.info(f"Dataset directory: {dataset_dir}")
            return True

        # If it's a file, extract it
        if local_path.is_file():
            # Determine format
            if local_path.suffix == ".gz" or str(local_path).endswith(".tar.gz"):
                file_format = "tar.gz"
            elif local_path.suffix == ".zip":
                file_format = "zip"
            else:
                logger.error(f"Unsupported file format: {local_path.suffix}")
                return False

            logger.info(f"Extracting {local_path} to {dataset_dir}")
            return self._extract_dataset(local_path, file_format)

        return False


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

  # Set up dataset from local file (for manually downloaded data like MegaHan97K)
  python scripts/download_research_datasets.py --dataset megahan97k --setup-local /path/to/megahan97k.tar.gz
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
        default=Path("dataset/research_datasets"),
        help="Output directory for downloads (default: dataset/research_datasets)",
    )
    parser.add_argument(
        "--setup-local",
        type=str,
        metavar="LOCAL_PATH",
        help="Set up dataset from local file/directory (use with --dataset)",
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

    # Handle --setup-local
    if args.setup_local:
        if not args.dataset:
            logger.error("--setup-local requires --dataset")
            return 1
        success = downloader.setup_local_dataset(args.dataset, args.setup_local)
        return 0 if success else 1

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
