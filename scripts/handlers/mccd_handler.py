"""
Handler for MCCD (Multi-attribute Chinese Calligraphy Character) dataset.

MCCD benchmark for calligraphic character recognition:
- Character images with style, period, and calligrapher attributes
- Multiple calligraphic styles (seal, clerical, regular, cursive, etc.)
- Historical periods and artist information
- Support for style transfer and historical analysis
- Recognition across multiple script styles

Metadata: style, period, calligrapher, artist_id, script_type
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lib.dataset_handlers import DatasetFormatHandler, DatasetManifest, PreprocessingConfig
from src.lib.logging_utils import setup_logger

logger = setup_logger(__name__)


class MCCDHandler(DatasetFormatHandler):
    """Handler for MCCD multi-attribute calligraphy dataset."""

    # Common calligraphic styles
    STYLES = ["seal", "clerical", "regular", "cursive", "running", "other"]

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize handler."""
        super().__init__(config)

    def download(self, output_dir: Path, force: bool = False) -> Path:
        """
        Download MCCD dataset.

        Args:
            output_dir: Directory to store dataset
            force: Force re-download

        Returns:
            Path to dataset directory
        """
        logger.info("MCCD download handled by download_research_datasets.py")
        dataset_path = output_dir / "mccd"
        dataset_path.mkdir(parents=True, exist_ok=True)
        return dataset_path

    def extract(self, archive_path: Path, output_dir: Path) -> Path:
        """
        Extract MCCD archive.

        Args:
            archive_path: Path to zip archive
            output_dir: Extract to this directory

        Returns:
            Path to extracted dataset
        """
        import zipfile

        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = output_dir / "mccd"

        try:
            logger.info(f"Extracting {archive_path} to {dataset_dir}")
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dataset_dir)
            logger.info("✓ Extracted MCCD")
            return dataset_dir
        except Exception as e:
            logger.error(f"Failed to extract MCCD: {e}")
            return None

    def parse_record(self, record: Any) -> Optional[Dict[str, Any]]:
        """
        Parse a single record.

        Args:
            record: Dict with image, label, style, period, calligrapher

        Returns:
            Dict with image_data, label, metadata
        """
        try:
            if isinstance(record, dict):
                image = record.get("image")
                label = record.get("label")

                if image is not None and label is not None:
                    return {
                        "image_data": image,
                        "label": int(label),
                        "metadata": {
                            "style": record.get("style", "unknown"),
                            "period": record.get("period"),
                            "calligrapher": record.get("calligrapher"),
                            "artist_id": record.get("artist_id"),
                        },
                    }
            return None
        except Exception as e:
            logger.warning(f"Failed to parse record: {e}")
            return None

    def iter_records(self, dataset_path: Path) -> List[Any]:
        """
        Iterate over all records in MCCD.

        Args:
            dataset_path: Path to dataset directory

        Yields:
            Records with image and calligraphic metadata
        """
        records = []

        # Look for metadata file
        metadata_files = list(dataset_path.glob("*.json")) + list(
            dataset_path.glob("metadata/*.json")
        )

        for metadata_file in metadata_files:
            try:
                with open(metadata_file, encoding="utf-8") as f:
                    metadata = json.load(f)

                # Metadata format can vary; try common patterns
                if isinstance(metadata, dict):
                    # Try standard structure
                    samples = metadata.get("samples", []) or metadata.get("data", [])

                    for sample in samples:
                        try:
                            image_file = Path(metadata_file.parent) / sample.get("image")

                            if image_file.exists():
                                image = np.array(Image.open(image_file).convert("L"))

                                label = sample.get("character_id")
                                style = sample.get("style", "unknown").lower()
                                period = sample.get("period")
                                calligrapher = sample.get("calligrapher")
                                artist_id = sample.get("artist_id")

                                # Normalize style
                                if style not in self.STYLES:
                                    style = "other"

                                records.append(
                                    {
                                        "image": image,
                                        "label": int(label) if label is not None else 0,
                                        "style": style,
                                        "period": period,
                                        "calligrapher": calligrapher,
                                        "artist_id": artist_id,
                                        "source_file": sample.get("image"),
                                    }
                                )

                        except Exception as e:
                            logger.warning(f"Failed to process sample: {e}")

                elif isinstance(metadata, list):
                    # List format
                    for sample in metadata:
                        try:
                            image_file = Path(metadata_file.parent) / sample.get("image")

                            if image_file.exists():
                                image = np.array(Image.open(image_file).convert("L"))

                                records.append(
                                    {
                                        "image": image,
                                        "label": int(sample.get("label", 0)),
                                        "style": sample.get("style", "unknown").lower(),
                                        "period": sample.get("period"),
                                        "calligrapher": sample.get("calligrapher"),
                                        "artist_id": sample.get("artist_id"),
                                        "source_file": sample.get("image"),
                                    }
                                )

                        except Exception as e:
                            logger.warning(f"Failed to process sample: {e}")

            except Exception as e:
                logger.warning(f"Failed to load metadata {metadata_file}: {e}")

        logger.info(f"Loaded {len(records)} records from MCCD")
        return records

    def get_dataset_manifest(self) -> DatasetManifest:
        """Get dataset manifest."""
        if self.manifest is None:
            self.manifest = DatasetManifest(
                dataset_id="mccd",
                name="MCCD",
                source_url="https://github.com/PRIS-CV/MCCD",
                format_type="images_with_metadata_json",
                num_classes=3500,  # Approximate
                num_samples=50000,  # Approximate
                description="Multi-attribute Chinese calligraphy character dataset with style and period information",
                languages=["zh"],
                character_sets=["hanzi"],
                download_size_mb=300,
                extracted_size_mb=400,
                tags=["calligraphy", "style", "period", "historical", "attributes"],
                year_published=2025,
                papers=["MCCD: PRIS-CV"],
            )

        return self.manifest
