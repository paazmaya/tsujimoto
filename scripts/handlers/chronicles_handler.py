"""
Handler for Chronicles-OCR dataset - cross-temporal historical character recognition.

Chronicles-OCR benchmark for evaluating vision-language models on:
- Seven Chinese historical scripts (tortoise shell → calligraphy)
- 2,800 strictly balanced images
- Cross-period character spotting
- Fine-grained historical recognition
- Character evolution tracking

Metadata: historical script type, period, topological features
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lib.dataset_handlers import DatasetFormatHandler, DatasetManifest, PreprocessingConfig
from src.lib.logging_utils import setup_logger

logger = setup_logger(__name__)


class ChroniclesOCRHandler(DatasetFormatHandler):
    """Handler for Chronicles-OCR historical character dataset."""

    # Seven Chinese historical scripts
    HISTORICAL_SCRIPTS = [
        "oracle_bone",  # Tortoise shell/bone
        "bronze",  # Bronze inscriptions
        "small_seal",  # Small seal script
        "clerical",  # Clerical script
        "regular",  # Regular script (modern foundation)
        "cursive",  # Cursive script
        "calligraphy",  # Calligraphic variations
    ]

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize handler."""
        super().__init__(config)

    def download(self, output_dir: Path, force: bool = False) -> Path:
        """
        Download Chronicles-OCR dataset.

        Args:
            output_dir: Directory to store dataset
            force: Force re-download

        Returns:
            Path to dataset directory
        """
        logger.info("Chronicles-OCR download handled by download_research_datasets.py")
        dataset_path = output_dir / "chronicles_ocr"
        dataset_path.mkdir(parents=True, exist_ok=True)
        return dataset_path

    def extract(self, archive_path: Path, output_dir: Path) -> Path:
        """
        Extract Chronicles-OCR archive.

        Args:
            archive_path: Path to zip archive
            output_dir: Extract to this directory

        Returns:
            Path to extracted dataset
        """
        import zipfile

        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = output_dir / "chronicles_ocr"

        try:
            logger.info(f"Extracting {archive_path} to {dataset_dir}")
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dataset_dir)
            logger.info("✓ Extracted Chronicles-OCR")
            return dataset_dir
        except Exception as e:
            logger.error(f"Failed to extract Chronicles-OCR: {e}")
            return None

    def parse_record(self, record: Any) -> Optional[Dict[str, Any]]:
        """
        Parse a single record.

        Args:
            record: Dict with image, label, script_type, period

        Returns:
            Dict with image_data, label, metadata
        """
        try:
            if isinstance(record, dict):
                image = record.get("image")
                label = record.get("label")
                script_type = record.get("script_type", "unknown")
                period = record.get("period")

                if image is not None and label is not None:
                    return {
                        "image_data": image,
                        "label": int(label),
                        "metadata": {
                            "script_type": script_type,
                            "historical_period": period,
                            "cross_temporal": True,
                        },
                    }
            return None
        except Exception as e:
            logger.warning(f"Failed to parse record: {e}")
            return None

    def iter_records(self, dataset_path: Path) -> List[Any]:
        """
        Iterate over all records in Chronicles-OCR.

        Args:
            dataset_path: Path to dataset directory

        Yields:
            Records with image, label, historical metadata
        """
        records = []

        # Iterate through script type directories
        for script_dir in dataset_path.glob("*/"):
            if not script_dir.is_dir():
                continue

            script_name = script_dir.name.lower()

            # Determine historical period from script type
            period = self._get_period_for_script(script_name)

            # Find image files
            image_files = sorted(script_dir.glob("*.png")) + sorted(script_dir.glob("*.jpg"))

            for img_file in image_files:
                try:
                    # Parse filename for label
                    # Assumed format: CHARACTER_ID.png
                    stem = img_file.stem
                    if stem.isdigit():
                        label = int(stem)
                    elif "_" in stem:
                        # Try extracting first numeric part
                        parts = stem.split("_")
                        if parts[0].isdigit():
                            label = int(parts[0])
                        else:
                            continue
                    else:
                        continue

                    # Load image
                    image = np.array(Image.open(img_file).convert("L"))

                    records.append(
                        {
                            "image": image,
                            "label": label,
                            "script_type": script_name,
                            "period": period,
                            "source_file": img_file.name,
                        }
                    )

                except Exception as e:
                    logger.warning(f"Failed to load {img_file}: {e}")

        logger.info(f"Loaded {len(records)} records from Chronicles-OCR")
        return records

    def _get_period_for_script(self, script_name: str) -> str:
        """Map script name to historical period."""
        mappings = {
            "oracle": "ancient",
            "bronze": "ancient",
            "seal": "classical",
            "small": "classical",
            "clerical": "middle",
            "regular": "modern",
            "cursive": "modern",
            "running": "modern",
            "calligraphy": "modern",
        }

        for key, period in mappings.items():
            if key in script_name.lower():
                return period

        return "unknown"

    def get_dataset_manifest(self) -> DatasetManifest:
        """Get dataset manifest."""
        if self.manifest is None:
            self.manifest = DatasetManifest(
                dataset_id="chronicles_ocr",
                name="Chronicles-OCR",
                source_url="https://github.com/VT-NLP/Chronicles",
                format_type="images_directory",
                num_classes=2400,  # Approximate unique characters
                num_samples=2800,
                description="Cross-temporal perception benchmark for historical Chinese character evolution",
                languages=["zh"],
                character_sets=["hanzi"],
                download_size_mb=200,
                extracted_size_mb=250,
                tags=["historical", "cross-temporal", "evolution", "benchmark"],
                year_published=2026,
                papers=["Chronicles-OCR: VT-NLP"],
            )

        return self.manifest
