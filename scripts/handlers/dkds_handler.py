"""
Handler for DKDS (Degraded Kuzushiji Documents with Seals) dataset.

DKDS benchmark for degraded historical Japanese handwriting:
- Character and seal detection
- Document binarization
- Degradation types: blur, stains, seals, contrast changes
- Kuzushiji-specific character set

Metadata preserved: degradation type, seal presence, historical period
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


class DKDSHandler(DatasetFormatHandler):
    """Handler for DKDS degraded kuzushiji dataset."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize handler."""
        super().__init__(config)
        self.degradation_types = ["blur", "stains", "seals", "contrast", "binarization", "none"]

    def download(self, output_dir: Path, force: bool = False) -> Path:
        """
        Download DKDS dataset.

        Args:
            output_dir: Directory to store dataset
            force: Force re-download

        Returns:
            Path to dataset directory
        """
        logger.info("DKDS download handled by download_research_datasets.py")
        dataset_path = output_dir / "dkds"
        dataset_path.mkdir(parents=True, exist_ok=True)
        return dataset_path

    def extract(self, archive_path: Path, output_dir: Path) -> Path:
        """
        Extract DKDS archive.

        Args:
            archive_path: Path to zip archive
            output_dir: Extract to this directory

        Returns:
            Path to extracted dataset
        """
        import zipfile

        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = output_dir / "dkds"

        try:
            logger.info(f"Extracting {archive_path} to {dataset_dir}")
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dataset_dir)
            logger.info("✓ Extracted DKDS")
            return dataset_dir
        except Exception as e:
            logger.error(f"Failed to extract DKDS: {e}")
            return None

    def parse_record(self, record: Any) -> Optional[Dict[str, Any]]:
        """
        Parse a single record (image + label + metadata).

        Args:
            record: Dict with image, label, degradation_type, seal_present

        Returns:
            Dict with image_data, label, metadata
        """
        try:
            if isinstance(record, dict):
                image = record.get("image")
                label = record.get("label")
                degradation = record.get("degradation_type", "none")
                seal = record.get("seal_present", False)

                if image is not None and label is not None:
                    return {
                        "image_data": image,
                        "label": int(label),
                        "metadata": {
                            "degradation_type": degradation,
                            "seal_present": seal,
                            "is_kuzushiji": True,
                        },
                    }
            return None
        except Exception as e:
            logger.warning(f"Failed to parse record: {e}")
            return None

    def iter_records(self, dataset_path: Path) -> List[Any]:
        """
        Iterate over all records in DKDS.

        Args:
            dataset_path: Path to dataset directory

        Yields:
            Records with image, label, degradation metadata
        """
        records = []

        # Look for image subdirectories
        image_dirs = sorted([d for d in dataset_path.glob("*") if d.is_dir()])

        for image_dir in image_dirs:
            # Get degradation type from directory name
            degradation_type = image_dir.name.lower()

            # Find all image files
            image_files = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))

            for img_file in image_files:
                try:
                    # Parse filename for label
                    # Assumed format: CHARACTER_ID_VARIANT.png
                    parts = img_file.stem.split("_")
                    if len(parts) >= 2 and parts[0].isdigit():
                        label = int(parts[0])

                        # Load image
                        image = np.array(Image.open(img_file).convert("L"))

                        # Check for seal interference
                        seal_present = "seal" in degradation_type or "seal" in img_file.name.lower()

                        records.append(
                            {
                                "image": image,
                                "label": label,
                                "degradation_type": degradation_type,
                                "seal_present": seal_present,
                                "source_file": img_file.name,
                            }
                        )

                except Exception as e:
                    logger.warning(f"Failed to load {img_file}: {e}")

        logger.info(f"Loaded {len(records)} records from DKDS")
        return records

    def get_dataset_manifest(self) -> DatasetManifest:
        """Get dataset manifest."""
        if self.manifest is None:
            self.manifest = DatasetManifest(
                dataset_id="dkds",
                name="DKDS (Degraded Kuzushiji Documents)",
                source_url="https://github.com/RuiyangJu/DKDS",
                format_type="images_directory",
                num_classes=3831,  # Kuzushiji characters
                num_samples=8000,  # Approximate
                description="Degraded historical Japanese handwriting with seal interference",
                languages=["ja"],
                character_sets=["kuzushiji", "kanji"],
                download_size_mb=500,
                extracted_size_mb=600,
                tags=["japanese", "historical", "degraded", "seals", "binarization"],
                year_published=2025,
                papers=["DKDS: RuiyangJu et al."],
            )

        return self.manifest
