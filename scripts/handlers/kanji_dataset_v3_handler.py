"""
Handler for Kanji Dataset v3 - expanded kanji character recognition dataset.

Kanji Dataset v3 characteristics:
- Version 3 with expanded character coverage
- Enhanced preprocessing and normalization
- Improved metadata structure
- Multiple script variations and writing styles
- Comprehensive character classification

Metadata: character code, Unicode mapping, script variant, quality score
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


class KanjiDatasetV3Handler(DatasetFormatHandler):
    """Handler for Kanji Dataset v3 character recognition dataset."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize handler."""
        super().__init__(config)

    def download(self, output_dir: Path, force: bool = False) -> Path:
        """
        Download Kanji Dataset v3.

        Args:
            output_dir: Directory to store dataset
            force: Force re-download

        Returns:
            Path to dataset directory
        """
        logger.info("Kanji Dataset v3 download handled by download_research_datasets.py")
        logger.info("Dataset is downloaded from Hugging Face: Ayphoss/kanji-dataset-v3")
        dataset_path = output_dir / "kanji_dataset_v3"
        dataset_path.mkdir(parents=True, exist_ok=True)
        return dataset_path

    def extract(self, archive_path: Path, output_dir: Path) -> Path:
        """
        Extract Kanji Dataset v3 archive.

        Args:
            archive_path: Path to archive
            output_dir: Extract to this directory

        Returns:
            Path to extracted dataset
        """
        import tarfile
        import zipfile

        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = output_dir / "kanji_dataset_v3"

        try:
            logger.info(f"Extracting {archive_path} to {dataset_dir}")

            if str(archive_path).endswith(".tar.gz") or str(archive_path).endswith(".tar"):
                with tarfile.open(archive_path, "r:*") as tf:
                    # Safely extract with path filtering
                    for member in tf.getmembers():
                        if member.name.startswith("/") or ".." in member.name:
                            logger.warning(f"Skipping unsafe path: {member.name}")
                            continue
                        tf.extract(member, dataset_dir)
            else:
                with zipfile.ZipFile(archive_path, "r") as zf:
                    # Safely extract with path filtering
                    for member in zf.namelist():
                        if member.startswith("/") or ".." in member:
                            logger.warning(f"Skipping unsafe path: {member}")
                            continue
                        zf.extract(member, dataset_dir)

            logger.info("✓ Extracted Kanji Dataset v3")
            return dataset_dir
        except Exception as e:
            logger.error(f"Failed to extract Kanji Dataset v3: {e}")
            return None

    def parse_record(self, record: Any) -> Optional[Dict[str, Any]]:
        """
        Parse a single record.

        Args:
            record: Dict with image, label, character_code, script_variant

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
                            "character_code": record.get("character_code"),
                            "unicode": record.get("unicode"),
                            "script_variant": record.get("script_variant", "standard"),
                            "quality_score": record.get("quality_score", 1.0),
                            "character_type": "kanji",
                            "version": 3,
                        },
                    }
            return None
        except Exception as e:
            logger.warning(f"Failed to parse record: {e}")
            return None

    def iter_records(self, dataset_path: Path) -> List[Any]:
        """
        Iterate over all records in Kanji Dataset v3.

        Args:
            dataset_path: Path to dataset directory

        Yields:
            Records with kanji character image and metadata
        """
        records = []

        # Try to load from HuggingFace datasets cache first
        try:
            # Check for HF cache structure (Ayphoss___kanji-dataset-v3/default/0.0.0/.../)
            cache_dirs = list(dataset_path.glob("*/default/0.0.0/*/"))
            if cache_dirs:
                cache_path = cache_dirs[0]
                arrow_files = list(cache_path.glob("*.arrow"))

                if arrow_files:
                    try:
                        from datasets import load_dataset

                        # Load using datasets library
                        dataset = load_dataset(
                            "arrow",
                            data_files={"train": str(arrow_files[0])},
                            split="train",
                        )

                        for item in dataset:
                            # Handle both PIL images and numpy arrays
                            if "image" in item:
                                if hasattr(item["image"], "convert"):
                                    image = np.array(item["image"].convert("L"))
                                else:
                                    image = np.array(item["image"])

                                label = item.get("label", 0)

                                records.append(
                                    {
                                        "image": image,
                                        "label": label,
                                        "character_code": str(
                                            item.get("character_code", "unknown")
                                        ),
                                        "unicode": str(item.get("unicode", "unknown")),
                                        "script_variant": item.get("script_variant", "standard"),
                                        "quality_score": float(item.get("quality_score", 1.0)),
                                    }
                                )

                        if records:
                            logger.info(f"Loaded {len(records)} records from Arrow format")
                            return records
                    except Exception as e:
                        logger.warning(f"Failed to load Arrow with datasets lib: {e}")

                        # Fallback: use pyarrow directly
                        try:
                            import pyarrow as pa

                            table = pa.ipc.open_file(str(arrow_files[0])).read_all()

                            for i in range(len(table)):
                                row = {col: table[col][i].as_py() for col in table.column_names}

                                if "image" in row:
                                    image = row["image"]
                                    if isinstance(image, bytes):
                                        from PIL import Image as PILImage

                                        image = np.array(
                                            PILImage.open(__import__("io").BytesIO(image)).convert(
                                                "L"
                                            )
                                        )
                                    elif not isinstance(image, np.ndarray):
                                        image = np.array(image)

                                    label = row.get("label", 0)

                                    records.append(
                                        {
                                            "image": image,
                                            "label": label,
                                            "character_code": str(
                                                row.get("character_code", "unknown")
                                            ),
                                            "unicode": str(row.get("unicode", "unknown")),
                                            "script_variant": row.get("script_variant", "standard"),
                                            "quality_score": float(row.get("quality_score", 1.0)),
                                        }
                                    )

                            if records:
                                logger.info(
                                    f"Loaded {len(records)} records from Arrow format (pyarrow)"
                                )
                                return records
                        except Exception as e2:
                            logger.warning(f"Failed to load Arrow with pyarrow: {e2}")
        except Exception as e:
            logger.debug(f"No Arrow cache found: {e}")

        # Fallback: look for metadata file that might contain structure info
        list(dataset_path.glob("metadata.json")) + list(dataset_path.glob("*/metadata.json"))

        # Look for image files organized by character or category
        image_dirs = [d for d in dataset_path.glob("*") if d.is_dir()]

        if not image_dirs:
            # If no subdirectories, look for images directly
            image_dirs = [dataset_path]

        for img_dir in image_dirs:
            character_label = img_dir.name if img_dir != dataset_path else "unknown"

            # Find all image files
            image_files = []
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
                image_files.extend(img_dir.glob(ext))

            for idx, img_file in enumerate(sorted(image_files)):
                try:
                    image = np.array(Image.open(img_file).convert("L"))

                    # Extract metadata from filename or use defaults
                    label = idx

                    records.append(
                        {
                            "image": image,
                            "label": label,
                            "character_code": character_label,
                            "unicode": character_label,
                            "script_variant": "standard",
                            "quality_score": 1.0,
                            "source_file": img_file.name,
                        }
                    )

                except Exception as e:
                    logger.warning(f"Failed to process image {img_file}: {e}")

        return records

    def get_manifest(self, dataset_path: Path) -> Optional[DatasetManifest]:
        """
        Get dataset manifest with statistics.

        Args:
            dataset_path: Path to dataset directory

        Returns:
            DatasetManifest with metadata
        """
        try:
            records = self.iter_records(dataset_path)

            manifest = DatasetManifest(
                name="Kanji Dataset v3",
                description="Kanji character dataset version 3 with expanded coverage",
                num_samples=len(records),
                num_classes=len({r.get("label") for r in records if r}),
                tags=["kanji", "japanese", "character-recognition", "v3"],
                source_url="https://huggingface.co/datasets/Ayphoss/kanji-dataset-v3",
            )

            return manifest
        except Exception as e:
            logger.error(f"Failed to get manifest: {e}")
            return None

    def get_dataset_manifest(self) -> Optional[DatasetManifest]:
        """
        Get dataset manifest (required by abstract base class).

        Returns:
            DatasetManifest with metadata
        """
        return DatasetManifest(
            name="Kanji Dataset v3",
            description="Expanded kanji dataset version 3 from Hugging Face (Ayphoss/kanji-dataset-v3)",
            num_samples=4000,  # Approximate
            num_classes=3036,  # Standard kanji count
            tags=["kanji", "japanese", "character-recognition", "huggingface", "v3"],
            source_url="https://huggingface.co/datasets/Ayphoss/kanji-dataset-v3",
        )
