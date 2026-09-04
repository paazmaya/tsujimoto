"""
Handler for Kanji dataset - character recognition with diverse writing styles.

Kanji dataset characteristics:
- Diverse kanji character variations
- Multiple writing styles and stroke patterns
- Character-level annotations and metadata
- Suitable for robust character recognition
- Handwriting and printed variations

Metadata: character code, Unicode mapping, writing style, stroke complexity
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


class KanjiHandler(DatasetFormatHandler):
    """Handler for Kanji character recognition dataset."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize handler."""
        super().__init__(config)

    def download(self, output_dir: Path, force: bool = False) -> Path:
        """
        Download Kanji dataset.

        Args:
            output_dir: Directory to store dataset
            force: Force re-download

        Returns:
            Path to dataset directory
        """
        logger.info("Kanji download handled by download_research_datasets.py")
        logger.info("Dataset is downloaded from Hugging Face: jmonas/kanji")
        dataset_path = output_dir / "kanji"
        dataset_path.mkdir(parents=True, exist_ok=True)
        return dataset_path

    def extract(self, archive_path: Path, output_dir: Path) -> Path:
        """
        Extract Kanji dataset archive.

        Args:
            archive_path: Path to archive
            output_dir: Extract to this directory

        Returns:
            Path to extracted dataset
        """
        import tarfile
        import zipfile

        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = output_dir / "kanji"

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

            logger.info("✓ Extracted Kanji")
            return dataset_dir
        except Exception as e:
            logger.error(f"Failed to extract Kanji: {e}")
            return None

    def parse_record(self, record: Any) -> Optional[Dict[str, Any]]:
        """
        Parse a single record.

        Args:
            record: Dict with image, label, character_code, writing_style

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
                            "writing_style": record.get("writing_style", "mixed"),
                            "stroke_complexity": record.get("stroke_complexity", "unknown"),
                            "character_type": "kanji",
                        },
                    }
            return None
        except Exception as e:
            logger.warning(f"Failed to parse record: {e}")
            return None

    def iter_records(self, dataset_path: Path) -> List[Any]:
        """
        Iterate over all records in Kanji dataset.

        Args:
            dataset_path: Path to dataset directory

        Yields:
            Records with kanji character image and metadata
        """
        records = []

        # Try to load from HuggingFace datasets cache first
        try:
            # Check for HF cache structure (jmonas___kanji/default/0.0.0/.../)
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

                                # Get label from text field (kanji character -> Unicode value)
                                if "text" in item and item["text"]:
                                    label = ord(
                                        item["text"][0]
                                    )  # Convert first character to Unicode
                                else:
                                    label = item.get("label", 0)

                                records.append(
                                    {
                                        "image": image,
                                        "label": label,
                                        "character_code": str(
                                            item.get("character_code", item.get("text", "unknown"))
                                        ),
                                        "unicode": str(label),
                                        "writing_style": item.get("writing_style", "standard"),
                                        "stroke_complexity": item.get(
                                            "stroke_complexity", "unknown"
                                        ),
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

                            for batch in [table.to_batches()]:
                                for i in range(len(batch)):
                                    row = {col: batch[col][i].as_py() for col in batch.column_names}

                                    if "image" in row:
                                        image = row["image"]
                                        if isinstance(image, bytes):
                                            from PIL import Image as PILImage

                                            image = np.array(
                                                PILImage.open(
                                                    __import__("io").BytesIO(image)
                                                ).convert("L")
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
                                                "writing_style": row.get(
                                                    "writing_style", "standard"
                                                ),
                                                "stroke_complexity": row.get(
                                                    "stroke_complexity", "unknown"
                                                ),
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

        # Fallback: look for image files organized by character or category
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

                    # Detect writing style from filename patterns
                    writing_style = "mixed"
                    if "handwriting" in img_file.name.lower():
                        writing_style = "handwriting"
                    elif "printed" in img_file.name.lower():
                        writing_style = "printed"
                    elif "cursive" in img_file.name.lower():
                        writing_style = "cursive"

                    records.append(
                        {
                            "image": image,
                            "label": label,
                            "character_code": character_label,
                            "unicode": character_label,
                            "writing_style": writing_style,
                            "stroke_complexity": "unknown",
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
                name="Kanji",
                description="Kanji character dataset with diverse writing styles",
                num_samples=len(records),
                num_classes=len({r.get("label") for r in records if r}),
                tags=["kanji", "japanese", "character-recognition", "styles"],
                source_url="https://huggingface.co/datasets/jmonas/kanji",
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
            name="Kanji",
            description="Kanji character dataset with diverse writing styles from Hugging Face (jmonas/kanji)",
            num_samples=2500,  # Approximate
            num_classes=3036,  # Standard kanji count
            tags=["kanji", "japanese", "character-recognition", "huggingface", "writing-styles"],
            source_url="https://huggingface.co/datasets/jmonas/kanji",
        )
