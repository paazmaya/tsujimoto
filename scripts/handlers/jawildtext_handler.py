"""
Handler for JaWildText dataset - Japanese scene text understanding.

JaWildText benchmark for in-the-wild Japanese text:
- 3,241 instances from 2,961 images captured in Japan
- 3,643 unique character types (~1.12M annotated characters)
- Mixed scripts, vertical writing, diverse layouts
- Dense VQA, receipt key extraction, handwriting OCR
- Realistic capture conditions (angle, distance, lighting)

Metadata: character position, bounding box, layout context, script type
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


class JaWildTextHandler(DatasetFormatHandler):
    """Handler for JaWildText Japanese scene text dataset."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize handler."""
        super().__init__(config)

    def download(self, output_dir: Path, force: bool = False) -> Path:
        """
        Download JaWildText dataset.

        Args:
            output_dir: Directory to store dataset
            force: Force re-download

        Returns:
            Path to dataset directory
        """
        logger.info("JaWildText download handled by download_research_datasets.py")
        logger.info("Dataset is downloaded from Hugging Face: llm-jp/jawildtext")
        dataset_path = output_dir / "jawildtext"
        dataset_path.mkdir(parents=True, exist_ok=True)
        return dataset_path

    def extract(self, archive_path: Path, output_dir: Path) -> Path:
        """
        Extract JaWildText archive.

        Args:
            archive_path: Path to zip archive
            output_dir: Extract to this directory

        Returns:
            Path to extracted dataset
        """
        import zipfile

        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = output_dir / "jawildtext"

        try:
            logger.info(f"Extracting {archive_path} to {dataset_dir}")
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dataset_dir)
            logger.info("✓ Extracted JaWildText")
            return dataset_dir
        except Exception as e:
            logger.error(f"Failed to extract JaWildText: {e}")
            return None

    def parse_record(self, record: Any) -> Optional[Dict[str, Any]]:
        """
        Parse a single record.

        Args:
            record: Dict with image, label, bbox, script_type, layout_info

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
                            "bbox": record.get("bbox"),
                            "script_type": record.get("script_type", "mixed"),
                            "is_vertical": record.get("is_vertical", False),
                            "captured_in_japan": True,
                        },
                    }
            return None
        except Exception as e:
            logger.warning(f"Failed to parse record: {e}")
            return None

    def iter_records(self, dataset_path: Path) -> List[Any]:
        """
        Iterate over all records in JaWildText.

        Args:
            dataset_path: Path to dataset directory

        Yields:
            Records with image and scene text metadata
        """
        records = []

        # Look for annotations file (JSON)
        annotation_files = list(dataset_path.glob("*.json")) + list(
            dataset_path.glob("annotations/*.json")
        )

        for anno_file in annotation_files:
            try:
                with open(anno_file, encoding="utf-8") as f:
                    annotations = json.load(f)

                # Annotations typically in COCO format or custom format
                # Try to extract images and character-level annotations

                if isinstance(annotations, dict):
                    # COCO-style format
                    images_info = annotations.get("images", [])
                    annotations_list = annotations.get("annotations", [])

                    for anno in annotations_list:
                        try:
                            image_id = anno.get("image_id")
                            category_id = anno.get("category_id", 0)

                            # Find corresponding image
                            image_info = next(
                                (img for img in images_info if img.get("id") == image_id), None
                            )

                            if image_info:
                                image_file = Path(anno_file.parent) / image_info.get("file_name")

                                if image_file.exists():
                                    image = np.array(Image.open(image_file).convert("L"))
                                    bbox = anno.get("bbox")

                                    # Detect layout characteristics
                                    is_vertical = self._detect_vertical_text(bbox)

                                    records.append(
                                        {
                                            "image": image,
                                            "label": int(category_id),
                                            "bbox": bbox,
                                            "script_type": "japanese",
                                            "is_vertical": is_vertical,
                                            "source_file": image_info.get("file_name"),
                                        }
                                    )

                        except Exception as e:
                            logger.warning(f"Failed to process annotation: {e}")

                elif isinstance(annotations, list):
                    # List-style format
                    for anno in annotations:
                        try:
                            image_file = Path(anno_file.parent) / anno.get("image")
                            if image_file.exists():
                                image = np.array(Image.open(image_file).convert("L"))
                                label = anno.get("label", 0)
                                bbox = anno.get("bbox")
                                is_vertical = self._detect_vertical_text(bbox)

                                records.append(
                                    {
                                        "image": image,
                                        "label": int(label),
                                        "bbox": bbox,
                                        "script_type": "japanese",
                                        "is_vertical": is_vertical,
                                        "source_file": anno.get("image"),
                                    }
                                )

                        except Exception as e:
                            logger.warning(f"Failed to process annotation: {e}")

            except Exception as e:
                logger.warning(f"Failed to load annotations {anno_file}: {e}")

        logger.info(f"Loaded {len(records)} records from JaWildText")
        return records

    def _detect_vertical_text(self, bbox: Optional[List]) -> bool:
        """Detect if text is vertical based on bounding box."""
        if not bbox or len(bbox) < 4:
            return False

        x, y, w, h = bbox[:4]
        # Taller than wide indicates vertical orientation
        return h > w

    def get_dataset_manifest(self) -> DatasetManifest:
        """Get dataset manifest."""
        if self.manifest is None:
            self.manifest = DatasetManifest(
                dataset_id="jawildtext",
                name="JaWildText",
                source_url="https://huggingface.co/datasets/llm-jp/jawildtext",
                format_type="images_with_annotations_json",
                num_classes=3643,
                num_samples=1120000,
                description="Japanese scene text understanding benchmark with dense VQA and receipt extraction",
                languages=["ja"],
                character_sets=["hiragana", "katakana", "kanji", "ascii"],
                download_size_mb=800,
                extracted_size_mb=1000,
                tags=[
                    "japanese",
                    "scene-text",
                    "in-the-wild",
                    "vqa",
                    "mixed-scripts",
                    "vertical-text",
                ],
                year_published=2026,
                papers=["JaWildText: Maeda & Okazaki"],
            )

        return self.manifest
