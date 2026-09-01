"""
Handler for Stroke-level Handwriting Database.

Stroke-level database for online handwriting research:
- 42 writers × 1,200 characters per writer = 50,400 samples
- Pen trajectory data (coordinate sequences)
- Writing order and stroke timing information
- Support for sequence models (RNN, Transformer, attention)
- Writer variation and personalization studies

Metadata: writer_id, character_id, trajectory_file, num_strokes, writing_time
"""

import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lib.dataset_handlers import DatasetFormatHandler, DatasetManifest, PreprocessingConfig
from src.lib.logging_utils import setup_logger

logger = setup_logger(__name__)


class StrokeDatabaseHandler(DatasetFormatHandler):
    """Handler for stroke-level handwriting database."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize handler."""
        super().__init__(config)

    def download(self, output_dir: Path, force: bool = False) -> Path:
        """
        Download stroke database.

        Args:
            output_dir: Directory to store dataset
            force: Force re-download

        Returns:
            Path to dataset directory
        """
        logger.info("Stroke Database download handled by download_research_datasets.py")
        dataset_path = output_dir / "stroke_database"
        dataset_path.mkdir(parents=True, exist_ok=True)
        return dataset_path

    def extract(self, archive_path: Path, output_dir: Path) -> Path:
        """
        Extract stroke database archive.

        Args:
            archive_path: Path to zip archive
            output_dir: Extract to this directory

        Returns:
            Path to extracted dataset
        """
        import zipfile

        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = output_dir / "stroke_database"

        try:
            logger.info(f"Extracting {archive_path} to {dataset_dir}")
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dataset_dir)
            logger.info("✓ Extracted Stroke Database")
            return dataset_dir
        except Exception as e:
            logger.error(f"Failed to extract Stroke Database: {e}")
            return None

    def parse_record(self, record: Any) -> Optional[Dict[str, Any]]:
        """
        Parse a single record.

        Args:
            record: Dict with image (rasterized), label, trajectory_file, writer_id

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
                            "writer_id": record.get("writer_id"),
                            "trajectory_file": record.get("trajectory_file"),
                            "num_strokes": record.get("num_strokes"),
                            "writing_time_ms": record.get("writing_time_ms"),
                            "has_trajectory": record.get("trajectory_file") is not None,
                        },
                    }
            return None
        except Exception as e:
            logger.warning(f"Failed to parse record: {e}")
            return None

    def iter_records(self, dataset_path: Path) -> List[Any]:
        """
        Iterate over all records in stroke database.

        Args:
            dataset_path: Path to dataset directory

        Yields:
            Records with image (rasterized from trajectory) and trajectory metadata
        """
        records = []

        # Look for metadata/index file
        metadata_files = list(dataset_path.glob("*.json")) + list(dataset_path.glob("index/*.json"))

        for metadata_file in metadata_files:
            try:
                with open(metadata_file, encoding="utf-8") as f:
                    metadata = json.load(f)

                if isinstance(metadata, dict):
                    samples = metadata.get("samples", [])

                    for sample in samples:
                        try:
                            writer_id = sample.get("writer_id")
                            character_id = sample.get("character_id")
                            trajectory_file = sample.get("trajectory_file")

                            if trajectory_file:
                                trajectory_path = Path(metadata_file.parent) / trajectory_file

                                if trajectory_path.exists():
                                    # Load trajectory and rasterize to image
                                    image = self._rasterize_trajectory(trajectory_path)

                                    if image is not None:
                                        records.append(
                                            {
                                                "image": image,
                                                "label": int(character_id),
                                                "writer_id": int(writer_id) if writer_id else None,
                                                "trajectory_file": trajectory_file,
                                                "num_strokes": sample.get("num_strokes"),
                                                "writing_time_ms": sample.get("writing_time_ms"),
                                                "source_file": trajectory_file,
                                            }
                                        )

                        except Exception as e:
                            logger.warning(f"Failed to process sample: {e}")

                elif isinstance(metadata, list):
                    for sample in metadata:
                        try:
                            trajectory_file = sample.get("trajectory_file")

                            if trajectory_file:
                                trajectory_path = Path(metadata_file.parent) / trajectory_file

                                if trajectory_path.exists():
                                    image = self._rasterize_trajectory(trajectory_path)

                                    if image is not None:
                                        records.append(
                                            {
                                                "image": image,
                                                "label": int(sample.get("label", 0)),
                                                "writer_id": sample.get("writer_id"),
                                                "trajectory_file": trajectory_file,
                                                "num_strokes": sample.get("num_strokes"),
                                                "writing_time_ms": sample.get("writing_time_ms"),
                                                "source_file": trajectory_file,
                                            }
                                        )

                        except Exception as e:
                            logger.warning(f"Failed to process sample: {e}")

            except Exception as e:
                logger.warning(f"Failed to load metadata {metadata_file}: {e}")

        logger.info(f"Loaded {len(records)} records from Stroke Database")
        return records

    def _rasterize_trajectory(
        self,
        trajectory_file: Path,
        target_size: int = 64,
    ) -> Optional[np.ndarray]:
        """
        Rasterize pen trajectory to image.

        Args:
            trajectory_file: Path to trajectory file (.npy or binary)
            target_size: Output image size

        Returns:
            Grayscale image array or None if failed
        """
        try:
            import cv2

            # Try loading as .npy first
            if trajectory_file.suffix == ".npy":
                trajectory = np.load(trajectory_file)
            else:
                # Try reading as binary coordinate sequence
                with open(trajectory_file, "rb") as f:
                    # Assume format: x, y, pen_up/down, timestamp
                    # This is format-dependent and may need adjustment
                    coords = []
                    while True:
                        data = f.read(12)  # 3 floats × 4 bytes
                        if not data:
                            break
                        try:
                            x, y, pen_state = struct.unpack("fff", data)
                            coords.append([x, y])
                        except:
                            break
                    trajectory = np.array(coords)

            if trajectory.size == 0:
                return None

            # Create blank image
            image = np.zeros((target_size, target_size), dtype=np.uint8)

            # Normalize coordinates to image space
            if trajectory.shape[0] > 0:
                min_x, min_y = trajectory.min(axis=0)
                max_x, max_y = trajectory.max(axis=0)

                width = max_x - min_x + 1
                height = max_y - min_y + 1

                if width > 0 and height > 0:
                    # Scale to fit image
                    scale = min((target_size - 2) / width, (target_size - 2) / height)
                    normalized = (trajectory - [min_x, min_y]) * scale + 1

                    # Draw trajectory on image
                    for i in range(len(normalized) - 1):
                        pt1 = tuple(normalized[i].astype(int))
                        pt2 = tuple(normalized[i + 1].astype(int))
                        cv2.line(image, pt1, pt2, 255, 2)

            return image

        except Exception as e:
            logger.warning(f"Failed to rasterize trajectory {trajectory_file}: {e}")
            return None

    def get_dataset_manifest(self) -> DatasetManifest:
        """Get dataset manifest."""
        if self.manifest is None:
            self.manifest = DatasetManifest(
                dataset_id="stroke_database",
                name="Stroke-level Handwriting Database",
                source_url="https://github.com/shirooo39/stroke-dataset",
                format_type="trajectory_files_with_metadata",
                num_classes=1200,  # Characters
                num_samples=50400,  # 42 writers × 1,200 chars
                description="Stroke-level pen trajectory data from 42 writers for online handwriting research",
                languages=["ja", "zh"],
                character_sets=["kanji", "hiragana", "katakana"],
                download_size_mb=400,
                extracted_size_mb=500,
                tags=["trajectory", "online-handwriting", "stroke-level", "writer-variation"],
                year_published=2025,
                papers=["Stroke Database: Xu et al."],
            )

        return self.manifest
