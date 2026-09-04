"""
Handler for MegaHan97K dataset - 97,455 Chinese character classes.

MegaHan97K is a large-scale benchmark supporting:
- Handwritten characters (from multiple writers)
- Historical/variants characters
- Synthetic characters
- Long-tail and rare character evaluation

Format: NumPy .npz files with structure:
  - X: Image array [N, H, W] or [N, H, W, C]
  - y: Labels [N]
  - metadata: Additional per-sample information
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lib.dataset_handlers import DatasetFormatHandler, DatasetManifest, PreprocessingConfig
from src.lib.logging_utils import setup_logger

logger = setup_logger(__name__)


class MegaHan97KHandler(DatasetFormatHandler):
    """Handler for MegaHan97K dataset."""

    def __init__(self, config: Optional[PreprocessingConfig] = None, subset: str = "all"):
        """
        Initialize handler.

        Args:
            config: Preprocessing configuration
            subset: Dataset subset - "top-1000", "top-3000", or "all" (default)
        """
        super().__init__(config)
        self.subset = subset
        self.subset_limits = {
            "top-1000": 1000,
            "top-3000": 3000,
            "all": 97455,
        }
        if subset not in self.subset_limits:
            logger.warning(f"Unknown subset {subset}, using 'all'")
            self.subset = "all"

    def download(self, output_dir: Path, force: bool = False) -> Path:
        """
        Download MegaHan97K dataset.

        Note: This is a placeholder. Actual download should be handled by
        scripts/download_research_datasets.py

        Args:
            output_dir: Directory to store dataset
            force: Force re-download

        Returns:
            Path to dataset directory
        """
        logger.info("MegaHan97K download handled by download_research_datasets.py")
        dataset_path = output_dir / "megahan97k"
        dataset_path.mkdir(parents=True, exist_ok=True)
        return dataset_path

    def extract(self, archive_path: Path, output_dir: Path) -> Path:
        """
        Extract MegaHan97K archive.

        Args:
            archive_path: Path to tar.gz archive
            output_dir: Extract to this directory

        Returns:
            Path to extracted dataset
        """
        import tarfile

        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = output_dir / "megahan97k"

        try:
            logger.info(f"Extracting {archive_path} to {dataset_dir}")
            with tarfile.open(archive_path, "r:gz") as tf:
                # Safely extract with path filtering
                for member in tf.getmembers():
                    if member.name.startswith("/") or ".." in member.name:
                        logger.warning(f"Skipping unsafe path: {member.name}")
                        continue
                    tf.extract(member, dataset_dir)
            logger.info("✓ Extracted MegaHan97K")
            return dataset_dir
        except Exception as e:
            logger.error(f"Failed to extract MegaHan97K: {e}")
            return None

    def parse_record(self, record: Any) -> Optional[Dict[str, Any]]:
        """
        Parse a single record (image + label).

        Args:
            record: Tuple of (image, label)

        Returns:
            Dict with image_data, label, metadata
        """
        try:
            if isinstance(record, (tuple, list)) and len(record) >= 2:
                image, label = record[0], record[1]

                return {
                    "image_data": image,
                    "label": int(label),
                    "metadata": {
                        "subset": self.subset,
                        "character_id": int(label),
                    },
                }
            return None
        except Exception as e:
            logger.warning(f"Failed to parse record: {e}")
            return None

    def iter_records(self, dataset_path: Path) -> List[Any]:
        """
        Iterate over all records in MegaHan97K.

        Args:
            dataset_path: Path to dataset directory with .npz files

        Yields:
            Tuples of (image, label)
        """
        records = []

        # Find all .npz files
        npz_files = sorted(dataset_path.glob("*.npz"))

        if not npz_files:
            logger.warning(f"No .npz files found in {dataset_path}")
            return records

        limit = self.subset_limits[self.subset]
        total = 0

        for npz_file in npz_files:
            try:
                data = np.load(npz_file)
                x = data["X"]  # Images
                y = data["y"]  # Labels

                for i in range(len(y)):
                    if total >= limit:
                        break

                    records.append((x[i], y[i]))
                    total += 1

                if total >= limit:
                    break

            except Exception as e:
                logger.warning(f"Failed to load {npz_file}: {e}")

        logger.info(f"Loaded {len(records)} records from MegaHan97K ({self.subset})")
        return records

    def get_dataset_manifest(self) -> DatasetManifest:
        """Get dataset manifest."""
        if self.manifest is None:
            num_classes = self.subset_limits[self.subset]

            self.manifest = DatasetManifest(
                dataset_id=f"megahan97k_{self.subset}",
                name=f"MegaHan97K ({self.subset})",
                source_url="https://github.com/SCUT-DLVCLab/MegaHan97K",
                format_type="numpy_npz",
                num_classes=num_classes,
                num_samples=500000,  # Approximate
                description="Large-scale benchmark with 97,455 Chinese character categories",
                languages=["zh"],
                character_sets=["hanzi"],
                download_size_mb=5000,
                extracted_size_mb=6000,
                tags=["zero-shot", "long-tail", "chinese", "large-scale"],
                year_published=2025,
                papers=["MegaHan97K: SCUT-DLVCLAB"],
            )

        return self.manifest
