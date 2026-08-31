"""
Dataset metadata generation utilities.

Consolidates chunk metadata generation and root-level metadata creation.
Eliminates duplication from generate_chunk_metadata.py and create_dataset_metadata.py.

Example:
    >>> from src.lib.metadata_generator import ChunkMetadataGenerator, RootMetadataGenerator
    >>> chunk_gen = ChunkMetadataGenerator()
    >>> chunk_gen.generate_for_dataset('dataset/etl9g')
    >>> root_gen = RootMetadataGenerator()
    >>> root_gen.create_root_metadata('dataset')
"""

import json
from pathlib import Path
from typing import Dict, Optional

from src.lib.logging_utils import setup_logger

logger = setup_logger(__name__)


class ChunkMetadataGenerator:
    """Generate chunk_info.json metadata for datasets."""

    DEFAULT_DATASETS = ["etl1", "etl6", "etl7", "etl8g", "etl9g", "combined_all_etl"]
    CHUNK_FILENAME_PATTERN = "{dataset_name}_chunk_*.npz"

    def generate_for_dataset(self, dataset_dir: Path, force: bool = False) -> bool:
        """
        Generate chunk_info.json for a specific dataset.

        Discovers all chunk files with pattern {dataset_name}_chunk_XX.npz
        and creates metadata file.

        Args:
            dataset_dir: Path to dataset directory
            force: If True, overwrite existing chunk_info.json

        Returns:
            True if successful, False otherwise
        """
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            logger.warning(f"⏭️  Dataset directory not found: {dataset_path}")
            return False

        chunk_info_path = dataset_path / "chunk_info.json"

        # Skip if already exists (unless forced)
        if chunk_info_path.exists() and not force:
            logger.info(f"✅ {dataset_path.name}: chunk_info.json already exists")
            return True

        # Get dataset name from directory
        dataset_name = dataset_path.name

        # Find chunk files
        chunk_files = sorted(
            dataset_path.glob(self.CHUNK_FILENAME_PATTERN.format(dataset_name=dataset_name))
        )

        if not chunk_files:
            logger.debug(f"⏭️  No chunk files found in {dataset_path}")
            return False

        # Create metadata
        num_chunks = len(chunk_files)
        chunk_info = {
            "dataset_name": dataset_name,
            "num_chunks": num_chunks,
            "chunk_files": [f.name for f in chunk_files],
        }

        # Write metadata
        with open(chunk_info_path, "w", encoding="utf-8") as f:
            json.dump(chunk_info, f, indent=2)

        logger.info(f"✅ Generated: {dataset_path.name}/chunk_info.json ({num_chunks} chunks)")
        return True

    def generate_all(self, data_dir: str = "dataset", force: bool = False) -> Dict[str, bool]:
        """
        Generate chunk metadata for all standard datasets.

        Args:
            data_dir: Root dataset directory
            force: If True, overwrite existing chunk_info.json files

        Returns:
            Dict mapping dataset name to generation success status
        """
        data_path = Path(data_dir)

        if not data_path.exists():
            logger.error(f"❌ Dataset directory not found: {data_path}")
            return {}

        results = {}
        generated_count = 0
        existing_count = 0

        for dataset_name in self.DEFAULT_DATASETS:
            dataset_dir = data_path / dataset_name

            if not dataset_dir.exists():
                logger.debug(f"⏭️  Skipping {dataset_name} (directory not found)")
                results[dataset_name] = False
                continue

            success = self.generate_for_dataset(dataset_dir, force=force)
            results[dataset_name] = success

            if success:
                chunk_info_path = dataset_dir / "chunk_info.json"
                if chunk_info_path.exists():
                    generated_count += 1
                else:
                    existing_count += 1

        # Summary
        if generated_count == 0:
            if existing_count > 0:
                logger.info(
                    f"✅ All datasets already have chunk_info.json ({existing_count} dataset(s))"
                )
                logger.info("   Your datasets are ready for training!")
            else:
                logger.warning(
                    "⚠️  No chunk metadata files were generated. No datasets with chunks found."
                )
        else:
            logger.info(f"✅ Generated {generated_count} chunk metadata files")

        return results

    def validate_chunks(self, dataset_dir: Path) -> bool:
        """
        Validate that all chunk files referenced in metadata exist.

        Args:
            dataset_dir: Path to dataset directory

        Returns:
            True if all chunks exist, False otherwise
        """
        chunk_info_path = Path(dataset_dir) / "chunk_info.json"

        if not chunk_info_path.exists():
            logger.warning(f"⚠️  chunk_info.json not found: {chunk_info_path}")
            return False

        try:
            with open(chunk_info_path, encoding="utf-8") as f:
                chunk_info = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in chunk_info.json: {e}")
            return False

        # Validate all chunk files exist
        dataset_dir = Path(dataset_dir)
        missing_files = []

        for chunk_file in chunk_info.get("chunk_files", []):
            chunk_path = dataset_dir / chunk_file
            if not chunk_path.exists():
                missing_files.append(chunk_file)

        if missing_files:
            logger.error(f"❌ Missing chunk files: {missing_files}")
            return False

        logger.info(f"✅ All {len(chunk_info['chunk_files'])} chunk files verified")
        return True


class RootMetadataGenerator:
    """Generate root-level dataset metadata.json."""

    DATASET_PRIORITY = [
        "combined_all_etl",
        "etl9g",
        "etl8g",
        "etl7",
        "etl6",
        "etl1",
    ]

    def create_root_metadata(self, data_dir: str = "dataset", force: bool = False) -> bool:
        """
        Create root-level metadata.json for training scripts.

        Aggregates information from the best available dataset in priority order.

        Args:
            data_dir: Root dataset directory
            force: If True, overwrite existing root metadata.json

        Returns:
            True if successful, False otherwise
        """
        data_path = Path(data_dir)

        if not data_path.exists():
            logger.error(f"❌ Dataset directory not found: {data_path}")
            return False

        # Check for existing root metadata
        root_metadata_path = data_path / "metadata.json"
        if root_metadata_path.exists() and not force:
            logger.info("✅ Root metadata.json already exists")
            return True

        # Find best available dataset
        selected_dataset = None
        selected_path = None

        for dataset_name in self.DATASET_PRIORITY:
            dataset_dir = data_path / dataset_name
            metadata_file = dataset_dir / "metadata.json"

            if metadata_file.exists():
                selected_dataset = dataset_name
                selected_path = metadata_file
                break

        if selected_dataset is None or selected_path is None:
            logger.error("❌ No dataset metadata found. Please prepare datasets first:")
            logger.error("   uv run python scripts/prepare_dataset.py")
            return False

        # Load the selected dataset's metadata
        try:
            with open(selected_path, encoding="utf-8") as f:
                dataset_metadata = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in {selected_path}: {e}")
            return False

        # Create root metadata
        root_metadata = {
            "primary_dataset": selected_dataset,
            "num_classes": dataset_metadata.get("num_classes", 0),
            "total_samples": dataset_metadata.get("total_samples", 0),
            "target_size": dataset_metadata.get("target_size", 64),
        }

        # Save root metadata
        with open(root_metadata_path, "w", encoding="utf-8") as f:
            json.dump(root_metadata, f, indent=2)

        logger.info("✅ Created root metadata.json")
        logger.info(f"   Using dataset: {selected_dataset}")
        logger.info(f"   Classes: {root_metadata['num_classes']}")
        logger.info(f"   Samples: {root_metadata['total_samples']}")
        return True

    def get_metadata_info(self, metadata_path: Path) -> Optional[Dict]:
        """
        Load and return metadata information.

        Args:
            metadata_path: Path to metadata.json file

        Returns:
            Dict with metadata content, or None if not found
        """
        metadata_path = Path(metadata_path)

        if not metadata_path.exists():
            logger.warning(f"⚠️  Metadata file not found: {metadata_path}")
            return None

        try:
            with open(metadata_path, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in {metadata_path}: {e}")
            return None

    def verify_metadata(self, data_dir: str = "dataset") -> bool:
        """
        Verify that root metadata.json exists and is valid.

        Args:
            data_dir: Root dataset directory

        Returns:
            True if metadata is valid, False otherwise
        """
        data_path = Path(data_dir)
        metadata_path = data_path / "metadata.json"

        if not metadata_path.exists():
            logger.error(f"❌ Root metadata.json not found: {metadata_path}")
            return False

        metadata = self.get_metadata_info(metadata_path)
        if metadata is None:
            return False

        # Check required fields
        required_fields = ["primary_dataset", "num_classes", "total_samples"]
        missing_fields = [f for f in required_fields if f not in metadata]

        if missing_fields:
            logger.error(f"❌ Metadata missing required fields: {missing_fields}")
            return False

        logger.info("✅ Root metadata.json is valid")
        logger.info(f"   Primary dataset: {metadata['primary_dataset']}")
        logger.info(f"   Classes: {metadata['num_classes']}")
        logger.info(f"   Samples: {metadata['total_samples']}")
        return True


class DatasetMetadataManager:
    """High-level manager for all dataset metadata operations."""

    def __init__(self, data_dir: str = "dataset"):
        """
        Initialize metadata manager.

        Args:
            data_dir: Root dataset directory
        """
        self.data_dir = Path(data_dir)
        self.chunk_gen = ChunkMetadataGenerator()
        self.root_gen = RootMetadataGenerator()

    def initialize_all_metadata(self, force: bool = False) -> bool:
        """
        Initialize all required metadata files for dataset.

        Generates chunk metadata for all datasets and root metadata.

        Args:
            force: If True, overwrite existing metadata files

        Returns:
            True if all operations successful
        """
        logger.info("=" * 70)
        logger.info("INITIALIZING ALL DATASET METADATA")
        logger.info("=" * 70)

        # Generate chunk metadata for all datasets
        logger.info("\n📂 Generating chunk metadata...")
        chunk_results = self.chunk_gen.generate_all(str(self.data_dir), force=force)

        successful_chunks = sum(1 for v in chunk_results.values() if v)
        logger.info(f"✓ Chunk metadata: {successful_chunks}/{len(chunk_results)} successful")

        # Generate root metadata
        logger.info("\n📂 Generating root metadata...")
        root_success = self.root_gen.create_root_metadata(str(self.data_dir), force=force)

        # Verify
        logger.info("\n🔍 Verifying metadata...")
        root_valid = self.root_gen.verify_metadata(str(self.data_dir))

        logger.info("=" * 70)
        if root_success and root_valid:
            logger.info("✅ All metadata initialization successful!")
            return True
        else:
            logger.error("❌ Some metadata initialization steps failed")
            return False
