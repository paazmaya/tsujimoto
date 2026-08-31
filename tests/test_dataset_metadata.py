"""
Tests for dataset metadata generation utilities.

Tests for:
- Chunk metadata generation
- Root metadata creation
- Metadata validation
- Dataset metadata manager workflows
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.lib.metadata_generator import (
    ChunkMetadataGenerator,
    DatasetMetadataManager,
    RootMetadataGenerator,
)


class TestChunkMetadataGenerator:
    """Test chunk metadata generation."""

    @pytest.fixture
    def generator(self):
        """Create a chunk metadata generator instance."""
        return ChunkMetadataGenerator()

    @pytest.fixture
    def temp_dataset(self):
        """Create a temporary dataset with chunk files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "test_dataset"
            dataset_dir.mkdir()

            # Create sample chunk files
            for i in range(3):
                chunk_file = dataset_dir / f"test_dataset_chunk_{i:02d}.npz"
                chunk_file.touch()

            yield dataset_dir

    def test_generate_for_dataset(self, generator, temp_dataset):
        """Test chunk metadata generation for single dataset."""
        success = generator.generate_for_dataset(temp_dataset)

        assert success is True
        chunk_info_path = temp_dataset / "chunk_info.json"
        assert chunk_info_path.exists()

        # Verify content
        with open(chunk_info_path, encoding="utf-8") as f:
            chunk_info = json.load(f)

        assert chunk_info["dataset_name"] == "test_dataset"
        assert chunk_info["num_chunks"] == 3
        assert len(chunk_info["chunk_files"]) == 3

    def test_generate_for_nonexistent_dataset(self, generator):
        """Test handling of nonexistent dataset directory."""
        success = generator.generate_for_dataset(Path("nonexistent_dir"))
        assert success is False

    def test_generate_skips_existing_metadata(self, generator, temp_dataset):
        """Test that existing metadata is not overwritten by default."""
        # Generate metadata first time
        success1 = generator.generate_for_dataset(temp_dataset)
        assert success1 is True

        # Modify the file
        chunk_info_path = temp_dataset / "chunk_info.json"
        with open(chunk_info_path, "w", encoding="utf-8") as f:
            json.dump({"modified": True}, f)

        # Try to generate again (should skip)
        success2 = generator.generate_for_dataset(temp_dataset, force=False)
        assert success2 is True

        # Verify file wasn't overwritten
        with open(chunk_info_path, encoding="utf-8") as f:
            chunk_info = json.load(f)
        assert chunk_info.get("modified") is True

    def test_generate_force_overwrites(self, generator, temp_dataset):
        """Test force parameter overwrites existing metadata."""
        # Generate metadata first time
        generator.generate_for_dataset(temp_dataset)

        # Modify the file
        chunk_info_path = temp_dataset / "chunk_info.json"
        with open(chunk_info_path, "w", encoding="utf-8") as f:
            json.dump({"modified": True}, f)

        # Force regenerate
        success = generator.generate_for_dataset(temp_dataset, force=True)
        assert success is True

        # Verify file was overwritten
        with open(chunk_info_path, encoding="utf-8") as f:
            chunk_info = json.load(f)
        assert "num_chunks" in chunk_info

    def test_generate_no_chunks(self, generator):
        """Test handling of dataset with no chunk files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "empty_dataset"
            dataset_dir.mkdir()

            success = generator.generate_for_dataset(dataset_dir)
            assert success is False
            assert not (dataset_dir / "chunk_info.json").exists()

    def test_generate_all_datasets(self, generator):
        """Test generating metadata for all datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create multiple datasets with chunks
            for dataset_name in ["etl1", "etl9g"]:
                dataset_dir = data_dir / dataset_name
                dataset_dir.mkdir()
                for i in range(2):
                    chunk_file = dataset_dir / f"{dataset_name}_chunk_{i:02d}.npz"
                    chunk_file.touch()

            # Create empty dataset
            empty_dir = data_dir / "empty"
            empty_dir.mkdir()

            results = generator.generate_all(str(data_dir))

            # Check results
            assert results["etl1"] is True
            assert results["etl9g"] is True
            assert (data_dir / "etl1" / "chunk_info.json").exists()
            assert (data_dir / "etl9g" / "chunk_info.json").exists()

    def test_validate_chunks_success(self, generator, temp_dataset):
        """Test successful chunk validation."""
        generator.generate_for_dataset(temp_dataset)
        success = generator.validate_chunks(temp_dataset)
        assert success is True

    def test_validate_chunks_missing_metadata(self, generator):
        """Test validation when chunk_info.json is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir)
            success = generator.validate_chunks(dataset_dir)
            assert success is False

    def test_validate_chunks_missing_files(self, generator):
        """Test validation when chunk files are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir)
            dataset_dir.mkdir(exist_ok=True)

            # Create metadata pointing to nonexistent files
            chunk_info = {
                "dataset_name": "test",
                "num_chunks": 2,
                "chunk_files": ["test_chunk_00.npz", "test_chunk_01.npz"],
            }

            with open(dataset_dir / "chunk_info.json", "w", encoding="utf-8") as f:
                json.dump(chunk_info, f)

            # Don't create actual chunk files
            success = generator.validate_chunks(dataset_dir)
            assert success is False

    def test_validate_chunks_invalid_json(self, generator):
        """Test validation with invalid JSON in chunk_info.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir)
            dataset_dir.mkdir(exist_ok=True)

            # Write invalid JSON
            with open(dataset_dir / "chunk_info.json", "w") as f:
                f.write("{ invalid json")

            success = generator.validate_chunks(dataset_dir)
            assert success is False


class TestRootMetadataGenerator:
    """Test root metadata generation."""

    @pytest.fixture
    def generator(self):
        """Create a root metadata generator instance."""
        return RootMetadataGenerator()

    @pytest.fixture
    def temp_datasets(self):
        """Create temporary datasets with metadata files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create datasets in priority order
            for dataset_name in ["etl9g", "etl8g"]:
                dataset_dir = data_dir / dataset_name
                dataset_dir.mkdir()

                metadata = {
                    "dataset_name": dataset_name,
                    "num_classes": 3000,
                    "total_samples": 50000,
                    "target_size": 64,
                }

                with open(dataset_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(metadata, f)

            yield data_dir

    def test_create_root_metadata(self, generator, temp_datasets):
        """Test root metadata creation."""
        success = generator.create_root_metadata(str(temp_datasets))

        assert success is True
        root_metadata_path = temp_datasets / "metadata.json"
        assert root_metadata_path.exists()

        # Verify content (should use etl9g as it comes first in priority)
        with open(root_metadata_path, encoding="utf-8") as f:
            root_metadata = json.load(f)

        assert root_metadata["primary_dataset"] == "etl9g"
        assert root_metadata["num_classes"] == 3000
        assert root_metadata["total_samples"] == 50000
        assert root_metadata["target_size"] == 64

    def test_create_root_metadata_uses_priority(self, generator):
        """Test that root metadata creation respects dataset priority."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create combined_all_etl (highest priority)
            combined_dir = data_dir / "combined_all_etl"
            combined_dir.mkdir()
            with open(combined_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "num_classes": 43000,
                        "total_samples": 500000,
                        "target_size": 64,
                    },
                    f,
                )

            # Create etl9g (lower priority)
            etl9g_dir = data_dir / "etl9g"
            etl9g_dir.mkdir()
            with open(etl9g_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"num_classes": 3000, "total_samples": 50000, "target_size": 64},
                    f,
                )

            success = generator.create_root_metadata(str(data_dir))
            assert success is True

            # Should use combined_all_etl (higher priority)
            with open(data_dir / "metadata.json", encoding="utf-8") as f:
                root_metadata = json.load(f)
            assert root_metadata["primary_dataset"] == "combined_all_etl"

    def test_create_root_metadata_skips_existing(self, generator, temp_datasets):
        """Test that existing root metadata is not overwritten by default."""
        # Create metadata first time
        generator.create_root_metadata(str(temp_datasets))

        # Modify it
        with open(temp_datasets / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({"modified": True}, f)

        # Try to create again without force
        success = generator.create_root_metadata(str(temp_datasets), force=False)
        assert success is True

        # Verify it wasn't overwritten
        with open(temp_datasets / "metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
        assert metadata.get("modified") is True

    def test_create_root_metadata_force_overwrites(self, generator, temp_datasets):
        """Test force parameter overwrites existing root metadata."""
        # Create metadata first time
        generator.create_root_metadata(str(temp_datasets))

        # Modify it
        with open(temp_datasets / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({"modified": True}, f)

        # Force recreate
        success = generator.create_root_metadata(str(temp_datasets), force=True)
        assert success is True

        # Verify it was overwritten
        with open(temp_datasets / "metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
        assert "primary_dataset" in metadata

    def test_create_root_metadata_no_datasets(self, generator):
        """Test handling when no dataset metadata exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            success = generator.create_root_metadata(str(data_dir))
            assert success is False

    def test_get_metadata_info(self, generator, temp_datasets):
        """Test loading metadata information."""
        # First create root metadata
        generator.create_root_metadata(str(temp_datasets))

        # Load it back
        metadata = generator.get_metadata_info(temp_datasets / "metadata.json")

        assert metadata is not None
        assert metadata["primary_dataset"] == "etl9g"
        assert metadata["num_classes"] == 3000

    def test_get_metadata_info_nonexistent(self, generator):
        """Test loading nonexistent metadata."""
        metadata = generator.get_metadata_info(Path("nonexistent.json"))
        assert metadata is None

    def test_verify_metadata_valid(self, generator, temp_datasets):
        """Test metadata verification with valid metadata."""
        generator.create_root_metadata(str(temp_datasets))
        success = generator.verify_metadata(str(temp_datasets))
        assert success is True

    def test_verify_metadata_missing(self, generator):
        """Test verification when metadata is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            success = generator.verify_metadata(tmpdir)
            assert success is False

    def test_verify_metadata_invalid_json(self, generator):
        """Test verification with invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            with open(data_dir / "metadata.json", "w") as f:
                f.write("{ invalid json")

            success = generator.verify_metadata(tmpdir)
            assert success is False

    def test_verify_metadata_missing_fields(self, generator):
        """Test verification when required fields are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            with open(data_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump({"primary_dataset": "etl9g"}, f)

            success = generator.verify_metadata(tmpdir)
            assert success is False


class TestDatasetMetadataManager:
    """Test high-level metadata manager."""

    @pytest.fixture
    def manager(self):
        """Create a dataset metadata manager instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DatasetMetadataManager(tmpdir)
            yield manager

    def test_initialize_all_metadata(self):
        """Test initialization of all metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test datasets
            data_dir = Path(tmpdir)

            for dataset_name in ["etl1", "etl9g"]:
                dataset_dir = data_dir / dataset_name
                dataset_dir.mkdir()

                # Create chunk files
                for i in range(2):
                    chunk_file = dataset_dir / f"{dataset_name}_chunk_{i:02d}.npz"
                    chunk_file.touch()

                # Create metadata
                with open(dataset_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "num_classes": 3000,
                            "total_samples": 50000,
                            "target_size": 64,
                        },
                        f,
                    )

            manager = DatasetMetadataManager(str(data_dir))
            success = manager.initialize_all_metadata()

            assert success is True
            assert (data_dir / "metadata.json").exists()
            assert (data_dir / "etl1" / "chunk_info.json").exists()
            assert (data_dir / "etl9g" / "chunk_info.json").exists()


class TestIntegrationMetadata:
    """Integration tests for metadata generation."""

    def test_complete_metadata_workflow(self):
        """Test complete workflow from empty directory to fully initialized metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create dataset structure
            etl9g_dir = data_dir / "etl9g"
            etl9g_dir.mkdir()

            # Create chunk files
            for i in range(3):
                chunk_file = etl9g_dir / f"etl9g_chunk_{i:02d}.npz"
                chunk_file.touch()

            # Create dataset metadata
            with open(etl9g_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset_name": "etl9g",
                        "num_classes": 3000,
                        "total_samples": 50000,
                        "target_size": 64,
                    },
                    f,
                )

            manager = DatasetMetadataManager(str(data_dir))

            # Initialize all metadata
            success = manager.initialize_all_metadata()

            assert success is True

            # Verify chunk metadata
            with open(etl9g_dir / "chunk_info.json", encoding="utf-8") as f:
                chunk_info = json.load(f)
            assert chunk_info["num_chunks"] == 3

            # Verify root metadata
            with open(data_dir / "metadata.json", encoding="utf-8") as f:
                root_metadata = json.load(f)
            assert root_metadata["primary_dataset"] == "etl9g"

            # Verify all metadata
            assert manager.root_gen.verify_metadata(str(data_dir)) is True
