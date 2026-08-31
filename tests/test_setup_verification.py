"""Tests for setup verification module.

Tests verify:
- Python version checking
- Virtual environment detection
- Dependency verification
- GPU/CUDA setup verification
- System resource checks
- Dataset structure validation
- ETL9G dataset validation
- Training script discovery
- Training time estimation
- Unified check orchestration
"""

import tempfile
from pathlib import Path

import torch

from src.lib.setup_verification import (
    SetupVerifier,
    verify_dependencies,
    verify_gpu_setup,
    verify_system_resources,
)

# ============================================================================
# Test SetupVerifier Class
# ============================================================================


class TestSetupVerifierInitialization:
    """Test SetupVerifier initialization."""

    def test_init_default(self):
        """Test default initialization."""
        verifier = SetupVerifier()
        assert verifier.data_dir == Path("dataset")
        assert verifier.verbose is True
        assert isinstance(verifier.check_results, dict)

    def test_init_custom_data_dir(self):
        """Test initialization with custom data directory."""
        verifier = SetupVerifier(data_dir="custom_data")
        assert verifier.data_dir == Path("custom_data")

    def test_init_verbose_flag(self):
        """Test initialization with verbose flag."""
        verifier = SetupVerifier(verbose=False)
        assert verifier.verbose is False


class TestPythonVersionVerification:
    """Test Python version verification."""

    def test_python_version_ok(self):
        """Test Python version check passes."""
        verifier = SetupVerifier(verbose=False)
        result = verifier.verify_python_version(min_version=(3, 0))
        assert result is True

    def test_python_version_too_old(self):
        """Test Python version check fails for very old version."""
        verifier = SetupVerifier(verbose=False)
        result = verifier.verify_python_version(min_version=(5, 0))
        assert result is False

    def test_python_version_custom(self):
        """Test Python version check with custom minimum."""
        verifier = SetupVerifier(verbose=False)
        result = verifier.verify_python_version(min_version=(3, 8))
        # Should pass if running Python 3.8+
        assert isinstance(result, bool)


class TestVirtualEnvironmentVerification:
    """Test virtual environment detection."""

    def test_virtual_environment_check(self):
        """Test virtual environment check returns boolean."""
        verifier = SetupVerifier(verbose=False)
        result = verifier.verify_virtual_environment()
        assert isinstance(result, bool)


class TestDependencyVerification:
    """Test dependency verification."""

    def test_verify_dependencies_default(self):
        """Test dependency verification with default packages."""
        verifier = SetupVerifier(verbose=False)
        results = verifier.verify_dependencies()

        assert isinstance(results, dict)
        assert "torch" in results
        assert "numpy" in results
        assert isinstance(results["torch"], bool)
        assert results["torch"] is True  # torch should be available (test env dependency)

    def test_verify_dependencies_custom(self):
        """Test dependency verification with custom packages."""
        verifier = SetupVerifier(verbose=False)
        results = verifier.verify_dependencies(packages=["torch", "os"])

        assert len(results) == 2
        assert "torch" in results
        # "os" is a builtin, might not be found by importlib.util.find_spec

    def test_verify_dependencies_convenience_function(self):
        """Test convenience function for dependency verification."""
        results = verify_dependencies(packages=["torch"])
        assert isinstance(results, dict)
        assert "torch" in results


class TestGPUVerification:
    """Test GPU/CUDA verification."""

    def test_verify_gpu_setup_returns_dict(self):
        """Test GPU setup check returns dictionary."""
        verifier = SetupVerifier(verbose=False)
        results = verifier.verify_gpu_setup()

        assert isinstance(results, dict)
        assert "cuda_available" in results

    def test_verify_gpu_setup_cuda_available(self):
        """Test GPU setup when CUDA is available."""
        verifier = SetupVerifier(verbose=False)
        results = verifier.verify_gpu_setup()

        if torch.cuda.is_available():
            assert results["cuda_available"] is True
            assert "gpu_count" in results
            assert results["gpu_count"] > 0
        else:
            assert results["cuda_available"] is False

    def test_verify_gpu_setup_convenience_function(self):
        """Test convenience function for GPU setup verification."""
        results = verify_gpu_setup()
        assert isinstance(results, dict)
        assert "cuda_available" in results


class TestSystemResourcesVerification:
    """Test system resources verification."""

    def test_verify_system_resources_returns_dict(self):
        """Test system resources check returns dictionary."""
        verifier = SetupVerifier(verbose=False)
        results = verifier.verify_system_resources()

        assert isinstance(results, dict)
        assert "disk_available_gb" in results
        assert "disk_total_gb" in results

    def test_verify_system_resources_disk_values(self):
        """Test system resources have valid disk values."""
        verifier = SetupVerifier(verbose=False)
        results = verifier.verify_system_resources()

        assert results["disk_available_gb"] >= 0
        assert results["disk_total_gb"] > 0
        assert results["disk_available_gb"] <= results["disk_total_gb"]

    def test_verify_system_resources_convenience_function(self):
        """Test convenience function for system resources verification."""
        results = verify_system_resources()
        assert isinstance(results, dict)
        assert "disk_available_gb" in results


class TestDatasetStructureVerification:
    """Test dataset structure verification."""

    def test_dataset_structure_no_dataset(self):
        """Test dataset structure check when no dataset exists."""
        verifier = SetupVerifier(data_dir="/nonexistent", verbose=False)
        results = verifier.verify_dataset_structure()

        assert results["dataset_found"] is False
        assert results["primary_dataset"] is None
        assert results["available_datasets"] == []

    def test_dataset_structure_with_mock(self):
        """Test dataset structure check with mock dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock dataset structure
            data_dir = Path(tmpdir)
            etl9g_dir = data_dir / "etl9g"
            etl9g_dir.mkdir()

            # Create metadata
            metadata_file = data_dir / "metadata.json"
            metadata_file.write_text('{"test": "data"}')

            verifier = SetupVerifier(data_dir=str(data_dir), verbose=False)
            results = verifier.verify_dataset_structure()

            assert results["dataset_found"] is True
            assert results["primary_dataset"] == "etl9g"
            assert results["metadata_exists"] is True

    def test_dataset_priority_order(self):
        """Test dataset priority ordering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            # Create multiple datasets
            (data_dir / "etl1").mkdir()
            (data_dir / "etl9g").mkdir()
            (data_dir / "combined_all_etl").mkdir()

            verifier = SetupVerifier(data_dir=str(data_dir), verbose=False)
            results = verifier.verify_dataset_structure()

            # Should pick highest priority: combined_all_etl
            assert results["primary_dataset"] == "combined_all_etl"


class TestETL9GVerification:
    """Test ETL9G dataset verification."""

    def test_etl9g_not_found(self):
        """Test ETL9G check when directory doesn't exist."""
        verifier = SetupVerifier(data_dir="/nonexistent", verbose=False)
        results = verifier.verify_etl9g_datasets()

        assert results["etl9g_found"] is False
        assert len(results["errors"]) > 0

    def test_etl9g_with_mock_structure(self):
        """Test ETL9G check with mock structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            etl9g_dir = data_dir / "etl9g"
            etl9g_dir.mkdir()

            # Create mock metadata and chunk_info
            metadata = {"class_1": "character_1"}
            chunk_info = {"chunks": []}

            with open(data_dir / "metadata.json", "w") as f:
                import json

                json.dump(metadata, f)

            with open(etl9g_dir / "chunk_info.json", "w") as f:
                import json

                json.dump(chunk_info, f)

            verifier = SetupVerifier(data_dir=str(data_dir), verbose=False)
            results = verifier.verify_etl9g_datasets()

            assert results["etl9g_found"] is True
            assert results["metadata_valid"] is True
            assert results["chunks_valid"] is True


class TestTrainingScriptsVerification:
    """Test training scripts verification."""

    def test_training_scripts_check_returns_dict(self):
        """Test training scripts check returns dictionary."""
        verifier = SetupVerifier(verbose=False)
        results = verifier.verify_training_scripts()

        assert isinstance(results, dict)
        # Should have entries for each script
        assert len(results) > 0

    def test_training_scripts_keys(self):
        """Test training scripts have expected keys."""
        verifier = SetupVerifier(verbose=False)
        results = verifier.verify_training_scripts()

        expected_scripts = [
            "train_cnn_model.py",
            "train_rnn.py",
            "train_vit.py",
            "train_hiercode.py",
            "train_qat.py",
            "train_radical_rnn.py",
            "train_hiercode_higita.py",
        ]

        for script in expected_scripts:
            assert script in results


class TestTrainingTimeEstimation:
    """Test training time estimation."""

    def test_training_time_estimate_returns_dict(self):
        """Test training time estimation returns dictionary."""
        verifier = SetupVerifier(verbose=False)
        results = verifier.estimate_training_time()

        assert isinstance(results, dict)
        assert len(results) > 0

    def test_training_time_estimate_with_custom_params(self):
        """Test training time estimation with custom parameters."""
        verifier = SetupVerifier(verbose=False)
        results = verifier.estimate_training_time(num_samples=1000000, batch_size=64)

        assert isinstance(results, dict)

    def test_training_time_estimates_have_reasonable_values(self):
        """Test training time estimates have reasonable string values."""
        verifier = SetupVerifier(verbose=False)
        results = verifier.estimate_training_time()

        for _key, value in results.items():
            assert isinstance(value, str)
            # Should contain time units
            assert any(unit in value for unit in ["hour", "day", "second"])


class TestUnifiedCheckOrchestration:
    """Test unified check orchestration."""

    def test_run_all_checks_returns_dict(self):
        """Test run_all_checks returns comprehensive dictionary."""
        verifier = SetupVerifier(verbose=False)
        results = verifier.run_all_checks(check_training_scripts=False)

        assert isinstance(results, dict)
        assert "python_version_ok" in results
        assert "dependencies_ok" in results
        assert "gpu" in results
        assert "system" in results
        assert "all_ok" in results

    def test_run_all_checks_with_training_scripts(self):
        """Test run_all_checks with training scripts check."""
        verifier = SetupVerifier(verbose=False)
        results = verifier.run_all_checks(check_training_scripts=True)

        assert "training_scripts" in results

    def test_run_all_checks_summary(self):
        """Test run_all_checks produces summary."""
        verifier = SetupVerifier(verbose=False)
        results = verifier.run_all_checks(check_training_scripts=False)

        # Should have overall pass/fail
        assert "all_ok" in results
        assert isinstance(results["all_ok"], bool)

    def test_run_all_checks_with_mock_dataset(self):
        """Test run_all_checks with mock dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            etl9g_dir = data_dir / "etl9g"
            etl9g_dir.mkdir()

            verifier = SetupVerifier(data_dir=str(data_dir), verbose=False)
            results = verifier.run_all_checks(check_training_scripts=False)

            assert "dataset" in results
            assert results["dataset"]["dataset_found"] is True


# ============================================================================
# Integration Tests
# ============================================================================


class TestSetupVerifierIntegration:
    """Integration tests for setup verifier."""

    def test_full_verification_workflow(self):
        """Test complete verification workflow."""
        verifier = SetupVerifier(verbose=False)

        # Run individual checks
        python_ok = verifier.verify_python_version()
        deps = verifier.verify_dependencies(packages=["torch"])
        gpu_info = verifier.verify_gpu_setup()
        system_info = verifier.verify_system_resources()

        assert isinstance(python_ok, bool)
        assert isinstance(deps, dict)
        assert isinstance(gpu_info, dict)
        assert isinstance(system_info, dict)

    def test_verifier_produces_consistent_results(self):
        """Test verifier produces consistent results across calls."""
        verifier = SetupVerifier(verbose=False)

        results1 = verifier.verify_dependencies(packages=["torch"])
        results2 = verifier.verify_dependencies(packages=["torch"])

        assert results1 == results2

    def test_multiple_verifiers_independent(self):
        """Test multiple verifier instances are independent."""
        verifier1 = SetupVerifier(data_dir="data1", verbose=False)
        verifier2 = SetupVerifier(data_dir="data2", verbose=False)

        assert verifier1.data_dir != verifier2.data_dir
        # Verify they're separate instances
        assert verifier1 is not verifier2
        assert verifier1.check_results is not verifier2.check_results
