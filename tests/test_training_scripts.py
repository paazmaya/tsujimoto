#!/usr/bin/env python3
"""Unit tests for unified training script and model utilities."""

import pytest


class TestUnifiedTrainingScript:
    """Tests for unified train.py script."""

    def test_train_py_module_imports(self):
        """Test that train.py module can be imported."""
        from scripts import train

        assert train is not None

    def test_train_py_has_subcommands(self):
        """Test that train.py has all 12 expected base and advanced subcommands."""
        import click

        from scripts.train import train

        assert isinstance(train, click.Group)

        subcommands = list(train.commands.keys())
        expected = [
            "cnn",
            "rnn",
            "vit",
            "hiercode",
            "qat",
            "hiercode-higita",
            "glhpn",
            "dtrnet",
            "degradation",
            "trajectory",
            "multigranular",
            "restoration-pipeline",
        ]
        for cmd in expected:
            assert cmd in subcommands, f"Missing subcommand: {cmd}"

    @pytest.mark.parametrize(
        "cmd",
        [
            "cnn",
            "rnn",
            "vit",
            "hiercode",
            "qat",
            "hiercode-higita",
            "glhpn",
            "dtrnet",
            "degradation",
            "trajectory",
            "multigranular",
            "restoration-pipeline",
        ],
    )
    def test_train_py_subcommand_help(self, cmd: str):
        """Test that every subcommand in train.py renders --help cleanly via CliRunner."""
        from click.testing import CliRunner

        from scripts.train import train

        runner = CliRunner()
        result = runner.invoke(train, [cmd, "--help"])
        assert result.exit_code == 0, f"{cmd} --help failed: {result.output}"
        assert "--epochs" in result.output

    def test_train_py_end_to_end_smoke_with_mocked_loader(self, tmp_path):
        """Test invoking cnn and glhpn subcommands end-to-end on synthetic batches."""
        from unittest.mock import patch

        import torch
        from click.testing import CliRunner
        from torch.utils.data import DataLoader, TensorDataset

        from scripts.train import train

        ds = TensorDataset(torch.randn(8, 1, 64, 64), torch.randint(0, 5, (8,)))
        loader = DataLoader(ds, batch_size=4)

        runner = CliRunner()
        with patch(
            "scripts.train._resolve_dataset_and_loaders",
            return_value=("synthetic", loader, loader, "val", 5),
        ):
            res_cnn = runner.invoke(
                train,
                [
                    "cnn",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--checkpoint-dir",
                    str(tmp_path / "cnn_ckpt"),
                ],
            )
            assert res_cnn.exit_code == 0, res_cnn.output

            res_glhpn = runner.invoke(
                train,
                [
                    "glhpn",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--embedding-dim",
                    "32",
                    "--top-k",
                    "4",
                    "--checkpoint-dir",
                    str(tmp_path / "glhpn_ckpt"),
                ],
            )
            assert res_glhpn.exit_code == 0, res_glhpn.output


class TestMeasureCO2:
    """Extended tests for CO2 emissions measurement."""

    def test_get_system_info_returns_dict(self):
        """Test that get_system_info returns dictionary."""
        from scripts.measure_co2_emissions import get_system_info

        info = get_system_info()
        assert isinstance(info, dict)

    def test_system_info_keys(self):
        """Test all required keys in system info."""
        from scripts.measure_co2_emissions import get_system_info

        info = get_system_info()
        required_keys = [
            "timestamp",
            "platform",
            "python_version",
            "cpu_count",
            "memory_total_gb",
            "torch_version",
        ]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"

    def test_system_info_valid_values(self):
        """Test that system info contains valid values."""
        from scripts.measure_co2_emissions import get_system_info

        info = get_system_info()
        assert isinstance(info["cpu_count"], int)
        assert info["cpu_count"] > 0
        assert isinstance(info["memory_total_gb"], (int, float))
        assert info["memory_total_gb"] > 0
        assert isinstance(info["torch_version"], str)
        assert len(info["torch_version"]) > 0


class TestGenerateMapping:
    """Extended tests for character mapping generation."""

    def test_jis_to_unicode_hiragana(self):
        """Test hiragana conversion from JIS."""
        from src.lib.character_mapping import JISConverter

        converter = JISConverter()
        result = converter.jis_to_unicode("2421")
        assert isinstance(result, str)
        assert len(result) >= 1

    def test_jis_to_unicode_katakana(self):
        """Test katakana conversion from JIS."""
        from src.lib.character_mapping import JISConverter

        converter = JISConverter()
        result = converter.jis_to_unicode("2521")
        assert isinstance(result, str)
        assert len(result) >= 1

    def test_jis_to_unicode_kanji(self):
        """Test kanji conversion from JIS."""
        from src.lib.character_mapping import JISConverter

        converter = JISConverter()
        result = converter.jis_to_unicode("3021")
        assert isinstance(result, str)
        assert len(result) >= 1

    def test_jis_to_unicode_invalid(self):
        """Test invalid JIS code handling."""
        from src.lib.character_mapping import JISConverter

        converter = JISConverter()
        result = converter.jis_to_unicode("ZZZZ")
        assert isinstance(result, str)
        # Should return placeholder for invalid codes
        assert "JIS" in result or result.startswith("[")

    def test_estimate_stroke_count_single_char(self):
        """Test stroke count for single character."""
        from src.lib.character_mapping import JISConverter

        converter = JISConverter()
        result = converter.estimate_stroke_count("a")
        assert isinstance(result, int)
        assert result >= 1

    def test_estimate_stroke_count_hiragana(self):
        """Test stroke count for hiragana."""
        from src.lib.character_mapping import JISConverter

        converter = JISConverter()
        result = converter.estimate_stroke_count("あ")
        assert isinstance(result, int)
        assert 1 <= result <= 25

    def test_estimate_stroke_count_kanji(self):
        """Test stroke count for kanji."""
        from src.lib.character_mapping import JISConverter

        converter = JISConverter()
        result = converter.estimate_stroke_count("漢")
        assert isinstance(result, int)
        assert 1 <= result <= 25

    def test_estimate_stroke_count_empty(self):
        """Test stroke count for empty string."""
        from src.lib.character_mapping import JISConverter

        converter = JISConverter()
        result = converter.estimate_stroke_count("")
        assert isinstance(result, int)

    def test_estimate_stroke_count_multi_char(self):
        """Test stroke count for multi-character string."""
        from src.lib.character_mapping import JISConverter

        converter = JISConverter()
        result = converter.estimate_stroke_count("abc")
        assert isinstance(result, int)


class TestSetupVerification:
    """Extended tests for setup verification."""

    def test_setup_verifier_module_imports(self):
        """Test SetupVerifier module imports."""
        from src.lib.setup_verification import SetupVerifier

        assert SetupVerifier is not None

    def test_verify_virtual_environment(self):
        """Test virtual environment verification."""
        try:
            from src.lib.setup_verification import SetupVerifier

            verifier = SetupVerifier(verbose=False)
            result = verifier.verify_virtual_environment()
            assert isinstance(result, bool)
        except (ImportError, AttributeError):
            pytest.skip("Function not available")

    def test_verify_gpu_setup(self):
        """Test GPU availability verification."""
        try:
            from src.lib.setup_verification import SetupVerifier

            verifier = SetupVerifier(verbose=False)
            result = verifier.verify_gpu_setup()
            assert isinstance(result, dict)
        except (ImportError, AttributeError):
            pytest.skip("Function not available")

    def test_verify_dependencies(self):
        """Test dependency verification."""
        try:
            from src.lib.setup_verification import SetupVerifier

            verifier = SetupVerifier(verbose=False)
            result = verifier.verify_dependencies()
            assert isinstance(result, dict)
        except (ImportError, AttributeError):
            pytest.skip("Function not available")


class TestDatasetsAndRouter:
    """Tests for src.lib.datasets, src.lib.dataset_router, and safe archive extraction."""

    def test_kanji_dataset_handles_grayscale_pil_image(self):
        """Verify KanjiDataset.__getitem__ converts 2D grayscale PIL images to (1, H, W) tensors."""
        import numpy as np
        import torch
        from PIL import Image

        from src.lib.datasets import KanjiDataset, create_synthetic_kanji_dataset

        synth = create_synthetic_kanji_dataset(num_samples=20, num_classes=10, image_size=64)
        assert "train" in synth and "validation" in synth and "test" in synth

        ds = KanjiDataset(synth["train"])
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (1, 64, 64)
        assert 0 <= int(label) < 10

        # Also test direct grayscale PIL item
        gray_pil = Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L")
        custom_ds = KanjiDataset([{"image": gray_pil, "label": 3}])
        tensor_img, lbl = custom_ds[0]
        assert tensor_img.shape == (1, 32, 32)
        assert lbl == 3

    def test_research_dataset_loader_deterministic_non_overlapping_splits(self, tmp_path):
        """Verify ResearchDatasetLoader train and test splits are disjoint across calls."""
        from unittest.mock import patch

        import numpy as np

        from src.lib.datasets import ResearchDatasetLoader

        x = np.arange(20 * 64 * 64, dtype=np.float32).reshape(20, 64 * 64)
        y = np.arange(20, dtype=np.int64)

        loader = ResearchDatasetLoader("kanji", data_dir=tmp_path, num_workers=0)
        with patch.object(loader, "load_dataset", return_value={"X": x, "y": y}):
            train_dl = loader.get_dataloader(batch_size=4, split="train", shuffle=False)
            test_dl = loader.get_dataloader(batch_size=4, split="test", shuffle=False)

            train_labels = set()
            for _, labels in train_dl:
                train_labels.update(labels.tolist())

            test_labels = set()
            for _, labels in test_dl:
                test_labels.update(labels.tolist())

            assert len(train_labels) == 16
            assert len(test_labels) == 4
            assert train_labels.isdisjoint(test_labels)

    def test_safe_extract_tar_and_zip_reject_traversal_and_symlinks(self, tmp_path):
        """Verify safe_extract_tar and safe_extract_zip block path traversal and symlinks."""
        import io
        import tarfile
        import zipfile

        from src.lib.system import safe_extract_tar, safe_extract_zip

        # 1. Malicious TAR with ../ traversal
        evil_tar = tmp_path / "evil.tar.gz"
        with tarfile.open(evil_tar, "w:gz") as tf:
            info = tarfile.TarInfo(name="../escaped.txt")
            payload = b"pwned"
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))

        with pytest.raises(ValueError, match="Unsafe tar member"):
            safe_extract_tar(evil_tar, tmp_path / "out_tar")

        # 2. Malicious TAR with symlink
        symlink_tar = tmp_path / "symlink.tar.gz"
        with tarfile.open(symlink_tar, "w:gz") as tf:
            info = tarfile.TarInfo(name="link_to_etc")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tf.addfile(info)

        with pytest.raises(ValueError, match="Unsafe tar member type"):
            safe_extract_tar(symlink_tar, tmp_path / "out_symlink")

        # 3. Malicious ZIP with ../ traversal
        evil_zip = tmp_path / "evil.zip"
        with zipfile.ZipFile(evil_zip, "w") as zf:
            zf.writestr("../../escaped.txt", "pwned")

        with pytest.raises(ValueError, match="Unsafe zip member"):
            safe_extract_zip(evil_zip, tmp_path / "out_zip")

    def test_onnx_analysis_and_dataset_router_and_cli(self):
        """Verify ONNXModelAnalyzer, DatasetRouter, and CLI utilities."""
        from scripts.training_args import get_variant_args
        from src.lib.cli import CLIContext, create_app
        from src.lib.dataset_router import DatasetRouter
        from src.lib.onnx_analysis import PoolingComparisonAnalyzer

        router = DatasetRouter()
        available = router.list_datasets()
        assert "etl" in available and "research" in available
        assert "etl9g" in available["etl"]
        assert "megahan97k" in available["research"]

        cnn_args = get_variant_args("cnn")
        assert "epochs" in cnn_args and "num_classes" in cnn_args
        assert cnn_args["num_classes"].default is None

        app = create_app(name="test-cli")
        assert app is not None
        with CLIContext("unit-test-op") as ctx:
            assert ctx.operation == "unit-test-op"

        assert PoolingComparisonAnalyzer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
