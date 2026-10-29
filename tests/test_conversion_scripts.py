#!/usr/bin/env python3
"""Unit and integration tests for model conversion, export, and quantization scripts."""

import struct
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from scripts import analyze_model, convert_model, verify_setup
from src.lib.conversion import load_model_checkpoint, quantize_model
from src.lib.model_export import ModelExporter
from src.lib.models import LightweightKanjiNet


@pytest.fixture
def sample_checkpoint(tmp_path: Path) -> Path:
    """Create a real LightweightKanjiNet checkpoint on disk."""
    model = LightweightKanjiNet(num_classes=12, input_channels=1, image_size=64)
    ckpt_path = tmp_path / "sample_cnn.pth"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1}, ckpt_path)
    return ckpt_path


class TestConvertModelCLI:
    """Tests for unified scripts/convert_model.py CLI and functions."""

    def test_parser_defaults_and_options(self):
        """Test create_parser parses required and optional conversion flags."""
        parser = convert_model.create_parser()
        args = parser.parse_args(
            [
                "--checkpoint",
                "model.pth",
                "--format",
                "gguf",
                "--quantization",
                "q4_k",
                "--model-type",
                "cnn",
                "--num-classes",
                "12",
            ]
        )
        assert args.checkpoint == "model.pth"
        assert args.format == "gguf"
        assert args.quantization == "q4_k"
        assert args.model_type == "cnn"
        assert args.num_classes == 12

    def test_load_checkpoint_extracts_state_dict(self, sample_checkpoint: Path):
        """Test load_checkpoint returns a valid state dict from trainer checkpoint."""
        state_dict = convert_model.load_checkpoint(str(sample_checkpoint), device="cpu")
        assert isinstance(state_dict, dict)
        assert "fc2.weight" in state_dict
        assert state_dict["fc2.weight"].shape[0] == 12

    def test_load_model_checkpoint_all_types(self, tmp_path: Path):
        """Test load_model_checkpoint instantiates real models from src.lib.models."""
        model, num_classes, info = load_model_checkpoint(str(self._save_cnn(tmp_path)), "cnn")
        assert isinstance(model, LightweightKanjiNet)
        assert num_classes == 16
        assert info["is_quantized"] is False

    @staticmethod
    def _save_cnn(tmp_path: Path) -> Path:
        model = LightweightKanjiNet(num_classes=16)
        p = tmp_path / "cnn16.pth"
        torch.save(model.state_dict(), p)
        return p

    def test_cli_main_pytorch_export(self, sample_checkpoint: Path, tmp_path: Path):
        """Test convert_model.main() exports PyTorch format end-to-end."""
        out_file = tmp_path / "exported_model.pth"
        test_argv = [
            "convert_model.py",
            "--checkpoint",
            str(sample_checkpoint),
            "--format",
            "pytorch",
            "--quantization",
            "none",
            "--model-type",
            "cnn",
            "--output",
            str(out_file),
            "--device",
            "cpu",
        ]
        with patch.object(sys, "argv", test_argv):
            convert_model.main()

        assert out_file.exists()
        assert out_file.with_suffix(".json").exists()
        loaded = torch.load(out_file, map_location="cpu", weights_only=True)
        assert "fc2.weight" in loaded

    def test_cli_main_safetensors_export(self, sample_checkpoint: Path, tmp_path: Path):
        """Test convert_model.main() exports SafeTensors format with INT8 quantization."""
        out_file = tmp_path / "exported_model.safetensors"
        test_argv = [
            "convert_model.py",
            "--checkpoint",
            str(sample_checkpoint),
            "--format",
            "safetensors",
            "--quantization",
            "int8",
            "--model-type",
            "cnn",
            "--output",
            str(out_file),
            "--device",
            "cpu",
        ]
        with patch.object(sys, "argv", test_argv):
            convert_model.main()

        assert out_file.exists()
        assert out_file.stat().st_size > 0

    def test_cli_main_gguf_export_writes_valid_gguf_header(
        self, sample_checkpoint: Path, tmp_path: Path
    ):
        """Test convert_model.main() writes a binary GGUF v3 file with magic header."""
        out_file = tmp_path / "exported_model.gguf"
        test_argv = [
            "convert_model.py",
            "--checkpoint",
            str(sample_checkpoint),
            "--format",
            "gguf",
            "--quantization",
            "q4_k",
            "--model-type",
            "cnn",
            "--output",
            str(out_file),
            "--device",
            "cpu",
        ]
        with patch.object(sys, "argv", test_argv):
            convert_model.main()

        assert out_file.exists()
        raw = out_file.read_bytes()
        assert raw[:4] == b"GGUF"
        version, tensor_count, kv_count = struct.unpack("<IQQ", raw[4:24])
        assert version == 3
        assert tensor_count > 0
        assert kv_count == 5


class TestGGUFAndQuantizationFormats:
    """Tests for GGUF quantization levels and unified quantization."""

    @pytest.mark.parametrize("quant", ["f32", "f16", "q8_0", "q4_k", "q2_k"])
    def test_export_gguf_quantization_levels(self, tmp_path: Path, quant: str):
        """Verify ModelExporter.export_gguf produces valid GGUF v3 binary across quant levels."""
        model = LightweightKanjiNet(num_classes=10)
        exporter = ModelExporter(model, model_type="cnn", num_classes=10, device="cpu")
        out_path = tmp_path / f"model_{quant}.gguf"
        saved_path, metadata = exporter.export_gguf(str(out_path), quantization=quant)

        assert Path(saved_path).exists()
        data = Path(saved_path).read_bytes()
        assert data[:4] == b"GGUF"
        version = struct.unpack("<I", data[4:8])[0]
        assert version == 3
        assert metadata["export_format"] == "gguf"
        assert metadata["quantization"] == quant

    def test_quantize_model_int8_and_bfloat16(self):
        """Verify quantize_model works on CPU for int8 and bfloat16."""
        model = LightweightKanjiNet(num_classes=10)
        q_int8, meta_int8 = quantize_model(model, "int8", device="cpu")
        assert q_int8 is not None
        assert meta_int8["quantization_format"] == "int8"
        assert meta_int8["quantized_size_mb"] > 0

        model_bf16 = LightweightKanjiNet(num_classes=10)
        q_bf16, meta_bf16 = quantize_model(model_bf16, "bfloat16", device="cpu")
        assert meta_bf16["quantization_format"] == "bfloat16"
        assert next(q_bf16.parameters()).dtype == torch.bfloat16


class TestAllUnifiedScriptsLoadable:
    """Integration tests to ensure all current scripts import and expose their CLI parsers."""

    def test_all_current_scripts_import_and_have_parsers(self):
        """Verify convert_model, analyze_model, and verify_setup expose create_parser."""
        assert callable(convert_model.create_parser)
        assert callable(analyze_model.create_parser)
        assert callable(verify_setup.create_parser)
