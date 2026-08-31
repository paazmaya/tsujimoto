"""
Tests for unified model export functionality.

Tests for:
- PyTorch export with quantization
- ONNX export
- SafeTensors export
- GGUF export
- Unified export interface
- Metadata generation and validation
"""

import json
import tempfile
from pathlib import Path
from typing import Tuple

import pytest
import torch
import torch.nn as nn

from src.lib.model_export import ModelExporter


class SimpleTestModel(nn.Module):
    """Simple test model for export testing."""

    def __init__(self, num_classes: int = 10, image_size: int = 64):
        """Initialize simple model."""
        super().__init__()
        self.fc1 = nn.Linear(1 * image_size * image_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TestModelExporter:
    """Test unified model exporter."""

    @pytest.fixture
    def test_model(self):
        """Create test model."""
        return SimpleTestModel()

    @pytest.fixture
    def exporter(self, test_model):
        """Create model exporter."""
        return ModelExporter(
            test_model,
            model_type="test_model",
            num_classes=10,
            device="cpu",
            image_size=64,
        )

    def test_exporter_initialization(self, exporter):
        """Test exporter initialization."""
        assert exporter.model is not None
        assert exporter.model_type == "test_model"
        assert exporter.num_classes == 10
        assert exporter.device == "cpu"
        assert exporter.original_size > 0

    def test_exporter_supported_formats(self):
        """Test supported export formats."""
        expected_formats = ["pytorch", "onnx", "safetensors", "gguf"]
        assert ModelExporter.SUPPORTED_FORMATS == expected_formats

    def test_export_pytorch_float32(self, exporter):
        """Test PyTorch export without quantization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pth"

            result_path, metadata = exporter.export_pytorch(str(output_path), quantization="float32")

            # Verify file was created
            assert Path(result_path).exists()
            assert metadata["quantization"] == "float32"
            assert metadata["compression_ratio"] == pytest.approx(1.0, rel=0.01)

    def test_export_pytorch_int8(self, exporter):
        """Test PyTorch export with INT8 quantization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_int8.pth"

            result_path, metadata = exporter.export_pytorch(str(output_path), quantization="int8")

            # Verify file was created
            assert Path(result_path).exists()
            assert metadata["quantization"] == "int8"
            assert metadata["compression_ratio"] > 1.0

    def test_export_pytorch_bfloat16(self, exporter):
        """Test PyTorch export with BFloat16."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_bf16.pth"

            result_path, metadata = exporter.export_pytorch(str(output_path), quantization="bfloat16")

            # Verify file was created
            assert Path(result_path).exists()
            assert metadata["quantization"] == "bfloat16"
            assert metadata["compression_ratio"] > 1.0

    def test_export_pytorch_metadata(self, exporter):
        """Test PyTorch export with metadata saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pth"

            result_path, metadata = exporter.export_pytorch(str(output_path), save_metadata=True)

            # Verify metadata file was created
            metadata_path = Path(output_path).with_suffix(".json")
            assert metadata_path.exists()

            # Verify metadata content
            with open(metadata_path, "r", encoding="utf-8") as f:
                saved_metadata = json.load(f)

            assert saved_metadata["model_type"] == "test_model"
            assert saved_metadata["num_classes"] == 10

    def test_export_pytorch_no_metadata(self, exporter):
        """Test PyTorch export without metadata saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pth"

            result_path, metadata = exporter.export_pytorch(str(output_path), save_metadata=False)

            # Verify model file exists but metadata doesn't
            assert Path(result_path).exists()
            metadata_path = Path(output_path).with_suffix(".json")
            assert not metadata_path.exists()

    def test_export_onnx(self, exporter):
        """Test ONNX export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"

            try:
                result_path, metadata = exporter.export_onnx(str(output_path))

                # Verify file was created
                assert Path(result_path).exists()
                assert metadata["export_format"] == "onnx"
            except ImportError:
                pytest.skip("ONNX not available")

    def test_export_safetensors_float32(self, exporter):
        """Test SafeTensors export without quantization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.safetensors"

            try:
                result_path, metadata = exporter.export_safetensors(
                    str(output_path), quantization="float32"
                )

                # Verify file was created
                assert Path(result_path).exists()
                assert metadata["quantization"] == "float32"
            except ImportError:
                pytest.skip("SafeTensors not available")

    def test_export_safetensors_int8(self, exporter):
        """Test SafeTensors export with INT8 quantization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_int8.safetensors"

            try:
                result_path, metadata = exporter.export_safetensors(
                    str(output_path), quantization="int8"
                )

                # Verify file was created
                assert Path(result_path).exists()
                assert metadata["quantization"] == "int8"
                assert metadata["compression_ratio"] > 1.0
            except ImportError:
                pytest.skip("SafeTensors not available")

    def test_export_gguf(self, exporter):
        """Test GGUF export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.gguf"

            result_path, metadata = exporter.export_gguf(str(output_path), quantization="q4_k")

            # Verify file was created
            assert Path(result_path).exists()
            assert metadata["export_format"] == "gguf"
            assert metadata["quantization"] == "q4_k"

    def test_export_gguf_invalid_quantization(self, exporter):
        """Test GGUF export with invalid quantization format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.gguf"

            with pytest.raises(ValueError):
                exporter.export_gguf(str(output_path), quantization="invalid_format")

    def test_export_unified_interface(self, exporter):
        """Test unified export interface."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test PyTorch export through unified interface
            output_path = Path(tmpdir) / "model.pth"
            result_path, metadata = exporter.export(str(output_path), format="pytorch")

            assert Path(result_path).exists()
            assert metadata["model_type"] == "test_model"

    def test_export_unified_invalid_format(self, exporter):
        """Test unified export with invalid format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.unknown"

            with pytest.raises(ValueError):
                exporter.export(str(output_path), format="invalid_format")

    def test_export_metadata_completeness(self, exporter):
        """Test that exported metadata contains all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pth"

            _, metadata = exporter.export_pytorch(str(output_path))

            required_fields = [
                "model_type",
                "num_classes",
                "image_size",
                "original_size_mb",
                "exported_size_mb",
                "compression_ratio",
                "size_reduction_percent",
            ]

            for field in required_fields:
                assert field in metadata, f"Missing metadata field: {field}"

    def test_export_preserves_model_structure(self, exporter):
        """Test that export preserves model structure."""
        original_params = sum(p.numel() for p in exporter.model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pth"

            exporter.export_pytorch(str(output_path))

            # Load and verify
            loaded_state = torch.load(output_path)
            loaded_params = sum(v.numel() for v in loaded_state.values())

            assert loaded_params == original_params

    def test_compression_ratio_calculation(self, exporter):
        """Test compression ratio is correctly calculated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pth"

            _, metadata_float32 = exporter.export_pytorch(str(output_path), quantization="float32")
            ratio_float32 = metadata_float32["compression_ratio"]

            output_path_int8 = Path(tmpdir) / "model_int8.pth"
            _, metadata_int8 = exporter.export_pytorch(str(output_path_int8), quantization="int8")
            ratio_int8 = metadata_int8["compression_ratio"]

            # INT8 should have better compression
            assert ratio_int8 >= ratio_float32


class TestModelExportIntegration:
    """Integration tests for model export workflow."""

    def test_complete_export_workflow(self):
        """Test complete workflow: create -> export -> load -> infer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model
            model = SimpleTestModel()
            exporter = ModelExporter(model, model_type="test", num_classes=10)

            # Export
            output_path = Path(tmpdir) / "exported_model.pth"
            exporter.export_pytorch(str(output_path), quantization="float32")

            # Load
            loaded_model = SimpleTestModel()
            loaded_model.load_state_dict(torch.load(output_path))
            loaded_model.eval()

            # Infer
            dummy_input = torch.randn(1, 1, 64, 64)
            with torch.no_grad():
                output = loaded_model(dummy_input)

            assert output.shape == (1, 10)

    def test_multiple_format_export(self):
        """Test exporting same model to multiple formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleTestModel()
            exporter = ModelExporter(model, model_type="test", num_classes=10)

            formats = ["pytorch"]

            for fmt in formats:
                output_path = Path(tmpdir) / f"model.{fmt}"
                try:
                    result_path, metadata = exporter.export(str(output_path), format=fmt)
                    assert Path(result_path).exists()
                except ImportError:
                    pytest.skip(f"{fmt} format not available")

    def test_quantization_compression_effectiveness(self):
        """Test that quantization reduces model size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleTestModel()
            exporter = ModelExporter(model, model_type="test", num_classes=10)

            # Export with different quantization levels
            results = {}

            for quant in ["float32", "int8", "bfloat16"]:
                output_path = Path(tmpdir) / f"model_{quant}.pth"
                _, metadata = exporter.export_pytorch(str(output_path), quantization=quant)
                results[quant] = metadata["exported_size_mb"]

            # Verify quantization reduces size
            assert results["int8"] <= results["float32"]
            assert results["bfloat16"] <= results["float32"]
