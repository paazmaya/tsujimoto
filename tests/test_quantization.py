"""
Tests for unified model quantization utilities.

Tests for:
- INT8 quantization
- 4-bit NF4 quantization (requires BitsAndBytes)
- 4-bit FP4 quantization (requires BitsAndBytes)
- BFloat16 conversion
- Unified quantize_model() interface
- Compression ratio calculations
"""

import tempfile
from pathlib import Path
from typing import Tuple

import pytest
import torch
import torch.nn as nn

from src.lib.conversion import (
    calculate_compression_ratio,
    quantize_model,
    quantize_model_bfloat16,
    quantize_model_int8,
    quantize_tensor_to_f16,
    quantize_tensor_to_q4,
    quantize_tensor_to_q8,
)


class SimpleTestModel(nn.Module):
    """Simple test model for quantization testing."""

    def __init__(self, num_classes: int = 10):
        """Initialize simple model."""
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TestCompressionRatio:
    """Test compression ratio calculation."""

    def test_compression_4x(self):
        """Test 4x compression ratio calculation."""
        ratio, percent = calculate_compression_ratio(1000, 250)
        assert ratio == pytest.approx(4.0, rel=0.01)
        assert percent == pytest.approx(75.0, rel=0.1)

    def test_compression_2x(self):
        """Test 2x compression ratio calculation."""
        ratio, percent = calculate_compression_ratio(1000, 500)
        assert ratio == pytest.approx(2.0, rel=0.01)
        assert percent == pytest.approx(50.0, rel=0.1)

    def test_compression_zero_size(self):
        """Test compression with zero original size."""
        ratio, percent = calculate_compression_ratio(0, 250)
        assert ratio == 0.0
        assert percent == 0.0

    def test_compression_no_reduction(self):
        """Test when sizes are equal."""
        ratio, percent = calculate_compression_ratio(1000, 1000)
        assert ratio == pytest.approx(1.0, rel=0.01)
        assert percent == pytest.approx(0.0, rel=0.1)


class TestTensorQuantization:
    """Test individual tensor quantization functions."""

    def test_quantize_tensor_to_q4(self):
        """Test Q4 quantization of a tensor."""
        tensor = torch.randn(100, 50)
        packed_bytes, scale, min_val, shape = quantize_tensor_to_q4(tensor)

        # Check return types
        assert isinstance(packed_bytes, bytes)
        assert isinstance(scale, float)
        assert isinstance(min_val, float)
        assert shape == (100, 50)

        # Check size reduction (roughly 8x for 4-bit)
        original_bytes = tensor.numel() * 4  # float32
        compressed_ratio = original_bytes / len(packed_bytes)
        assert compressed_ratio > 4.0  # At least 4x compression

    def test_quantize_tensor_to_q8(self):
        """Test Q8 quantization of a tensor."""
        tensor = torch.randn(100, 50)
        quantized_bytes, scale, min_val, shape = quantize_tensor_to_q8(tensor)

        assert isinstance(quantized_bytes, bytes)
        assert isinstance(scale, float)
        assert isinstance(min_val, float)
        assert shape == (100, 50)

        # Q8 should give 4x compression
        original_bytes = tensor.numel() * 4
        compressed_ratio = original_bytes / len(quantized_bytes)
        assert compressed_ratio > 3.0

    def test_quantize_tensor_to_f16(self):
        """Test F16 conversion of a tensor."""
        tensor = torch.randn(100, 50)
        f16_bytes, scale, min_val, shape = quantize_tensor_to_f16(tensor)

        assert isinstance(f16_bytes, bytes)
        assert scale == 1.0
        assert min_val == 0.0
        assert shape == (100, 50)

        # F16 should give 2x compression
        original_bytes = tensor.numel() * 4
        compressed_ratio = original_bytes / len(f16_bytes)
        assert compressed_ratio == pytest.approx(2.0, rel=0.01)

    def test_quantize_constant_tensor(self):
        """Test quantization of constant (uniform) tensor."""
        tensor = torch.ones(100, 50) * 5.0
        packed_bytes, scale, min_val, shape = quantize_tensor_to_q4(tensor)

        # Should still work (scale = 0 case)
        assert isinstance(packed_bytes, bytes)
        assert shape == (100, 50)


class TestINT8Quantization:
    """Test INT8 quantization."""

    @pytest.fixture
    def test_model(self):
        """Create test model."""
        return SimpleTestModel()

    def test_int8_quantization_cpu(self, test_model):
        """Test INT8 quantization on CPU."""
        quantized_model, orig_size, quant_size = quantize_model_int8(test_model, device="cpu")

        assert quantized_model is not None
        assert orig_size > 0
        assert quant_size is None  # For dynamic quantization, actual size measured after save

    def test_int8_quantization_cuda_available(self, test_model):
        """Test INT8 quantization on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        quantized_model, orig_size, quant_size = quantize_model_int8(test_model, device="cuda")
        assert quantized_model is not None
        assert orig_size > 0

    def test_int8_model_evaluation(self, test_model):
        """Test that quantized model can perform inference."""
        quantized_model, _, _ = quantize_model_int8(test_model, device="cpu")

        # Test forward pass
        dummy_input = torch.randn(1, 784)
        with torch.no_grad():
            output = quantized_model(dummy_input)

        assert output.shape == (1, 10)
        assert not torch.isnan(output).any()


class TestBFloat16Quantization:
    """Test BFloat16 conversion."""

    @pytest.fixture
    def test_model(self):
        """Create test model."""
        return SimpleTestModel()

    def test_bfloat16_conversion(self, test_model):
        """Test BFloat16 conversion."""
        bf16_model = quantize_model_bfloat16(test_model)

        # Check that model parameters are in BFloat16
        for param in bf16_model.parameters():
            assert param.dtype == torch.bfloat16

    def test_bfloat16_inference(self, test_model):
        """Test inference with BFloat16 model."""
        bf16_model = quantize_model_bfloat16(test_model)

        dummy_input = torch.randn(1, 784).to(torch.bfloat16)
        with torch.no_grad():
            output = bf16_model(dummy_input)

        assert output.shape == (1, 10)
        assert output.dtype == torch.bfloat16


class TestUnifiedQuantizeModel:
    """Test unified quantize_model interface."""

    @pytest.fixture
    def test_model(self):
        """Create test model."""
        return SimpleTestModel()

    def test_quantize_int8_format(self, test_model):
        """Test unified API with INT8 format."""
        quant_model, metadata = quantize_model(test_model, "int8", device="cpu")

        assert quant_model is not None
        assert metadata["quantization_format"] == "int8"
        assert "compression_ratio" in metadata
        assert "original_size_mb" in metadata
        assert "quantized_size_mb" in metadata

    def test_quantize_bfloat16_format(self, test_model):
        """Test unified API with BFloat16 format."""
        quant_model, metadata = quantize_model(test_model, "bfloat16", device="cpu")

        assert quant_model is not None
        assert metadata["quantization_format"] == "bfloat16"
        assert metadata["compression_target"] == "~2x"

    def test_quantize_no_quantization(self, test_model):
        """Test unified API with no quantization."""
        quant_model, metadata = quantize_model(test_model, "float32", device="cpu")

        assert quant_model is not None
        assert metadata["quantization_format"] == "float32"
        assert metadata["compression_ratio"] == pytest.approx(1.0, rel=0.01)

    def test_quantize_none_format(self, test_model):
        """Test unified API with 'none' format."""
        quant_model, metadata = quantize_model(test_model, "none", device="cpu")

        assert quant_model is not None
        assert metadata["quantization_format"] == "none"

    def test_quantize_invalid_format(self, test_model):
        """Test unified API with invalid format."""
        with pytest.raises(ValueError):
            quantize_model(test_model, "invalid_format")

    def test_quantize_metadata_completeness(self, test_model):
        """Test that metadata contains all required fields."""
        quant_model, metadata = quantize_model(test_model, "int8", device="cpu")

        required_fields = [
            "quantization_format",
            "device",
            "method",
            "compression_target",
            "original_size_mb",
            "quantized_size_mb",
            "compression_ratio",
            "size_reduction_percent",
        ]

        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"

    def test_quantize_preserves_architecture(self, test_model):
        """Test that quantization preserves model architecture."""
        original_params = sum(p.numel() for p in test_model.parameters())

        quant_model, _ = quantize_model(test_model, "int8", device="cpu")
        quantized_params = sum(p.numel() for p in quant_model.parameters())

        assert original_params == quantized_params

    def test_quantize_inference_shapes(self, test_model):
        """Test that quantized model produces correct output shapes."""
        quant_model, _ = quantize_model(test_model, "int8", device="cpu")

        # Test different batch sizes
        for batch_size in [1, 8, 16]:
            dummy_input = torch.randn(batch_size, 784)
            with torch.no_grad():
                output = quant_model(dummy_input)

            assert output.shape == (batch_size, 10)
            assert not torch.isnan(output).any()


class TestQuantizationBitsAndBytes:
    """Tests for BitsAndBytes 4-bit quantization (requires bitsandbytes library)."""

    @pytest.fixture
    def has_bitsandbytes(self):
        """Check if BitsAndBytes is available."""
        try:
            import bitsandbytes  # noqa: F401

            return True
        except ImportError:
            return False

    @pytest.fixture
    def test_model(self):
        """Create test model."""
        return SimpleTestModel()

    def test_quantize_4bit_nf4_available(self, test_model, has_bitsandbytes):
        """Test 4-bit NF4 quantization if BitsAndBytes available."""
        if not has_bitsandbytes:
            pytest.skip("BitsAndBytes not available")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        quant_model, metadata = quantize_model(test_model, "4bit_nf4", device="cuda")

        assert quant_model is not None
        assert metadata["quantization_format"] == "4bit_nf4"
        assert "4-bit NF4" in metadata["method"]

    def test_quantize_4bit_fp4_available(self, test_model, has_bitsandbytes):
        """Test 4-bit FP4 quantization if BitsAndBytes available."""
        if not has_bitsandbytes:
            pytest.skip("BitsAndBytes not available")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        quant_model, metadata = quantize_model(test_model, "4bit_fp4", device="cuda")

        assert quant_model is not None
        assert metadata["quantization_format"] == "4bit_fp4"
        assert "4-bit FP4" in metadata["method"]

    def test_quantize_4bit_nf4_missing_library(self, test_model, has_bitsandbytes):
        """Test error handling when BitsAndBytes missing."""
        if has_bitsandbytes:
            pytest.skip("BitsAndBytes is available")

        with pytest.raises((ImportError, ModuleNotFoundError)):
            quantize_model(test_model, "4bit_nf4", device="cpu")


class TestIntegrationQuantization:
    """Integration tests for quantization workflow."""

    def test_complete_quantization_workflow(self):
        """Test complete workflow: train -> quantize -> save -> load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "quantized_model.pth"

            # Create and quantize model
            original_model = SimpleTestModel()
            quant_model, metadata = quantize_model(original_model, "int8", device="cpu")

            # Save quantized model
            torch.save(quant_model.state_dict(), model_path)

            # Verify file was created
            assert model_path.exists()

            # Load and test
            loaded_model = SimpleTestModel()
            loaded_model.load_state_dict(torch.load(model_path))
            loaded_model.eval()

            # Test inference
            dummy_input = torch.randn(1, 784)
            with torch.no_grad():
                output = loaded_model(dummy_input)

            assert output.shape == (1, 10)

    def test_multiple_quantization_formats(self):
        """Test quantizing same model with different formats."""
        model = SimpleTestModel()
        formats = ["float32", "int8", "bfloat16"]

        results = {}
        for fmt in formats:
            quant_model, metadata = quantize_model(model, fmt, device="cpu")
            results[fmt] = metadata

            # All should produce valid quantization
            assert "compression_ratio" in metadata

        # Verify different compression ratios
        assert results["float32"]["compression_ratio"] <= results["int8"]["compression_ratio"]
        assert results["bfloat16"]["compression_ratio"] <= results["int8"]["compression_ratio"]
