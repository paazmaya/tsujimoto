"""
Unified model export interface supporting multiple formats.

Consolidates model conversion logic from convert_to_*.py scripts.
Provides single entry point for exporting to GGUF, ONNX, SafeTensors, or PyTorch.

Example:
    >>> from src.lib.model_export import ModelExporter
    >>> exporter = ModelExporter(model, num_classes=43427)
    >>> exporter.export_onnx('model.onnx')
    >>> exporter.export_safetensors('model.safetensors', quantization='int8')
    >>> exporter.export_gguf('model.gguf', quantization='q4_k')
"""

import json
import logging
import struct
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from src.lib.conversion import (
    calculate_compression_ratio,
    dequantize_state_dict,
    get_model_date,
    log_conversion_summary,
    quantize_model,
    quantize_state_dict_int8,
    quantize_tensor_to_f16,
    quantize_tensor_to_q4,
    quantize_tensor_to_q8,
)
from src.lib.onnx import validate_onnx_model

logger = logging.getLogger(__name__)


def _write_gguf_string(buf: bytearray, text: str) -> None:
    """Append a GGUF-encoded UTF-8 string (uint64 length + bytes) to buffer."""
    encoded = text.encode("utf-8")
    buf.extend(struct.pack("<Q", len(encoded)))
    buf.extend(encoded)


def _write_gguf_kv(buf: bytearray, key: str, value_type: int, packed_value: bytes) -> None:
    """Append a GGUF metadata key-value entry to buffer."""
    _write_gguf_string(buf, key)
    buf.extend(struct.pack("<I", value_type))
    buf.extend(packed_value)


class ModelExporter:
    """Unified model exporter supporting multiple formats."""

    SUPPORTED_FORMATS = ["pytorch", "onnx", "safetensors", "gguf"]
    GGUF_QUANTIZATION_OPTIONS = ["f32", "f16", "q8_0", "q6_k", "q5_k", "q4_k", "q3_k", "q2_k"]

    def __init__(
        self,
        model: nn.Module,
        model_type: str = "unknown",
        num_classes: int = 43427,
        device: str = "cuda",
        image_size: int = 64,
    ):
        """
        Initialize model exporter.

        Args:
            model: PyTorch model to export
            model_type: Type of model (cnn, rnn, vit, etc.)
            num_classes: Number of output classes
            device: Device for computations (cuda or cpu)
            image_size: Input image size (default: 64)
        """
        self.model = model
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = device
        self.image_size = image_size
        self.model_date = get_model_date(str(model))

        # Calculate original model size
        self.original_state = model.state_dict()
        self.original_size = sum(
            v.numel() * v.element_size() for v in self.original_state.values() if v is not None
        )

    def export_pytorch(
        self,
        output_path: str,
        quantization: str = "float32",
        save_metadata: bool = True,
    ) -> Tuple[str, Dict]:
        """
        Export model to PyTorch format with optional quantization.

        Args:
            output_path: Path to save model (.pth or .pt)
            quantization: Quantization format (float32, int8, 4bit_nf4, 4bit_fp4, bfloat16)
            save_metadata: Whether to save accompanying metadata JSON

        Returns:
            Tuple of (output_path, metadata_dict)
        """
        logger.info("=" * 70)
        logger.info(f"EXPORTING TO PYTORCH ({quantization})")
        logger.info("=" * 70)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Quantize if needed
        export_model = self.model
        quant_metadata = {}

        if quantization not in ("float32", "none"):
            export_model, quant_metadata = quantize_model(self.model, quantization, self.device)

        # Save model
        torch.save(export_model.state_dict(), str(output_path))
        exported_size = output_path.stat().st_size

        # Create metadata
        metadata = {
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "image_size": self.image_size,
            "quantization": quantization,
            "export_date": self.model_date,
            "original_size_mb": self.original_size / 1e6,
            "exported_size_mb": exported_size / 1e6,
        }

        if quant_metadata:
            metadata.update(quant_metadata)

        # Calculate compression
        ratio, percent = calculate_compression_ratio(self.original_size, exported_size)
        metadata["compression_ratio"] = round(ratio, 2)
        metadata["size_reduction_percent"] = round(percent, 1)

        # Save metadata if requested
        if save_metadata:
            metadata_path = output_path.with_suffix(".json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"✓ Metadata saved: {metadata_path}")

        log_conversion_summary(
            self.original_size, exported_size, self.model_type, "pytorch", quantization
        )

        return str(output_path), metadata

    def export_onnx(
        self,
        output_path: str,
        quantization: str = "float32",
        opset_version: int = 14,
        save_metadata: bool = True,
    ) -> Tuple[str, Dict]:
        """
        Export model to ONNX format.

        Args:
            output_path: Path to save ONNX model (.onnx)
            quantization: Quantization format for preprocessing (float32, int8, bfloat16)
            opset_version: ONNX opset version
            save_metadata: Whether to save accompanying metadata JSON

        Returns:
            Tuple of (output_path, metadata_dict)

        Note:
            ONNX export requires dequantized model for compatibility.
            Quantization is applied before export but may not be preserved in ONNX.
        """
        logger.info("=" * 70)
        logger.info(f"EXPORTING TO ONNX ({quantization})")
        logger.info("=" * 70)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare model for ONNX
        export_model = self.model
        export_model.eval()

        # Dequantize if model is quantized
        state_dict = export_model.state_dict()
        state_dict = dequantize_state_dict(state_dict)
        export_model.load_state_dict(state_dict)

        # Move to CPU for ONNX export
        export_model = export_model.cpu()

        # Create dummy input for export
        dummy_input = torch.randn(1, 1, self.image_size, self.image_size)

        try:
            # Export to ONNX
            torch.onnx.export(
                export_model,
                dummy_input,
                str(output_path),
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )

            # Validate ONNX model
            if validate_onnx_model(str(output_path)):
                logger.info(f"✓ ONNX export successful: {output_path}")
            else:
                logger.warning("⚠ ONNX model may have validation issues")

        except Exception as e:
            logger.error(f"✗ ONNX export failed: {e}")
            raise

        exported_size = output_path.stat().st_size

        # Create metadata
        metadata = {
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "image_size": self.image_size,
            "export_format": "onnx",
            "opset_version": opset_version,
            "export_date": self.model_date,
            "original_size_mb": self.original_size / 1e6,
            "exported_size_mb": exported_size / 1e6,
        }

        ratio, percent = calculate_compression_ratio(self.original_size, exported_size)
        metadata["compression_ratio"] = round(ratio, 2)
        metadata["size_reduction_percent"] = round(percent, 1)

        # Save metadata if requested
        if save_metadata:
            metadata_path = output_path.with_suffix(".json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"✓ Metadata saved: {metadata_path}")

        log_conversion_summary(
            self.original_size, exported_size, self.model_type, "onnx", "float32"
        )

        return str(output_path), metadata

    def export_safetensors(
        self,
        output_path: str,
        quantization: str = "float32",
        save_metadata: bool = True,
    ) -> Tuple[str, Dict]:
        """
        Export model to SafeTensors format.

        Args:
            output_path: Path to save SafeTensors model (.safetensors)
            quantization: Quantization format (float32, int8, bfloat16, 4bit_nf4, 4bit_fp4)
            save_metadata: Whether to save accompanying metadata JSON

        Returns:
            Tuple of (output_path, metadata_dict)
        """
        logger.info("=" * 70)
        logger.info(f"EXPORTING TO SAFETENSORS ({quantization})")
        logger.info("=" * 70)

        try:
            from safetensors.torch import save_file
        except ImportError as e:
            logger.error(f"✗ SafeTensors not installed: {e}")
            logger.info("   Install with: pip install safetensors")
            raise

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Quantize if needed
        export_model = self.model
        quant_metadata: Dict[str, Any] = {}

        if quantization not in ("float32", "none"):
            if quantization == "int8":
                # SafeTensors requires pure torch.Tensor dict (not dynamic QuantizedLinear packed params)
                raw_state = dequantize_state_dict(self.model.state_dict())
                state_dict, quant_metadata = quantize_state_dict_int8(raw_state)
            else:
                export_model, quant_metadata = quantize_model(self.model, quantization, self.device)
                state_dict = export_model.state_dict()
        else:
            state_dict = dequantize_state_dict(export_model.state_dict())

        # Ensure all tensors are contiguous torch.Tensor instances for SafeTensors
        clean_state_dict = {
            k: v.contiguous() for k, v in state_dict.items() if isinstance(v, torch.Tensor)
        }

        # Save with SafeTensors
        save_file(clean_state_dict, str(output_path))

        exported_size = output_path.stat().st_size

        # Create metadata
        metadata = {
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "image_size": self.image_size,
            "export_format": "safetensors",
            "quantization": quantization,
            "export_date": self.model_date,
            "original_size_mb": self.original_size / 1e6,
            "exported_size_mb": exported_size / 1e6,
        }

        if quant_metadata:
            metadata.update(quant_metadata)

        ratio, percent = calculate_compression_ratio(self.original_size, exported_size)
        metadata["compression_ratio"] = round(ratio, 2)
        metadata["size_reduction_percent"] = round(percent, 1)

        # Save metadata if requested
        if save_metadata:
            metadata_path = output_path.with_suffix(".json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"✓ Metadata saved: {metadata_path}")

        log_conversion_summary(
            self.original_size, exported_size, self.model_type, "safetensors", quantization
        )

        return str(output_path), metadata

    def export_gguf(
        self,
        output_path: str,
        quantization: str = "q4_k",
        save_metadata: bool = True,
    ) -> Tuple[str, Dict]:
        """
        Export model to GGUF format for CPU inference.

        GGUF is optimized for CPU inference with tools like llama.cpp.

        Quantization options:
        - f32: 32-bit float (no quantization, largest)
        - f16: 16-bit float (2x compression)
        - q8_0: 8-bit quantization (4x compression)
        - q6_k: 6-bit K-quant (8x compression, high quality)
        - q5_k: 5-bit K-quant (10x compression)
        - q4_k: 4-bit K-quant (12x compression, recommended)
        - q3_k: 3-bit K-quant (16x compression)
        - q2_k: 2-bit K-quant (20x compression, lossy)

        Args:
            output_path: Path to save GGUF model (.gguf)
            quantization: Quantization format (default: q4_k)
            save_metadata: Whether to save accompanying metadata JSON

        Returns:
            Tuple of (output_path, metadata_dict)
        """
        logger.info("=" * 70)
        logger.info(f"EXPORTING TO GGUF ({quantization})")
        logger.info("=" * 70)

        if quantization not in self.GGUF_QUANTIZATION_OPTIONS:
            raise ValueError(
                f"Invalid GGUF quantization '{quantization}'. "
                f"Must be one of: {', '.join(self.GGUF_QUANTIZATION_OPTIONS)}"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare model
        export_model = self.model.cpu()
        export_model.eval()

        state_dict = dequantize_state_dict(export_model.state_dict())
        tensor_items = [(k, v) for k, v in state_dict.items() if isinstance(v, torch.Tensor)]

        logger.info("✓ Model prepared for GGUF export")
        logger.info(f"  Quantization: {quantization}")
        logger.info(f"  Number of tensors: {len(tensor_items)}")

        # GGUF v3 binary serialization
        # GGML tensor type codes: F32=0, F16=1, Q4_0=2, Q8_0=8, Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14
        ggml_type_map = {
            "f32": 0,
            "f16": 1,
            "q8_0": 8,
            "q2_k": 10,
            "q3_k": 11,
            "q4_k": 12,
            "q5_k": 13,
            "q6_k": 14,
        }
        target_ggml_type = ggml_type_map[quantization]

        header_buf = bytearray()
        # Magic 'GGUF' + Version 3 (uint32) + tensor_count (uint64) + kv_count (uint64)
        header_buf.extend(b"GGUF")
        header_buf.extend(struct.pack("<I", 3))
        header_buf.extend(struct.pack("<Q", len(tensor_items)))

        # Metadata KV pairs (5 entries)
        # GGUF value types: UINT32=4, FLOAT32=6, STRING=8
        header_buf.extend(struct.pack("<Q", 5))

        def _pack_gguf_str(s: str) -> bytes:
            b = s.encode("utf-8")
            return struct.pack("<Q", len(b)) + b

        _write_gguf_kv(header_buf, "general.architecture", 8, _pack_gguf_str(self.model_type))
        _write_gguf_kv(header_buf, "general.quantization", 8, _pack_gguf_str(quantization))
        _write_gguf_kv(header_buf, "kanji.num_classes", 4, struct.pack("<I", int(self.num_classes)))
        _write_gguf_kv(header_buf, "kanji.image_size", 4, struct.pack("<I", int(self.image_size)))
        _write_gguf_kv(header_buf, "general.file_type", 4, struct.pack("<I", target_ggml_type))

        # Encode tensor data and build tensor info records
        tensor_infos_buf = bytearray()
        tensor_data_buf = bytearray()
        alignment = 32

        for name, tensor in tensor_items:
            # Quantize / encode tensor payload
            if quantization == "f32" or tensor.ndim < 2:
                raw_bytes = tensor.detach().float().cpu().numpy().tobytes()
                ggml_type = 0  # F32 for 1D biases/norms or f32 mode
            elif quantization == "f16":
                raw_bytes, _, _, _ = quantize_tensor_to_f16(tensor.detach())
                ggml_type = 1
            elif quantization == "q8_0":
                q_bytes, scale, min_val, _ = quantize_tensor_to_q8(tensor.detach())
                raw_bytes = struct.pack("<ff", scale, min_val) + q_bytes
                ggml_type = 8
            else:
                # 4-bit / K-quant family
                q_bytes, scale, min_val, _ = quantize_tensor_to_q4(tensor.detach())
                raw_bytes = struct.pack("<ff", scale, min_val) + q_bytes
                ggml_type = target_ggml_type

            # Align offset in tensor_data_buf to 32 bytes
            pad = (alignment - (len(tensor_data_buf) % alignment)) % alignment
            if pad:
                tensor_data_buf.extend(b"\x00" * pad)
            offset = len(tensor_data_buf)
            tensor_data_buf.extend(raw_bytes)

            # Write tensor descriptor: name, n_dims (uint32), dims (uint64 * n_dims), type (uint32), offset (uint64)
            _write_gguf_string(tensor_infos_buf, name)
            shape = tuple(tensor.shape) if tensor.ndim > 0 else (1,)
            tensor_infos_buf.extend(struct.pack("<I", len(shape)))
            for dim in shape:
                tensor_infos_buf.extend(struct.pack("<Q", int(dim)))
            tensor_infos_buf.extend(struct.pack("<I", ggml_type))
            tensor_infos_buf.extend(struct.pack("<Q", offset))

        # Combine header + tensor_infos, pad to alignment, then append tensor_data_buf
        combined_header = header_buf + tensor_infos_buf
        header_pad = (alignment - (len(combined_header) % alignment)) % alignment
        if header_pad:
            combined_header.extend(b"\x00" * header_pad)

        with open(output_path, "wb") as f:
            f.write(combined_header)
            f.write(tensor_data_buf)

        exported_size = output_path.stat().st_size

        # Create metadata
        metadata = {
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "image_size": self.image_size,
            "export_format": "gguf",
            "quantization": quantization,
            "export_date": self.model_date,
            "original_size_mb": self.original_size / 1e6,
            "exported_size_mb": exported_size / 1e6,
        }

        ratio, percent = calculate_compression_ratio(self.original_size, exported_size)
        metadata["compression_ratio"] = round(ratio, 2)
        metadata["size_reduction_percent"] = round(percent, 1)

        # Save metadata if requested
        if save_metadata:
            metadata_path = output_path.with_suffix(".json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"✓ Metadata saved: {metadata_path}")

        log_conversion_summary(
            self.original_size, exported_size, self.model_type, "gguf", quantization
        )

        return str(output_path), metadata

    def export(
        self,
        output_path: str,
        export_format: str = "pytorch",
        quantization: str = "float32",
        **kwargs,
    ) -> Tuple[str, Dict]:
        """
        Export model to specified format with unified interface.

        Args:
            output_path: Path to save model
            export_format: Export format (pytorch, onnx, safetensors, gguf)
            quantization: Quantization format specific to export format
            **kwargs: Additional format-specific arguments

        Returns:
            Tuple of (output_path, metadata_dict)

        Raises:
            ValueError: If format not recognized
        """
        export_format = export_format.lower()

        if export_format == "pytorch":
            return self.export_pytorch(output_path, quantization, **kwargs)
        elif export_format == "onnx":
            return self.export_onnx(output_path, quantization, **kwargs)
        elif export_format == "safetensors":
            return self.export_safetensors(output_path, quantization, **kwargs)
        elif export_format == "gguf":
            return self.export_gguf(output_path, quantization, **kwargs)
        else:
            raise ValueError(
                f"Unknown export format '{export_format}'. "
                f"Must be one of: {', '.join(self.SUPPORTED_FORMATS)}"
            )
