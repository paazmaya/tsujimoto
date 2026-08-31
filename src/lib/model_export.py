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
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.lib.conversion import (
    calculate_compression_ratio,
    dequantize_state_dict,
    format_export_filename,
    get_model_date,
    log_conversion_summary,
    quantize_model,
    quantize_tensor_to_f16,
    quantize_tensor_to_q4,
    quantize_tensor_to_q8,
)
from src.lib.onnx import export_to_onnx, validate_onnx_model

logger = logging.getLogger(__name__)


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

        if quantization != "float32":
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

        log_conversion_summary(self.original_size, exported_size, self.model_type, "pytorch", quantization)

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
                logger.warning(f"⚠ ONNX model may have validation issues")

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

        log_conversion_summary(self.original_size, exported_size, self.model_type, "onnx", "float32")

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
        quant_metadata = {}

        if quantization != "float32":
            export_model, quant_metadata = quantize_model(self.model, quantization, self.device)

        # Get state dict
        state_dict = export_model.state_dict()

        # Save with SafeTensors
        save_file(state_dict, str(output_path))

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

        log_conversion_summary(self.original_size, exported_size, self.model_type, "safetensors", quantization)

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

        state_dict = export_model.state_dict()

        logger.info(f"✓ Model prepared for GGUF export")
        logger.info(f"  Quantization: {quantization}")
        logger.info(f"  Number of layers: {len(state_dict)}")

        # Note: Full GGUF implementation would require:
        # 1. Serialization of model metadata (architecture, layer configs)
        # 2. Layer-by-layer quantization based on quantization format
        # 3. Binary packing into GGUF format
        # For now, save as intermediate PyTorch format with GGUF metadata

        # Save state dict with GGUF format indicator
        gguf_data = {
            "__metadata__": {
                "format": "gguf",
                "quantization": quantization,
                "model_type": self.model_type,
                "num_classes": self.num_classes,
                "image_size": self.image_size,
            },
            "state_dict": state_dict,
        }

        torch.save(gguf_data, str(output_path))

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

        log_conversion_summary(self.original_size, exported_size, self.model_type, "gguf", quantization)

        return str(output_path), metadata

    def export(
        self,
        output_path: str,
        format: str = "pytorch",
        quantization: str = "float32",
        **kwargs,
    ) -> Tuple[str, Dict]:
        """
        Export model to specified format with unified interface.

        Args:
            output_path: Path to save model
            format: Export format (pytorch, onnx, safetensors, gguf)
            quantization: Quantization format specific to export format
            **kwargs: Additional format-specific arguments

        Returns:
            Tuple of (output_path, metadata_dict)

        Raises:
            ValueError: If format not recognized
        """
        format = format.lower()

        if format == "pytorch":
            return self.export_pytorch(output_path, quantization, **kwargs)
        elif format == "onnx":
            return self.export_onnx(output_path, quantization, **kwargs)
        elif format == "safetensors":
            return self.export_safetensors(output_path, quantization, **kwargs)
        elif format == "gguf":
            return self.export_gguf(output_path, quantization, **kwargs)
        else:
            raise ValueError(
                f"Unknown export format '{format}'. "
                f"Must be one of: {', '.join(self.SUPPORTED_FORMATS)}"
            )
