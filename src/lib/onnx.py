"""
ONNX conversion and optimization utilities.

Provides functions for exporting models to ONNX and converting between formats.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    import onnx

    try:
        import onnxruntime
    except ImportError:
        import onnxruntime_gpu as onnxruntime  # type: ignore

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX libraries not available. Install with: pip install onnx onnxruntime")


def export_to_onnx(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    output_path: str,
    opset_version: int = 14,
    use_external_data_format: bool = False,
) -> bool:
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        input_shape: Input tensor shape (e.g., (1, 1, 64, 64) for single 64x64 image)
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
        use_external_data_format: Use external data format for large models

    Returns:
        bool: True if export successful, False otherwise
    """
    if not ONNX_AVAILABLE:
        logger.error("ONNX export requires: pip install onnx onnxruntime")
        return False

    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create dummy input
        dummy_input = torch.randn(*input_shape)

        # Export model
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            use_external_data_format=use_external_data_format,
        )

        logger.info(f"✓ ONNX export successful: {output_path}")
        return True

    except Exception as e:
        logger.error(f"✗ ONNX export failed: {e}")
        return False


def validate_onnx_model(onnx_path: str) -> bool:
    """
    Validate ONNX model integrity.

    Args:
        onnx_path: Path to ONNX model

    Returns:
        bool: True if model is valid, False otherwise
    """
    if not ONNX_AVAILABLE:
        logger.error("ONNX validation requires: pip install onnx")
        return False

    try:
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info(f"✓ ONNX model is valid: {onnx_path}")
        return True
    except Exception as e:
        logger.error(f"✗ ONNX validation failed: {e}")
        return False


def get_onnx_model_info(onnx_path: str) -> Optional[Dict]:
    """
    Extract information from ONNX model.

    Args:
        onnx_path: Path to ONNX model

    Returns:
        dict: Model info including inputs, outputs, and parameters, or None if failed
    """
    if not ONNX_AVAILABLE:
        logger.error("ONNX inspection requires: pip install onnx")
        return None

    try:
        onnx_model = onnx.load(str(onnx_path))
        graph = onnx_model.graph

        # Extract input/output info
        inputs = [
            {
                "name": inp.name,
                "shape": [dim.dim_value for dim in inp.type.tensor_type.shape.dim],
                "dtype": inp.type.tensor_type.data_type,
            }
            for inp in graph.input
        ]

        outputs = [
            {
                "name": out.name,
                "shape": [dim.dim_value for dim in out.type.tensor_type.shape.dim],
                "dtype": out.type.tensor_type.data_type,
            }
            for out in graph.output
        ]

        info = {
            "inputs": inputs,
            "outputs": outputs,
            "num_parameters": len(graph.initializer),
        }

        logger.info(f"ONNX Model Info: {info}")
        return info

    except Exception as e:
        logger.error(f"Failed to extract ONNX info: {e}")
        return None


def test_onnx_inference(onnx_path: str, input_data: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Test ONNX model inference.

    Args:
        onnx_path: Path to ONNX model
        input_data: Input tensor for testing

    Returns:
        Output tensor from inference, or None if failed
    """
    if not ONNX_AVAILABLE:
        logger.error("ONNX inference requires: pip install onnxruntime")
        return None

    try:
        session = onnxruntime.InferenceSession(str(onnx_path))

        # Prepare input
        input_name = session.get_inputs()[0].name
        input_np = input_data.cpu().numpy()

        # Run inference
        outputs = session.run(None, {input_name: input_np})
        output_tensor = torch.from_numpy(outputs[0])

        logger.info(f"✓ ONNX inference successful. Output shape: {output_tensor.shape}")
        return output_tensor

    except Exception as e:
        logger.error(f"✗ ONNX inference failed: {e}")
        return None
