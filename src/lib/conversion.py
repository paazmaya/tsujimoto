"""
Model conversion utilities for export and quantization operations.

Provides reusable functions for:
- Model checkpoint loading with automatic class inference
- State dict quantization/dequantization
- Date extraction from model files
- Metadata handling
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def get_model_date(model_path: str) -> str:
    """Extract the date when the model was created from checkpoint file metadata.

    Returns ISO date string (YYYY-MM-DD) from model's modification time.

    Args:
        model_path: Path to model checkpoint file

    Returns:
        str: ISO date string (YYYY-MM-DD), or current date if extraction fails
    """
    try:
        if Path(model_path).exists():
            # Get file modification time
            mod_time = os.path.getmtime(model_path)
            date_obj = datetime.fromtimestamp(mod_time)
            return date_obj.strftime("%Y-%m-%d")
    except Exception as e:
        logger.warning(f"Could not extract date from model: {e}")

    # Fallback to current date if extraction fails
    return datetime.now().strftime("%Y-%m-%d")


def dequantize_state_dict(state_dict: dict) -> dict:
    """Dequantize INT8 tensors for ONNX compatibility.

    Converts quantized tensors back to float32 for ONNX export.

    Args:
        state_dict: PyTorch state dict that may contain quantized tensors

    Returns:
        dict: State dict with dequantized tensors
    """
    dequantized_state = {}
    quantized_count = 0

    for key, value in state_dict.items():
        if hasattr(value, "dequantize"):
            dequantized_state[key] = value.dequantize()
            quantized_count += 1
        else:
            dequantized_state[key] = value

    if quantized_count > 0:
        logger.info(f"✓ Dequantized {quantized_count} tensors for ONNX compatibility")

    return dequantized_state


def infer_num_classes_from_state_dict(state_dict: dict, default: int = 3036) -> int:
    """Infer number of classes from model state dict.

    Searches for classifier layer weights to determine output dimensions.

    Args:
        state_dict: PyTorch state dict
        default: Default num_classes if inference fails (default: 3036)

    Returns:
        int: Inferred number of classes

    Example:
        >>> num_classes = infer_num_classes_from_state_dict(model.state_dict())
        >>> print(f"Model has {num_classes} output classes")
    """
    classifier_weights = {}

    for key in state_dict.keys():
        if isinstance(state_dict[key], torch.Tensor) and "classifier" in key and "weight" in key:
            # Extract layer number if present (e.g., "classifier.4" -> 4)
            parts = key.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                layer_idx = int(parts[1])
                classifier_weights[layer_idx] = (key, state_dict[key].shape[0])

    # Use highest indexed classifier layer (typically the output layer)
    if classifier_weights:
        max_layer = max(classifier_weights.keys())
        key, num_classes = classifier_weights[max_layer]
        logger.info(f"  → Found output classifier in '{key}': {num_classes} classes")
        return num_classes

    logger.warning(f"  → Could not infer num_classes, using default: {default}")
    return default


def load_num_classes_from_config(model_path: str, model_type: str, default: int = 3036) -> int:
    """Load number of classes from model config file.

    Args:
        model_path: Path to model checkpoint
        model_type: Model type (used to find config file)
        default: Default num_classes if config not found (default: 3036)

    Returns:
        int: Number of classes from config or default
    """
    config_path = Path(model_path).parent / f"{model_type}_config.json"

    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config_dict = json.load(f)
                num_classes = config_dict.get("num_classes", default)
                logger.info(f"  → Found in config.json: {num_classes} classes")
                return num_classes
        except Exception as e:
            logger.warning(f"  → Failed to load config: {e}, using default: {default}")

    return default


def load_model_checkpoint(
    model_path: str,
    model_type: str,
) -> Tuple[torch.nn.Module, int, dict]:
    """Load model from checkpoint and infer num_classes.

    Handles both wrapped checkpoints (with 'model_state_dict') and raw state dicts.
    Automatically detects INT8 quantization and dequantizes if needed.

    Args:
        model_path: Path to model checkpoint
        model_type: Model type (hiercode, cnn, rnn, qat, vit, radical-rnn, hiercode-higita)

    Returns:
        Tuple of (model, num_classes, checkpoint_info)

    Raises:
        FileNotFoundError: If model file not found
        ValueError: If model type not supported

    Example:
        >>> model, num_classes, info = load_model_checkpoint(
        ...     "training/cnn/best_model.pth", "cnn"
        ... )
        >>> print(f"Model: {model_type}, Classes: {num_classes}")
    """
    # Import here to avoid circular imports
    from .config import (
        CNNConfig,
        HierCodeConfig,
        QATConfig,
        RadicalRNNConfig,
        RNNConfig,
        ViTConfig,
    )

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        logger.error(f"✗ Model path not found: {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info("→ Loading model from %s...", model_path)

    # Load checkpoint
    checkpoint = torch.load(model_path_obj, map_location="cpu")

    # Extract model_state_dict if checkpoint is wrapped
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    else:
        model_state_dict = checkpoint

    # Check if model is INT8 quantized
    is_quantized = any(hasattr(v, "dequantize") for v in model_state_dict.values())

    # Infer num_classes from state dict, fallback to config file
    num_classes = infer_num_classes_from_state_dict(model_state_dict)
    if num_classes == 3036:  # If using default
        num_classes = load_num_classes_from_config(model_path, model_type)

    # Import model architectures
    try:
        # Try importing from scripts directory (for conversion scripts)
        import sys
        from pathlib import Path as PathlibPath

        scripts_path = PathlibPath(__file__).parent.parent.parent / "scripts"
        if scripts_path.exists():
            sys.path.insert(0, str(scripts_path))

        from train_cnn_model import LightweightKanjiNet  # noqa: E402
        from train_hiercode import HierCodeClassifier  # noqa: E402
        from train_qat import QuantizableLightweightKanjiNet  # noqa: E402
        from train_radical_rnn import RadicalRNNClassifier  # noqa: E402
        from train_rnn import KanjiRNN  # noqa: E402
        from train_vit import VisionTransformer  # noqa: E402
    except ImportError as e:
        logger.error(f"Failed to import model architectures: {e}")
        raise

    # Create model based on type
    if model_type == "hiercode":
        config = HierCodeConfig(num_classes=num_classes)
        model = HierCodeClassifier(num_classes=num_classes, config=config)
    elif model_type == "hiercode-higita":
        config = HierCodeConfig(num_classes=num_classes)
        try:
            from train_hiercode_higita import HierCodeWithHiGITA

            model = HierCodeWithHiGITA(num_classes=num_classes)
        except ImportError:
            logger.error("HierCode-HiGITA not available")
            raise
    elif model_type == "cnn":
        config = CNNConfig(num_classes=num_classes)
        model = LightweightKanjiNet(num_classes=num_classes)
    elif model_type == "qat":
        config = QATConfig(num_classes=num_classes)
        model = QuantizableLightweightKanjiNet(num_classes=num_classes)
    elif model_type == "rnn":
        config = RNNConfig(num_classes=num_classes)
        model = KanjiRNN(num_classes=num_classes)
    elif model_type == "radical-rnn":
        config = RadicalRNNConfig(num_classes=num_classes)
        model = RadicalRNNClassifier(num_classes=num_classes, config=config)
    elif model_type == "vit":
        config = ViTConfig(num_classes=num_classes)
        model = VisionTransformer(num_classes=num_classes, config=config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Handle quantized models - dequantize for ONNX compatibility
    state_dict_to_load = model_state_dict
    if is_quantized:
        logger.info("→ Model is INT8 quantized, dequantizing for export...")
        state_dict_to_load = dequantize_state_dict(model_state_dict)

    # Load weights
    try:
        model.load_state_dict(state_dict_to_load)
    except RuntimeError:
        model.load_state_dict(state_dict_to_load, strict=False)
        logger.warning("⚠ Loaded model with strict=False (some keys may not match)")

    model.eval()

    return model, num_classes, {"is_quantized": is_quantized, "original_model": model_state_dict}


def calculate_model_size(model: torch.nn.Module, include_gradients: bool = False) -> float:
    """Calculate total model size in bytes.

    Sums the memory footprint of all model parameters, with optional gradient size.

    Args:
        model: PyTorch model
        include_gradients: Whether to include gradient memory (default: False)

    Returns:
        float: Total model size in bytes

    Example:
        >>> model = MyModel()
        >>> size_bytes = calculate_model_size(model)
        >>> size_mb = size_bytes / 1e6
        >>> print(f"Model size: {size_mb:.2f} MB")
    """
    size = 0
    for param in model.parameters():
        size += param.numel() * param.element_size()
        if include_gradients and param.grad is not None:
            size += param.grad.numel() * param.grad.element_size()
    return size


def calculate_compression_ratio(original_size: float, compressed_size: float) -> tuple:
    """Calculate compression ratio and percentage reduction.

    Args:
        original_size: Original size in bytes
        compressed_size: Compressed size in bytes

    Returns:
        Tuple of (ratio, percent_reduction)
        - ratio: Compression ratio (e.g., 4.0 means 4x compression)
        - percent_reduction: Size reduction as percentage (0-100)

    Example:
        >>> ratio, percent = calculate_compression_ratio(1000, 250)
        >>> print(f"{ratio:.1f}x compression, {percent:.1f}% reduction")
        4.0x compression, 75.0% reduction
    """
    if original_size <= 0:
        return 0.0, 0.0

    ratio = original_size / compressed_size
    percent = 100 * (1 - compressed_size / original_size)
    return ratio, percent


def quantize_model_int8(
    model: torch.nn.Module, device: str = "cuda"
) -> Tuple[torch.nn.Module, float, float]:
    """Apply INT8 dynamic quantization to model.

    Converts weights to 8-bit integers with ~4x size reduction.

    Args:
        model: PyTorch model to quantize
        device: Device to use (cuda recommended)

    Returns:
        Tuple of (quantized_model, original_size_bytes, quantized_size_bytes)

    Example:
        >>> model = train_model()
        >>> quant_model, orig_size, quant_size = quantize_model_int8(model)
        >>> ratio = orig_size / quant_size
        >>> print(f"Quantization: {ratio:.1f}x compression")
    """
    import warnings

    logger.info("🔄 Quantizing model to INT8 (dynamic)...")

    model = model.to(device)
    model.eval()

    # Calculate original size
    original_state = model.state_dict()
    original_size = sum(
        v.numel() * v.element_size() for v in original_state.values() if v is not None
    )
    logger.info("  → Original model size: %.2f MB", original_size / 1e6)

    # Dynamic quantization requires CPU
    logger.info("  → Applying INT8 quantization...")
    model_cpu = model.cpu()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )

    quantized_model = quantized_model.to(device)
    logger.info("✓ INT8 quantization complete")

    # For dynamic quantization, actual size measured after saving
    quantized_size = 0  # Will be updated when saved

    return quantized_model, original_size, quantized_size


def quantize_state_dict_int8(state_dict: dict) -> tuple:
    """Quantize state dict to INT8 format with scale/offset storage.

    Converts weights to 8-bit integers and stores scale factors separately.

    Args:
        state_dict: PyTorch state dict

    Returns:
        Tuple of (quantized_state_dict, metadata)

    Example:
        >>> state_dict = model.state_dict()
        >>> quant_state, meta = quantize_state_dict_int8(state_dict)
        >>> print(f"Quantized {meta['num_tensors']} tensors")
    """
    logger.info("  → Applying INT8 quantization to state dict...")
    quantized_state = {}
    quantized_count = 0

    for key, value in state_dict.items():
        if "weight" in key and value.dim() == 2:  # Linear layer weights
            # Compute quantization parameters
            min_val = value.min()
            max_val = value.max()
            scale = (max_val - min_val) / 255.0

            # Quantize to int8 range
            quantized = torch.round((value - min_val) / scale).clamp(0, 255).to(torch.uint8)

            # Store quantized weights
            quantized_state[key] = quantized

            # Store scale and offset
            scale_key = key.replace("weight", "weight_scale")
            zero_key = key.replace("weight", "weight_zero")
            quantized_state[scale_key] = scale.unsqueeze(0)
            quantized_state[zero_key] = min_val.unsqueeze(0)
            quantized_count += 1

        elif "bias" in key:
            # Keep biases in float32
            quantized_state[key] = value
        else:
            quantized_state[key] = value

    logger.info(f"    ✓ INT8 quantization complete - {quantized_count} tensors quantized")
    return quantized_state, {"quantization_type": "int8", "num_tensors": quantized_count}


def save_model_with_metadata(
    model: torch.nn.Module,
    output_path: str,
    model_info: dict,
    metadata: dict = None,
) -> tuple:
    """Save model checkpoint with accompanying metadata JSON.

    Args:
        model: PyTorch model to save
        output_path: Path to save model (.pth or .pt file)
        model_info: Dict with model information (model_type, num_classes, etc.)
        metadata: Optional additional metadata dict

    Returns:
        Tuple of (saved_path, file_size_bytes)

    Example:
        >>> info = {"model_type": "cnn", "num_classes": 3036}
        >>> saved_path, size = save_model_with_metadata(model, "model.pth", info)
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    logger.info("→ Saving model to %s...", output_path)
    torch.save(model.state_dict(), str(output_path_obj))

    # Get file size
    file_size = output_path_obj.stat().st_size
    logger.info("✓ Model saved: %.2f MB", file_size / 1e6)

    # Save metadata
    full_metadata = {
        "model_type": model_info.get("model_type", "unknown"),
        "num_classes": model_info.get("num_classes", 3036),
        "size_mb": file_size / 1e6,
        "size_bytes": file_size,
    }

    if metadata:
        full_metadata.update(metadata)

    metadata_path = output_path_obj.with_suffix(".json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(full_metadata, f, indent=2)

    logger.info("✓ Metadata saved: %s", metadata_path)

    return str(output_path_obj), file_size


def quantize_tensor_to_q4(tensor: torch.Tensor) -> Tuple[bytes, float, float, tuple]:
    """Quantize a single tensor to 4-bit (Q4) format for GGUF export.

    Packs two 4-bit values into each byte for 8x compression.

    Args:
        tensor: PyTorch tensor to quantize

    Returns:
        Tuple of (packed_bytes, scale, min_val, original_shape)

    Example:
        >>> tensor = torch.randn(1000, 100)
        >>> packed, scale, min_val, shape = quantize_tensor_to_q4(tensor)
        >>> print(f"Packed {shape} to {len(packed)} bytes")
    """
    import numpy as np

    orig_shape = tensor.shape
    tensor_flat = tensor.float().cpu().flatten().numpy()

    min_val = float(tensor_flat.min())
    max_val = float(tensor_flat.max())

    # Avoid division by zero
    if max_val == min_val:
        scale = 1.0
    else:
        scale = (max_val - min_val) / 15.0  # 4-bit has 16 values (0-15)

    # Quantize to 4-bit
    quantized = np.round((tensor_flat - min_val) / scale).clip(0, 15).astype(np.uint8)

    # Pack two 4-bit values into one byte
    packed = np.zeros(len(quantized) // 2 + (len(quantized) % 2), dtype=np.uint8)
    for i in range(0, len(quantized) - 1, 2):
        packed[i // 2] = (quantized[i] << 4) | quantized[i + 1]

    if len(quantized) % 2:
        packed[-1] = quantized[-1] << 4

    return bytes(packed), scale, min_val, orig_shape


def quantize_tensor_to_q8(tensor: torch.Tensor) -> Tuple[bytes, float, float, tuple]:
    """Quantize a single tensor to 8-bit (Q8) format for GGUF export.

    Args:
        tensor: PyTorch tensor to quantize

    Returns:
        Tuple of (quantized_bytes, scale, min_val, original_shape)
    """
    import numpy as np

    orig_shape = tensor.shape
    tensor_flat = tensor.float().cpu().flatten().numpy()

    min_val = float(tensor_flat.min())
    max_val = float(tensor_flat.max())

    if max_val == min_val:
        scale = 1.0
    else:
        scale = (max_val - min_val) / 255.0

    quantized = np.round((tensor_flat - min_val) / scale).clip(0, 255).astype(np.uint8)

    return bytes(quantized), scale, min_val, orig_shape


def quantize_tensor_to_f16(tensor: torch.Tensor) -> Tuple[bytes, float, float, tuple]:
    """Convert tensor to 16-bit float (F16) format for GGUF export.

    Args:
        tensor: PyTorch tensor to convert

    Returns:
        Tuple of (float16_bytes, scale, min_val, original_shape)
    """
    orig_shape = tensor.shape
    f16_array = tensor.half().cpu().numpy().astype(dtype="float16")
    return bytes(f16_array.tobytes()), 1.0, 0.0, orig_shape


def format_export_filename(base_name: str, model_date: str, quantization: str) -> str:
    """Format export filename with date and quantization method.

    Args:
        base_name: Base filename without extension (e.g., "cnn_model")
        model_date: ISO date string (YYYY-MM-DD)
        quantization: Quantization method (e.g., "int8", "q4_k", "nf4")

    Returns:
        str: Formatted filename (no extension)

    Example:
        >>> filename = format_export_filename("cnn", "2026-01-28", "int8")
        >>> print(filename)
        cnn_2026-01-28_int8
    """
    if quantization.lower() == "float32" or quantization.lower() == "f32":
        return f"{base_name}_{model_date}"
    else:
        return f"{base_name}_{model_date}_{quantization}"


def log_conversion_summary(
    original_size: float,
    converted_size: float,
    model_type: str,
    format_type: str,
    quantization: str = "float32",
) -> None:
    """Log a summary of model conversion results.

    Args:
        original_size: Original model size in bytes
        converted_size: Converted model size in bytes
        model_type: Type of model (cnn, rnn, etc.)
        format_type: Export format (onnx, safetensors, gguf)
        quantization: Quantization method used (default: float32)
    """
    ratio, percent = calculate_compression_ratio(original_size, converted_size)

    logger.info("=" * 60)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 60)
    logger.info("Model type: %s", model_type)
    logger.info("Export format: %s", format_type)
    logger.info("Quantization: %s", quantization)
    logger.info("-" * 60)
    logger.info("Original size: %.2f MB", original_size / 1e6)
    logger.info("Converted size: %.2f MB", converted_size / 1e6)
    logger.info("Compression: %.1fx (%+.1f%%)", ratio, -percent)
    logger.info("=" * 60)
