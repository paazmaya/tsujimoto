#!/usr/bin/env python3
"""
ONNX Conversion for ETL9G Kanji Recognition Model
Converts trained PyTorch model to ONNX format with backend-specific optimizations
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import generate_export_path, infer_model_type, setup_logger

logger = setup_logger(__name__)

# Suppress PyTorch's TypedStorage deprecation warning (internal, not in user code)
warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage.*")

# Import the correct model architecture from training
try:
    from train_cnn_model import LightweightKanjiNet as OriginalLightweightKanjiNet
except ImportError:
    # Handle case when running from scripts directory
    sys.path.append(str(Path(__file__).parent))
    from train_cnn_model import LightweightKanjiNet as OriginalLightweightKanjiNet

try:
    import onnxruntime

    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False


class LightweightKanjiNet(OriginalLightweightKanjiNet):
    """Extended LightweightKanjiNet with configurable pooling for different backends."""

    def __init__(self, num_classes: int, image_size: int = 64, pooling_type: str = "adaptive_avg"):
        # Initialize the original model first
        super().__init__(num_classes, image_size)

        # Override the pooling layer based on target backend compatibility
        if pooling_type == "adaptive_avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)  # Original: GlobalAveragePool in ONNX
        elif pooling_type == "adaptive_max":
            self.global_pool = nn.AdaptiveMaxPool2d(1)  # Alternative: GlobalMaxPool in ONNX
        elif pooling_type == "fixed_avg":
            self.global_pool = nn.AvgPool2d(
                kernel_size=4, stride=1, padding=0
            )  # Compatible: AveragePool in ONNX
        elif pooling_type == "fixed_max":
            self.global_pool = nn.MaxPool2d(
                kernel_size=4, stride=1, padding=0
            )  # Compatible: MaxPool in ONNX
        else:
            logger.warning("Unknown pooling type: %s, using adaptive_avg", pooling_type)
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Note: For the trained model, the input after conv4 is 4x4, so both adaptive and fixed pooling
        # with kernel_size=4 will produce 1x1 output, maintaining the same classifier input size (256)


def generate_output_filename(base_name, image_size, backend, suffix):
    """Generate consistent filename with configuration details."""
    model_type = infer_model_type(base_name)
    export_dir = generate_export_path(model_type)
    return str(
        export_dir / f"{base_name}_etl9g_{image_size}x{image_size}_3036classes_{backend}{suffix}"
    )


def export_to_onnx(
    model_path,
    onnx_path,
    image_size,
    pooling_type="adaptive_avg",
    target_backend="tract",
):
    """Export PyTorch model to ONNX with backend-specific optimizations

    Auto-generates filename if onnx_path is None.

    Backend Compatibility Analysis:
    ===============================

    Sonos Tract vs ORT-Tract (https://ort.pyke.io/backends/tract) Comparison:
    -----------------------------------
    1. **Sonos Tract (Direct)**:
       - Full Tract library with ~85% ONNX operator coverage
       - Supports opset versions 9-18
       - Direct access to all Tract optimizations
       - Designed for embedded/edge deployment

    2. **ORT-Tract (via ort crate)**:
       - ONNX Runtime API wrapper around Tract
       - Same underlying engine but through ort interface
       - Limited to ort-supported APIs (tensor operations only)
       - More constrained feature set for compatibility

    Model Architecture -> ONNX Operators -> Backend Support:
    -------------------------------------------------------
    ✅ **UNIVERSALLY SUPPORTED** (Both Tract & ORT-Tract):
    - Depthwise Convolution -> Conv (with groups) -> ✅ Full Support
    - Pointwise Convolution -> Conv (1x1) -> ✅ Full Support
    - Batch Normalization -> BatchNormalization -> ✅ Full Support
    - ReLU Activations -> Relu -> ✅ Full Support
    - Fixed Pooling -> AveragePool/MaxPool -> ✅ Full Support

    ⚠️  **TRACT ONLY** (Direct Tract, NOT ORT-Tract):
    - Global Pooling -> GlobalAveragePool/GlobalMaxPool -> ❌ ORT-Tract Unsupported

    Export Method Selection:
    -----------------------
    - `tract`: Uses dynamo=True (modern torch.export) + opset 12
    - `ort-tract`: Uses dynamo=False (legacy TorchScript) + opset 11
    - `strict`: Uses dynamo=False (legacy TorchScript) + opset 11

    The new dynamo=True export (PyTorch 2.9+) provides better optimization
    and support for advanced features, but we use it selectively for compatibility.
    =========================
    """

    # Validate inputs
    if not Path(model_path).exists():
        logger.error(f"❌ Model not found: {model_path}")
        return None

    if not ONNXRUNTIME_AVAILABLE:
        logger.error("❌ ONNX Runtime not available")
        return None

    # Generate default filename if not provided
    if onnx_path is None:
        onnx_path = generate_output_filename("kanji_model", image_size, target_backend, ".onnx")

    logger.info(f"📂 Loading model: {model_path}")
    logger.info(f"🎯 Target backend: {target_backend}")
    logger.info(f"📊 Image size: {image_size}x{image_size}")

    # Choose export method and opset version based on target backend
    if target_backend == "ort-tract":
        # ORT-Tract: Use more conservative settings for API compatibility
        opset_version = 11  # More conservative for ort compatibility
        use_dynamo = False  # Conservative export for better compatibility
        # Override pooling for ORT-Tract compatibility - GlobalAveragePool not supported
        if pooling_type in ["adaptive_avg", "adaptive_max"]:
            original_pooling = pooling_type
            pooling_type = "fixed_avg" if pooling_type == "adaptive_avg" else "fixed_max"
    elif target_backend == "strict":
        # Strict mode: Maximum compatibility - also avoid GlobalAveragePool
        opset_version = 11
        use_dynamo = False  # Legacy export for maximum compatibility
        if pooling_type in ["adaptive_avg", "adaptive_max"]:
            original_pooling = pooling_type
            pooling_type = "fixed_avg" if pooling_type == "adaptive_avg" else "fixed_max"
    else:
        # Direct Tract: Use newer opset for better optimization
        opset_version = 12  # Newer opset for direct tract
        use_dynamo = True  # Modern export for better optimization

    # Initialize model architecture with proper pooling
    # ETL9G dataset has exactly 3,036 character classes (fixed)
    NUM_CLASSES = 3036
    model = LightweightKanjiNet(NUM_CLASSES, image_size, pooling_type)

    # Load trained weights
    checkpoint = torch.load(model_path, map_location="cpu")

    # For ORT-Tract compatibility, we need to ensure ALL pooling layers are correct
    if target_backend in ["ort-tract", "strict"] and pooling_type == "fixed_avg":
        # Replace ALL AdaptiveAvgPool2d layers with fixed pooling for backend compatibility

        # Replace the main global pooling layer
        if hasattr(model, "global_pool"):
            model.global_pool = nn.AvgPool2d(4, stride=1, padding=0)

        # Replace pooling layers in channel attention modules with correct kernel sizes
        # Based on architecture: conv3->8x8, conv4->4x4, conv5->4x4
        attention_pool_sizes = {
            "attention3": 8,  # After conv3: 8x8 spatial size
            "attention4": 4,  # After conv4: 4x4 spatial size
            "attention5": 4,  # After conv5: 4x4 spatial size
        }

        for name, module in model.named_modules():
            if hasattr(module, "global_pool") and any(
                att_name in name for att_name in attention_pool_sizes
            ):
                # Find which attention module this is
                for att_name, kernel_size in attention_pool_sizes.items():
                    if att_name in name:
                        module.global_pool = nn.AvgPool2d(kernel_size, stride=1, padding=0)
                        break

        # Load all weights except pooling layers which we've reconfigured
        state_dict = checkpoint.copy()

        # Remove any pooling layer weights that might conflict
        keys_to_remove = [k for k in state_dict.keys() if "global_pool" in k]
        for key in keys_to_remove:
            del state_dict[key]

        # Load the remaining weights
        model.load_state_dict(state_dict, strict=False)
    else:
        # Standard loading for other backends
        model.load_state_dict(checkpoint)

    model = model.to("cpu")  # Explicitly move model to CPU
    model.eval()

    # Create dummy input for ONNX export
    # The trained model expects 2D image input (1, 1, 64, 64), not flattened
    dummy_input = torch.randn(1, 1, image_size, image_size)

    logger.info("📤 Exporting to ONNX...")

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,  # Both backends benefit from constant folding ✅
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            if target_backend != "strict"
            else None,  # Disable dynamic axes for ultra-conservative mode
            dynamo=use_dynamo,
            verbose=False,
        )
        logger.info(f"✓ ONNX export complete: {onnx_path}")
    except Exception as e:
        if use_dynamo:
            logger.warning("⚠️  Retrying export without dynamo...")
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
                if target_backend != "strict"
                else None,
                verbose=False,
            )
            logger.info(f"✓ ONNX export complete: {onnx_path}")
        else:
            logger.error(f"❌ ONNX export failed: {e}")
            raise

    # Optimize ONNX model
    logger.info("🔧 Optimizing ONNX model...")

    try:
        # Basic optimization using session options
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Test inference to ensure model works
        session = onnxruntime.InferenceSession(str(onnx_path), sess_options)

        # Get model info
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape

        # Test inference
        test_input = torch.randn(1, 1, image_size, image_size).numpy()
        outputs = session.run(None, {"input": test_input})

    except Exception as e:  # noqa: S110
        logger.error(f"❌ ONNX validation failed: {e}")

    # Create character mapping for inference
    mapping_path = onnx_path.replace(".onnx", "_mapping.json")

    # Load the character mapping
    mapping_file = "kanji_etl9g_mapping.json"
    try:
        with open(mapping_file, encoding="utf-8") as f:
            character_mapping = json.load(f)
        logger.info(f"✓ Loaded character mapping: {mapping_file}")

        # Convert mapping to the format expected by create_character_mapping
        char_details = {}
        for class_id, char_info in character_mapping.items():
            # Skip metadata entries that aren't class mappings
            if not class_id.isdigit():
                continue

            char_details[char_info.get("jis_code", f"{class_id:04X}")] = {
                "class_idx": int(class_id),
                "character": char_info.get("character", f"[{class_id}]"),
                "stroke_count": char_info.get("stroke_count", 8),
            }

    except FileNotFoundError:
        logger.warning(f"⚠️  Character mapping not found: {mapping_file}")
        # Fallback to dataset mapping
        try:
            with open("dataset/character_mapping.json", encoding="utf-8") as f:
                char_details = json.load(f)
            logger.info("✓ Using fallback character mapping")
        except FileNotFoundError:
            logger.warning("⚠️  No character mapping found (inference may lack character info)")
            char_details = {}

    # Create comprehensive mapping
    mapping = create_character_mapping(
        char_details,
        {
            "image_size": image_size,
            "pooling_type": pooling_type,
            "target_backend": target_backend,
            "opset_version": opset_version,
            "export_method": "torch.export" if use_dynamo else "TorchScript",
        },
        character_mapping if "character_mapping" in locals() else None,
    )

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Saved character mapping: {mapping_path}")
    logger.info("=" * 70)
    logger.info("✅ ONNX export complete!")
    logger.info("=" * 70)

    return onnx_path


def create_character_mapping(char_details, model_info, character_mapping=None):
    """Create character mapping for ETL9G dataset (3,036 classes)"""
    NUM_CLASSES = 3036

    logger.info(f"📊 Creating character mapping for {NUM_CLASSES} classes...")

    # If character mapping is provided directly, use it
    if character_mapping:
        characters = {}
        mapping_chars = character_mapping.get("characters", {})
        for class_idx in range(NUM_CLASSES):
            class_str = str(class_idx)
            char_info = mapping_chars.get(class_str, {})

            if char_info:
                characters[class_str] = {
                    "character": char_info.get("character", f"[{class_idx}]"),
                    "jis_code": char_info.get("jis_code", f"{class_idx:04X}"),
                    "stroke_count": char_info.get("stroke_count", 8),
                }
            else:
                characters[class_str] = {
                    "character": f"[{class_idx}]",
                    "jis_code": f"{class_idx:04X}",
                    "stroke_count": 8,
                }

        # Complete mapping structure
        mapping = {
            "model_info": model_info,
            "num_classes": NUM_CLASSES,
            "characters": characters,
            "statistics": {
                "total_characters": len(characters),
                "mapped_characters": len(
                    [c for c in characters.values() if not c["character"].startswith("[")]
                ),
                "creation_timestamp": int(time.time()),
            },
        }
        return mapping

    def jis_to_unicode(jis_code):
        """Convert JIS X 0208 area/code format to Unicode character."""
        try:
            # Convert hex string to integer
            jis_int = int(jis_code, 16)

            # Extract area (high byte) and code (low byte)
            area = (jis_int >> 8) & 0xFF
            code = jis_int & 0xFF

            # JIS X 0208 to Unicode mapping
            if area == 0x24:  # Hiragana
                if 0x21 <= code <= 0x73:
                    return chr(0x3041 + (code - 0x21))
            elif area == 0x25:  # Katakana
                if 0x21 <= code <= 0x76:
                    return chr(0x30A1 + (code - 0x21))
            elif 0x30 <= area <= 0x4F:  # Kanji
                # Simplified kanji mapping - this is a basic approximation
                # Real JIS X 0208 to Unicode requires full conversion tables
                base_offset = (area - 0x30) * 94 + (code - 0x21)
                return chr(0x4E00 + base_offset)  # CJK Unified Ideographs base

            return f"[JIS:{jis_code}]"

        except (ValueError, OverflowError):
            return f"[JIS:{jis_code}]"

    def estimate_stroke_count(jis_code, character):
        """Estimate stroke count based on JIS code and character"""
        try:
            jis_int = int(jis_code, 16)
            area = (jis_int >> 8) & 0xFF
            code = jis_int & 0xFF

            # Hiragana typically have 1-4 strokes
            if area == 0x24 or area == 0x30:
                # Simple estimation based on position
                return min(4, max(1, (code - 0x21) % 4 + 1))

            # Katakana typically have 1-5 strokes
            elif area == 0x25:
                return min(5, max(1, (code - 0x21) % 5 + 1))

            # Kanji stroke count estimation
            elif 0x30 <= area <= 0x4F:
                # JIS ordering roughly follows stroke count/complexity
                jis_linear = ((area - 0x30) * 94) + (code - 0x21)

                # Rough stroke count estimation
                if jis_linear < 200:
                    return min(8, max(1, jis_linear // 25 + 1))  # 1-8 strokes
                elif jis_linear < 800:
                    return min(12, max(8, (jis_linear - 200) // 75 + 8))  # 8-12 strokes
                elif jis_linear < 1500:
                    return min(16, max(12, (jis_linear - 800) // 100 + 12))  # 12-16 strokes
                else:
                    return min(25, max(16, (jis_linear - 1500) // 150 + 16))  # 16+ strokes

            return 1  # Default fallback

        except ValueError:
            return 1

    # estimate_stroke_count function previously defined above (line 390), not redefined here

    # Create class-to-JIS mapping
    class_to_jis = {}
    jis_to_class = {}

    for jis_code, details in char_details.items():
        class_idx = details["class_idx"]
        class_to_jis[str(class_idx)] = jis_code
        jis_to_class[jis_code] = class_idx

    # Create comprehensive character mapping
    characters = {}

    for class_idx in range(NUM_CLASSES):
        class_str = str(class_idx)

        if class_str in class_to_jis:
            jis_code = class_to_jis[class_str]

            # Get the actual character using JIS to Unicode conversion
            character = jis_to_unicode(jis_code)
            stroke_count = estimate_stroke_count(jis_code, character)

            characters[class_str] = {
                "character": character,
                "jis_code": jis_code,
                "stroke_count": stroke_count,
            }
        else:
            # Fallback for missing mappings - this should be rare with proper dataset
            characters[class_str] = {
                "character": f"[Missing-{class_idx}]",
                "jis_code": "unknown",
                "stroke_count": 1,
            }

    # Complete mapping structure
    mapping = {
        "model_info": model_info,
        "num_classes": NUM_CLASSES,
        "characters": characters,
        "statistics": {
            "total_characters": len(characters),
            "hiragana_count": len(
                [c for c in characters.values() if c["jis_code"].startswith("30")]
            ),
            "kanji_count": len(
                [
                    c
                    for c in characters.values()
                    if c["jis_code"].startswith(
                        (
                            "31",
                            "32",
                            "33",
                            "34",
                            "35",
                            "36",
                            "37",
                            "38",
                            "39",
                            "3A",
                            "3B",
                            "3C",
                            "3D",
                            "3E",
                            "3F",
                            "40",
                            "41",
                            "42",
                            "43",
                            "44",
                        )
                    )
                ]
            ),
            "average_stroke_count": sum(c["stroke_count"] for c in characters.values())
            / len(characters)
            if characters
            else 0,
        },
    }

    return mapping


# Note: create_character_mapping is defined above (line 322) - no redefinition needed


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch kanji model to ONNX")
    parser.add_argument(
        "--model-path",
        type=str,
        default="training/cnn/best_kanji_model.pth",
        help="Path to the trained PyTorch model",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=None,
        help="Output path for ONNX model (auto-generated if not specified)",
    )
    parser.add_argument("--image-size", type=int, default=64, help="Image size used in training")
    parser.add_argument(
        "--pooling-type",
        type=str,
        default="adaptive_avg",
        choices=["adaptive_avg", "adaptive_max", "fixed_avg", "fixed_max"],
        help="Pooling layer configuration",
    )
    parser.add_argument(
        "--target-backend",
        type=str,
        default="tract",
        choices=["tract", "ort-tract", "strict"],
        help="Target inference backend",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("CONVERT PYTORCH MODEL TO ONNX FORMAT")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Backend: {args.target_backend}")
    logger.info(f"Pooling: {args.pooling_type}")
    logger.info("=" * 70)

    # Note: Output path should be specified via --onnx-path
    # Directory will be created by ONNX export function

    if args.onnx_path is None:
        onnx_filename = generate_output_filename(
            "kanji_model",
            args.image_size,
            args.target_backend,
            ".onnx",
        )
    else:
        onnx_filename = args.onnx_path

    result = export_to_onnx(
        args.model_path,
        onnx_filename,
        args.image_size,
        args.pooling_type,
        args.target_backend,
    )

    if result:
        logger.info("✅ ONNX export successful!")
        if args.target_backend == "tract":
            logger.info("   Backend: Tract (GlobalAveragePool compatible)")
        elif args.target_backend == "ort-tract":
            logger.info("   Backend: ORT-Tract (AveragePool compatible)")
        else:  # strict
            logger.info("   Backend: Strict ONNX")
        logger.info("=" * 70)

    else:
        logger.error("❌ ONNX export failed")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
