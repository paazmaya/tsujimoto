#!/usr/bin/env python3
"""
Real ONNX Operation Comparison - Show actual operations from our models
"""

import os
import sys
from pathlib import Path

import onnx

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import setup_logger

logger = setup_logger(__name__)


def show_real_onnx_operations():
    """Show the actual ONNX operations from our converted models"""

    models = {
        "Direct Tract (GlobalAveragePool)": "training/cnn/exports/kanji_model_etl9g_64x64_3036classes_tract.onnx",
        "ORT-Tract (Fixed AveragePool)": "training/cnn/exports/kanji_model_etl9g_64x64_3036classes_ort-tract.onnx",
    }

    logger.info("Comparing ONNX operation implementations...")
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            logger.warning("Model not found: %s", model_path)
            continue

        try:
            model = onnx.load(model_path)
            logger.info("\n%s:", model_name)

            # Find pooling operations
            pooling_ops = []
            for node in model.graph.node:
                if "pool" in node.op_type.lower():
                    pooling_ops.append(node)

            logger.info("  Pooling operations: %d", len(pooling_ops))
            for i, node in enumerate(pooling_ops, 1):
                if node.op_type == "GlobalAveragePool":
                    logger.info("    [%d] %s - adapts to input size", i, node.op_type)

                elif node.op_type == "AveragePool":
                    kernel_shape = None
                    strides = None
                    pads = None
                    for attr in node.attribute:
                        if attr.name == "kernel_shape":
                            kernel_shape = list(attr.ints)
                        elif attr.name == "strides":
                            strides = list(attr.ints)
                        elif attr.name == "pads":
                            pads = list(attr.ints)
                    logger.info(
                        "    [%d] %s - kernel=%s, stride=%s, padding=%s",
                        i,
                        node.op_type,
                        kernel_shape,
                        strides,
                        pads,
                    )

        except Exception as e:  # noqa: S110
            logger.error("Error processing %s: %s", model_name, str(e))


def explain_compatibility_impact():
    """Explain the practical impact of the differences"""

    logger.info("\n=== Compatibility Impact ===")
    logger.info("GlobalAveragePool:")
    logger.info("  • Pros: Flexible, adapts to any input size")
    logger.info("  • Cons: Not all backends support it (esp. embedded/mobile)")
    logger.info("  • Backends: PyTorch, ONNX Runtime, some mobile runtimes")
    logger.info("\nAveragePool with fixed kernel:")
    logger.info("  • Pros: Widely supported, predictable behavior")
    logger.info("  • Cons: Requires exact input size match")
    logger.info("  • Backends: Tract, ORT, CoreML, TensorFlow Lite")


if __name__ == "__main__":
    show_real_onnx_operations()
    explain_compatibility_impact()
