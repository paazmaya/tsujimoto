#!/usr/bin/env python3
"""
ONNX Model Inspector - Check what operations are used in ONNX models
"""

import os
import sys
from collections import Counter
from pathlib import Path

import onnx

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import setup_logger

logger = setup_logger(__name__)


def inspect_onnx_model(model_path):
    """Inspect ONNX model and show all operations used"""
    if not os.path.exists(model_path):
        logger.warning("Model not found: %s", model_path)
        return

    try:
        # Load the ONNX model
        model = onnx.load(model_path)
        logger.info("Loaded ONNX model: %s", os.path.basename(model_path))

        # Get all operations used in the model
        operations = []
        pooling_ops = []

        for node in model.graph.node:
            operations.append(node.op_type)
            if "pool" in node.op_type.lower():
                pooling_ops.append(
                    {
                        "op": node.op_type,
                        "name": node.name,
                        "inputs": list(node.input),
                        "outputs": list(node.output),
                        "attributes": {attr.name: attr for attr in node.attribute},
                    }
                )

        # Count operations
        op_counts = Counter(operations)
        logger.info("Operations used: %d unique types", len(op_counts))

        for op, count in sorted(op_counts.items()):
            icon = "🔴" if "GlobalAveragePool" in op else "🟢" if "pool" in op.lower() else "⚪"
            logger.info("  %s %s: %d", icon, op, count)

        # Check specifically for pooling operations
        if pooling_ops:
            logger.info("Pooling operations found: %d", len(pooling_ops))
            for i, pool_op in enumerate(pooling_ops, 1):
                logger.info("  [%d] %s (%s)", i, pool_op["op"], pool_op["name"])
                if pool_op["attributes"]:
                    for attr_name, attr in pool_op["attributes"].items():
                        if attr.type == onnx.AttributeProto.INTS:
                            logger.debug("    %s: %s", attr_name, list(attr.ints))
                        elif attr.type == onnx.AttributeProto.INT:
                            logger.debug("    %s: %d", attr_name, attr.i)
                        elif attr.type == onnx.AttributeProto.STRING:
                            logger.debug("    %s: %s", attr_name, attr.s.decode("utf-8"))
                        else:
                            logger.debug("    %s: %s", attr_name, str(attr))

        # Check for problematic operations
        problematic_ops = [op for op in operations if op in ["GlobalAveragePool", "GlobalMaxPool"]]
        if problematic_ops:
            logger.warning("Problematic pooling ops found: %s", set(problematic_ops))
        else:
            logger.debug("No problematic operations detected")

    except Exception as e:  # noqa: S110
        logger.error("Error inspecting model: %s", str(e))


def main():
    """Main function to inspect all ONNX models"""
    models_to_check = [
        "training/cnn/exports/kanji_model_etl9g_64x64_3036classes_tract.onnx",
        "training/cnn/exports/kanji_model_etl9g_64x64_3036classes_ort-tract.onnx",
        "training/cnn/exports/kanji_model_etl9g_64x64_3036classes_strict.onnx",
    ]

    logger.info("Inspecting %d ONNX models...", len(models_to_check))
    for model_file in models_to_check:
        if os.path.exists(model_file):
            inspect_onnx_model(model_file)
        else:
            logger.debug("Model file not found: %s", model_file)


if __name__ == "__main__":
    main()
