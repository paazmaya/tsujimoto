#!/usr/bin/env python3
"""
GlobalAveragePool vs AveragePool - Demonstration of the differences
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import setup_logger

logger = setup_logger(__name__)


def demonstrate_pooling_differences():
    """Demonstrate the differences between GlobalAveragePool and AveragePool"""

    # Create sample input tensor: (batch=1, channels=128, height=8, width=8)
    sample_input = torch.randn(1, 128, 8, 8)
    logger.info("Input shape: %s", sample_input.shape)

    # 1. GlobalAveragePool (AdaptiveAvgPool2d(1))

    global_pool = nn.AdaptiveAvgPool2d(1)
    global_output = global_pool(sample_input)
    logger.info("GlobalAveragePool output: %s", global_output.shape)

    # 2. AveragePool with same effect

    avg_pool_8x8 = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
    avg_output_8x8 = avg_pool_8x8(sample_input)
    logger.info("AvgPool2d(8×8) output: %s", avg_output_8x8.shape)

    # Verify they produce the same result
    is_close = torch.allclose(global_output, avg_output_8x8, atol=1e-6)
    logger.info("Results match: %s", is_close)

    # 3. Show what happens with wrong kernel size
    avg_pool_4x4 = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
    try:
        out_4x4 = avg_pool_4x4(sample_input)
        logger.info("AvgPool2d(4×4) on 8×8 input: %s", out_4x4.shape)
    except Exception as e:
        logger.warning("AvgPool2d(4×4) on 8×8 input failed: %s", str(e))

    # 4. Different input size demonstration
    different_input = torch.randn(1, 128, 4, 4)  # 4x4 instead of 8x8
    logger.info("Different input shape: %s", different_input.shape)

    # GlobalAveragePool adapts automatically
    adaptive_out = global_pool(different_input)
    logger.info("GlobalAveragePool on 4×4: %s (adapts automatically)", adaptive_out.shape)

    # AveragePool with 8x8 kernel fails on 4x4 input (already shown above)

    # But AveragePool with 4x4 kernel works
    avg_pool_4x4_correct = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
    correct_out = avg_pool_4x4_correct(different_input)
    logger.info("AvgPool2d(4×4) on 4×4: %s (must match input size)", correct_out.shape)


def explain_onnx_differences():
    """Explain why this matters for ONNX export and backend compatibility"""

    logger.info("\n=== Spatial dimensions through the network ===")
    spatial_sizes = {
        "Input": "64×64",
        "After conv1 (stride=2)": "32×32",
        "After conv2 (stride=2)": "16×16",
        "After conv3 (stride=2)": "8×8  ← attention3 uses AvgPool2d(8)",
        "After conv4 (stride=2)": "4×4  ← attention4 uses AvgPool2d(4)",
        "After conv5 (stride=1)": "4×4  ← attention5 uses AvgPool2d(4), main pool uses AvgPool2d(4)",
    }

    for stage, size in spatial_sizes.items():
        logger.info("  %s: %s", stage, size)


def show_onnx_operation_details():
    """Show the actual ONNX operation differences"""

    logger.info("\n=== Key ONNX differences ===")
    logger.info("• GlobalAveragePool: Size adapts to input (flexible)")
    logger.info("• AveragePool: Fixed kernel size (must match spatial dims)")


if __name__ == "__main__":
    demonstrate_pooling_differences()
    explain_onnx_differences()
    show_onnx_operation_details()
