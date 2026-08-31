"""Unified ONNX model analysis CLI tool.

Consolidates analysis functionality from:
- inspect_onnx_model.py
- onnx_operations_comparison.py
- pooling_comparison.py

Provides comprehensive ONNX model inspection and analysis.

Usage:
    python analyze_model.py --model model.onnx
    python analyze_model.py --model model.onnx --inspect operations
    python analyze_model.py --model model.onnx --inspect pooling
    python analyze_model.py --model model.onnx --output analysis.json

"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib.logging_utils import setup_logger
from src.lib.onnx_analysis import ONNXModelAnalyzer, PoolingComparisonAnalyzer

logger = setup_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for model analysis CLI.

    Returns:
        ArgumentParser instance

    """
    parser = argparse.ArgumentParser(
        description="Analyze ONNX models and compare operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Analyze ONNX model:
    python analyze_model.py --model model.onnx
  
  Inspect specific operations:
    python analyze_model.py --model model.onnx --inspect operations
  
  Compare pooling implementations:
    python analyze_model.py --model model.onnx --inspect pooling
  
  Save analysis to JSON:
    python analyze_model.py --model model.onnx --output analysis.json
        """,
    )

    # Model file
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to ONNX model file",
    )

    # Analysis type
    parser.add_argument(
        "--inspect",
        type=str,
        choices=["structure", "operations", "pooling", "all"],
        default="structure",
        help="Type of analysis to perform (default: structure)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for analysis results (JSON)",
    )

    # Comparison
    parser.add_argument(
        "--compare-pooling",
        action="store_true",
        help="Compare different pooling implementations",
    )

    # Verbosity
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser


def analyze_model(model_path: str, inspect_type: str) -> dict:
    """Analyze ONNX model.

    Args:
        model_path: Path to ONNX model file
        inspect_type: Type of inspection (structure, operations, pooling, all)

    Returns:
        Analysis results dictionary

    Raises:
        FileNotFoundError: If model file doesn't exist

    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Loading model: {model_path}")

    results = {}

    # Create analyzer
    analyzer = ONNXModelAnalyzer(str(model_path))

    if inspect_type in ["structure", "all"]:
        logger.info("Analyzing model structure...")
        analysis = analyzer.analyze()
        results["structure"] = analysis
        analyzer.print_summary()

    if inspect_type in ["operations", "all"]:
        logger.info("Analyzing operations...")
        ops_analysis = analyzer._analyze_operations()
        results["operations"] = ops_analysis
        logger.info(f"  Found {len(ops_analysis)} operation types")

    if inspect_type in ["pooling", "all"]:
        logger.info("Analyzing pooling operations...")
        pooling_analysis = PoolingComparisonAnalyzer.analyze_pooling_differences()
        results["pooling_analysis"] = pooling_analysis

    return results


def main():
    """Main CLI entry point for model analysis."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("=" * 70)
    logger.info("ONNX MODEL ANALYSIS TOOL")
    logger.info("=" * 70)

    try:
        # Analyze model
        results = analyze_model(args.model, args.inspect)

        # Save output if specified
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"✓ Analysis saved to: {output_path}")

        logger.info("=" * 70)
        logger.info("✓ Model analysis completed successfully!")
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"✗ Error during analysis: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
