"""
ONNX model analysis and inspection utilities.

Consolidates ONNX analysis functions from inspect_onnx_model.py,
onnx_operations_comparison.py, and pooling_comparison.py.

Example:
    >>> from src.lib.onnx_analysis import ONNXModelAnalyzer
    >>> analyzer = ONNXModelAnalyzer('model.onnx')
    >>> analyzer.analyze()
    >>> analyzer.get_model_info()
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import onnx
    from onnx import helper

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX not available. Install with: pip install onnx")


class ONNXModelAnalyzer:
    """Analyze ONNX model structure and operations."""

    def __init__(self, model_path: str):
        """
        Initialize ONNX model analyzer.

        Args:
            model_path: Path to ONNX model file
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not installed. Install with: pip install onnx")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        try:
            self.model = onnx.load(str(model_path))
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise

        self.graph = self.model.graph
        self.info = {}

    def get_model_info(self) -> Dict:
        """
        Get high-level ONNX model information.

        Returns:
            Dict with model metadata
        """
        if not self.info:
            self.analyze()

        return self.info

    def analyze(self) -> Dict:
        """
        Perform complete model analysis.

        Returns:
            Dict with comprehensive analysis results
        """
        logger.info("=" * 70)
        logger.info("ANALYZING ONNX MODEL")
        logger.info("=" * 70)

        self.info = {
            "model_path": str(self.model_path),
            "ir_version": self.model.ir_version,
            "opset_version": self._get_opset_version(),
            "producer_name": self.model.producer_name,
            "inputs": self._analyze_inputs(),
            "outputs": self._analyze_outputs(),
            "nodes": self._analyze_nodes(),
            "initializers": self._analyze_initializers(),
            "operations": self._analyze_operations(),
        }

        logger.info(f"✓ Model analysis complete")
        logger.info(f"  IR Version: {self.info['ir_version']}")
        logger.info(f"  Opset Version: {self.info['opset_version']}")
        logger.info(f"  Inputs: {len(self.info['inputs'])}")
        logger.info(f"  Outputs: {len(self.info['outputs'])}")
        logger.info(f"  Nodes: {self.info['nodes']['count']}")
        logger.info(f"  Unique Operations: {len(self.info['operations']['by_type'])}")

        return self.info

    def _get_opset_version(self) -> Optional[int]:
        """Get ONNX opset version."""
        try:
            for opset in self.model.opset_import:
                if opset.domain == "" or opset.domain == "ai.onnx":
                    return opset.version
        except Exception:
            pass
        return None

    def _analyze_inputs(self) -> List[Dict]:
        """Analyze model inputs."""
        inputs = []
        for inp in self.graph.input:
            shape = [d.dim_value if d.dim_value > 0 else "?" for d in inp.type.tensor_type.shape.dim]
            dtype = self._get_tensor_type_name(inp.type.tensor_type.elem_type)

            inputs.append({
                "name": inp.name,
                "shape": shape,
                "dtype": dtype,
            })

        logger.info(f"  Inputs: {len(inputs)}")
        for inp in inputs:
            logger.info(f"    - {inp['name']}: {inp['shape']} ({inp['dtype']})")

        return inputs

    def _analyze_outputs(self) -> List[Dict]:
        """Analyze model outputs."""
        outputs = []
        for out in self.graph.output:
            shape = [d.dim_value if d.dim_value > 0 else "?" for d in out.type.tensor_type.shape.dim]
            dtype = self._get_tensor_type_name(out.type.tensor_type.elem_type)

            outputs.append({
                "name": out.name,
                "shape": shape,
                "dtype": dtype,
            })

        logger.info(f"  Outputs: {len(outputs)}")
        for out in outputs:
            logger.info(f"    - {out['name']}: {out['shape']} ({out['dtype']})")

        return outputs

    def _analyze_nodes(self) -> Dict:
        """Analyze model nodes."""
        nodes_by_type = {}
        total_nodes = len(self.graph.node)

        for node in self.graph.node:
            op_type = node.op_type
            if op_type not in nodes_by_type:
                nodes_by_type[op_type] = []
            nodes_by_type[op_type].append({
                "name": node.name,
                "op_type": op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
            })

        logger.info(f"  Nodes: {total_nodes}")
        for op_type, nodes in sorted(nodes_by_type.items()):
            logger.info(f"    - {op_type}: {len(nodes)}")

        return {
            "count": total_nodes,
            "by_type": {k: len(v) for k, v in nodes_by_type.items()},
            "nodes": nodes_by_type,
        }

    def _analyze_initializers(self) -> Dict:
        """Analyze model parameters."""
        initializers = {}
        total_params = 0

        for init in self.graph.initializer:
            shape = list(init.dims)
            dtype = self._get_tensor_type_name(init.data_type)
            numel = 1
            for d in shape:
                numel *= d
            total_params += numel

            initializers[init.name] = {
                "shape": shape,
                "dtype": dtype,
                "numel": numel,
            }

        logger.info(f"  Initializers (Parameters): {len(initializers)}")
        logger.info(f"  Total Parameters: {total_params:,}")

        return {
            "count": len(initializers),
            "total_params": total_params,
            "parameters": initializers,
        }

    def _analyze_operations(self) -> Dict:
        """Analyze operation types used in model."""
        ops_by_type = {}

        for node in self.graph.node:
            op_type = node.op_type
            if op_type not in ops_by_type:
                ops_by_type[op_type] = {
                    "op_type": op_type,
                    "count": 0,
                    "attributes": set(),
                }
            ops_by_type[op_type]["count"] += 1
            for attr in node.attribute:
                ops_by_type[op_type]["attributes"].add(attr.name)

        # Convert sets to lists for JSON serialization
        for op_type in ops_by_type:
            ops_by_type[op_type]["attributes"] = list(ops_by_type[op_type]["attributes"])

        logger.info(f"  Unique Operations: {len(ops_by_type)}")
        for op_type, op_info in sorted(ops_by_type.items()):
            logger.info(f"    - {op_type}: {op_info['count']} occurrences")

        return {
            "count": len(ops_by_type),
            "by_type": ops_by_type,
        }

    def _get_tensor_type_name(self, elem_type: int) -> str:
        """Get human-readable tensor type name."""
        type_map = {
            1: "float32",
            2: "uint8",
            3: "int8",
            4: "uint16",
            5: "int16",
            6: "int32",
            7: "int64",
            8: "string",
            9: "bool",
            10: "float16",
            11: "double",
            12: "uint32",
            13: "uint64",
            14: "complex64",
            15: "complex128",
        }
        return type_map.get(elem_type, f"unknown({elem_type})")

    def print_summary(self) -> None:
        """Print model analysis summary."""
        if not self.info:
            self.analyze()

        print("\n" + "=" * 70)
        print("ONNX MODEL SUMMARY")
        print("=" * 70)
        print(f"Model Path: {self.info['model_path']}")
        print(f"IR Version: {self.info['ir_version']}")
        print(f"Opset Version: {self.info['opset_version']}")
        print(f"Producer: {self.info['producer_name']}")
        print(f"\nInputs: {len(self.info['inputs'])}")
        for inp in self.info['inputs']:
            print(f"  - {inp['name']}: {inp['shape']} ({inp['dtype']})")
        print(f"\nOutputs: {len(self.info['outputs'])}")
        for out in self.info['outputs']:
            print(f"  - {out['name']}: {out['shape']} ({out['dtype']})")
        print(f"\nNodes: {self.info['nodes']['count']}")
        print(f"Parameters: {self.info['initializers']['total_params']:,}")
        print(f"Unique Operations: {self.info['operations']['count']}")
        print("=" * 70 + "\n")

    def save_analysis(self, output_path: str) -> None:
        """
        Save analysis results to JSON file.

        Args:
            output_path: Path to save analysis results
        """
        if not self.info:
            self.analyze()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert sets to lists for JSON serialization
        analysis = self._make_json_serializable(self.info)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)

        logger.info(f"✓ Analysis saved: {output_path}")

    def _make_json_serializable(self, obj):
        """Recursively convert non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        else:
            return obj


class PoolingComparisonAnalyzer:
    """Compare different pooling implementations in ONNX models."""

    @staticmethod
    def analyze_pooling_differences() -> Dict:
        """
        Analyze differences between GlobalAveragePool and FixedAveragePool.

        Returns:
            Dict with comparison analysis
        """
        comparison = {
            "GlobalAveragePool": {
                "description": "Standard ONNX global average pooling",
                "input": "4D tensor (batch, channels, height, width)",
                "output": "2D tensor (batch, channels)",
                "operation": "Averages across all spatial dimensions",
                "supported_shapes": "Any (H, W)",
                "performance": "Optimized in most frameworks",
                "export_compatibility": "Excellent",
                "dtype_support": ["float32", "float16", "int32", "int64"],
            },
            "FixedAveragePool": {
                "description": "Fixed-size average pooling",
                "input": "4D tensor (batch, channels, height, width)",
                "output": "4D tensor (batch, channels, H_out, W_out)",
                "operation": "Average pools with fixed kernel and stride",
                "supported_shapes": "Depends on pool_size and strides",
                "performance": "Good, but requires additional convolution",
                "export_compatibility": "Good",
                "dtype_support": ["float32", "float16"],
            },
            "MaxPool": {
                "description": "Max pooling with fixed size",
                "input": "4D tensor (batch, channels, height, width)",
                "output": "4D tensor (batch, channels, H_out, W_out)",
                "operation": "Max pools with fixed kernel and stride",
                "supported_shapes": "Depends on pool_size and strides",
                "performance": "Optimized in all frameworks",
                "export_compatibility": "Excellent",
                "dtype_support": ["float32", "float16", "int32", "int64"],
            },
        }

        return comparison

    @staticmethod
    def get_onnx_pooling_operations() -> Dict:
        """
        Get real ONNX pooling operations with their implementations.

        Returns:
            Dict describing ONNX pooling operations
        """
        operations = {
            "GlobalAveragePool": {
                "onnx_name": "GlobalAveragePool",
                "attributes": {},
                "inputs": 1,
                "outputs": 1,
                "description": "Applies global average pooling to input tensor",
            },
            "AveragePool": {
                "onnx_name": "AveragePool",
                "attributes": [
                    "auto_pad",
                    "ceil_mode",
                    "count_include_pad",
                    "kernels",
                    "pads",
                    "strides",
                ],
                "inputs": 1,
                "outputs": 1,
                "description": "Average pooling with configurable kernel and stride",
            },
            "MaxPool": {
                "onnx_name": "MaxPool",
                "attributes": [
                    "auto_pad",
                    "ceil_mode",
                    "dilations",
                    "kernel_shape",
                    "pads",
                    "storage_order",
                    "strides",
                ],
                "inputs": 1,
                "outputs": 1,  # Or 2 if indices are requested
                "description": "Max pooling with configurable kernel and stride",
            },
        }

        return operations

    @staticmethod
    def explain_onnx_compatibility() -> Dict:
        """
        Explain compatibility impacts of different pooling operations.

        Returns:
            Dict with compatibility information
        """
        compatibility = {
            "GlobalAveragePool": {
                "onnx_runtime": "Full support",
                "tensorflow": "tf.nn.global_average_pool2d equivalent",
                "pytorch": "torch.nn.functional.adaptive_avg_pool2d",
                "tvm": "Fully supported",
                "ort_providers": [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                    "CoreMLExecutionProvider",
                ],
                "notes": "Universal operation, highly compatible",
            },
            "AveragePool": {
                "onnx_runtime": "Full support",
                "tensorflow": "tf.nn.avg_pool equivalent",
                "pytorch": "torch.nn.functional.avg_pool2d",
                "tvm": "Fully supported",
                "ort_providers": [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
                "notes": "Standard operation, good compatibility",
            },
            "MaxPool": {
                "onnx_runtime": "Full support",
                "tensorflow": "tf.nn.max_pool equivalent",
                "pytorch": "torch.nn.functional.max_pool2d",
                "tvm": "Fully supported",
                "ort_providers": [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                    "CoreMLExecutionProvider",
                ],
                "notes": "Universal operation, excellent compatibility",
            },
        }

        return compatibility
