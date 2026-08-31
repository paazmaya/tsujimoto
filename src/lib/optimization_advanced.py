"""
Model optimization utilities for ONNX export, quantization, and pruning.

This module provides tools for optimizing models for deployment:
- ONNX export with optimization
- Dynamic and static quantization
- Structured and unstructured pruning
- Model compression analysis

Example Usage:
    >>> from src.lib.optimization_advanced import ModelOptimizer
    >>>
    >>> optimizer = ModelOptimizer(model, device='cuda')
    >>>
    >>> # Export to ONNX with optimization
    >>> optimizer.export_onnx(
    ...     output_path="model.onnx",
    ...     input_shape=(1, 3, 64, 64),
    ...     optimize=True
    ... )
    >>>
    >>> # Quantize the model
    >>> quantized_model = optimizer.quantize(
    ...     method="dynamic",
    ...     backend="qnnpack"
    ... )
    >>>
    >>> # Prune the model
    >>> pruned_model = optimizer.prune(
    ...     amount=0.2,  # 20% sparsity
    ...     method="structured"
    ... )
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .logging_utils import setup_logger

logger = setup_logger(__name__)


class ModelOptimizer:
    """
    Comprehensive model optimization toolkit.

    Handles ONNX export, quantization, pruning, and compression analysis.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
    ):
        """
        Initialize model optimizer.

        Args:
            model: PyTorch model to optimize
            device: Device to use (cpu, cuda, mps)
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        logger.info(f"ModelOptimizer initialized (device={device})")

    def export_onnx(
        self,
        output_path: Path,
        input_shape: Tuple = (1, 3, 64, 64),
        optimize: bool = True,
        simplify: bool = False,
    ) -> Path:
        """
        Export model to ONNX format.

        Args:
            output_path: Path to save ONNX model
            input_shape: Shape of input tensor for tracing
            optimize: Whether to apply ONNX optimizations
            simplify: Whether to simplify the graph (requires onnxsimplifier)

        Returns:
            Path to exported ONNX model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Exporting model to ONNX: {output_path} "
            f"(input_shape={input_shape}, optimize={optimize})"
        )

        try:
            # Create dummy input
            dummy_input = torch.randn(*input_shape, device=self.device)

            # Export to ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=14,
                do_constant_folding=optimize,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                }
                if optimize
                else None,
                verbose=False,
            )

            logger.info("✓ ONNX export successful")

            # Optionally simplify with onnxsimplifier
            if simplify:
                try:
                    import onnxsimplifier

                    simplified_path = output_path.parent / f"simplified_{output_path.name}"
                    model_opt, check_ok = onnxsimplifier.simplify(str(output_path))

                    import onnx

                    onnx.save(model_opt, str(simplified_path))
                    logger.info(f"✓ ONNX simplified: {simplified_path} (check_ok={check_ok})")

                except ImportError:
                    logger.warning("onnxsimplifier not installed, skipping simplification")

            return output_path

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise

    def quantize(
        self,
        method: str = "dynamic",
        backend: str = "qnnpack",
        calibration_loader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """
        Quantize model to reduce size and improve inference speed.

        Args:
            method: "dynamic" or "static"
            backend: Quantization backend (qnnpack, fbgemm)
            calibration_loader: DataLoader for static quantization calibration

        Returns:
            Quantized model
        """
        logger.info(f"Quantizing model (method={method}, backend={backend})")

        try:
            if method == "dynamic":
                return self._quantize_dynamic(backend)
            elif method == "static":
                return self._quantize_static(backend, calibration_loader)
            else:
                raise ValueError(f"Unknown quantization method: {method}")

        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            raise

    def _quantize_dynamic(self, backend: str) -> nn.Module:
        """Apply dynamic quantization."""
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8,
        )

        logger.info(f"✓ Dynamic quantization applied (backend={backend})")
        return quantized_model

    def _quantize_static(
        self,
        backend: str,
        calibration_loader: Optional[DataLoader],
    ) -> nn.Module:
        """Apply static quantization with calibration."""
        if calibration_loader is None:
            raise ValueError("calibration_loader required for static quantization")

        # Prepare model for static quantization
        self.model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.quantization.prepare(self.model, inplace=True)

        # Calibrate on sample data
        logger.info("Calibrating model on sample data...")
        with torch.no_grad():
            for batch, _ in calibration_loader:
                batch = batch.to(self.device)
                self.model(batch)

        # Convert to quantized model
        torch.quantization.convert(self.model, inplace=True)

        logger.info(f"✓ Static quantization applied (backend={backend})")
        return self.model

    def prune(
        self,
        amount: float = 0.2,
        method: str = "structured",
        layer_names: Optional[list] = None,
    ) -> nn.Module:
        """
        Prune model to reduce parameters and FLOPs.

        Args:
            amount: Fraction of weights to prune (0.0-1.0)
            method: "structured" or "unstructured"
            layer_names: Specific layers to prune, or None for all

        Returns:
            Pruned model
        """
        logger.info(
            f"Pruning model (amount={amount}, method={method}, layers={layer_names or 'all'})"
        )

        try:
            # Get layers to prune
            layers_to_prune = self._get_layers_to_prune(layer_names)

            if method == "structured":
                return self._prune_structured(layers_to_prune, amount)
            elif method == "unstructured":
                return self._prune_unstructured(layers_to_prune, amount)
            else:
                raise ValueError(f"Unknown pruning method: {method}")

        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            raise

    def _get_layers_to_prune(self, layer_names: Optional[list]) -> list:
        """Get list of layers to prune."""
        if layer_names:
            return [(self.model, name) for name in layer_names if hasattr(self.model, name)]

        # Default: all Linear and Conv layers
        layers = []
        for _name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                layers.append((module, "weight"))

        return layers

    def _prune_structured(self, layers: list, amount: float) -> nn.Module:
        """Apply structured pruning (channel-level)."""
        import torch.nn.utils.prune as prune

        for layer, name in layers:
            prune.ln_structured(layer, name=name, amount=amount, n=2, dim=0)

        logger.info(f"✓ Structured pruning applied (amount={amount})")
        return self.model

    def _prune_unstructured(self, layers: list, amount: float) -> nn.Module:
        """Apply unstructured pruning (weight-level)."""
        import torch.nn.utils.prune as prune

        for layer, name in layers:
            prune.l1_unstructured(layer, name=name, amount=amount)

        # Remove pruning reparameterization to make permanent
        for layer, name in layers:
            prune.remove(layer, name)

        logger.info(f"✓ Unstructured pruning applied (amount={amount})")
        return self.model

    def get_compression_stats(self) -> Dict:
        """
        Get model compression statistics.

        Returns:
            Dictionary with size, sparsity, and parameter count
        """
        original_size = sum(p.numel() for p in self.model.parameters())
        original_bytes = original_size * 4  # Float32

        # Count pruned parameters
        pruned_params = 0
        for _name, param in self.model.named_parameters():
            if hasattr(param, "data_mask"):
                pruned_params += (param.data_mask == 0).sum().item()

        sparsity = pruned_params / original_size if original_size > 0 else 0

        return {
            "total_parameters": original_size,
            "original_size_bytes": original_bytes,
            "original_size_mb": original_bytes / (1024 * 1024),
            "pruned_parameters": pruned_params,
            "sparsity": sparsity,
            "remaining_parameters": original_size - pruned_params,
        }

    def benchmark_inference(
        self,
        input_shape: Tuple = (1, 3, 64, 64),
        num_runs: int = 100,
    ) -> Dict:
        """
        Benchmark model inference speed.

        Args:
            input_shape: Input tensor shape
            num_runs: Number of inference runs

        Returns:
            Benchmark results (latency, throughput)
        """
        import time

        logger.info(f"Benchmarking inference (num_runs={num_runs})")

        dummy_input = torch.randn(*input_shape, device=self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                self.model(dummy_input)

        # Benchmark
        torch.cuda.synchronize() if "cuda" in self.device else None

        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                self.model(dummy_input)

        torch.cuda.synchronize() if "cuda" in self.device else None
        elapsed_time = time.time() - start_time

        latency_ms = (elapsed_time / num_runs) * 1000
        throughput = num_runs / elapsed_time

        stats = {
            "total_time_s": elapsed_time,
            "num_runs": num_runs,
            "latency_ms": latency_ms,
            "throughput_samples_per_sec": throughput,
            "device": self.device,
        }

        logger.info(
            f"✓ Benchmark complete: {latency_ms:.2f}ms latency, {throughput:.1f} samples/sec"
        )

        return stats


def create_optimizer(
    model: nn.Module,
    device: str = "cpu",
) -> ModelOptimizer:
    """
    Factory function to create a model optimizer.

    Args:
        model: PyTorch model
        device: Device to use

    Returns:
        ModelOptimizer instance
    """
    return ModelOptimizer(model, device=device)
