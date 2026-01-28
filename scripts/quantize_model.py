#!/usr/bin/env python3
"""
Post-Training INT8 Quantization for Kanji Models
Converts trained PyTorch models to INT8 for efficient deployment
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import (
    get_dataset_directory,
    load_chunked_dataset,
    setup_logger,
    verify_and_setup_gpu,
)

logger = setup_logger(__name__)

# Suppress PyTorch's TypedStorage deprecation warning (internal, not in user code)
warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage.*")

# Add scripts to path for imports
sys.path.append(str(Path(__file__).parent))
from train_cnn_model import LightweightKanjiNet  # noqa: E402
from train_hiercode import HierCodeClassifier  # noqa: E402
from train_qat import QuantizableLightweightKanjiNet  # noqa: E402
from train_radical_rnn import RadicalRNNClassifier  # noqa: E402
from train_rnn import KanjiRNN  # noqa: E402
from train_vit import VisionTransformer  # noqa: E402


def quantize_model_int8(model: nn.Module, device: str = "cuda", model_name: str = "quantized"):
    """
    Convert model to INT8 using dynamic quantization.

    This uses dynamic quantization (QAT-free):
    - No retraining or calibration required
    - Quantizes weights only (activations stay float32)
    - Reduces model size by ~4x
    - Minimal accuracy loss
    - Works cross-platform (Windows, Linux, Mac)

    Args:
        model: PyTorch model to quantize
        device: Device to use (always cuda)
        model_name: Name for logging
    """

    logger.info("🔄 Quantizing %s model to INT8 (dynamic)...", model_name)

    # Keep model on GPU for all calculations
    model = model.to(device)
    model.eval()

    # Calculate original size
    original_state = model.state_dict()
    original_size = sum(
        v.numel() * v.element_size() for v in original_state.values() if v is not None
    )
    logger.info("  → Original model size: %.2f MB", original_size / 1e6)

    # Dynamic quantization requires CPU, so temporarily move only for this operation
    # PyTorch limitation: dynamic quantization kernel only available on CPU backend
    logger.info("  → Applying INT8 quantization...")
    model_cpu = model.cpu()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
    # Move quantized model back to GPU if needed for downstream operations
    quantized_model = quantized_model.to(device)
    logger.info("✓ INT8 quantization complete")

    # For dynamic quantization, the actual size is measured after saving
    # (packed parameters have a different memory layout)
    # Return a sentinel value that will be updated after saving
    quantized_size = None

    return quantized_model, original_size, quantized_size


def quantize_with_calibration(
    model: nn.Module,
    train_loader,
    device: str = "cuda",
    model_name: str = "quantized",
):
    """
    Quantize model with calibration using actual training data.

    This improves quantization accuracy by:
    - Using real data distribution for calibration
    - Computing optimal scale factors
    - Reducing accuracy loss from ~2-3% to <1%

    Args:
        model: PyTorch model to quantize
        train_loader: DataLoader for calibration
        device: Device to use (always cuda for GPU acceleration)
        model_name: Name for logging
    """

    logger.info("🔄 Quantizing %s model with calibration...", model_name)

    model = model.to(device)
    model.eval()

    # Calculate original size
    original_state = model.state_dict()
    original_size = sum(
        v.numel() * v.element_size() for v in original_state.values() if isinstance(v, torch.Tensor)
    )
    logger.info("  → Original model size: %.2f MB", original_size / 1e6)

    # Prepare for quantization
    qconfig = torch.quantization.get_default_qconfig("fbgemm")
    model.qconfig = qconfig  # type: ignore
    torch.quantization.prepare(model, inplace=True)

    # Calibration: Run model on subset of training data
    num_batches = min(100, len(train_loader))  # Use 100 batches for calibration
    logger.info("  → Running calibration on %d batches...", num_batches)
    with torch.no_grad():
        for i, (images, _labels) in enumerate(train_loader):
            if i >= num_batches:
                break

            images = images.to(device)

            # Forward pass to collect activation statistics
            if hasattr(model, "forward"):
                _ = model(images)

            if (i + 1) % 25 == 0:
                logger.info("    → Calibrated %d/%d batches", i + 1, num_batches)

    # Convert to quantized model
    logger.info("  → Converting to INT8...")
    torch.quantization.convert(model, inplace=True)
    logger.info("✓ Calibrated INT8 quantization complete")

    quantized_state = model.state_dict()
    quantized_size = sum(
        v.numel() * v.element_size()
        for v in quantized_state.values()
        if isinstance(v, torch.Tensor)
    )

    return model, original_size, quantized_size


def evaluate_quantized_model(model, test_loader, criterion, device: str = "cuda"):
    """
    Evaluate quantized model accuracy.

    Note: PyTorch dynamic quantization only has CPU kernels available.
    For GPU inference, export to ONNX format which supports GPU-accelerated int8 inference.

    Args:
        model: Quantized model
        test_loader: DataLoader for test set
        criterion: Loss function
        device: Device specified (note: inference runs on CPU due to PyTorch backend limitations)
    """
    logger.info("🧪 Evaluating quantized model...")

    # Dynamically quantized models only support CPU inference (PyTorch backend limitation)
    # For GPU inference in production, use ONNX export instead
    model = model.cpu()
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for _i, (images, labels) in enumerate(test_loader):
            # Keep on CPU for quantized model evaluation
            images = images.cpu()
            labels = labels.cpu()

            outputs = model(images)

            # Handle dict outputs (e.g., from Hi-GITA model)
            if isinstance(outputs, dict):
                logits = outputs.get("logits", outputs)
            else:
                logits = outputs

            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)
    logger.info("✓ Evaluation complete: Accuracy %.2f%%, Loss %.4f", accuracy, avg_loss)

    return accuracy, avg_loss


def main():
    parser = argparse.ArgumentParser(
        description="Post-Training INT8 Quantization for Kanji Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize HierCode model
  python quantize_model.py --model-path training/hiercode/hiercode_model_best.pth --model-type hiercode

  # Quantize with calibration
  python quantize_model.py --model-path training/cnn/best_kanji_model.pth --model-type cnn --calibrate

  # Evaluate quantized model
  python quantize_model.py --model-path training/cnn/best_kanji_model.pth --evaluate
        """,
    )

    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["hiercode", "hiercode-higita", "cnn", "qat", "rnn", "radical-rnn", "vit"],
        help="Model architecture type (required)",
    )
    # Note: Dataset auto-detected via get_dataset_directory() from optimization_config
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Use calibration with training data for better accuracy",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate quantized model on test set",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for quantized model (default: auto-generated)",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("POST-TRAINING INT8 QUANTIZATION")
    logger.info("=" * 70)
    logger.info(
        "Model: %s | Calibrate: %s | Evaluate: %s", args.model_type, args.calibrate, args.evaluate
    )

    # Verify GPU availability (required for quantization)
    device = verify_and_setup_gpu()

    # Load model

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error("✗ Model path not found: %s", model_path)
        return

    logger.info("→ Loading model from %s...", model_path)

    # Load checkpoint to infer num_classes from classifier layer
    checkpoint = torch.load(model_path, map_location="cpu")

    # Debug: show all checkpoint keys
    logger.info("→ Checkpoint contains %d keys", len(checkpoint))
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], torch.Tensor):
            logger.info("  Key: '%s' → shape %s", key, checkpoint[key].shape)
        else:
            logger.info("  Key: '%s' → type %s", key, type(checkpoint[key]))

    # Extract model_state_dict if checkpoint is wrapped
    if "model_state_dict" in checkpoint:
        logger.info("→ Checkpoint is wrapped format, extracting model_state_dict...")
        model_state_dict = checkpoint["model_state_dict"]
    else:
        model_state_dict = checkpoint

    # Infer num_classes from the OUTPUT classifier weight (last one)
    # Look for patterns: classifier.X.weight where X is the highest number
    num_classes = None
    classifier_weights = {}

    for key in model_state_dict.keys():
        if (
            isinstance(model_state_dict[key], torch.Tensor)
            and "classifier" in key
            and "weight" in key
        ):
            # Extract layer number if present (e.g., "classifier.4" -> 4)
            parts = key.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                layer_idx = int(parts[1])
                classifier_weights[layer_idx] = (key, model_state_dict[key].shape[0])

    # Use the highest indexed classifier layer (typically the output layer)
    if classifier_weights:
        max_layer = max(classifier_weights.keys())
        key, num_classes = classifier_weights[max_layer]
        logger.info(
            "  → Found output classifier in '%s': shape %s → %d classes",
            key,
            model_state_dict[key].shape,
            num_classes,
        )
    else:
        # Fallback: look for any large weight matrix
        for key in model_state_dict.keys():
            if (
                isinstance(model_state_dict[key], torch.Tensor)
                and len(model_state_dict[key].shape) >= 2
            ):
                if model_state_dict[key].shape[0] > 1000:  # Likely num_classes
                    num_classes = model_state_dict[key].shape[0]
                    logger.info(
                        "  → Inferred from large weight '%s': shape %s → %d classes",
                        key,
                        model_state_dict[key].shape,
                        num_classes,
                    )
                    break

    if num_classes is None:
        # Fallback: try to load config
        config_path = model_path.parent / f"{args.model_type}_config.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config_dict = json.load(f)
                num_classes = config_dict.get("num_classes", 3036)
                logger.info("  → Found in config.json: %d classes", num_classes)
        else:
            num_classes = 3036
            logger.warning("  → Using default: %d classes", num_classes)

    logger.info("→ Detected %d classes from model checkpoint", num_classes)

    # Create config and model based on type
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.lib.config import (
        CNNConfig,
        HierCodeConfig,
        OptimizationConfig,
        QATConfig,
        RadicalRNNConfig,
        RNNConfig,
        ViTConfig,
    )

    if args.model_type == "hiercode":
        config: OptimizationConfig = HierCodeConfig(num_classes=num_classes)
        model = HierCodeClassifier(num_classes=num_classes, config=config)  # type: ignore
    elif args.model_type == "hiercode-higita":
        config = HierCodeConfig(num_classes=num_classes)
        from hiercode_higita_enhancement import HierCodeWithHiGITA

        model = HierCodeWithHiGITA(num_classes=num_classes)
    elif args.model_type == "cnn":
        config = CNNConfig(num_classes=num_classes)
        model = LightweightKanjiNet(num_classes=num_classes)
    elif args.model_type == "qat":
        config = QATConfig(num_classes=num_classes)
        model = QuantizableLightweightKanjiNet(num_classes=num_classes)
    elif args.model_type == "rnn":
        config = RNNConfig(num_classes=num_classes)
        model = KanjiRNN(num_classes=num_classes)
    elif args.model_type == "radical-rnn":
        config = RadicalRNNConfig(num_classes=num_classes)
        model = RadicalRNNClassifier(num_classes=num_classes, config=config)
    elif args.model_type == "vit":
        config = ViTConfig(num_classes=num_classes)
        model = VisionTransformer(num_classes=num_classes, config=config)
    else:
        logger.error("✗ Unknown model type: %s", args.model_type)
        return

    # Load state dict with flexible key matching
    try:
        model.load_state_dict(model_state_dict)
        logger.info("✓ Model loaded successfully")
    except RuntimeError:
        # If there's a mismatch, try to detect the actual num_classes from state_dict
        # This handles cases where checkpoint was trained with different dataset size
        checkpoint_num_classes = None

        for key in model_state_dict.keys():
            if (
                "classifier" in key
                and "weight" in key
                and isinstance(model_state_dict[key], torch.Tensor)
            ):
                # Get the first dimension (output size) of any classifier weight
                checkpoint_num_classes = model_state_dict[key].shape[0]
                logger.warning(
                    "⚠ Detected num_classes mismatch: checkpoint has %d, model has %d",
                    checkpoint_num_classes,
                    num_classes,
                )
                break

        if checkpoint_num_classes and checkpoint_num_classes != num_classes:
            logger.info(
                "→ Recreating model with checkpoint's num_classes=%d", checkpoint_num_classes
            )
            # Recreate model with correct num_classes
            num_classes = checkpoint_num_classes

            if args.model_type == "hiercode":
                config = HierCodeConfig(num_classes=num_classes)
                model = HierCodeClassifier(num_classes=num_classes, config=config)  # type: ignore
            elif args.model_type == "hiercode-higita":
                config = HierCodeConfig(num_classes=num_classes)
                from hiercode_higita_enhancement import HierCodeWithHiGITA

                model = HierCodeWithHiGITA(num_classes=num_classes)
            elif args.model_type == "cnn":
                config = CNNConfig(num_classes=num_classes)
                model = LightweightKanjiNet(num_classes=num_classes)
            elif args.model_type == "qat":
                config = QATConfig(num_classes=num_classes)
                model = QuantizableLightweightKanjiNet(num_classes=num_classes)
            elif args.model_type == "rnn":
                config = RNNConfig(num_classes=num_classes)
                model = KanjiRNN(num_classes=num_classes)
            elif args.model_type == "radical-rnn":
                config = RadicalRNNConfig(num_classes=num_classes)
                model = RadicalRNNClassifier(num_classes=num_classes, config=config)
            elif args.model_type == "vit":
                config = ViTConfig(num_classes=num_classes)
                model = VisionTransformer(num_classes=num_classes, config=config)

            # Try loading again with new model
            try:
                model.load_state_dict(model_state_dict)
                logger.info("✓ Model loaded successfully with checkpoint's num_classes")
            except RuntimeError:
                model.load_state_dict(model_state_dict, strict=False)
                logger.warning("⚠ Loaded model with strict=False (some keys may not match)")
        else:
            # Original error, try strict=False
            model.load_state_dict(model_state_dict, strict=False)
            logger.warning("⚠ Loaded model with strict=False (some keys may not match)")

    # Quantize
    if args.calibrate:
        # Load dataset for calibration (auto-detected)
        logger.info("→ Loading dataset for calibration...")
        data_dir = str(get_dataset_directory())
        X, y = load_chunked_dataset(data_dir)

        # Split data (use prepare_dataset_and_loaders instead)
        import numpy as np
        from torch.utils.data import TensorDataset

        from src.lib import prepare_dataset_and_loaders

        def create_tensor_dataset(x, y):
            """Factory for TensorDataset"""
            if x.ndim == 2 and x.shape[1] == 4096:
                x = x.reshape(-1, 64, 64)
            if x.ndim == 3:
                x = x[:, np.newaxis, :, :]
            x_tensor = torch.from_numpy(x).float()
            y_tensor = torch.from_numpy(y).long()
            return TensorDataset(x_tensor, y_tensor)

        (train_X, train_y), num_classes, train_loader, val_loader = prepare_dataset_and_loaders(
            data_dir=data_dir,
            dataset_fn=create_tensor_dataset,
            batch_size=config.batch_size,
            sample_limit=None,
            logger=logger,
        )

        # Use train_loader for quantization (validation not needed for calibration)

        quantized_model, orig_size, quant_size = quantize_with_calibration(
            model, train_loader, device=device, model_name=args.model_type
        )
    else:
        quantized_model, orig_size, quant_size = quantize_model_int8(
            model, device=device, model_name=args.model_type
        )

    # Save quantized model
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = model_path.parent / f"quantized_{args.model_type}_int8.pth"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("→ Saving quantized model to %s...", output_path)
    torch.save(quantized_model.state_dict(), output_path)

    # Measure actual file size after saving (especially important for dynamic quantization)
    if quant_size is None and output_path.exists():
        quant_size = output_path.stat().st_size

    logger.info("✓ Quantized model saved")
    logger.info("  → Original size: %.2f MB", orig_size / 1e6)
    logger.info("  → Quantized size: %.2f MB", quant_size / 1e6 if quant_size else 0)
    if quant_size and orig_size:
        logger.info("  → Reduction: %.1f%%", 100 * (1 - quant_size / orig_size))

    # Evaluate if requested
    if args.evaluate:
        logger.info("→ Loading dataset for evaluation...")
        data_dir = str(get_dataset_directory())

        try:
            X, y = load_chunked_dataset(data_dir)

            # Filter to only include classes the model was trained on
            import numpy as np
            from torch.utils.data import DataLoader, TensorDataset

            valid_mask = y < num_classes
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) < len(y):
                logger.warning(
                    "⚠ Filtered dataset from %d to %d samples (model trained on %d classes, dataset has %d)",
                    len(y),
                    len(valid_indices),
                    num_classes,
                    y.max() + 1,
                )

            X_filtered = X[valid_indices]
            y_filtered = y[valid_indices]

            # Split data: 90% train, 9% val, 1% test
            n_samples = len(X_filtered)
            n_train = int(0.9 * n_samples)
            n_val = int(0.09 * n_samples)

            indices = np.random.permutation(n_samples)
            test_indices = indices[n_train + n_val :]

            X_test = X_filtered[test_indices]
            y_test = y_filtered[test_indices]

            logger.info(
                "  → Test set: %d samples, %d unique classes", len(y_test), len(np.unique(y_test))
            )

            # Convert to tensors
            if X_test.ndim == 2 and X_test.shape[1] == 4096:
                X_test = X_test.reshape(-1, 64, 64)
            if X_test.ndim == 3:
                X_test = X_test[:, np.newaxis, :, :]

            X_test_tensor = torch.from_numpy(X_test).float()
            y_test_tensor = torch.from_numpy(y_test).long()
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            # Evaluate
            criterion = nn.CrossEntropyLoss()
            accuracy, loss = evaluate_quantized_model(
                quantized_model, test_loader, criterion, device=device
            )

            # Save evaluation results
            results = {
                "model_type": args.model_type,
                "model_num_classes": num_classes,
                "dataset_num_classes": int(y.max()) + 1,
                "test_samples": int(len(y_test)),
                "original_size_mb": orig_size / 1e6,
                "quantized_size_mb": quant_size / 1e6 if quant_size else 0,
                "size_reduction": (orig_size - quant_size) / orig_size if quant_size else 0,
                "quantized_accuracy": accuracy,
                "quantized_loss": loss,
                "calibrated": args.calibrate,
            }

            results_path = model_path.parent / f"quantization_results_{args.model_type}.json"
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            logger.info("✓ Evaluation results saved to %s", results_path)

        except Exception as e:
            logger.error("✗ Evaluation failed: %s", str(e))
            logger.info("  Check that model was trained on the same dataset you're evaluating on")

    logger.info("=" * 70)
    logger.info("✓ Quantization complete!")


if __name__ == "__main__":
    main()
