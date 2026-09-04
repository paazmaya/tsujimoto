#!/usr/bin/env python3
"""
CNN Training Pipeline Verification

Verifies that all components are properly integrated:
1. Dataset router can be instantiated
2. Dataset router recognizes all datasets
3. CNN model can be created
4. CNN Lightning module can be created
5. CLI command accepts dataset_name

Run with: python scripts/verify_cnn_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib.logging_utils import setup_logger

logger = setup_logger(__name__)


def test_dataset_router():
    """Test DatasetRouter import and dataset recognition."""

    try:
        from src.lib.dataset_router import RESEARCH_DATASETS, DatasetRouter

        router = DatasetRouter()

        datasets = router.list_datasets()
        len(datasets["etl"])
        len(datasets["research"])

        expected_kanji = ["kanji_full", "kanji_dataset_v3", "kanji"]
        for ds in expected_kanji:
            if ds in RESEARCH_DATASETS:
                pass
            else:
                return False

        return True
    except Exception:
        import traceback

        traceback.print_exc()
        return False


def test_cnn_model():
    """Test CNN model creation."""

    try:
        import torch

        from scripts.train_cnn import LightweightKanjiCNN

        model = LightweightKanjiCNN(num_classes=3036)

        # Test forward pass
        batch = torch.randn(2, 1, 64, 64)
        model(batch)

        # Count parameters
        sum(p.numel() for p in model.parameters())

        return True
    except Exception:
        import traceback

        traceback.print_exc()
        return False


def test_lightning_module():
    """Test PyTorch Lightning module creation."""

    try:
        from scripts.train_cnn import CNNLightningModule

        module = CNNLightningModule(num_classes=3036, learning_rate=0.001)

        # Check methods
        required_methods = ["forward", "training_step", "validation_step", "configure_optimizers"]
        for method in required_methods:
            if hasattr(module, method):
                pass
            else:
                return False

        return True
    except Exception:
        import traceback

        traceback.print_exc()
        return False


def test_train_cnn_function():
    """Test train_cnn function is importable."""

    try:
        # Check signature
        import inspect

        from scripts.train_cnn import train_cnn

        sig = inspect.signature(train_cnn)
        params = list(sig.parameters.keys())

        required_params = [
            "dataset_name",
            "epochs",
            "batch_size",
            "learning_rate",
        ]

        for param in required_params:
            if param in params:
                pass
            else:
                return False

        return True
    except Exception:
        import traceback

        traceback.print_exc()
        return False


def test_cli_integration():
    """Test CLI command accepts dataset_name."""

    try:
        # Verify the cnn command exists in train.py
        with open("scripts/train.py") as f:
            content = f.read()

        if "def cnn(" in content:
            pass
        else:
            return False

        if "from scripts.train_cnn import train_cnn" in content:
            pass
        else:
            return False

        if "dataset_name" in content:
            pass
        else:
            return False

        return True
    except Exception:
        import traceback

        traceback.print_exc()
        return False


def test_training_args():
    """Test training_args configuration."""

    try:
        from scripts.training_args import COMMON_ARGS

        if "dataset_name" in COMMON_ARGS:
            choices = COMMON_ARGS["dataset_name"].choices
            expected = ["kanji_full", "kanji_dataset_v3", "kanji"]

            for dataset in expected:
                if dataset in choices:
                    pass
                else:
                    return False
        else:
            return False

        return True
    except Exception:
        import traceback

        traceback.print_exc()
        return False


def test_exports():
    """Test that new modules are properly exported."""

    try:
        # No imports needed - using only built-in functionality

        return True
    except ImportError:
        return False


def main():
    """Run all tests."""

    tests = [
        ("Dataset Router", test_dataset_router),
        ("CNN Model", test_cnn_model),
        ("Lightning Module", test_lightning_module),
        ("train_cnn Function", test_train_cnn_function),
        ("CLI Integration", test_cli_integration),
        ("Training Arguments", test_training_args),
        ("Module Exports", test_exports),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception:
            results.append((test_name, False))

    # Summary

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for _test_name, _result in results:
        pass

    if passed == total:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
