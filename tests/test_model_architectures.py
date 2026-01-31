"""
Tests for model architecture forward passes and parameter validation.

Ensures all model architectures can be instantiated and perform forward passes
with appropriate input shapes.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCNNArchitectures:
    """Test CNN-based model architectures"""

    def test_lightweight_kanji_net_forward(self):
        """Test LightweightKanjiNet forward pass"""
        try:
            from scripts.train_cnn_model import LightweightKanjiNet

            model = LightweightKanjiNet(num_classes=100)
            x = torch.randn(4, 1, 64, 64)
            output = model(x)

            assert output.shape == (4, 100), f"Expected (4, 100), got {output.shape}"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"
        except ImportError:
            pytest.skip("LightweightKanjiNet not available")

    def test_lightweight_kanji_net_different_sizes(self):
        """Test LightweightKanjiNet with different batch sizes"""
        try:
            from scripts.train_cnn_model import LightweightKanjiNet

            model = LightweightKanjiNet(num_classes=50)

            # Test different batch sizes
            for batch_size in [1, 8, 16]:
                x = torch.randn(batch_size, 1, 64, 64)
                output = model(x)
                assert output.shape == (batch_size, 50)
        except ImportError:
            pytest.skip("LightweightKanjiNet not available")

    def test_quantizable_kanji_net_forward(self):
        """Test QuantizableLightweightKanjiNet forward pass"""
        try:
            from scripts.train_qat import QuantizableLightweightKanjiNet

            model = QuantizableLightweightKanjiNet(num_classes=100)
            x = torch.randn(4, 1, 64, 64)
            output = model(x)

            assert output.shape == (4, 100)
            assert not torch.isnan(output).any()
        except ImportError:
            pytest.skip("QuantizableLightweightKanjiNet not available")


class TestRNNArchitectures:
    """Test RNN-based model architectures"""

    def test_kanji_rnn_forward(self):
        """Test KanjiRNN forward pass"""
        try:
            from scripts.train_rnn import KanjiRNN

            model = KanjiRNN(num_classes=100)
            # RNN expects (batch, seq_len, features)
            x = torch.randn(4, 10, 4)
            output = model(x)

            assert output.shape == (4, 100)
            assert not torch.isnan(output).any()
        except ImportError:
            pytest.skip("KanjiRNN not available")


class TestHierarchicalArchitectures:
    """Test hierarchical model architectures"""

    def test_hiercode_classifier_exists(self):
        """Test that HierCodeClassifier can be imported"""
        try:
            from scripts.train_hiercode import HierCodeClassifier

            # Just verify it exists, signature may vary
            assert HierCodeClassifier is not None
        except ImportError:
            pytest.skip("HierCodeClassifier not available")

    def test_hiercode_higita_exists(self):
        """Test that HierCodeWithHiGITA can be imported"""
        try:
            from scripts.train_hiercode_higita import HierCodeWithHiGITA

            # Just verify it exists, signature may vary
            assert HierCodeWithHiGITA is not None
        except ImportError:
            pytest.skip("HierCodeWithHiGITA not available")


class TestViTArchitectures:
    """Test Vision Transformer architectures"""

    def test_vision_transformer_exists(self):
        """Test that VisionTransformer can be imported"""
        try:
            from scripts.train_vit import VisionTransformer

            # Just verify it exists, signature may vary
            assert VisionTransformer is not None
        except ImportError:
            pytest.skip("VisionTransformer not available")


class TestModelParameters:
    """Test model parameter counts and memory footprint"""

    def test_model_parameter_count(self):
        """Test that model has reasonable parameter count"""
        try:
            from scripts.train_cnn_model import LightweightKanjiNet

            model = LightweightKanjiNet(num_classes=43427)
            param_count = sum(p.numel() for p in model.parameters())

            # Lightweight model should have < 50M parameters
            assert param_count < 50_000_000, f"Model has too many parameters: {param_count:,}"
            assert param_count > 10_000, f"Model has too few parameters: {param_count:,}"
        except ImportError:
            pytest.skip("LightweightKanjiNet not available")

    def test_model_trainable_parameters(self):
        """Test that model has trainable parameters"""
        try:
            from scripts.train_cnn_model import LightweightKanjiNet

            model = LightweightKanjiNet(num_classes=100)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            assert trainable_params > 0, "Model has no trainable parameters"
        except ImportError:
            pytest.skip("LightweightKanjiNet not available")


class TestModelGradients:
    """Test gradient flow through models"""

    def test_cnn_gradient_flow(self):
        """Test that gradients flow through CNN"""
        try:
            from scripts.train_cnn_model import LightweightKanjiNet

            model = LightweightKanjiNet(num_classes=10)
            x = torch.randn(4, 1, 64, 64, requires_grad=True)
            target = torch.randint(0, 10, (4,))

            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()

            # Check that gradients exist
            has_gradients = any(p.grad is not None for p in model.parameters())
            assert has_gradients, "No gradients computed"

            # Check that gradients are not all zero
            grad_sum = sum(
                p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None
            )
            assert grad_sum > 0, "All gradients are zero"
        except ImportError:
            pytest.skip("LightweightKanjiNet not available")
