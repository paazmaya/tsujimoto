"""
Tests for model architecture forward passes, parameter validation, and gradient flow.

Directly tests all production model architectures defined in ``src/lib/models.py``.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib.models import (
    HierCodeHiGITA,
    HierCodeNet,
    KanjiRNN,
    KanjiViT,
    LightweightKanjiNet,
    QuantizableLightweightKanjiNet,
)


class TestCNNArchitectures:
    """Test CNN-based model architectures."""

    def test_lightweight_kanji_net_forward(self):
        """Test LightweightKanjiNet forward pass shape and numerical stability."""
        model = LightweightKanjiNet(num_classes=100, input_channels=1, image_size=64)
        x = torch.randn(4, 1, 64, 64)
        output = model(x)

        assert output.shape == (4, 100), f"Expected (4, 100), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_lightweight_kanji_net_different_sizes(self):
        """Test LightweightKanjiNet with different batch sizes."""
        model = LightweightKanjiNet(num_classes=50)
        model.eval()

        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, 1, 64, 64)
            output = model(x)
            assert output.shape == (batch_size, 50)

    def test_quantizable_kanji_net_forward(self):
        """Test QuantizableLightweightKanjiNet forward pass with QuantStub/DeQuantStub."""
        model = QuantizableLightweightKanjiNet(num_classes=100)
        x = torch.randn(4, 1, 64, 64)
        output = model(x)

        assert output.shape == (4, 100)
        assert not torch.isnan(output).any()
        assert hasattr(model, "quant")
        assert hasattr(model, "dequant")


class TestRNNArchitectures:
    """Test RNN-based model architectures."""

    @pytest.mark.parametrize("rnn_type", ["lstm", "gru", "rnn"])
    def test_kanji_rnn_forward_cell_types(self, rnn_type: str):
        """Test KanjiRNN forward pass across LSTM, GRU, and vanilla RNN cell types."""
        model = KanjiRNN(
            num_classes=100,
            hidden_size=64,
            num_layers=2,
            rnn_type=rnn_type,
            input_channels=1,
            image_size=64,
        )
        x = torch.randn(4, 1, 64, 64)
        output = model(x)

        assert output.shape == (4, 100), f"Expected (4, 100), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_kanji_rnn_gradient_flow(self):
        """Test gradient flow through KanjiRNN."""
        model = KanjiRNN(num_classes=16, hidden_size=32, num_layers=1, rnn_type="lstm")
        x = torch.randn(4, 1, 64, 64, requires_grad=True)
        target = torch.randint(0, 16, (4,))

        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        assert x.grad is not None
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


class TestHierarchicalArchitectures:
    """Test hierarchical codebook model architectures."""

    @pytest.mark.parametrize("batch_size", [2, 8])
    def test_hiercode_net_forward_and_gradients(self, batch_size: int):
        """Test HierCodeNet forward pass, nearest codebook selection, and gradient flow."""
        model = HierCodeNet(
            num_classes=100,
            codebook_size=64,
            num_levels=3,
            input_channels=1,
            image_size=64,
        )
        x = torch.randn(batch_size, 1, 64, 64, requires_grad=True)
        output = model(x)

        assert output.shape == (batch_size, 100)
        assert not torch.isnan(output).any()

        target = torch.randint(0, 100, (batch_size,))
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        assert x.grad is not None
        assert model.conv1.weight.grad is not None
        assert model.codebooks[0].weight.grad is not None

    @pytest.mark.parametrize("batch_size", [2, 8])
    def test_hiercode_higita_forward_and_gradients(self, batch_size: int):
        """Test HierCodeHiGITA preserves batch dimension and differentiates input images."""
        model = HierCodeHiGITA(
            num_classes=50,
            codebook_size=64,
            input_channels=1,
            image_size=64,
        )
        model.eval()
        x = torch.randn(batch_size, 1, 64, 64, requires_grad=True)
        output = model(x)

        assert output.shape == (batch_size, 50)
        assert not torch.isnan(output).any()
        # Distinct inputs in a batch should produce distinct logits
        assert not torch.allclose(output[0], output[1])

        model.train()
        out_train = model(x)
        target = torch.randint(0, 50, (batch_size,))
        loss = torch.nn.functional.cross_entropy(out_train, target)
        loss.backward()
        assert x.grad is not None
        assert model.conv1.weight.grad is not None


class TestViTArchitectures:
    """Test Vision Transformer architectures."""

    @pytest.mark.parametrize("patch_size", [4, 8, 16])
    def test_kanji_vit_patch_sizes(self, patch_size: int):
        """Test KanjiViT forward pass with various configured patch sizes."""
        model = KanjiViT(
            num_classes=32,
            image_size=64,
            patch_size=patch_size,
            dim=64,
            depth=2,
            num_heads=4,
            mlp_dim=128,
            input_channels=1,
        )
        x = torch.randn(3, 1, 64, 64)
        output = model(x)

        assert output.shape == (3, 32)
        assert not torch.isnan(output).any()

    def test_kanji_vit_invalid_patch_size_raises(self):
        """Test KanjiViT rejects image_size not divisible by patch_size."""
        with pytest.raises(AssertionError):
            KanjiViT(num_classes=10, image_size=64, patch_size=7)


class TestModelParameters:
    """Test model parameter counts and memory footprint."""

    def test_model_parameter_count(self):
        """Test that model has reasonable parameter count."""
        model = LightweightKanjiNet(num_classes=43427)
        param_count = sum(p.numel() for p in model.parameters())

        assert param_count < 150_000_000, f"Model has too many parameters: {param_count:,}"
        assert param_count > 10_000, f"Model has too few parameters: {param_count:,}"

    def test_model_trainable_parameters(self):
        """Test that all 6 models have trainable parameters."""
        models = [
            LightweightKanjiNet(num_classes=20),
            KanjiRNN(num_classes=20, hidden_size=32, num_layers=1),
            KanjiViT(num_classes=20, patch_size=8, dim=32, depth=1, num_heads=4, mlp_dim=64),
            HierCodeNet(num_classes=20, codebook_size=32),
            QuantizableLightweightKanjiNet(num_classes=20),
            HierCodeHiGITA(num_classes=20, codebook_size=32),
        ]
        for model in models:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert trainable_params > 0, f"{type(model).__name__} has no trainable parameters"


class TestModelGradients:
    """Test gradient flow through models."""

    def test_cnn_gradient_flow(self):
        """Test that gradients flow through LightweightKanjiNet."""
        model = LightweightKanjiNet(num_classes=10)
        x = torch.randn(4, 1, 64, 64, requires_grad=True)
        target = torch.randint(0, 10, (4,))

        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients, "No gradients computed"

        grad_sum = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        assert grad_sum > 0, "All gradients are zero"
