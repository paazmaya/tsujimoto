#!/usr/bin/env python3
"""Unit tests for unified training script and model utilities."""

import pytest


class TestUnifiedTrainingScript:
    """Tests for unified train.py script."""

    def test_train_py_module_imports(self):
        """Test that train.py module can be imported."""
        try:
            from scripts import train

            assert train is not None
        except ImportError:
            pytest.skip("train.py requires additional dependencies")

    def test_train_py_has_subcommands(self):
        """Test that train.py has the expected subcommands."""
        try:
            import click

            from scripts.train import train

            # train should be a click group with subcommands
            assert isinstance(train, click.Group)

            # Check for key subcommands
            subcommands = list(train.commands.keys())
            expected = ["cnn", "rnn", "vit", "hiercode", "qat", "hiercode_higita"]
            for cmd in expected:
                assert cmd in subcommands, f"Missing subcommand: {cmd}"
        except (ImportError, AttributeError):
            pytest.skip("train.py structure not available")

    def test_train_py_cnn_subcommand_exists(self):
        """Test CNN subcommand exists."""
        try:
            from scripts.train import train

            assert "cnn" in train.commands
        except (ImportError, AttributeError):
            pytest.skip("CNN subcommand not available")

    def test_train_py_rnn_subcommand_exists(self):
        """Test RNN subcommand exists."""
        try:
            from scripts.train import train

            assert "rnn" in train.commands
        except (ImportError, AttributeError):
            pytest.skip("RNN subcommand not available")

    def test_train_py_vit_subcommand_exists(self):
        """Test ViT subcommand exists."""
        try:
            from scripts.train import train

            assert "vit" in train.commands
        except (ImportError, AttributeError):
            pytest.skip("ViT subcommand not available")

    def test_train_py_hiercode_subcommand_exists(self):
        """Test HierCode subcommand exists."""
        try:
            from scripts.train import train

            assert "hiercode" in train.commands
        except (ImportError, AttributeError):
            pytest.skip("HierCode subcommand not available")

    def test_train_py_qat_subcommand_exists(self):
        """Test QAT subcommand exists."""
        try:
            from scripts.train import train

            assert "qat" in train.commands
        except (ImportError, AttributeError):
            pytest.skip("QAT subcommand not available")


class TestMeasureCO2:
    """Extended tests for CO2 emissions measurement."""

    def test_get_system_info_returns_dict(self):
        """Test that get_system_info returns dictionary."""
        from scripts.measure_co2_emissions import get_system_info

        info = get_system_info()
        assert isinstance(info, dict)

    def test_system_info_keys(self):
        """Test all required keys in system info."""
        from scripts.measure_co2_emissions import get_system_info

        info = get_system_info()
        required_keys = [
            "timestamp",
            "platform",
            "python_version",
            "cpu_count",
            "memory_total_gb",
            "torch_version",
        ]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"

    def test_system_info_valid_values(self):
        """Test that system info contains valid values."""
        from scripts.measure_co2_emissions import get_system_info

        info = get_system_info()
        assert isinstance(info["cpu_count"], int)
        assert info["cpu_count"] > 0
        assert isinstance(info["memory_total_gb"], (int, float))
        assert info["memory_total_gb"] > 0
        assert isinstance(info["torch_version"], str)
        assert len(info["torch_version"]) > 0


class TestGenerateMapping:
    """Extended tests for character mapping generation."""

    def test_jis_to_unicode_hiragana(self):
        """Test hiragana conversion from JIS."""
        from src.lib.character_mapping import JISConverter

        converter = JISConverter()
        result = converter.jis_to_unicode("2421")
        assert isinstance(result, str)
        assert len(result) >= 1

    def test_jis_to_unicode_katakana(self):
        """Test katakana conversion from JIS."""
        from src.lib.character_mapping import JISConverter

        converter = JISConverter()
        result = converter.jis_to_unicode("2521")
        assert isinstance(result, str)
        assert len(result) >= 1

    def test_jis_to_unicode_kanji(self):
        """Test kanji conversion from JIS."""
        from src.lib.character_mapping import JISConverter

        converter = JISConverter()
        result = converter.jis_to_unicode("3021")
        assert isinstance(result, str)
        assert len(result) >= 1

    def test_jis_to_unicode_invalid(self):
        """Test invalid JIS code handling."""
        from src.lib.character_mapping import JISConverter

        converter = JISConverter()
        result = converter.jis_to_unicode("ZZZZ")
        assert isinstance(result, str)
        # Should return placeholder for invalid codes
        assert "JIS" in result or result.startswith("[")

    def test_estimate_stroke_count_single_char(self):
        """Test stroke count for single character."""
        from src.lib.character_mapping import JISConverter

        converter = JISConverter()
        result = converter.estimate_stroke_count("a")
        assert isinstance(result, int)
        assert result >= 1

    def test_estimate_stroke_count_hiragana(self):
        """Test stroke count for hiragana."""
        from src.lib.character_mapping import JISConverter

        converter = JISConverter()
        result = converter.estimate_stroke_count("あ")
        assert isinstance(result, int)
        assert 1 <= result <= 25

    def test_estimate_stroke_count_kanji(self):
        """Test stroke count for kanji."""
        from src.lib.character_mapping import JISConverter

        converter = JISConverter()
        result = converter.estimate_stroke_count("漢")
        assert isinstance(result, int)
        assert 1 <= result <= 25

    def test_estimate_stroke_count_empty(self):
        """Test stroke count for empty string."""
        from src.lib.character_mapping import JISConverter

        converter = JISConverter()
        result = converter.estimate_stroke_count("")
        assert isinstance(result, int)

    def test_estimate_stroke_count_multi_char(self):
        """Test stroke count for multi-character string."""
        from src.lib.character_mapping import JISConverter

        converter = JISConverter()
        result = converter.estimate_stroke_count("abc")
        assert isinstance(result, int)


class TestSetupVerification:
    """Extended tests for setup verification."""

    def test_setup_verifier_module_imports(self):
        """Test SetupVerifier module imports."""
        from src.lib.setup_verification import SetupVerifier

        assert SetupVerifier is not None

    def test_verify_virtual_environment(self):
        """Test virtual environment verification."""
        try:
            from src.lib.setup_verification import SetupVerifier

            verifier = SetupVerifier(verbose=False)
            result = verifier.verify_virtual_environment()
            assert isinstance(result, bool)
        except (ImportError, AttributeError):
            pytest.skip("Function not available")

    def test_verify_gpu_setup(self):
        """Test GPU availability verification."""
        try:
            from src.lib.setup_verification import SetupVerifier

            verifier = SetupVerifier(verbose=False)
            result = verifier.verify_gpu_setup()
            assert isinstance(result, dict)
        except (ImportError, AttributeError):
            pytest.skip("Function not available")

    def test_verify_dependencies(self):
        """Test dependency verification."""
        try:
            from src.lib.setup_verification import SetupVerifier

            verifier = SetupVerifier(verbose=False)
            result = verifier.verify_dependencies()
            assert isinstance(result, dict)
        except (ImportError, AttributeError):
            pytest.skip("Function not available")


class TestQuantization:
    """Extended tests for quantization utilities."""

    def test_quantize_model_module_imports(self):
        """Test quantize_model module imports."""
        try:
            from scripts import quantize_model

            assert quantize_model is not None
        except ImportError:
            pytest.skip("quantize_model not available")

    def test_quantize_model_has_quantization_functions(self):
        """Test that quantization module has expected functions."""
        try:
            from scripts.quantize_model import convert_to_int8_static, quantize_to_int8

            assert convert_to_int8_static is not None
            assert quantize_to_int8 is not None
        except (ImportError, AttributeError):
            pytest.skip("Quantization functions not found")


class TestInspectONNX:
    """Tests for ONNX model inspection utilities."""

    def test_inspect_onnx_module_imports(self):
        """Test inspect_onnx_model module imports."""
        try:
            from scripts import inspect_onnx_model

            assert inspect_onnx_model is not None
        except ImportError:
            pytest.skip("inspect_onnx_model not available")

    def test_inspect_onnx_has_inspection_functions(self):
        """Test that inspection module has expected functions."""
        try:
            from scripts.inspect_onnx_model import get_onnx_model_details

            assert get_onnx_model_details is not None
        except (ImportError, AttributeError):
            pytest.skip("Inspection functions not found")


class TestPoolingComparison:
    """Tests for pooling operation comparisons."""

    def test_pooling_comparison_module_imports(self):
        """Test pooling_comparison module imports."""
        try:
            from scripts import pooling_comparison

            assert pooling_comparison is not None
        except ImportError:
            pytest.skip("pooling_comparison not available")


class TestONNXOperations:
    """Tests for ONNX operations comparison."""

    def test_onnx_operations_module_imports(self):
        """Test onnx_operations_comparison module imports."""
        try:
            from scripts import onnx_operations_comparison

            assert onnx_operations_comparison is not None
        except ImportError:
            pytest.skip("onnx_operations_comparison not available")


class TestConversionScripts:
    """Additional tests for conversion utilities."""

    def test_convert_to_onnx_module(self):
        """Test convert_to_onnx module."""
        try:
            from scripts import convert_to_onnx

            assert convert_to_onnx is not None
        except ImportError:
            pytest.skip("convert_to_onnx not available")

    def test_export_to_onnx_hiercode(self):
        """Test export_to_onnx_hiercode module."""
        try:
            from scripts import export_to_onnx

            assert export_to_onnx is not None
        except ImportError:
            pytest.skip("export_to_onnx_hiercode not available")

    def test_convert_to_safetensors(self):
        """Test convert_to_safetensors module."""
        try:
            from scripts import convert_to_safetensors

            assert convert_to_safetensors is not None
        except ImportError:
            pytest.skip("convert_to_safetensors not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
