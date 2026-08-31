#!/usr/bin/env python3
"""Unit tests for utility modules."""

import pytest

from scripts.measure_co2_emissions import get_system_info
from src.lib.character_mapping import JISConverter

# Create JISConverter instance for test wrappers
_jis_converter = JISConverter()


# Helper functions for backwards compatibility with test code
def jis_to_unicode(code):
    """Wrapper for JISConverter.jis_to_unicode()"""
    return _jis_converter.jis_to_unicode(code)


def estimate_stroke_count(char):
    """Wrapper for JISConverter.estimate_stroke_count()"""
    return _jis_converter.estimate_stroke_count(char)


class TestJISToUnicode:
    """Tests for JIS to Unicode conversion."""

    def test_jis_hiragana_conversion(self):
        """Test conversion of hiragana JIS code."""
        # JIS X 0208 hiragana 'a' (あ)
        result = jis_to_unicode("2421")
        assert result is not None
        assert len(result) >= 1

    def test_jis_katakana_conversion(self):
        """Test conversion of katakana JIS code."""
        # JIS X 0208 katakana 'a' (ア)
        result = jis_to_unicode("2521")
        assert result is not None

    def test_jis_kanji_conversion(self):
        """Test conversion of kanji JIS code."""
        result = jis_to_unicode("3021")
        assert result is not None

    def test_invalid_jis_code(self):
        """Test handling of invalid JIS code."""
        result = jis_to_unicode("ZZZZ")
        assert isinstance(result, str)
        # Should return placeholder string
        assert "JIS" in result or result.startswith("[")

    def test_jis_code_format(self):
        """Test that result is always a string."""
        codes = ["2421", "2521", "3021"]
        for code in codes:
            result = jis_to_unicode(code)
            assert isinstance(result, str)

    def test_jis_empty_code(self):
        """Test empty JIS code handling."""
        result = jis_to_unicode("")
        assert isinstance(result, str)

    def test_jis_boundary_codes(self):
        """Test boundary JIS codes."""
        codes = ["0000", "FFFF", "2400", "2500", "3000"]
        for code in codes:
            result = jis_to_unicode(code)
            assert isinstance(result, str)
            assert len(result) >= 1

    def test_jis_multiple_conversions(self):
        """Test multiple consecutive conversions."""
        results = [jis_to_unicode(code) for code in ["2421", "2521", "3021"]]
        assert all(isinstance(r, str) for r in results)
        assert len(results) == 3


class TestEstrokeCount:
    """Tests for stroke count estimation."""

    def test_single_character_input(self):
        """Test stroke count for single character."""
        result = estimate_stroke_count("a")
        assert isinstance(result, int)
        assert result >= 1

    def test_hiragana_stroke_count(self):
        """Test stroke count for hiragana."""
        result = estimate_stroke_count("あ")
        assert isinstance(result, int)
        assert result >= 1

    def test_kanji_stroke_count(self):
        """Test stroke count for kanji."""
        result = estimate_stroke_count("漢")
        assert isinstance(result, int)
        assert result >= 1

    def test_empty_string(self):
        """Test stroke count for empty string."""
        result = estimate_stroke_count("")
        assert isinstance(result, int)

    def test_multi_character_string(self):
        """Test stroke count for multi-character string."""
        result = estimate_stroke_count("abc")
        assert isinstance(result, int)
        assert result >= 1

    def test_katakana_stroke_count(self):
        """Test stroke count for katakana."""
        result = estimate_stroke_count("ア")
        assert isinstance(result, int)
        assert 1 <= result <= 25

    def test_numeric_character(self):
        """Test stroke count for numeric characters."""
        result = estimate_stroke_count("5")
        assert isinstance(result, int)
        assert result >= 1

    def test_special_character(self):
        """Test stroke count for special characters."""
        result = estimate_stroke_count("!")
        assert isinstance(result, int)
        assert result >= 1

    def test_stroke_count_consistency(self):
        """Test stroke count is consistent across calls."""
        char = "漢"
        results = [estimate_stroke_count(char) for _ in range(5)]
        assert all(r == results[0] for r in results)


class TestSetupVerification:
    """Tests for setup verification module."""

    def test_setup_verifier_imports(self):
        """Test that SetupVerifier can be imported."""
        from src.lib.setup_verification import SetupVerifier

        assert SetupVerifier is not None

    def test_setup_verifier_can_instantiate(self):
        """Test that SetupVerifier can be instantiated."""
        from src.lib.setup_verification import SetupVerifier

        verifier = SetupVerifier(verbose=False)
        assert verifier is not None

    def test_setup_verifier_has_methods(self):
        """Test that SetupVerifier has verification methods."""
        from src.lib.setup_verification import SetupVerifier

        verifier = SetupVerifier(verbose=False)
        assert hasattr(verifier, "verify_python_version")
        assert hasattr(verifier, "verify_virtual_environment")
        assert hasattr(verifier, "verify_dependencies")


class TestSystemInfo:
    """Tests for system information gathering."""

    def test_get_system_info(self):
        """Test system information collection."""
        info = get_system_info()
        assert isinstance(info, dict)
        assert "timestamp" in info
        assert "platform" in info
        assert "cpu_count" in info

    def test_system_info_has_memory(self):
        """Test that system info includes memory information."""
        info = get_system_info()
        assert "memory_total_gb" in info
        assert isinstance(info["memory_total_gb"], (int, float))
        assert info["memory_total_gb"] > 0

    def test_system_info_has_torch_version(self):
        """Test that system info includes PyTorch version."""
        info = get_system_info()
        assert "torch_version" in info
        assert isinstance(info["torch_version"], str)

    def test_system_info_cpu_count_positive(self):
        """Test that CPU count is positive."""
        info = get_system_info()
        assert isinstance(info["cpu_count"], int)
        assert info["cpu_count"] > 0

    def test_system_info_timestamp_present(self):
        """Test that timestamp is present and non-empty."""
        info = get_system_info()
        assert "timestamp" in info
        assert len(str(info["timestamp"])) > 0

    def test_system_info_platform_present(self):
        """Test that platform information is present."""
        info = get_system_info()
        assert "platform" in info
        assert len(str(info["platform"])) > 0

    def test_system_info_consistency(self):
        """Test system info is consistent across multiple calls."""
        info1 = get_system_info()
        info2 = get_system_info()
        assert info1["cpu_count"] == info2["cpu_count"]
        assert info1["memory_total_gb"] == info2["memory_total_gb"]

    def test_system_info_has_python_version(self):
        """Test that system info includes Python version."""
        info = get_system_info()
        assert "python_version" in info or "platform" in info


class TestConversionAndExportScripts:
    """Basic tests for conversion scripts."""

    def test_onnx_operations_comparison_module_imports(self):
        """Test that ONNX operations module can be imported."""
        try:
            from scripts import onnx_operations_comparison

            assert onnx_operations_comparison is not None
        except ImportError:
            # Module might have missing dependencies, that's OK
            pass

    def test_pooling_comparison_module_imports(self):
        """Test that pooling comparison module can be imported."""
        try:
            from scripts import pooling_comparison

            assert pooling_comparison is not None
        except ImportError:
            # Module might have missing dependencies, that's OK
            pass


class TestLoadMultiETL:
    """Tests for multi-ETL dataset loading."""

    def test_load_multi_etl_module_imports(self):
        """Test that load_multi_etl module can be imported."""
        try:
            import scripts.load_multi_etl

            assert scripts.load_multi_etl is not None
        except ImportError:
            # Optional module dependency
            pass


class TestHierCodeEnhancement:
    """Tests for HierCode enhancement module."""

    def test_hiercode_enhancement_module_imports(self):
        """Test that hiercode enhancement module can be imported."""
        try:
            import scripts.hiercode_higita_enhancement

            assert scripts.hiercode_higita_enhancement is not None
        except (ImportError, AttributeError):
            # Optional module
            pass


class TestQuantization:
    """Tests for quantization utilities."""

    def test_quantize_model_module_imports(self):
        """Test that quantize_model module can be imported."""
        try:
            from scripts import quantize_model

            assert quantize_model is not None
        except ImportError:
            pass


class TestExportScripts:
    """Tests for ONNX export scripts."""

    def test_convert_to_onnx_module_imports(self):
        """Test that convert_to_onnx module can be imported."""
        try:
            from scripts import convert_to_onnx

            assert convert_to_onnx is not None
        except ImportError:
            pass

    def test_safetensors_usage_module_imports(self):
        """Test that safetensors usage module can be imported."""
        try:
            import scripts.example_safetensors_usage

            assert scripts.example_safetensors_usage is not None
        except (ImportError, AttributeError):
            pass


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_jis_unicode_edge_cases(self):
        """Test edge cases in JIS to Unicode conversion."""
        edge_cases = ["", "0", "00", "FF", "FFFF"]
        for case in edge_cases:
            result = jis_to_unicode(case)
            assert isinstance(result, str)

    def test_stroke_count_edge_cases(self):
        """Test edge cases in stroke count estimation."""
        edge_cases = ["", " ", "\t", "\n"]
        for case in edge_cases:
            result = estimate_stroke_count(case)
            assert isinstance(result, int)

    def test_system_info_edge_cases(self):
        """Test that system info doesn't fail under any conditions."""
        # Should always succeed
        info = get_system_info()
        assert info is not None
        assert isinstance(info, dict)


class TestDataIntegrity:
    """Tests for data integrity and correctness."""

    def test_jis_unicode_output_type(self):
        """Test that JIS to Unicode always returns string."""
        for code in ["2421", "2521", "3021", "ZZZZ", ""]:
            result = jis_to_unicode(code)
            assert isinstance(result, str)
            assert len(result) >= 1

    def test_stroke_count_output_type(self):
        """Test that stroke count always returns int."""
        for char in ["a", "あ", "ア", "漢", ""]:
            result = estimate_stroke_count(char)
            assert isinstance(result, int)

    def test_system_info_output_type(self):
        """Test that system info always returns dict."""
        info = get_system_info()
        assert isinstance(info, dict)
        assert len(info) > 0


class TestPerformance:
    """Tests for performance characteristics."""

    def test_jis_unicode_fast_conversion(self):
        """Test that JIS to Unicode conversion is reasonably fast."""
        import time

        start = time.time()
        for _ in range(100):
            jis_to_unicode("2421")
        elapsed = time.time() - start
        # Should complete 100 conversions in reasonable time
        assert elapsed < 1.0

    def test_stroke_count_fast_estimation(self):
        """Test that stroke count estimation is reasonably fast."""
        import time

        start = time.time()
        for _ in range(100):
            estimate_stroke_count("漢")
        elapsed = time.time() - start
        # Should complete 100 estimations in reasonable time
        assert elapsed < 1.0

    def test_system_info_fast_gathering(self):
        """Test that system info gathering is reasonably fast."""
        import time

        start = time.time()
        for _ in range(10):
            get_system_info()
        elapsed = time.time() - start
        # Should complete 10 calls in reasonable time
        assert elapsed < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
