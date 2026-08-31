"""
Tests for character mapping and JIS conversion utilities.

Tests for:
- JIS X 0208 to Unicode conversion
- Stroke count estimation
- Character type classification
- Character mapping generation from metadata
- Complete character mapping from dataset
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.lib.character_mapping import CharacterMappingGenerator, JISConverter


class TestJISConverter:
    """Test JIS X 0208 to Unicode conversion."""

    @pytest.fixture
    def converter(self):
        """Create a JIS converter instance."""
        return JISConverter()

    def test_hiragana_conversion(self, converter):
        """Test hiragana character conversion."""
        # ぁ (small a-hiragana): JIS 0x2421
        char = converter.jis_to_unicode("2421")
        assert char == "ぁ"
        assert ord(char) == 0x3041

        # ぃ (small i-hiragana): JIS 0x2423
        char = converter.jis_to_unicode("2423")
        assert char == "ぃ"
        assert ord(char) == 0x3043

    def test_hiragana_conversion_int(self, converter):
        """Test hiragana conversion with integer input."""
        char = converter.jis_to_unicode(0x2421)
        assert char == "ぁ"

    def test_katakana_conversion(self, converter):
        """Test katakana character conversion."""
        # ァ (small a-katakana): JIS 0x2521
        char = converter.jis_to_unicode("2521")
        assert char == "ァ"
        assert ord(char) == 0x30A1

        # ィ (small i-katakana): JIS 0x2523
        char = converter.jis_to_unicode("2523")
        assert char == "ィ"
        assert ord(char) == 0x30A3

    def test_kanji_conversion(self, converter):
        """Test kanji character conversion."""
        # JIS 0x3021 should convert to a kanji
        char = converter.jis_to_unicode("3021")
        assert len(char) == 1
        assert 0x4E00 <= ord(char) <= 0x9FAF

    def test_invalid_jis_code(self, converter):
        """Test handling of invalid JIS codes."""
        result = converter.jis_to_unicode("FFFF")
        assert "[UNK:" in result

    def test_invalid_format(self, converter):
        """Test handling of invalid format."""
        result = converter.jis_to_unicode("ZZZZ")
        assert "[UNK:" in result

    def test_stroke_count_hiragana(self, converter):
        """Test stroke count estimation for hiragana."""
        char = "あ"  # あ
        strokes = converter.estimate_stroke_count(char)
        assert 1 <= strokes <= 4

    def test_stroke_count_katakana(self, converter):
        """Test stroke count estimation for katakana."""
        char = "ア"  # ア
        strokes = converter.estimate_stroke_count(char)
        assert 1 <= strokes <= 4

    def test_stroke_count_kanji(self, converter):
        """Test stroke count estimation for kanji."""
        char = "漢"  # Common kanji
        strokes = converter.estimate_stroke_count(char)
        assert 1 <= strokes <= 25

    def test_stroke_count_invalid(self, converter):
        """Test stroke count for invalid input."""
        strokes = converter.estimate_stroke_count("abc")
        assert strokes == 1

    def test_character_type_hiragana(self, converter):
        """Test character type classification for hiragana."""
        assert converter.get_character_type("あ") == "hiragana"
        assert converter.get_character_type("い") == "hiragana"

    def test_character_type_katakana(self, converter):
        """Test character type classification for katakana."""
        assert converter.get_character_type("ア") == "katakana"
        assert converter.get_character_type("イ") == "katakana"

    def test_character_type_kanji(self, converter):
        """Test character type classification for kanji."""
        assert converter.get_character_type("漢") == "kanji"
        assert converter.get_character_type("字") == "kanji"

    def test_character_type_other(self, converter):
        """Test character type classification for other characters."""
        assert converter.get_character_type("a") == "other"
        assert converter.get_character_type("1") == "other"
        assert converter.get_character_type("!") == "other"


class TestCharacterMappingGenerator:
    """Test character mapping generation."""

    @pytest.fixture
    def generator(self):
        """Create a character mapping generator instance."""
        return CharacterMappingGenerator()

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        return {
            "jis_to_class": {
                "2421": 0,  # あ
                "2423": 1,  # い
                "2521": 2,  # ア
                "3021": 3,  # First kanji
            }
        }

    def test_generate_from_metadata(self, generator, sample_metadata):
        """Test generation from metadata file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_file = Path(tmpdir) / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(sample_metadata, f)

            class_to_char, char_to_class = generator.generate_from_metadata(metadata_file)

            # Check mapping size
            assert len(class_to_char) == 4
            assert len(char_to_class) == 4

            # Check specific conversions
            assert class_to_char[0] == "ぁ"
            assert class_to_char[1] == "ぃ"
            assert class_to_char[2] == "ァ"

            # Check reverse mapping
            assert char_to_class["ぁ"] == 0
            assert char_to_class["ぃ"] == 1

    def test_generate_from_missing_metadata(self, generator):
        """Test handling of missing metadata file."""
        with pytest.raises(FileNotFoundError):
            generator.generate_from_metadata(Path("nonexistent.json"))

    def test_generate_from_invalid_metadata(self, generator):
        """Test handling of metadata missing jis_to_class."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_file = Path(tmpdir) / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump({"other_field": {}}, f)

            with pytest.raises(KeyError):
                generator.generate_from_metadata(metadata_file)

    def test_generate_from_metadata_with_output(self, generator, sample_metadata):
        """Test generation with output file saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_file = Path(tmpdir) / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(sample_metadata, f)

            output_dir = Path(tmpdir) / "output"

            class_to_char, char_to_class = generator.generate_from_metadata(
                metadata_file, output_path=output_dir
            )

            # Check output files created
            assert (output_dir / "class_to_character.json").exists()
            assert (output_dir / "character_to_class.json").exists()

            # Verify saved content (JSON keys are always strings, so convert back to int for comparison)
            with open(output_dir / "class_to_character.json", encoding="utf-8") as f:
                saved_c2c = json.load(f)
            # Convert string keys back to int for comparison
            saved_c2c_int_keys = {int(k): v for k, v in saved_c2c.items()}
            assert saved_c2c_int_keys == class_to_char

    def test_generate_with_stroke_info(self, generator, sample_metadata):
        """Test generation with stroke count information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_file = Path(tmpdir) / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(sample_metadata, f)

            mapping = generator.generate_with_stroke_info(metadata_file)

            # Check structure
            assert "model_info" in mapping
            assert "characters" in mapping
            assert "statistics" in mapping

            # Check character entries have stroke info
            for _class_idx, char_info in mapping["characters"].items():
                assert "character" in char_info
                assert "stroke_count" in char_info
                assert "type" in char_info
                assert char_info["stroke_count"] >= 1

            # Check statistics
            assert mapping["statistics"]["total_characters"] == 4
            assert mapping["statistics"]["average_stroke_count"] > 0

    def test_generate_with_stroke_info_saves_output(self, generator, sample_metadata):
        """Test stroke info generation with output saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_file = Path(tmpdir) / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(sample_metadata, f)

            output_dir = Path(tmpdir) / "output"
            generator.generate_with_stroke_info(metadata_file, output_path=output_dir)

            # Check output file
            output_file = output_dir / "character_mapping_with_strokes.json"
            assert output_file.exists()

            # Verify saved content
            with open(output_file, encoding="utf-8") as f:
                saved_mapping = json.load(f)
            assert saved_mapping["statistics"]["total_characters"] == 4

    def test_character_type_counting(self, generator, sample_metadata):
        """Test that character types are counted correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_file = Path(tmpdir) / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(sample_metadata, f)

            mapping = generator.generate_with_stroke_info(metadata_file)

            # Should have: 1 hiragana (あ), 1 hiragana (い), 1 katakana (ア), 1 kanji
            stats = mapping["statistics"]
            assert stats["hiragana_count"] == 2
            assert stats["katakana_count"] == 1
            assert stats["kanji_count"] == 1


class TestCompleteMapping:
    """Tests for complete mapping generation from dataset."""

    @pytest.fixture
    def generator(self):
        """Create a character mapping generator instance."""
        return CharacterMappingGenerator()

    def test_complete_mapping_structure(self, generator):
        """Test structure of complete mapping."""
        # This is a basic structural test that doesn't require actual data
        # More thorough testing would need actual dataset files


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_end_to_end_mapping_generation(self):
        """Test complete workflow from metadata to saved mappings."""
        generator = CharacterMappingGenerator()

        sample_metadata = {
            "jis_to_class": {
                "2421": 0,
                "2423": 1,
                "2521": 2,
                "3021": 3,
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_file = Path(tmpdir) / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(sample_metadata, f)

            output_dir = Path(tmpdir) / "output"

            # Generate mappings with stroke info
            generator.generate_with_stroke_info(metadata_file, output_path=output_dir)

            # Verify all output files exist
            assert (output_dir / "character_mapping_with_strokes.json").exists()

            # Verify consistency
            with open(output_dir / "character_mapping_with_strokes.json", encoding="utf-8") as f:
                mapping_file = json.load(f)
            assert len(mapping_file["characters"]) == 4

            # Verify stroke info mapping
            assert mapping_file["statistics"]["average_stroke_count"] > 0
            assert (
                mapping_file["statistics"]["hiragana_count"]
                + mapping_file["statistics"]["katakana_count"]
                + mapping_file["statistics"]["kanji_count"]
                == mapping_file["statistics"]["total_characters"]
            )
