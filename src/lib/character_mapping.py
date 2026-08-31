"""
Unified character mapping and encoding utilities for ETL9G dataset.

This module consolidates JIS X 0208 to Unicode conversion and character
mapping generation. Eliminates duplication from generate_mapping.py,
create_class_mapping.py, and generate_complete_class_mapping.py.

Example:
    >>> from src.lib.character_mapping import JISConverter, CharacterMappingGenerator
    >>> converter = JISConverter()
    >>> char = converter.jis_to_unicode('3021')  # Kanji
    >>> generator = CharacterMappingGenerator()
    >>> mapping = generator.generate_from_metadata('path/to/metadata.json')
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from src.lib.logging_utils import setup_logger

logger = setup_logger(__name__)


class JISConverter:
    """Convert JIS X 0208 codes to Unicode characters with stroke estimation."""

    # JIS X 0208 area code ranges
    HIRAGANA_AREA = 0x24  # Hiragana
    KATAKANA_AREA = 0x25  # Katakana
    KANJI_AREA_START = 0x30  # Kanji start
    KANJI_AREA_END = 0x4F  # Kanji end

    # Unicode base ranges
    HIRAGANA_BASE = 0x3041  # Unicode Hiragana start
    KATAKANA_BASE = 0x30A1  # Unicode Katakana start
    KANJI_BASE = 0x4E00  # CJK Unified Ideographs base

    # Code ranges within JIS areas
    HIRAGANA_CODE_END = 0x73
    KATAKANA_CODE_END = 0x76
    JIS_CODE_START = 0x21
    JIS_AREA_WIDTH = 94

    def jis_to_unicode(self, jis_code_input) -> str:
        """
        Convert JIS X 0208 code to Unicode character.

        Supports both string and integer formats:
        - String: '3021' (hex), '0x3021' (hex with prefix)
        - Integer: 0x3021

        JIS X 0208 structure:
        - Hiragana (area 0x24): code 0x21-0x73
        - Katakana (area 0x25): code 0x21-0x76
        - Kanji (area 0x30-0x4F): code 0x21-0x7E

        Args:
            jis_code_input: JIS X 0208 code (str or int format)

        Returns:
            Unicode character or placeholder string "[UNK:XXXX]"
        """
        try:
            # Parse input format
            if isinstance(jis_code_input, str):
                jis_int = int(jis_code_input, 16)
            else:
                jis_int = jis_code_input

            # Extract area (high byte) and code (low byte)
            area = (jis_int >> 8) & 0xFF
            code = jis_int & 0xFF

            # Hiragana (ひらがな)
            if area == self.HIRAGANA_AREA:
                if self.JIS_CODE_START <= code <= self.HIRAGANA_CODE_END:
                    return chr(self.HIRAGANA_BASE + (code - self.JIS_CODE_START))

            # Katakana (カタカナ)
            elif area == self.KATAKANA_AREA:
                if self.JIS_CODE_START <= code <= self.KATAKANA_CODE_END:
                    return chr(self.KATAKANA_BASE + (code - self.JIS_CODE_START))

            # Kanji (漢字)
            elif self.KANJI_AREA_START <= area <= self.KANJI_AREA_END:
                base_offset = (area - self.KANJI_AREA_START) * self.JIS_AREA_WIDTH + (
                    code - self.JIS_CODE_START
                )
                return chr(self.KANJI_BASE + base_offset)

            return f"[UNK:{jis_int:04X}]"

        except (ValueError, OverflowError) as e:
            logger.debug(f"Failed to convert JIS code {jis_code_input}: {e}")
            return f"[UNK:{jis_code_input}]"

    def estimate_stroke_count(self, character: str) -> int:
        """
        Estimate stroke count for a character.

        Uses character type and Unicode code point position to estimate
        typical stroke counts:
        - Hiragana: 1-4 strokes
        - Katakana: 1-4 strokes
        - Kanji: 1-25 strokes

        Args:
            character: Single Unicode character

        Returns:
            Estimated stroke count (1-25)
        """
        if len(character) != 1:
            return 1

        code_point = ord(character)

        # Hiragana: typically 1-4 strokes
        if 0x3041 <= code_point <= 0x3096:
            return max(1, 1 + ((code_point - 0x3041) % 4))

        # Katakana: typically 1-4 strokes
        elif 0x30A1 <= code_point <= 0x30FC:
            return max(1, 1 + ((code_point - 0x30A1) % 4))

        # Kanji: typically 1-25 strokes (based on code point position)
        elif 0x4E00 <= code_point <= 0x9FAF:
            base_strokes = 1 + ((code_point - 0x4E00) % 20)
            return min(base_strokes, 25)

        return 1

    def get_character_type(self, character: str) -> str:
        """
        Classify character type.

        Returns:
            One of: 'hiragana', 'katakana', 'kanji', 'other'
        """
        if len(character) != 1:
            return "other"

        code_point = ord(character)

        if 0x3041 <= code_point <= 0x3096:
            return "hiragana"
        elif 0x30A1 <= code_point <= 0x30FC:
            return "katakana"
        elif 0x4E00 <= code_point <= 0x9FAF:
            return "kanji"
        else:
            return "other"


class CharacterMappingGenerator:
    """Generate and manage character mappings from JIS codes and metadata."""

    def __init__(self):
        """Initialize converter."""
        self.converter = JISConverter()

    def generate_from_metadata(
        self, metadata_path: Path, output_path: Optional[Path] = None
    ) -> Tuple[Dict[int, str], Dict[str, int]]:
        """
        Generate character mappings from metadata.json.

        Loads JIS-to-class mapping from metadata and converts to Unicode.

        Args:
            metadata_path: Path to metadata.json file
            output_path: Optional path to save mappings JSON

        Returns:
            Tuple of (class_to_char, char_to_class) dicts

        Raises:
            FileNotFoundError: If metadata file not found
            KeyError: If metadata missing 'jis_to_class' key
        """
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        logger.info(f"📂 Loading metadata from {metadata_path}...")

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        if "jis_to_class" not in metadata:
            raise KeyError("Metadata missing 'jis_to_class' mapping")

        jis_to_class = metadata["jis_to_class"]
        logger.info(f"✓ Loaded {len(jis_to_class)} JIS codes")

        # Build bidirectional mapping
        class_to_char = {}
        char_to_class = {}
        conversion_errors = 0

        for jis_hex_str, class_idx in jis_to_class.items():
            try:
                char = self.converter.jis_to_unicode(jis_hex_str)
                class_to_char[class_idx] = char
                char_to_class[char] = class_idx
            except (ValueError, OverflowError) as e:
                logger.warning(f"⚠ Failed to convert JIS {jis_hex_str}: {e}")
                conversion_errors += 1

        logger.info(f"✓ Created mappings for {len(class_to_char)} classes")
        if conversion_errors > 0:
            logger.warning(f"⚠ Conversion errors: {conversion_errors}")

        # Optionally save mappings
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            c2c_file = output_path / "class_to_character.json"
            with open(c2c_file, "w", encoding="utf-8") as f:
                json.dump(class_to_char, f, ensure_ascii=False, indent=2)
            logger.info(f"✓ Saved class_to_character: {c2c_file}")

            chr2c_file = output_path / "character_to_class.json"
            with open(chr2c_file, "w", encoding="utf-8") as f:
                json.dump(char_to_class, f, ensure_ascii=False, indent=2)
            logger.info(f"✓ Saved character_to_class: {chr2c_file}")

        return class_to_char, char_to_class

    def generate_complete_mapping(
        self, data_dir: str = "dataset", output_dir: Optional[Path] = None
    ) -> Tuple[Dict, Dict]:
        """
        Generate complete class-to-character mapping from actual dataset.

        Scans the dataset to find all unique classes (including variants)
        and creates mappings. Solves the issue where metadata only contains
        standard JIS codes but actual data has 40,000+ variant classes.

        Args:
            data_dir: Path to dataset directory
            output_dir: Optional output directory for mappings

        Returns:
            Tuple of (class_to_character, character_to_class) dicts
        """
        from src.lib.training import load_chunked_dataset

        logger.info("=" * 70)
        logger.info("GENERATING COMPLETE CHARACTER MAPPING")
        logger.info("=" * 70)

        # Load dataset to find all unique classes
        logger.info("📂 Loading combined_all_etl dataset...")
        _, y = load_chunked_dataset(f"{data_dir}/combined_all_etl")

        unique_classes = np.unique(y)
        logger.info(f"✓ Loaded {len(y):,} samples with {len(unique_classes):,} unique classes")
        logger.info(f"  Class range: {unique_classes.min()} - {unique_classes.max()}")

        # Try to load standard JIS mapping from metadata
        metadata_path = Path(data_dir) / "combined_all_etl" / "metadata.json"
        class_to_jis = {}

        if metadata_path.exists():
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    metadata = json.load(f)
                jis_to_class = metadata.get("jis_to_class", {})
                class_to_jis = {int(v): k for k, v in jis_to_class.items()}
                logger.info(f"✓ Loaded {len(class_to_jis)} standard JIS mappings")
            except Exception as e:
                logger.warning(f"⚠ Could not load metadata: {e}")
        else:
            logger.warning(f"⚠ Metadata not found: {metadata_path}")

        # Create complete mapping
        class_to_character = {}
        character_to_class = {}
        known_count = 0
        unknown_count = 0

        for class_idx in unique_classes:
            class_idx = int(class_idx)

            if class_idx in class_to_jis:
                # Standard JIS character
                try:
                    jis_hex = class_to_jis[class_idx]
                    char = self.converter.jis_to_unicode(jis_hex)
                    class_to_character[class_idx] = char
                    character_to_class[char] = class_idx
                    known_count += 1
                except Exception as e:
                    logger.debug(f"⚠ Failed to convert class {class_idx}: {e}")
                    class_to_character[class_idx] = f"[CLASS:{class_idx}]"
                    unknown_count += 1
            else:
                # Unknown variant/non-standard character
                class_to_character[class_idx] = f"[CLASS:{class_idx}]"
                character_to_class[f"[CLASS:{class_idx}]"] = class_idx
                unknown_count += 1

        logger.info(f"✓ Created mappings for {len(class_to_character):,} classes")
        logger.info(f"  - Known (JIS): {known_count:,}")
        logger.info(f"  - Unknown/Variants: {unknown_count:,}")

        # Determine output directory
        if output_dir is None:
            training_dirs = list(Path("training").glob("*/checkpoints"))
            if training_dirs:
                output_dir = max(training_dirs, key=lambda p: p.stat().st_mtime)
            else:
                output_dir = Path("training/complete_mapping")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save mappings and statistics
        c2c_file = output_path / "class_to_character_complete.json"
        with open(c2c_file, "w", encoding="utf-8") as f:
            json.dump(class_to_character, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved class_to_character: {c2c_file}")

        chr2c_file = output_path / "character_to_class_complete.json"
        with open(chr2c_file, "w", encoding="utf-8") as f:
            json.dump(character_to_class, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved character_to_class: {chr2c_file}")

        stats = {
            "total_classes": len(class_to_character),
            "known_jis_classes": known_count,
            "unknown_variant_classes": unknown_count,
            "jis_percentage": 100 * known_count / len(class_to_character),
            "description": "Complete bidirectional mapping for all classes in combined_all_etl",
        }

        stats_file = output_path / "mapping_stats_complete.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"✓ Saved statistics: {stats_file}")

        logger.info("=" * 70)
        logger.info("✓ Complete mapping generation successful!")
        logger.info(f"  Output directory: {output_path}")

        return class_to_character, character_to_class

    def generate_with_stroke_info(
        self, metadata_path: Path, output_path: Optional[Path] = None
    ) -> Dict:
        """
        Generate character mapping with detailed stroke information.

        Args:
            metadata_path: Path to metadata.json file
            output_path: Optional path to save enriched mapping

        Returns:
            Mapping dict with characters and stroke counts
        """
        class_to_char, _ = self.generate_from_metadata(metadata_path)

        mapping = {
            "model_info": {
                "dataset": "ETL9G",
                "total_classes": len(class_to_char),
                "description": "Character mapping with stroke counts",
            },
            "characters": {},
            "statistics": {
                "total_characters": len(class_to_char),
                "hiragana_count": 0,
                "katakana_count": 0,
                "kanji_count": 0,
                "total_stroke_count": 0,
            },
        }

        # Process each character with stroke info
        for class_idx, char in class_to_char.items():
            stroke_count = self.converter.estimate_stroke_count(char)
            char_type = self.converter.get_character_type(char)

            mapping["characters"][str(class_idx)] = {
                "character": char,
                "stroke_count": stroke_count,
                "type": char_type,
            }

            # Update statistics
            mapping["statistics"]["total_stroke_count"] += stroke_count
            if char_type == "hiragana":
                mapping["statistics"]["hiragana_count"] += 1
            elif char_type == "katakana":
                mapping["statistics"]["katakana_count"] += 1
            elif char_type == "kanji":
                mapping["statistics"]["kanji_count"] += 1

        # Calculate averages
        total_chars = mapping["statistics"]["total_characters"]
        if total_chars > 0:
            mapping["statistics"]["average_stroke_count"] = round(
                mapping["statistics"]["total_stroke_count"] / total_chars, 1
            )

        # Optionally save
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            output_file = output_path / "character_mapping_with_strokes.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
            logger.info(f"✓ Saved mapping with strokes: {output_file}")

        return mapping
