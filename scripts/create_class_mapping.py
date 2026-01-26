#!/usr/bin/env python3
"""
Create Character Mapping from Class Indices

This script creates a bidirectional mapping between:
- Class indices (0-43527) → Unicode characters
- Unicode characters → Class indices

Should be run immediately after training completes and before model conversion.
Generates class_to_character.json for inference pipelines.

Usage:
    python scripts/create_class_mapping.py --metadata-path dataset/combined_all_etl/metadata.json
    python scripts/create_class_mapping.py --output-dir model_training/cnn/results/

Author: tsujimoto
Date: November 18, 2025
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import setup_logger

logger = setup_logger(__name__)


def jis_to_unicode(jis_code_int: int) -> str:
    """
    Convert JIS X 0208 code to Unicode character.

    JIS X 0208 format: area (high byte) + code (low byte)
    - Hiragana: area 0x24
    - Katakana: area 0x25
    - Kanji: area 0x30-0x4F
    """
    area = (jis_code_int >> 8) & 0xFF
    code = jis_code_int & 0xFF

    if area == 0x24:  # Hiragana
        if 0x21 <= code <= 0x73:
            return chr(0x3041 + (code - 0x21))

    elif area == 0x25:  # Katakana
        if 0x21 <= code <= 0x76:
            return chr(0x30A1 + (code - 0x21))

    elif 0x30 <= area <= 0x4F:  # Kanji
        base_offset = (area - 0x30) * 94 + (code - 0x21)
        return chr(0x4E00 + base_offset)  # CJK Unified Ideographs

    return f"[UNK:{jis_code_int:04X}]"


def create_class_mapping(metadata_path: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Create bidirectional character mappings from metadata.

    Returns:
        - class_to_char: {class_idx: character}
        - char_to_class: {character: class_idx}
    """
    logger.info(f"📂 Loading metadata from {metadata_path}...")

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    if "jis_to_class" not in metadata:
        logger.error("❌ Metadata missing 'jis_to_class' mapping")
        sys.exit(1)

    jis_to_class = metadata["jis_to_class"]
    logger.info(f"✓ Loaded {len(jis_to_class)} JIS codes")

    # Build class_idx -> character mapping
    class_to_char = {}
    char_to_class = {}
    conversion_errors = 0

    for jis_hex_str, class_idx in jis_to_class.items():
        try:
            jis_int = int(jis_hex_str, 16)
            char = jis_to_unicode(jis_int)

            class_to_char[class_idx] = char
            char_to_class[char] = class_idx

        except (ValueError, OverflowError) as e:
            logger.warning(f"⚠ Failed to convert JIS {jis_hex_str}: {e}")
            conversion_errors += 1

    logger.info(f"✓ Created mappings for {len(class_to_char)} classes")
    if conversion_errors > 0:
        logger.warning(f"⚠ Conversion errors: {conversion_errors}")

    return class_to_char, char_to_class


def save_mappings(
    class_to_char: Dict[int, str],
    char_to_class: Dict[str, int],
    output_dir: Path,
) -> None:
    """Save both mapping formats to JSON files."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save class_idx -> character (for model inference)
    class_map_path = output_dir / "class_to_character.json"
    with open(class_map_path, "w", encoding="utf-8") as f:
        # Convert int keys to strings for JSON compatibility
        json.dump({str(k): v for k, v in class_to_char.items()}, f, ensure_ascii=False, indent=2)
    logger.info(f"✓ Saved class→character mapping: {class_map_path}")

    # Save character -> class_idx (for training/debugging)
    char_map_path = output_dir / "character_to_class.json"
    with open(char_map_path, "w", encoding="utf-8") as f:
        json.dump(char_to_class, f, ensure_ascii=False, indent=2)
    logger.info(f"✓ Saved character→class mapping: {char_map_path}")

    # Save summary statistics
    stats_path = output_dir / "mapping_stats.json"
    stats = {
        "total_classes": len(class_to_char),
        "total_unique_chars": len(char_to_class),
        "created_at": __import__("datetime").datetime.now().isoformat(),
        "description": "Bidirectional character-class mapping for kanji recognition",
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"✓ Saved mapping statistics: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create character mapping from class indices to Unicode characters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/create_class_mapping.py
  python scripts/create_class_mapping.py --metadata-path dataset/combined_all_etl/metadata.json
  python scripts/create_class_mapping.py --output-dir model_training/cnn/results/
  python scripts/create_class_mapping.py --metadata-path dataset/combined_all_etl/metadata.json --output-dir model_training/cnn/results/
        """,
    )

    parser.add_argument(
        "--metadata-path",
        type=str,
        default="dataset/combined_all_etl/metadata.json",
        help="Path to dataset metadata with JIS-to-class mapping (default: dataset/combined_all_etl/metadata.json)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset",
        help="Output directory for mapping files (default: dataset)",
    )

    args = parser.parse_args()

    metadata_path = Path(args.metadata_path)
    output_dir = Path(args.output_dir)

    # Validate input
    if not metadata_path.exists():
        logger.error(f"❌ Metadata file not found: {metadata_path}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("CREATE CHARACTER MAPPING FROM CLASS INDICES")
    logger.info("=" * 70)

    # Create mappings
    class_to_char, char_to_class = create_class_mapping(metadata_path)

    # Save mappings
    save_mappings(class_to_char, char_to_class, output_dir)

    logger.info("=" * 70)
    logger.info("✅ Character mapping creation complete!")
    logger.info(f"   Output directory: {output_dir}")
    logger.info("   Files created:")
    logger.info("     - class_to_character.json (for inference)")
    logger.info("     - character_to_class.json (for debugging)")
    logger.info("     - mapping_stats.json (metadata)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
