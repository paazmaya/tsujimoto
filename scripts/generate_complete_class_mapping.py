#!/usr/bin/env python3
"""
Generate Complete Character Mapping from Actual Dataset Classes

Creates a bidirectional mapping for all 43,427 unique classes found in the
combined_all_etl dataset, not just the 3,129 standard JIS characters.

This solves the metadata mismatch issue where the metadata only contains
standard JIS codes but the actual data has 40,000+ additional variant classes.

Usage:
    uv run python scripts/generate_complete_class_mapping.py
    uv run python scripts/generate_complete_class_mapping.py --data-dir dataset
    uv run python scripts/generate_complete_class_mapping.py --output-dir training/hiercode_higita
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import load_chunked_dataset, setup_logger

logger = setup_logger(__name__)


def generate_complete_mapping(data_dir: str = "dataset", output_dir: str = None) -> Tuple[Dict, Dict]:
    """
    Generate complete class-to-character mapping from actual dataset.
    
    Args:
        data_dir: Path to dataset directory
        output_dir: Optional output directory for mappings (default: training/{model}/checkpoints/)
    
    Returns:
        Tuple of (class_to_character, character_to_class) dicts
    """
    logger.info("=" * 70)
    logger.info("GENERATING COMPLETE CHARACTER MAPPING")
    logger.info("=" * 70)
    
    # Load the actual dataset to find all unique classes
    logger.info("📂 Loading combined_all_etl dataset...")
    X, y = load_chunked_dataset(f"{data_dir}/combined_all_etl")
    
    unique_classes = np.unique(y)
    logger.info(f"✓ Loaded {len(y):,} samples with {len(unique_classes):,} unique classes")
    logger.info(f"  Class range: {unique_classes.min()} - {unique_classes.max()}")
    
    # Load standard JIS mapping if it exists (for classes 0-3128)
    jis_mapping = {}
    metadata_path = Path(data_dir) / "combined_all_etl" / "metadata.json"
    
    if metadata_path.exists():
        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
            jis_to_class = metadata.get("jis_to_class", {})
            
            # Reverse: class -> JIS hex
            class_to_jis = {int(v): k for k, v in jis_to_class.items()}
            logger.info(f"✓ Loaded {len(class_to_jis)} standard JIS mappings from metadata")
        except Exception as e:
            logger.warning(f"⚠ Could not load metadata: {e}")
            class_to_jis = {}
    else:
        logger.warning(f"⚠ Metadata not found: {metadata_path}")
        class_to_jis = {}
    
    # Create complete mapping
    class_to_character = {}
    character_to_class = {}
    known_count = 0
    unknown_count = 0
    
    for class_idx in unique_classes:
        class_idx = int(class_idx)
        
        if class_idx in class_to_jis:
            # Standard JIS character - convert to Unicode
            try:
                jis_hex = class_to_jis[class_idx]
                jis_int = int(jis_hex, 16)
                char = _jis_to_unicode(jis_int)
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
    
    # Save mappings
    if output_dir:
        output_path = Path(output_dir)
    else:
        # Try to find latest model training directory
        training_dirs = list(Path("training").glob("*/checkpoints"))
        if training_dirs:
            # Use the most recently modified
            output_path = max(training_dirs, key=lambda p: p.stat().st_mtime)
        else:
            output_path = Path("training/complete_mapping")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save class_to_character
    c2c_file = output_path / "class_to_character_complete.json"
    with open(c2c_file, "w", encoding="utf-8") as f:
        json.dump(class_to_character, f, ensure_ascii=False, indent=2)
    logger.info(f"✓ Saved class_to_character mapping: {c2c_file}")
    
    # Save character_to_class
    chr2c_file = output_path / "character_to_class_complete.json"
    with open(chr2c_file, "w", encoding="utf-8") as f:
        json.dump(character_to_class, f, ensure_ascii=False, indent=2)
    logger.info(f"✓ Saved character_to_class mapping: {chr2c_file}")
    
    # Save statistics
    stats = {
        "total_classes": len(class_to_character),
        "known_jis_classes": known_count,
        "unknown_variant_classes": unknown_count,
        "jis_percentage": 100 * known_count / len(class_to_character),
        "created_at": str(Path.cwd()),
        "description": "Complete bidirectional character-class mapping for all 43,427 classes in combined_all_etl"
    }
    
    stats_file = output_path / "mapping_stats_complete.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"✓ Saved mapping statistics: {stats_file}")
    
    logger.info("=" * 70)
    logger.info("✓ Complete mapping generation successful!")
    logger.info(f"  Output directory: {output_path}")
    
    return class_to_character, character_to_class


def _jis_to_unicode(jis_code_int: int) -> str:
    """Convert JIS X 0208 code to Unicode character."""
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
        return chr(0x4E00 + base_offset)

    return f"[UNK:{jis_code_int:04X}]"


def main():
    parser = argparse.ArgumentParser(
        description="Generate complete character mapping from actual dataset classes"
    )
    parser.add_argument(
        "--data-dir",
        default="dataset",
        help="Path to dataset directory (default: dataset)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for mappings (default: latest training dir)",
    )

    args = parser.parse_args()
    
    try:
        generate_complete_mapping(args.data_dir, args.output_dir)
    except Exception as e:
        logger.error(f"❌ Error generating mapping: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
