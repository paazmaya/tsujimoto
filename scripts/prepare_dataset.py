#!/usr/bin/env python3
"""
Unified ETLCDB Dataset Preparation - Auto-detects and processes all available datasets.

Automatically discovers ETL folders in the project root and:
1. Processes each available dataset individually
2. Combines them into a single unified dataset

Usage:
    python scripts/prepare_dataset.py                    # Auto-detect all
    python scripts/prepare_dataset.py --no-combine       # Process only
    python scripts/prepare_dataset.py --only etl9g etl8g # Specific datasets
"""

import argparse
import json
import struct
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import setup_logger

logger = setup_logger(__name__)


class ETLFormatHandler(ABC):
    """Abstract base class for ETLCDB format handlers"""

    def __init__(self):
        self.dataset_info = {}

    @abstractmethod
    def get_record_size(self) -> int:
        """Return the size of one record in bytes"""

    @abstractmethod
    def extract_record_info(self, record_data: bytes) -> Optional[dict]:
        """Extract information from a single record"""

    @abstractmethod
    def get_image_dimensions(self) -> tuple:
        """Return (width, height) of raw image"""


class ETL1Handler(ETLFormatHandler):
    """Handler for ETL1 format (M-type). Specs: 72x76 pixels, 4-bit grayscale, 99 Katakana + ASCII"""

    def get_record_size(self) -> int:
        return 2052

    def get_image_dimensions(self) -> tuple:
        return (72, 76)

    def extract_record_info(self, record_data: bytes) -> Optional[dict]:
        """ETL1 M-type format"""
        try:
            serial = struct.unpack(">H", record_data[0:2])[0]
            jis_code = struct.unpack(">H", record_data[2:4])[0]
            ascii_reading = record_data[4:12].decode("ascii", errors="ignore").strip()
            data_serial = struct.unpack(">I", record_data[12:16])[0]
            writer_id = record_data[18]
            image_data = record_data[32:2048]

            return {
                "serial": serial,
                "jis_code": jis_code,
                "ascii_reading": ascii_reading,
                "data_serial": data_serial,
                "writer_id": writer_id,
                "image_data": image_data,
                "format_type": "M-type",
            }
        except Exception:
            return None


class ETL2Handler(ETLFormatHandler):
    """Handler for ETL2 format (K-type). Specs: 60x60 pixels, 6-bit grayscale, 2,184 characters"""

    def get_record_size(self) -> int:
        return 1956

    def get_image_dimensions(self) -> tuple:
        return (60, 60)

    def extract_record_info(self, record_data: bytes) -> Optional[dict]:
        """ETL2 K-type format"""
        try:
            serial = struct.unpack(">H", record_data[0:2])[0]
            jis_code = struct.unpack(">H", record_data[2:4])[0]
            ascii_reading = record_data[4:12].decode("ascii", errors="ignore").strip()
            data_serial = struct.unpack(">I", record_data[12:16])[0]
            writer_id = record_data[18]
            image_data = record_data[64:2764]

            return {
                "serial": serial,
                "jis_code": jis_code,
                "ascii_reading": ascii_reading,
                "data_serial": data_serial,
                "writer_id": writer_id,
                "image_data": image_data,
                "format_type": "K-type",
            }
        except Exception:
            return None


class ETL3Handler(ETLFormatHandler):
    """Handler for ETL3 format (C-type). Specs: 72x76 pixels, 4-bit grayscale, 48 ASCII"""

    def get_record_size(self) -> int:
        return 2052

    def get_image_dimensions(self) -> tuple:
        return (72, 76)

    def extract_record_info(self, record_data: bytes) -> Optional[dict]:
        """ETL3 C-type format"""
        try:
            serial = struct.unpack(">H", record_data[0:2])[0]
            jis_code = struct.unpack(">H", record_data[2:4])[0]
            ascii_reading = record_data[4:12].decode("ascii", errors="ignore").strip()
            data_serial = struct.unpack(">I", record_data[12:16])[0]
            writer_id = record_data[18]
            image_data = record_data[32:2048]

            return {
                "serial": serial,
                "jis_code": jis_code,
                "ascii_reading": ascii_reading,
                "data_serial": data_serial,
                "writer_id": writer_id,
                "image_data": image_data,
                "format_type": "C-type",
            }
        except Exception:
            return None


class ETL4Handler(ETLFormatHandler):
    """Handler for ETL4 format (M-type). Specs: 64x63 pixels, 4-bit grayscale, 51 Hiragana"""

    def get_record_size(self) -> int:
        return 2052

    def get_image_dimensions(self) -> tuple:
        return (64, 63)

    def extract_record_info(self, record_data: bytes) -> Optional[dict]:
        """ETL4 M-type format"""
        try:
            serial = struct.unpack(">H", record_data[0:2])[0]
            jis_code = struct.unpack(">H", record_data[2:4])[0]
            ascii_reading = record_data[4:12].decode("ascii", errors="ignore").strip()
            data_serial = struct.unpack(">I", record_data[12:16])[0]
            writer_id = record_data[18]
            image_data = record_data[32:2048]

            return {
                "serial": serial,
                "jis_code": jis_code,
                "ascii_reading": ascii_reading,
                "data_serial": data_serial,
                "writer_id": writer_id,
                "image_data": image_data,
                "format_type": "M-type",
            }
        except Exception:
            return None


class ETL5Handler(ETLFormatHandler):
    """Handler for ETL5 format (M-type). Specs: 64x63 pixels, 4-bit grayscale, 51 Katakana"""

    def get_record_size(self) -> int:
        return 2052

    def get_image_dimensions(self) -> tuple:
        return (64, 63)

    def extract_record_info(self, record_data: bytes) -> Optional[dict]:
        """ETL5 M-type format"""
        try:
            serial = struct.unpack(">H", record_data[0:2])[0]
            jis_code = struct.unpack(">H", record_data[2:4])[0]
            ascii_reading = record_data[4:12].decode("ascii", errors="ignore").strip()
            data_serial = struct.unpack(">I", record_data[12:16])[0]
            writer_id = record_data[18]
            image_data = record_data[32:2048]

            return {
                "serial": serial,
                "jis_code": jis_code,
                "ascii_reading": ascii_reading,
                "data_serial": data_serial,
                "writer_id": writer_id,
                "image_data": image_data,
                "format_type": "M-type",
            }
        except Exception:
            return None


class ETL6Handler(ETLFormatHandler):
    """Handler for ETL6 format (M-type). Specs: 64x63 pixels, 4-bit grayscale, 114 chars"""

    def get_record_size(self) -> int:
        return 2052

    def get_image_dimensions(self) -> tuple:
        return (64, 63)

    def extract_record_info(self, record_data: bytes) -> Optional[dict]:
        """ETL6 M-type format"""
        try:
            serial = struct.unpack(">H", record_data[0:2])[0]
            jis_code = struct.unpack(">H", record_data[2:4])[0]
            ascii_reading = record_data[4:12].decode("ascii", errors="ignore").strip()
            data_serial = struct.unpack(">I", record_data[12:16])[0]
            writer_id = record_data[18]
            image_data = record_data[32:2048]

            return {
                "serial": serial,
                "jis_code": jis_code,
                "ascii_reading": ascii_reading,
                "data_serial": data_serial,
                "writer_id": writer_id,
                "image_data": image_data,
                "format_type": "M-type",
            }
        except Exception:
            return None


class ETL7Handler(ETLFormatHandler):
    """Handler for ETL7 format (M-type). Specs: 64x63 pixels, 4-bit grayscale, 48 Hiragana"""

    def get_record_size(self) -> int:
        return 2052

    def get_image_dimensions(self) -> tuple:
        return (64, 63)

    def extract_record_info(self, record_data: bytes) -> Optional[dict]:
        """ETL7 M-type format"""
        try:
            serial = struct.unpack(">H", record_data[0:2])[0]
            jis_code = struct.unpack(">H", record_data[2:4])[0]
            ascii_reading = record_data[4:12].decode("ascii", errors="ignore").strip()
            data_serial = struct.unpack(">I", record_data[12:16])[0]
            writer_id = record_data[18]
            image_data = record_data[32:2048]

            return {
                "serial": serial,
                "jis_code": jis_code,
                "ascii_reading": ascii_reading,
                "data_serial": data_serial,
                "writer_id": writer_id,
                "image_data": image_data,
                "format_type": "M-type",
            }
        except Exception:
            return None


class ETL8GHandler(ETLFormatHandler):
    """Handler for ETL8G format. Specs: 128x127 pixels, 4-bit grayscale, 956 chars"""

    def get_record_size(self) -> int:
        return 8199

    def get_image_dimensions(self) -> tuple:
        return (128, 127)

    def extract_record_info(self, record_data: bytes) -> Optional[dict]:
        """ETL8G grayscale format (bytes 61-8188 per official spec)"""
        try:
            serial = struct.unpack(">H", record_data[0:2])[0]
            jis_code = struct.unpack(">H", record_data[2:4])[0]
            ascii_reading = record_data[4:12].decode("ascii", errors="ignore").strip()
            data_serial = struct.unpack(">I", record_data[12:16])[0]
            writer_id = record_data[18]
            image_data = record_data[60:8188]

            return {
                "serial": serial,
                "jis_code": jis_code,
                "ascii_reading": ascii_reading,
                "data_serial": data_serial,
                "writer_id": writer_id,
                "image_data": image_data,
                "format_type": "ETL8G",
            }
        except Exception:
            return None


class ETL9GHandler(ETLFormatHandler):
    """Handler for ETL9G format. Specs: 128x127 pixels, 4-bit grayscale, 3,036 chars"""

    def get_record_size(self) -> int:
        return 8199

    def get_image_dimensions(self) -> tuple:
        return (128, 127)

    def extract_record_info(self, record_data: bytes) -> Optional[dict]:
        """ETL9G grayscale format"""
        try:
            serial = struct.unpack(">H", record_data[0:2])[0]
            jis_code = struct.unpack(">H", record_data[2:4])[0]
            ascii_reading = record_data[4:12].decode("ascii", errors="ignore").strip()
            data_serial = struct.unpack(">I", record_data[12:16])[0]
            writer_id = record_data[18]
            image_data = record_data[64:8192]

            return {
                "serial": serial,
                "jis_code": jis_code,
                "ascii_reading": ascii_reading,
                "data_serial": data_serial,
                "writer_id": writer_id,
                "image_data": image_data,
                "format_type": "ETL9G",
            }
        except Exception:
            return None


class ETLDatasetProcessor:
    """Unified processor for all ETLCDB formats"""

    HANDLERS = {
        "etl1": ETL1Handler,
        "etl2": ETL2Handler,
        "etl3": ETL3Handler,
        "etl4": ETL4Handler,
        "etl5": ETL5Handler,
        "etl6": ETL6Handler,
        "etl7": ETL7Handler,
        "etl8g": ETL8GHandler,
        "etl8b": ETL8GHandler,
        "etl9g": ETL9GHandler,
        "etl9b": ETL9GHandler,
    }

    DATASET_INFO = {
        "etl1": {
            "name": "ETL1",
            "classes": 99,
            "samples": 141319,
            "description": "Katakana + ASCII",
        },
        "etl2": {
            "name": "ETL2",
            "classes": 2184,
            "samples": 52796,
            "description": "Mixed Hiragana/Katakana/Kanji",
        },
        "etl3": {
            "name": "ETL3",
            "classes": 48,
            "samples": 9600,
            "description": "ASCII numerals and letters",
        },
        "etl4": {"name": "ETL4", "classes": 51, "samples": 6120, "description": "Hiragana"},
        "etl5": {"name": "ETL5", "classes": 51, "samples": 10608, "description": "Katakana"},
        "etl6": {
            "name": "ETL6",
            "classes": 114,
            "samples": 157662,
            "description": "Numerals, letters, Katakana, symbols",
        },
        "etl7": {"name": "ETL7", "classes": 48, "samples": 16800, "description": "Hiragana"},
        "etl8g": {
            "name": "ETL8G",
            "classes": 956,
            "samples": 152960,
            "description": "Educational Kanji (881) + Hiragana (75)",
        },
        "etl8b": {
            "name": "ETL8B",
            "classes": 956,
            "samples": 152960,
            "description": "ETL8 Binary format",
        },
        "etl9g": {
            "name": "ETL9G",
            "classes": 3036,
            "samples": 607200,
            "description": "JIS Level 1 Kanji (2965) + Hiragana (71)",
        },
        "etl9b": {
            "name": "ETL9B",
            "classes": 3036,
            "samples": 607200,
            "description": "ETL9 Binary format",
        },
    }

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def unpack_4bit_image(
        self, packed_data: bytes, width: int, height: int
    ) -> Optional[np.ndarray]:
        """Convert 4-bit packed grayscale image to 2D array"""
        try:
            unpacked = []
            for byte in packed_data:
                unpacked.append((byte >> 4) & 0xF)
                unpacked.append(byte & 0xF)

            pixels = unpacked[: width * height]
            image = np.array(pixels, dtype=np.uint8).reshape(height, width)
            return image * 17

        except Exception:
            return None

    def unpack_6bit_image(
        self, packed_data: bytes, width: int, height: int
    ) -> Optional[np.ndarray]:
        """Convert 6-bit packed grayscale image to 2D array (for ETL2)"""
        try:
            pixels = []
            i = 0

            while len(pixels) < width * height and i < len(packed_data):
                if i + 2 < len(packed_data):
                    b1 = packed_data[i]
                    b2 = packed_data[i + 1]
                    b3 = packed_data[i + 2]

                    pixels.append((b1 >> 2) & 0x3F)
                    pixels.append(((b1 & 0x3) << 4) | ((b2 >> 4) & 0xF))
                    pixels.append(((b2 & 0xF) << 2) | ((b3 >> 6) & 0x3))
                    pixels.append(b3 & 0x3F)

                    i += 3
                else:
                    break

            pixels = pixels[: width * height]
            image = np.array(pixels, dtype=np.uint8).reshape(height, width)
            return image * 4

        except Exception:
            return None

    def preprocess_image(
        self, image: Optional[np.ndarray], target_size: int = 64
    ) -> Optional[np.ndarray]:
        """Preprocess image for training"""
        if image is None:
            return None

        image_smooth = cv2.GaussianBlur(image, (3, 3), 0.5)
        image_resized = cv2.resize(
            image_smooth, (target_size, target_size), interpolation=cv2.INTER_AREA
        )
        image_normalized = image_resized.astype(np.float32) / 255.0

        return image_normalized

    def process_dataset(
        self,
        dataset_name: str,
        etl_dir: str,
        target_size: int = 64,
        max_workers: int = 4,
    ):
        """Process a single ETLCDB dataset"""

        dataset_name_lower = dataset_name.lower()

        if dataset_name_lower not in self.HANDLERS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        handler_class = self.HANDLERS[dataset_name_lower]
        handler = handler_class()

        etl_dir_path = Path(etl_dir)
        etl_files = sorted(
            [
                f
                for f in etl_dir_path.glob(f"{dataset_name}*")
                if f.is_file() and "INFO" not in f.name
            ]
        )

        if not etl_files:
            raise FileNotFoundError(f"No {dataset_name} files found in {etl_dir}")

        all_samples = []
        all_local_mappings = []
        record_size = handler.get_record_size()
        width, height = handler.get_image_dimensions()

        # Process files sequentially to avoid pickling issues
        for etl_file_path in tqdm(etl_files, desc=f"Processing {dataset_name} files"):
            samples = []
            local_jis_to_class = {}
            local_class_counter = 0

            try:
                with open(etl_file_path, "rb") as f:
                    record_count = 0

                    while True:
                        record_data = f.read(record_size)
                        if len(record_data) != record_size:
                            break

                        record_info = handler.extract_record_info(record_data)
                        if record_info is None:
                            continue

                        jis_code = record_info["jis_code"]

                        if jis_code == 0 or jis_code == 0xFFFF:
                            continue

                        if jis_code not in local_jis_to_class:
                            local_jis_to_class[jis_code] = local_class_counter
                            local_class_counter += 1

                        class_idx = local_jis_to_class[jis_code]

                        if "ETL2" in dataset_name.upper():
                            image = self.unpack_6bit_image(record_info["image_data"], width, height)
                        else:
                            image = self.unpack_4bit_image(record_info["image_data"], width, height)

                        if image is None:
                            continue

                        processed_image = self.preprocess_image(image, target_size)

                        if processed_image is None:
                            continue

                        samples.append(
                            {
                                "image": processed_image.flatten(),
                                "class_idx": class_idx,
                                "jis_code": jis_code,
                                "writer_id": record_info["writer_id"],
                                "ascii_reading": record_info["ascii_reading"],
                            }
                        )

                        record_count += 1

                all_samples.extend(samples)
                all_local_mappings.append(local_jis_to_class)
            except Exception:  # noqa: S110
                pass

        if not all_samples:
            raise ValueError(f"No samples processed from {dataset_name}")

        global_jis_to_class = self._merge_class_mappings(all_local_mappings)

        for sample in tqdm(all_samples, desc="Updating indices"):
            sample["class_idx"] = global_jis_to_class[sample["jis_code"]]

        X = np.array([sample["image"] for sample in all_samples], dtype=np.float32)
        y = np.array([sample["class_idx"] for sample in all_samples], dtype=np.int32)

        class_to_jis = {
            class_idx: f"{jis_code:04X}" for jis_code, class_idx in global_jis_to_class.items()
        }

        samples_per_class = defaultdict(int)
        for sample in all_samples:
            samples_per_class[sample["class_idx"]] += 1

        metadata = {
            "dataset_name": dataset_name,
            "num_classes": len(global_jis_to_class),
            "total_samples": len(all_samples),
            "target_size": target_size,
            "jis_to_class": {f"{k:04X}": v for k, v in global_jis_to_class.items()},
            "class_to_jis": class_to_jis,
            "samples_per_class": dict(samples_per_class),
            "dataset_info": {
                "source": self.DATASET_INFO[dataset_name_lower]["name"],
                "description": self.DATASET_INFO[dataset_name_lower]["description"],
                "expected_classes": self.DATASET_INFO[dataset_name_lower]["classes"],
                "expected_samples": self.DATASET_INFO[dataset_name_lower]["samples"],
                "actual_classes": len(global_jis_to_class),
                "actual_samples": len(all_samples),
                "files_processed": len(etl_files),
                "avg_samples_per_class": len(all_samples) / len(global_jis_to_class),
            },
        }

        output_subdir = self.output_dir / dataset_name_lower
        output_subdir.mkdir(parents=True, exist_ok=True)

        chunk_size = 50000
        if len(all_samples) > chunk_size:
            for i in range(0, len(all_samples), chunk_size):
                chunk_end = min(i + chunk_size, len(all_samples))
                chunk_X = np.array(
                    [sample["image"] for sample in all_samples[i:chunk_end]],
                    dtype=np.float32,
                )
                chunk_y = np.array(
                    [sample["class_idx"] for sample in all_samples[i:chunk_end]],
                    dtype=np.int32,
                )

                np.savez_compressed(
                    output_subdir / f"{dataset_name_lower}_chunk_{i // chunk_size:02d}.npz",
                    X=chunk_X,
                    y=chunk_y,
                )
        else:
            np.savez_compressed(
                output_subdir / f"{dataset_name_lower}_dataset.npz",
                X=X,
                y=y,
            )

        with open(output_subdir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return X, y, metadata

    @staticmethod
    def _merge_class_mappings(all_local_mappings):
        """Merge class mappings from all files into global mapping"""
        global_jis_to_class = {}
        global_class_counter = 0

        all_jis_codes = set()
        for local_mapping in all_local_mappings:
            all_jis_codes.update(local_mapping.keys())

        for jis_code in sorted(all_jis_codes):
            global_jis_to_class[jis_code] = global_class_counter
            global_class_counter += 1

        return global_jis_to_class

    def combine_datasets(self, dataset_names: list, output_name: str = "combined_etl"):
        """Combine multiple processed datasets into one unified dataset"""

        all_X = []
        all_y = []
        global_jis_to_class = {}
        current_class_idx = 0
        metadata_list = []
        combined_sources = []

        for dataset_name in dataset_names:
            dataset_lower = dataset_name.lower()
            dataset_dir = self.output_dir / dataset_lower

            if not dataset_dir.exists():
                continue

            metadata_file = dataset_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    metadata_list.append(metadata)
                    combined_sources.append(
                        f"{metadata['dataset_info']['source']} ({metadata['total_samples']} samples)"
                    )

            chunk_files = sorted(dataset_dir.glob(f"{dataset_lower}_chunk_*.npz"))
            if not chunk_files:
                chunk_files = sorted(dataset_dir.glob(f"{dataset_lower}_dataset.npz"))

            for chunk_file in chunk_files:
                data = np.load(chunk_file)
                X_chunk = data["X"]
                y_chunk = data["y"]

                y_chunk_offset = y_chunk + current_class_idx
                all_X.append(X_chunk)
                all_y.append(y_chunk_offset)

                local_metadata = None
                for md in metadata_list:
                    if md["dataset_name"].lower() == dataset_lower:
                        local_metadata = md
                        break

                if local_metadata:
                    for jis_hex, local_idx in local_metadata["jis_to_class"].items():
                        global_jis_to_class[jis_hex] = current_class_idx + local_idx

                current_class_idx += len(set(y_chunk))

        if not all_X:
            return

        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)

        combined_metadata = {
            "dataset_name": output_name,
            "num_classes": current_class_idx,
            "total_samples": len(X_combined),
            "target_size": 64,
            "datasets_combined": dataset_names,
            "jis_to_class": global_jis_to_class,
            "combined_info": {
                "source": "Combined ETLCDB datasets",
                "datasets": combined_sources,
            },
        }

        output_dir = self.output_dir / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        chunk_size = 50000
        if len(X_combined) > chunk_size:
            for i in range(0, len(X_combined), chunk_size):
                chunk_end = min(i + chunk_size, len(X_combined))
                np.savez_compressed(
                    output_dir / f"{output_name}_chunk_{i // chunk_size:02d}.npz",
                    X=X_combined[i:chunk_end],
                    y=y_combined[i:chunk_end],
                )
        else:
            np.savez_compressed(
                output_dir / f"{output_name}_dataset.npz",
                X=X_combined,
                y=y_combined,
            )

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(combined_metadata, f, indent=2)

    @staticmethod
    def find_etl_directories(root_dir: Path = Path(".")) -> dict:
        """Auto-detect available ETL directories"""
        available = {}

        for dataset_name in [
            "etl1",
            "etl2",
            "etl3",
            "etl4",
            "etl5",
            "etl6",
            "etl7",
            "etl8g",
            "etl9g",
        ]:
            dataset_dir = root_dir / dataset_name.upper()
            if dataset_dir.exists() and dataset_dir.is_dir():
                files = [f for f in dataset_dir.glob(f"{dataset_name.upper()}*") if f.is_file()]
                if files:
                    available[dataset_name] = str(dataset_dir)

        return available


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Unified ETLCDB Dataset Preparation - Auto-detects and processes all available datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect, process, and combine all available ETL datasets (DEFAULT)
  python scripts/prepare_dataset.py

  # Process specific datasets and combine them
  python scripts/prepare_dataset.py --only etl9g etl8g

  # Process datasets individually WITHOUT combining
  python scripts/prepare_dataset.py --no-combine

  # Custom output directory (combine is default)
  python scripts/prepare_dataset.py --output-dir my_datasets
        """,
    )

    parser.add_argument(
        "--only",
        nargs="+",
        help="Process only specific datasets (auto-detected by default)",
    )
    parser.add_argument(
        "--no-combine",
        action="store_true",
        help="Process datasets individually without combining (default: combine all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset",
        help="Output directory for processed datasets (default: dataset)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        help="Target image size (default: 64)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes (default: 4)",
    )

    args = parser.parse_args()

    logger.info("\n=== ETL Dataset Preparation ===")
    output_dir = Path(args.output_dir)
    processor = ETLDatasetProcessor(str(output_dir))

    # Auto-detect available ETL directories
    available_datasets = processor.find_etl_directories()
    logger.info(
        "Found %d available ETL datasets: %s",
        len(available_datasets),
        ", ".join(available_datasets.keys()),
    )

    if not available_datasets:
        logger.error("No ETL datasets found in project root")
        return 1

    # Determine which datasets to process
    if args.only:
        datasets_to_process = [d.lower() for d in args.only]
        missing = [d for d in datasets_to_process if d not in available_datasets]
        if missing:
            logger.error("Specified datasets not found: %s", ", ".join(missing))
            return 1
        logger.info("Processing specified datasets: %s", ", ".join(datasets_to_process))
    else:
        # Process in priority order (high-value datasets first)
        priority_order = ["etl9g", "etl8g", "etl7", "etl6", "etl5", "etl4", "etl3", "etl2", "etl1"]
        datasets_to_process = [d for d in priority_order if d in available_datasets]
        logger.info("Processing datasets in priority order: %s", ", ".join(datasets_to_process))

    if not datasets_to_process:
        logger.error("No datasets to process")
        return 1

    # Process each dataset
    processed_datasets = []
    for dataset_name in datasets_to_process:
        try:
            logger.info("\nProcessing %s...", dataset_name.upper())
            etl_dir = available_datasets[dataset_name]
            processor.process_dataset(
                dataset_name,
                etl_dir,
                target_size=args.size,
                max_workers=args.workers,
            )
            processed_datasets.append(dataset_name)
            logger.info("✓ %s completed", dataset_name.upper())
        except Exception as e:  # noqa: S110
            logger.error("✗ Error processing %s: %s", dataset_name.upper(), str(e))

    # Combine datasets (default behavior unless --no-combine specified)
    combined_dataset_name = None
    if not args.no_combine and len(processed_datasets) > 1:
        try:
            logger.info("\nCombining %d datasets into unified dataset...", len(processed_datasets))
            processor.combine_datasets(processed_datasets, output_name="combined_all_etl")
            combined_dataset_name = "combined_all_etl"
            logger.info("✓ Combined dataset created")
        except Exception as e:
            logger.error("Error combining datasets: %s", str(e))
            return 1
    elif args.no_combine:
        logger.info("Dataset combination skipped (--no-combine)")

    # Create root metadata.json for training scripts
    combined_metadata_path = output_dir / combined_dataset_name / "metadata.json"
    if combined_dataset_name and combined_metadata_path.exists():
        try:
            logger.info("\nCreating root metadata for training scripts...")
            with open(combined_metadata_path, encoding="utf-8") as f:
                combined_metadata = json.load(f)

            # Create root metadata with reference to primary dataset
            root_metadata = {
                "primary_dataset": combined_dataset_name,
                "num_classes": combined_metadata.get("num_classes"),
                "total_samples": combined_metadata.get("total_samples"),
                "target_size": combined_metadata.get("target_size"),
                "jis_to_class": combined_metadata.get("jis_to_class", {}),
            }

            root_metadata_path = output_dir / "metadata.json"
            with open(root_metadata_path, "w", encoding="utf-8") as f:
                json.dump(root_metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"✓ Root metadata created: {root_metadata_path}")
        except Exception as e:
            logger.warning(f"⚠ Could not create root metadata: {e}")

    # Summary
    logger.info("\n=== Summary ===")
    for dataset in processed_datasets:
        output_path = output_dir / dataset
        logger.info("✓ %s: %s", dataset.upper(), output_path)

    if not args.no_combine and len(processed_datasets) > 1:
        logger.info("✓ Combined: %s", output_dir / "combined_all_etl")

    logger.info("✓ All datasets prepared successfully!")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
