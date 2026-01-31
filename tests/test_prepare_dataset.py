#!/usr/bin/env python3
"""Unit tests for prepare_dataset module."""

import struct
import tempfile
from pathlib import Path

import pytest

from scripts.prepare_dataset import (
    ETL1Handler,
    ETL2Handler,
    ETL6Handler,
    ETL8GHandler,
    ETL9GHandler,
    ETLDatasetProcessor,
)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestETL1HandlerDefaults:
    """Tests for ETL1 format handler defaults."""

    def test_record_size(self):
        """Test ETL1 record size is 2052 bytes."""
        handler = ETL1Handler()
        assert handler.get_record_size() == 2052

    def test_image_dimensions(self):
        """Test ETL1 image dimensions are 72x76."""
        handler = ETL1Handler()
        width, height = handler.get_image_dimensions()
        assert width == 72
        assert height == 76


class TestETL1Handler:
    """Tests for ETL1 record extraction."""

    def test_extract_valid_record(self):
        """Test extracting a valid ETL1 record."""
        handler = ETL1Handler()
        # Create minimal valid record
        record = bytearray(2052)
        # Add serial (bytes 0-1)
        record[0:2] = struct.pack(">H", 12345)
        # Add JIS code (bytes 2-3)
        record[2:4] = struct.pack(">H", 0x3421)  # Valid JIS code
        # Add ASCII reading (bytes 4-11)
        record[4:12] = b"test    "
        # Add data serial (bytes 12-15)
        record[12:16] = struct.pack(">I", 100)
        # Add writer ID (byte 18)
        record[18] = 5

        result = handler.extract_record_info(bytes(record))

        assert result is not None
        assert result["serial"] == 12345
        assert result["jis_code"] == 0x3421
        assert result["format_type"] == "M-type"
        assert result["writer_id"] == 5


class TestETL2HandlerDefaults:
    """Tests for ETL2 format handler defaults."""

    def test_record_size(self):
        """Test ETL2 record size is 1956 bytes."""
        handler = ETL2Handler()
        assert handler.get_record_size() == 1956

    def test_image_dimensions(self):
        """Test ETL2 image dimensions are 60x60."""
        handler = ETL2Handler()
        width, height = handler.get_image_dimensions()
        assert width == 60
        assert height == 60


class TestETL6HandlerDefaults:
    """Tests for ETL6 format handler defaults."""

    def test_record_size(self):
        """Test ETL6 record size is 2052 bytes."""
        handler = ETL6Handler()
        assert handler.get_record_size() == 2052

    def test_image_dimensions(self):
        """Test ETL6 image dimensions are 64x63."""
        handler = ETL6Handler()
        width, height = handler.get_image_dimensions()
        assert width == 64
        assert height == 63


class TestETL8GHandlerDefaults:
    """Tests for ETL8G format handler defaults."""

    def test_record_size(self):
        """Test ETL8G record size is 8199 bytes."""
        handler = ETL8GHandler()
        assert handler.get_record_size() == 8199

    def test_image_dimensions(self):
        """Test ETL8G image dimensions are 128x127."""
        handler = ETL8GHandler()
        width, height = handler.get_image_dimensions()
        assert width == 128
        assert height == 127


class TestETL8GHandler:
    """Tests for ETL8G record extraction."""

    def test_extract_valid_record(self):
        """Test extracting a valid ETL8G record."""
        handler = ETL8GHandler()
        record = bytearray(8199)
        # Add serial (bytes 0-1)
        record[0:2] = struct.pack(">H", 54321)
        # Add JIS code (bytes 2-3)
        record[2:4] = struct.pack(">H", 0x3021)
        # Add ASCII reading (bytes 4-11)
        record[4:12] = b"etl8g   "
        # Add data serial (bytes 12-15)
        record[12:16] = struct.pack(">I", 200)
        # Add writer ID (byte 18)
        record[18] = 10

        result = handler.extract_record_info(bytes(record))

        assert result is not None
        assert result["serial"] == 54321
        assert result["jis_code"] == 0x3021
        assert result["format_type"] == "ETL8G"
        assert result["writer_id"] == 10


class TestETL9GHandlerDefaults:
    """Tests for ETL9G format handler defaults."""

    def test_record_size(self):
        """Test ETL9G record size is 8199 bytes."""
        handler = ETL9GHandler()
        assert handler.get_record_size() == 8199

    def test_image_dimensions(self):
        """Test ETL9G image dimensions are 128x127."""
        handler = ETL9GHandler()
        width, height = handler.get_image_dimensions()
        assert width == 128
        assert height == 127


class TestETL9GHandler:
    """Tests for ETL9G record extraction."""

    def test_extract_valid_record(self):
        """Test extracting a valid ETL9G record."""
        handler = ETL9GHandler()
        record = bytearray(8199)
        # Add serial (bytes 0-1) - use valid range for >H format
        record[0:2] = struct.pack(">H", 9999)
        # Add JIS code (bytes 2-3)
        record[2:4] = struct.pack(">H", 0xB341)
        # Add ASCII reading (bytes 4-11)
        record[4:12] = b"kanji   "
        # Add data serial (bytes 12-15)
        record[12:16] = struct.pack(">I", 300)
        # Add writer ID (byte 18)
        record[18] = 15

        result = handler.extract_record_info(bytes(record))

        assert result is not None
        assert result["serial"] == 9999
        assert result["jis_code"] == 0xB341
        assert result["format_type"] == "ETL9G"
        assert result["writer_id"] == 15


class TestHandlerErrorHandling:
    """Tests for error handling in handlers."""

    def test_etl1_handles_short_record(self):
        """Test ETL1 handler gracefully handles short records."""
        handler = ETL1Handler()
        short_record = b"short"
        result = handler.extract_record_info(short_record)
        assert result is None

    def test_etl9g_handles_short_record(self):
        """Test ETL9G handler gracefully handles short records."""
        handler = ETL9GHandler()
        short_record = b"short"
        result = handler.extract_record_info(short_record)
        assert result is None

    def test_etl1_handles_invalid_data(self):
        """Test ETL1 handler handles invalid data gracefully."""
        handler = ETL1Handler()
        invalid_record = b"\xff" * 2052
        # Should not raise exception, should return None or handle gracefully
        result = handler.extract_record_info(invalid_record)
        # Result could be dict or None, either is acceptable
        assert result is None or isinstance(result, dict)


class TestETLDatasetProcessor:
    """Tests for ETLDatasetProcessor initialization."""

    def test_processor_initialization(self, temp_output_dir):
        """Test that ETLDatasetProcessor can be initialized."""
        processor = ETLDatasetProcessor(temp_output_dir)
        assert processor is not None

    def test_processor_has_handlers(self, temp_output_dir):
        """Test that processor has handler registry."""
        processor = ETLDatasetProcessor(temp_output_dir)
        assert hasattr(processor, "HANDLERS")
        assert isinstance(processor.HANDLERS, dict)

    def test_processor_handlers_include_etl9g(self, temp_output_dir):
        """Test that processor includes ETL9G handler."""
        processor = ETLDatasetProcessor(temp_output_dir)
        assert "etl9g" in processor.HANDLERS

    def test_processor_handlers_include_etl8g(self, temp_output_dir):
        """Test that processor includes ETL8G handler."""
        processor = ETLDatasetProcessor(temp_output_dir)
        assert "etl8g" in processor.HANDLERS

    def test_processor_has_dataset_info(self, temp_output_dir):
        """Test that processor has dataset information."""
        processor = ETLDatasetProcessor(temp_output_dir)
        assert hasattr(processor, "DATASET_INFO")
        assert isinstance(processor.DATASET_INFO, dict)

    def test_dataset_info_includes_etl9g(self, temp_output_dir):
        """Test that dataset info includes ETL9G metadata."""
        processor = ETLDatasetProcessor(temp_output_dir)
        assert "etl9g" in processor.DATASET_INFO
        info = processor.DATASET_INFO["etl9g"]
        assert "name" in info
        assert "classes" in info
        assert info["classes"] == 3036  # ETL9G has 3036 classes

    def test_processor_creates_output_dir(self, temp_output_dir):
        """Test that processor creates output directory."""
        output_path = Path(temp_output_dir) / "new_subdir"
        ETLDatasetProcessor(str(output_path))
        assert output_path.exists()


class TestRecordExtraction:
    """Tests for record extraction details."""

    def test_etl1_extracts_image_data(self):
        """Test ETL1 handler extracts image data."""
        handler = ETL1Handler()
        record = bytearray(2052)
        # Fill image data section with test pattern
        for i in range(32, 2048):
            record[i] = i % 256

        result = handler.extract_record_info(bytes(record))

        assert result is not None
        assert "image_data" in result
        assert len(result["image_data"]) == 2016  # 2048 - 32

    def test_etl9g_extracts_image_data_from_correct_offset(self):
        """Test ETL9G handler extracts image from correct offset (64)."""
        handler = ETL9GHandler()
        record = bytearray(8199)
        # Fill image data section
        image_start = 64
        image_end = 8192
        for i in range(image_start, image_end):
            record[i] = (i - image_start) % 256

        result = handler.extract_record_info(bytes(record))

        assert result is not None
        assert "image_data" in result
        expected_size = image_end - image_start
        assert len(result["image_data"]) == expected_size


class TestDimensionConsistency:
    """Tests for dimension consistency."""

    def test_smaller_datasets_have_smaller_images(self):
        """Test that older ETL formats have smaller images than ETL9G."""
        etl1 = ETL1Handler()
        etl2 = ETL2Handler()
        etl9g = ETL9GHandler()

        w1, h1 = etl1.get_image_dimensions()
        w2, h2 = etl2.get_image_dimensions()
        w9g, h9g = etl9g.get_image_dimensions()

        assert w1 * h1 < w9g * h9g
        assert w2 * h2 < w9g * h9g

    def test_etl8g_etl9g_same_dimensions(self):
        """Test that ETL8G and ETL9G have same image dimensions."""
        etl8g = ETL8GHandler()
        etl9g = ETL9GHandler()

        assert etl8g.get_image_dimensions() == etl9g.get_image_dimensions()


class TestRecordSizeCalculations:
    """Tests for record size calculations."""

    def test_etl1_record_size_is_positive(self):
        """Test ETL1 record size is positive."""
        handler = ETL1Handler()
        assert handler.get_record_size() > 0

    def test_etl9g_record_larger_than_etl1(self):
        """Test ETL9G records are larger than ETL1 records."""
        etl1 = ETL1Handler()
        etl9g = ETL9GHandler()
        # ETL9G is a larger format than ETL1
        assert etl9g.get_record_size() > etl1.get_record_size()


class TestASCIIReadingExtraction:
    """Tests for ASCII reading extraction."""

    def test_etl1_ascii_reading_extraction(self):
        """Test ETL1 extracts ASCII reading correctly."""
        handler = ETL1Handler()
        record = bytearray(2052)
        # Add reading at bytes 4-11
        record[4:12] = b"testread"

        result = handler.extract_record_info(bytes(record))

        assert result is not None
        assert "ascii_reading" in result
        assert result["ascii_reading"] == "testread"

    def test_etl9g_ascii_reading_extraction(self):
        """Test ETL9G extracts ASCII reading correctly."""
        handler = ETL9GHandler()
        record = bytearray(8199)
        # Add reading at bytes 4-11
        record[4:12] = b"kanjichar"

        result = handler.extract_record_info(bytes(record))

        assert result is not None
        assert "ascii_reading" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
