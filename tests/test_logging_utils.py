#!/usr/bin/env python3
"""Extended tests for logging utilities to reach 100% coverage."""

import logging
import tempfile
from pathlib import Path

from src.lib.logging_utils import setup_logger


class TestSetupLogger:
    """Tests for setup_logger function."""

    def test_setup_logger_default(self):
        """Test setup_logger with default parameters."""
        logger = setup_logger("test_default")
        assert logger.name == "test_default"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_setup_logger_custom_level(self):
        """Test setup_logger with custom logging level."""
        logger = setup_logger("test_debug", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_setup_logger_with_file_output(self):
        """Test setup_logger with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logger("test_file", log_file=str(log_file))

            # Write a test message
            logger.info("Test message")

            # Close file handlers before cleanup
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

            # Verify file was created and content
            with open(log_file) as f:
                content = f.read()
                assert "Test message" in content

    def test_setup_logger_creates_log_directory(self):
        """Test that setup_logger creates parent directories for log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "nested" / "logs" / "test.log"
            logger = setup_logger("test_nested", log_file=str(log_file))

            logger.info("Nested test")
            # Close handlers
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            assert log_file.exists()
            assert log_file.parent.exists()

    def test_setup_logger_file_formatting(self):
        """Test that file handler has detailed formatting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "detailed.log"
            logger = setup_logger("test_format", log_file=str(log_file))

            logger.info("Detailed message")
            # Close handlers
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            with open(log_file) as f:
                content = f.read()
                # Should have timestamp, name, level
                assert "test_format" in content
                assert "INFO" in content
                assert "Detailed message" in content

    def test_setup_logger_no_duplicate_handlers(self):
        """Test that calling setup_logger twice doesn't add duplicate handlers."""
        logger_name = "test_no_dupe"

        logger1 = setup_logger(logger_name)
        initial_handler_count = len(logger1.handlers)

        logger2 = setup_logger(logger_name)
        assert len(logger2.handlers) == initial_handler_count

    def test_setup_logger_different_levels(self):
        """Test setup_logger with various logging levels."""
        levels = [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]

        for level in levels:
            logger = setup_logger(f"test_level_{level}", level=level)
            assert logger.level == level

    def test_setup_logger_console_handler_exists(self):
        """Test that console handler is added."""
        logger = setup_logger("test_console")

        # Check that at least one handler writes to stdout
        has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        assert has_stream_handler

    def test_setup_logger_file_and_console(self):
        """Test logger with both file and console handlers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "both.log"
            logger = setup_logger("test_both", log_file=str(log_file))

            # Should have both handlers
            assert len(logger.handlers) >= 2

            # Check types
            handler_types = [type(h).__name__ for h in logger.handlers]
            assert "StreamHandler" in handler_types
            assert "FileHandler" in handler_types
            # Close handlers
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)


class TestLoggerIntegration:
    """Integration tests for logger usage."""

    def test_logger_actual_logging(self):
        """Test that logger actually logs messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "integration.log"
            logger = setup_logger("test_integration", log_file=str(log_file))

            # Log various levels
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            # Close handlers
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            with open(log_file) as f:
                content = f.read()
                assert "Info message" in content
                assert "Warning message" in content
                assert "Error message" in content

    def test_logger_with_formatting_arguments(self):
        """Test logger with format arguments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "format.log"
            logger = setup_logger("test_fmt", log_file=str(log_file))

            logger.info("Processing %d items", 42)
            logger.info("Status: %s", "complete")
            # Close handlers
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            with open(log_file) as f:
                content = f.read()
                assert "Processing 42 items" in content
                assert "Status: complete" in content

    def test_multiple_loggers_independent(self):
        """Test that multiple loggers are independent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file1 = Path(tmpdir) / "logger1.log"
            log_file2 = Path(tmpdir) / "logger2.log"

            logger1 = setup_logger("independent_1", log_file=str(log_file1))
            logger2 = setup_logger("independent_2", log_file=str(log_file2))

            logger1.info("Message from logger 1")
            logger2.info("Message from logger 2")

            # Close handlers
            for handler in logger1.handlers[:]:
                handler.close()
                logger1.removeHandler(handler)
            for handler in logger2.handlers[:]:
                handler.close()
                logger2.removeHandler(handler)

            with open(log_file1) as f:
                content1 = f.read()
                assert "Message from logger 1" in content1
                assert "Message from logger 2" not in content1

            with open(log_file2) as f:
                content2 = f.read()
                assert "Message from logger 2" in content2
                assert "Message from logger 1" not in content2
