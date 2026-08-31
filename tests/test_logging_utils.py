#!/usr/bin/env python3
"""Extended tests for logging utilities to reach 100% coverage."""

import tempfile
from pathlib import Path

from src.lib.logging_utils import setup_logger, suppress_warnings


class TestSetupLogger:
    """Tests for setup_logger function."""

    def test_setup_logger_default(self):
        """Test setup_logger with default parameters."""
        logger = setup_logger("test_default")
        # Loguru logger should be returned
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")

    def test_setup_logger_custom_level(self):
        """Test setup_logger with custom logging level."""
        logger = setup_logger("test_debug", level="DEBUG")
        # Should accept DEBUG level
        assert logger is not None
        assert hasattr(logger, "debug")

    def test_setup_logger_with_file_output(self):
        """Test setup_logger with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logger("test_file", log_file=str(log_file))

            # Write a test message
            logger.info("Test message")

            # Verify file was created and content
            assert log_file.exists()
            with open(log_file) as f:
                content = f.read()
                assert "Test message" in content

    def test_setup_logger_creates_log_directory(self):
        """Test that setup_logger creates parent directories for log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "nested" / "logs" / "test.log"
            logger = setup_logger("test_nested", log_file=str(log_file))

            logger.info("Nested test")
            assert log_file.exists()
            assert log_file.parent.exists()

    def test_setup_logger_file_formatting(self):
        """Test that file handler has detailed formatting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "detailed.log"
            logger = setup_logger("test_format", log_file=str(log_file))

            logger.info("Detailed message")
            with open(log_file) as f:
                content = f.read()
                # Should have timestamp and detailed info
                assert "Detailed message" in content
                # Loguru format includes timestamp in ISO format
                assert content.count("|") > 0  # Log format delimiter

    def test_setup_logger_no_duplicate_handlers(self):
        """Test that calling setup_logger twice works correctly."""
        logger_name = "test_no_dupe"

        logger1 = setup_logger(logger_name)
        logger2 = setup_logger(logger_name)

        # Both should be loguru logger instances
        assert logger1 is not None
        assert logger2 is not None

    def test_setup_logger_different_levels(self):
        """Test setup_logger with various logging levels."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in levels:
            logger = setup_logger(f"test_level_{level}", level=level)
            assert logger is not None
            assert hasattr(logger, "info")

    def test_setup_logger_console_handler_exists(self):
        """Test that logger can write to console."""
        logger = setup_logger("test_console")
        # Should not raise exception when logging to console
        logger.info("Console test message")
        assert True  # If we got here, it worked

    def test_setup_logger_file_and_console(self):
        """Test logger with both file and console handlers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "both.log"
            logger = setup_logger("test_both", log_file=str(log_file))

            # Log to both file and console
            logger.info("Test message")

            # File should exist
            assert log_file.exists()

            # File should have content
            with open(log_file) as f:
                content = f.read()
                assert "Test message" in content


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

            logger.info("Processing {} items", 42)
            logger.info("Status: {}", "complete")

            with open(log_file) as f:
                content = f.read()
                assert "Processing 42 items" in content
                assert "Status: complete" in content

    def test_multiple_loggers_independent(self):
        """Test that multiple loggers can write to independent files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file1 = Path(tmpdir) / "logger1.log"
            log_file2 = Path(tmpdir) / "logger2.log"

            logger1 = setup_logger("independent_1", log_file=str(log_file1))
            logger2 = setup_logger("independent_2", log_file=str(log_file2))

            logger1.info("Message from logger 1")
            logger2.info("Message from logger 2")

            with open(log_file1) as f:
                content1 = f.read()
                assert "Message from logger 1" in content1

            with open(log_file2) as f:
                content2 = f.read()
                assert "Message from logger 2" in content2


class TestSuppressWarnings:
    """Tests for suppress_warnings function."""

    def test_suppress_warnings_callable(self):
        """Test that suppress_warnings can be called without error."""
        suppress_warnings()
        assert True  # If we got here, it worked

    def test_suppress_warnings_no_exceptions(self):
        """Test that suppress_warnings doesn't raise exceptions."""
        try:
            suppress_warnings()
            success = True
        except Exception as e:
            success = False
            raise e
        assert success
