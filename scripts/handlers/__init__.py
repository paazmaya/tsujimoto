"""
Format handlers for research datasets.

Provides unified format handling for:
- MegaHan97K: Large-scale Chinese character dataset
- DKDS: Degraded Kuzushiji documents
- Chronicles-OCR: Historical character evolution benchmark
- JaWildText: Japanese scene text
- MCCD: Multi-attribute calligraphy characters
- Stroke Database: Stroke-level handwriting data
"""

from .chronicles_handler import ChroniclesOCRHandler
from .dkds_handler import DKDSHandler
from .jawildtext_handler import JaWildTextHandler
from .mccd_handler import MCCDHandler
from .megahan97k_handler import MegaHan97KHandler
from .stroke_database_handler import StrokeDatabaseHandler

__all__ = [
    "MegaHan97KHandler",
    "DKDSHandler",
    "ChroniclesOCRHandler",
    "JaWildTextHandler",
    "MCCDHandler",
    "StrokeDatabaseHandler",
]
