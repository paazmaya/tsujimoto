"""
Format handlers for research datasets.

Provides unified format handling for:
- MegaHan97K: Large-scale Chinese character dataset
- DKDS: Degraded Kuzushiji documents
- Chronicles-OCR: Historical character evolution benchmark
- JaWildText: Japanese scene text
- MCCD: Multi-attribute calligraphy characters
- Stroke Database: Stroke-level handwriting data
- Kanji Full: Comprehensive kanji character dataset
- Kanji Dataset v3: Kanji dataset version 3 with expanded coverage
- Kanji: Kanji characters with diverse writing styles
"""

from .chronicles_handler import ChroniclesOCRHandler
from .dkds_handler import DKDSHandler
from .jawildtext_handler import JaWildTextHandler
from .kanji_dataset_v3_handler import KanjiDatasetV3Handler
from .kanji_full_handler import KanjiFulHandler
from .kanji_handler import KanjiHandler
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
    "KanjiFulHandler",
    "KanjiDatasetV3Handler",
    "KanjiHandler",
]
