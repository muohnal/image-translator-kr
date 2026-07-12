from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

from deep_translator.exceptions import BaseError as TranslationError
from deep_translator import GoogleTranslator

from ocr import OCRLine


ENGLISH_PATTERN = re.compile(r"[A-Za-z]")
KOREAN_PATTERN = re.compile(r"[가-힣]")

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Represents a translated OCR line and its display metadata."""

    file_name: str
    source_text: str
    translated_text: str | None
    confidence: float
    left: int
    top: int
    width: int
    height: int
    status: str
    error_message: str | None = None


def contains_meaningful_english(text: str) -> bool:
    """Return True when the text contains enough English to translate."""

    english_chars = len(ENGLISH_PATTERN.findall(text))
    korean_chars = len(KOREAN_PATTERN.findall(text))

    if english_chars == 0:
        return False
    return english_chars >= korean_chars


def translate_lines(
    lines: Sequence[OCRLine],
    file_name: str,
) -> list[TranslationResult]:
    """Translate OCR lines from English to Korean."""

    translator = GoogleTranslator(source="en", target="ko")
    results: list[TranslationResult] = []

    for line in lines:
        translated_text: str | None
        error_message: str | None
        status: str

        if not contains_meaningful_english(line.text):
            results.append(
                TranslationResult(
                    file_name=file_name,
                    source_text=line.text,
                    translated_text=None,
                    confidence=line.confidence,
                    left=line.left,
                    top=line.top,
                    width=line.width,
                    height=line.height,
                    status="skipped_non_english",
                    error_message=None,
                )
            )
            continue

        try:
            translated_text = translator.translate(line.text)
            error_message = None
            status = "success"
        except (TranslationError, ConnectionError, TimeoutError) as exc:
            logger.warning("Translation failed for %r: %s", line.text, exc)
            translated_text = None
            error_message = str(exc)
            status = "translation_failed"

        results.append(
            TranslationResult(
                file_name=file_name,
                source_text=line.text,
                translated_text=translated_text,
                confidence=line.confidence,
                left=line.left,
                top=line.top,
                width=line.width,
                height=line.height,
                status=status,
                error_message=error_message,
            )
        )

    return results
