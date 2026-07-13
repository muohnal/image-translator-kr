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


def clean_ocr_text(text: str) -> str:
    """Remove common OCR artifacts so the translator receives cleaner input."""

    tokens = text.split()
    # Drop tokens with no letters or digits (stray '|', '~', '—' from OCR noise).
    meaningful_tokens = [
        token for token in tokens if re.search(r"[A-Za-z0-9가-힣]", token)
    ]
    return " ".join(meaningful_tokens)


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
    """Translate OCR lines from English to Korean using a single batch request."""

    results: list[TranslationResult] = []
    pending: list[tuple[int, str]] = []

    for line in lines:
        cleaned_text = clean_ocr_text(line.text)

        if not cleaned_text or not contains_meaningful_english(cleaned_text):
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
                status="translation_failed",
                error_message=None,
            )
        )
        pending.append((len(results) - 1, cleaned_text))

    if not pending:
        return results

    translator = GoogleTranslator(source="en", target="ko")
    pending_texts = [text for _, text in pending]

    translations: list[str | None] | None
    try:
        translations = translator.translate_batch(pending_texts)
    except (TranslationError, ConnectionError, TimeoutError) as exc:
        logger.warning("Batch translation failed, falling back to per-line: %s", exc)
        translations = None

    if translations is not None:
        for (result_index, _), translated_text in zip(pending, translations):
            if translated_text:
                results[result_index].translated_text = translated_text
                results[result_index].status = "success"
            else:
                results[result_index].error_message = "빈 번역 결과"
        return results

    for result_index, cleaned_text in pending:
        try:
            results[result_index].translated_text = translator.translate(cleaned_text)
            results[result_index].status = "success"
        except (TranslationError, ConnectionError, TimeoutError) as exc:
            logger.warning("Translation failed for %r: %s", cleaned_text, exc)
            results[result_index].error_message = str(exc)

    return results
