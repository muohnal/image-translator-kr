"""Tests for OCR text cleaning and batch translation in translation.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import translation
from ocr import OCRLine


def make_line(text: str) -> OCRLine:
    return OCRLine(text=text, confidence=90.0, left=0, top=0, width=100, height=20)


class BatchTranslator:
    def __init__(self, *args, **kwargs):
        pass

    def translate_batch(self, texts):
        return [f"번역:{text}" for text in texts]

    def translate(self, text):
        raise AssertionError("per-line translate should not be called when batch works")


class BatchFailingTranslator:
    def __init__(self, *args, **kwargs):
        pass

    def translate_batch(self, texts):
        raise ConnectionError("batch down")

    def translate(self, text):
        return f"번역:{text}"


class DeadTranslator:
    def __init__(self, *args, **kwargs):
        pass

    def translate_batch(self, texts):
        raise ConnectionError("down")

    def translate(self, text):
        raise ConnectionError("down")


def test_clean_ocr_text_removes_noise_tokens():
    assert translation.clean_ocr_text("| Tea Tree ~ Serum |") == "Tea Tree Serum"


def test_clean_ocr_text_pure_noise_becomes_empty():
    assert translation.clean_ocr_text("|| ~~ ——") == ""


def test_contains_meaningful_english():
    assert translation.contains_meaningful_english("Hello world")
    assert not translation.contains_meaningful_english("안녕하세요")
    assert not translation.contains_meaningful_english("OK 안녕하세요 반갑습니다")


def test_translate_lines_batch_success(monkeypatch):
    monkeypatch.setattr(translation, "GoogleTranslator", BatchTranslator)
    results = translation.translate_lines(
        [make_line("Hello world"), make_line("안녕하세요")], "test.png"
    )
    assert results[0].status == "success"
    assert results[0].translated_text == "번역:Hello world"
    assert results[1].status == "skipped_non_english"
    assert results[1].translated_text is None


def test_translate_lines_falls_back_per_line(monkeypatch):
    monkeypatch.setattr(translation, "GoogleTranslator", BatchFailingTranslator)
    results = translation.translate_lines([make_line("Hello world")], "test.png")
    assert results[0].status == "success"
    assert results[0].translated_text == "번역:Hello world"


def test_translate_lines_records_failure(monkeypatch):
    monkeypatch.setattr(translation, "GoogleTranslator", DeadTranslator)
    results = translation.translate_lines([make_line("Hello world")], "test.png")
    assert results[0].status == "translation_failed"
    assert results[0].translated_text is None
    assert results[0].error_message


def test_translate_lines_no_translatable_lines_skips_network(monkeypatch):
    class ExplodingTranslator:
        def __init__(self, *args, **kwargs):
            raise AssertionError("translator should not be constructed")

    monkeypatch.setattr(translation, "GoogleTranslator", ExplodingTranslator)
    results = translation.translate_lines([make_line("안녕하세요")], "test.png")
    assert results[0].status == "skipped_non_english"
