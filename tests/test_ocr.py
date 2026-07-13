"""Tests for Tesseract path resolution in ocr.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ocr


def test_empty_path_falls_back_to_detection_or_default():
    result = ocr.resolve_tesseract_cmd("")
    assert result.endswith("tesseract") or result.endswith("tesseract.exe")


def test_whitespace_path_treated_as_empty():
    result = ocr.resolve_tesseract_cmd("   ")
    assert result.endswith("tesseract") or result.endswith("tesseract.exe")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path validation")
def test_windows_path_to_non_tesseract_binary_rejected():
    with pytest.raises(ValueError):
        ocr.resolve_tesseract_cmd(r"C:\Windows\System32\calc.exe")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path validation")
def test_windows_nonexistent_drive_path_rejected():
    with pytest.raises(ValueError):
        ocr.resolve_tesseract_cmd(r"C:\does\not\exist\tesseract.exe")


def test_valid_tesseract_file_accepted(tmp_path):
    binary_name = "tesseract.exe" if sys.platform == "win32" else "tesseract"
    fake_binary = tmp_path / binary_name
    fake_binary.write_bytes(b"")
    result = ocr.resolve_tesseract_cmd(str(fake_binary))
    assert result == str(fake_binary)


def make_word(text: str, left: int, width: int = 50, height: int = 20) -> dict:
    return {"text": text, "conf": 90.0, "left": left, "top": 0, "width": width, "height": height}


def test_split_line_keeps_adjacent_words_together():
    words = [make_word("Hello", 0), make_word("world", 60)]
    segments = ocr.split_line_on_gaps(words)
    assert len(segments) == 1
    assert [w["text"] for w in segments[0]] == ["Hello", "world"]


def test_split_line_separates_distant_words():
    # 1000px gap with 20px-tall words: noise scattered across a camera photo
    words = [make_word("Hello", 0), make_word("noise", 1100)]
    segments = ocr.split_line_on_gaps(words)
    assert len(segments) == 2


def test_extract_text_lines_splits_scattered_noise(monkeypatch):
    from PIL import Image

    def fake_image_to_data(image, lang, config, output_type):
        # Two words Tesseract wrongly grouped into one line, 1500px apart
        return {
            "text": ["left", "right"],
            "conf": ["90", "90"],
            "left": [0, 1550],
            "top": [100, 100],
            "width": [50, 50],
            "height": [100, 100],
            "block_num": [1, 1],
            "par_num": [1, 1],
            "line_num": [1, 1],
        }

    monkeypatch.setattr(ocr.pytesseract, "image_to_data", fake_image_to_data)
    lines = ocr.extract_text_lines(Image.new("L", (1600, 300)), min_confidence=20.0)
    assert len(lines) == 2
    assert all(line.width <= 100 for line in lines)
