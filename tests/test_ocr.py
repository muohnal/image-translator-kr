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
