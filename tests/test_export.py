"""Tests for CSV export safety and status localization in export.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import export
from translation import TranslationResult


def make_result(**overrides) -> TranslationResult:
    defaults = dict(
        file_name="t.png",
        source_text="Hello",
        translated_text="안녕",
        confidence=90.0,
        left=0,
        top=0,
        width=100,
        height=20,
        status="success",
    )
    defaults.update(overrides)
    return TranslationResult(**defaults)


def test_escape_csv_formula_prefixes():
    assert export.escape_csv_formula("=SUM(A1)") == "'=SUM(A1)"
    assert export.escape_csv_formula("+1") == "'+1"
    assert export.escape_csv_formula("-1") == "'-1"
    assert export.escape_csv_formula("@cmd") == "'@cmd"
    assert export.escape_csv_formula("hello") == "hello"


def test_status_labels_localized():
    dataframe = export.build_result_dataframe(
        [
            make_result(status="success"),
            make_result(status="skipped_non_english", translated_text=None),
            make_result(status="translation_failed", translated_text=None),
        ]
    )
    assert list(dataframe["상태"]) == ["번역 완료", "건너뜀(영어 아님)", "번역 실패"]


def test_unknown_status_passes_through():
    dataframe = export.build_result_dataframe([make_result(status="weird_status")])
    assert dataframe["상태"][0] == "weird_status"


def test_safe_csv_escapes_formula_in_source_text():
    dataframe = export.build_result_dataframe(
        [make_result(source_text="=HYPERLINK(evil)")]
    )
    csv_text = export.dataframe_to_safe_csv(dataframe).decode("utf-8-sig")
    assert "'=HYPERLINK(evil)" in csv_text


def test_safe_csv_uses_utf8_sig_encoding():
    dataframe = export.build_result_dataframe([make_result()])
    data = export.dataframe_to_safe_csv(dataframe)
    assert data[:3] == b"\xef\xbb\xbf"
