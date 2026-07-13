from __future__ import annotations

from typing import Sequence

import pandas as pd

from translation import TranslationResult


CSV_FORMULA_PREFIXES = ("=", "+", "-", "@")
STATUS_LABELS = {
    "success": "번역 완료",
    "skipped_non_english": "건너뜀(영어 아님)",
    "translation_failed": "번역 실패",
}


def escape_csv_formula(value: str) -> str:
    """Prefix a leading quote to prevent spreadsheet formula injection."""

    if value.startswith(CSV_FORMULA_PREFIXES):
        return f"'{value}"
    return value


def build_result_dataframe(results: Sequence[TranslationResult]) -> pd.DataFrame:
    """Convert translation results into a dataframe for display and export."""

    dataframe = pd.DataFrame(
        [
            {
                "파일명": result.file_name,
                "원문": result.source_text,
                "번역문": result.translated_text or "",
                "신뢰도": result.confidence,
                "상태": STATUS_LABELS.get(result.status, result.status),
                "오류": result.error_message or "",
            }
            for result in results
        ]
    )
    return dataframe


def dataframe_to_safe_csv(dataframe: pd.DataFrame) -> bytes:
    """Encode a dataframe as CSV bytes with formula-injection escaping applied."""

    safe_dataframe = dataframe.copy()
    for column in ("파일명", "원문", "번역문", "오류"):
        safe_dataframe[column] = safe_dataframe[column].astype(str).map(escape_csv_formula)
    return safe_dataframe.to_csv(index=False).encode("utf-8-sig")
