from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytesseract
import streamlit as st
from PIL import Image, UnidentifiedImageError

from export import build_result_dataframe, dataframe_to_safe_csv
from ocr import (
    MIN_CONFIDENCE,
    extract_english_lines,
    preprocess_image_for_ocr,
    resolve_tesseract_cmd,
)
from rendering import draw_preview, image_to_png_bytes
from translation import TranslationResult, translate_lines


MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


def process_uploaded_image(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
    min_confidence: float,
) -> tuple[Image.Image, Image.Image, list[TranslationResult]] | None:
    """Process one uploaded image and return source, preview, and translation results."""

    if uploaded_file.size > MAX_UPLOAD_BYTES:
        st.error(
            f"'{uploaded_file.name}' 파일이 너무 큽니다 "
            f"(최대 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB)."
        )
        return None

    try:
        image = Image.open(uploaded_file)
        if image.width * image.height > MAX_IMAGE_PIXELS:
            st.error(f"'{uploaded_file.name}' 이미지 해상도가 너무 큽니다.")
            return None
        image.load()
    except UnidentifiedImageError:
        st.error(f"'{uploaded_file.name}' 파일을 이미지로 읽을 수 없습니다.")
        return None

    preprocessed_image = preprocess_image_for_ocr(image)
    lines = extract_english_lines(preprocessed_image, min_confidence=min_confidence)
    if not lines:
        st.warning(
            f"'{uploaded_file.name}' 파일에서 조건에 맞는 영어 텍스트를 찾지 못했습니다."
        )
        return None

    results = translate_lines(lines, file_name=uploaded_file.name)
    preview = draw_preview(image, results)
    return image, preview, results


def configure_sidebar() -> tuple[float, str]:
    """Render sidebar controls and return user-selected settings."""

    with st.sidebar:
        st.header("설정")
        min_confidence = st.slider(
            "최소 OCR 신뢰도",
            min_value=0,
            max_value=100,
            value=int(MIN_CONFIDENCE),
        )
        tesseract_path = st.text_input(
            "Tesseract 실행 파일 경로(선택)",
            value="",
            placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        )

    return float(min_confidence), tesseract_path.strip()


def render_results(
    uploaded_file_name: str,
    image: Image.Image,
    preview: Image.Image,
    results: Sequence[TranslationResult],
    expanded: bool,
) -> None:
    """Render original and preview images for a processed upload."""

    translated_count = sum(1 for result in results if result.status == "success")
    skipped_count = sum(
        1 for result in results if result.status == "skipped_non_english"
    )
    preview_bytes = image_to_png_bytes(preview)

    with st.expander(f"이미지 미리보기: {uploaded_file_name}", expanded=expanded):
        st.caption(
            f"번역된 줄 {translated_count}개, 한글 또는 비영문으로 건너뛴 줄 {skipped_count}개"
        )
        original_tab, preview_tab = st.tabs(["원본", "번역 미리보기"])
        with original_tab:
            st.image(image, width="stretch")
        with preview_tab:
            st.image(preview, width="stretch")

        st.download_button(
            label=f"{uploaded_file_name} 번역 미리보기 PNG 다운로드",
            data=preview_bytes,
            file_name=f"{Path(uploaded_file_name).stem}_translated_preview.png",
            mime="image/png",
        )


def main() -> None:
    """Run the Streamlit OCR translation application."""

    st.set_page_config(page_title="이미지 영어-한국어 번역기", layout="wide")
    st.title("이미지 영어-한국어 번역기")
    st.write(
        "여러 이미지를 업로드하면 영어 텍스트를 OCR로 추출하고 한국어로 번역합니다."
    )
    st.caption(
        "⚠️ 추출된 텍스트는 번역을 위해 Google 번역(외부 서비스)으로 전송됩니다. "
        "민감한 개인정보나 기밀 문서가 포함된 이미지는 업로드하지 마세요."
    )

    min_confidence, tesseract_path = configure_sidebar()
    try:
        pytesseract.pytesseract.tesseract_cmd = resolve_tesseract_cmd(tesseract_path)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    uploaded_files = st.file_uploader(
        "이미지를 업로드하세요",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("번역할 이미지를 하나 이상 업로드하세요.")
        st.stop()

    all_results: list[TranslationResult] = []
    processed_count = 0

    try:
        for uploaded_file in uploaded_files:
            processed = process_uploaded_image(
                uploaded_file=uploaded_file,
                min_confidence=min_confidence,
            )
            if processed is None:
                continue

            image, preview, results = processed
            processed_count += 1
            all_results.extend(results)
            render_results(
                uploaded_file_name=uploaded_file.name,
                image=image,
                preview=preview,
                results=results,
                expanded=processed_count == 1,
            )
    except pytesseract.TesseractNotFoundError:
        st.error(
            "Tesseract OCR이 설치되어 있지 않거나 실행 파일 경로가 올바르지 않습니다. "
            "README를 확인하세요."
        )
        st.stop()

    if not all_results:
        st.warning("업로드한 모든 이미지에서 번역할 영어 텍스트를 찾지 못했습니다.")
        st.stop()

    st.success(f"{processed_count}개 이미지에서 번역 결과를 추출했습니다.")

    st.subheader("통합 번역 결과")
    dataframe = build_result_dataframe(all_results)
    st.dataframe(dataframe, width="stretch", hide_index=True)

    csv_bytes = dataframe_to_safe_csv(dataframe)
    st.download_button(
        label="통합 CSV 다운로드",
        data=csv_bytes,
        file_name="image_translation_ko.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
