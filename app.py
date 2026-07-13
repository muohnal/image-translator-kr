from __future__ import annotations

import logging
import os
from io import BytesIO
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


logger = logging.getLogger(__name__)


MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000

CUSTOM_STYLE = """
<style>
/* 모바일 터치 타깃 확대 및 버튼 스타일 */
.stButton > button,
.stDownloadButton > button {
    width: 100%;
    min-height: 48px;
    border-radius: 12px;
    font-size: 1rem;
    font-weight: 600;
}

/* 업로더 영역을 눈에 잘 띄게 */
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #3B82F6;
    border-radius: 12px;
}

/* 탭을 크게 - 모바일에서 누르기 쉽게 */
.stTabs [data-baseweb="tab"] {
    min-height: 44px;
    font-size: 1rem;
}

/* Streamlit 기본 장식 숨김 (배포 앱에서 불필요) */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* 모바일에서 본문 좌우 여백 축소로 화면 넓게 사용 */
@media (max-width: 640px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 2rem;
    }
}
</style>
"""


@st.cache_data(show_spinner=False, max_entries=20)
def run_translation_pipeline(
    file_bytes: bytes,
    file_name: str,
    min_confidence: float,
) -> tuple[Image.Image, Image.Image, list[TranslationResult]] | None:
    """Run OCR, translation, and preview rendering; cached per file content."""

    image = Image.open(BytesIO(file_bytes))
    image.load()

    preprocessed_image = preprocess_image_for_ocr(image)
    lines = extract_english_lines(preprocessed_image, min_confidence=min_confidence)
    if not lines:
        return None

    results = translate_lines(lines, file_name=file_name)
    preview = draw_preview(image, results)
    return image, preview, results


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

    file_bytes = uploaded_file.getvalue()

    try:
        probe = Image.open(BytesIO(file_bytes))
        if probe.width * probe.height > MAX_IMAGE_PIXELS:
            st.error(
                f"'{uploaded_file.name}' 이미지 해상도가 너무 큽니다 (최대 약 4천만 화소). "
                "스크린샷을 자르거나 축소한 뒤 다시 시도하세요."
            )
            return None
    except UnidentifiedImageError:
        st.error(f"'{uploaded_file.name}' 파일을 이미지로 읽을 수 없습니다.")
        return None

    processed = run_translation_pipeline(
        file_bytes, uploaded_file.name, min_confidence
    )
    if processed is None:
        st.warning(
            f"'{uploaded_file.name}' 파일에서 조건에 맞는 영어 텍스트를 찾지 못했습니다."
        )
        return None

    return processed


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
    failed_count = sum(
        1 for result in results if result.status == "translation_failed"
    )
    preview_bytes = image_to_png_bytes(preview)

    with st.expander(f"이미지 미리보기: {uploaded_file_name}", expanded=expanded):
        st.caption(
            f"번역 {translated_count}개 · 건너뜀(영어 아님) {skipped_count}개 · 실패 {failed_count}개"
        )
        if failed_count:
            st.warning(
                "일부 문장 번역에 실패했습니다. 잠시 후 다시 업로드해 보세요."
            )
        preview_tab, original_tab = st.tabs(["번역 미리보기", "원본"])
        with preview_tab:
            st.image(preview, width="stretch")
        with original_tab:
            st.image(image, width="stretch")

        st.download_button(
            label="번역 미리보기 PNG 저장",
            data=preview_bytes,
            file_name=f"{Path(uploaded_file_name).stem}_translated_preview.png",
            mime="image/png",
            key=f"download_{uploaded_file_name}",
        )


def main() -> None:
    """Run the Streamlit OCR translation application."""

    st.set_page_config(
        page_title="이미지 영어-한국어 번역기",
        page_icon="🌐",
        layout="centered",
    )
    st.markdown(CUSTOM_STYLE, unsafe_allow_html=True)
    st.title("이미지 영어→한국어 번역기")
    st.write("스마트폰 스크린샷 속 영어를 자동으로 찾아 한국어로 번역해 드립니다.")
    st.markdown(
        "1. 아래에서 이미지를 업로드하세요\n"
        "2. 번역이 입혀진 미리보기를 확인하세요\n"
        "3. 필요하면 PNG/CSV로 저장하세요"
    )
    st.caption(
        "⚠️ 추출된 텍스트는 번역을 위해 Google 번역(외부 서비스)으로 전송됩니다. "
        "민감한 개인정보나 기밀 문서가 포함된 이미지는 업로드하지 마세요."
    )

    try:
        pytesseract.pytesseract.tesseract_cmd = resolve_tesseract_cmd(
            os.environ.get("TESSERACT_CMD", "")
        )
    except ValueError as exc:
        logger.error("Invalid TESSERACT_CMD: %s", exc)
        st.error("서버의 텍스트 인식 엔진 설정에 문제가 있습니다. 잠시 후 다시 시도해 주세요.")
        st.stop()

    st.caption(
        f"PNG·JPG·WEBP·BMP 지원, 장당 최대 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB / 약 4천만 화소"
    )
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
    total_files = len(uploaded_files)
    progress = st.progress(0.0, text="이미지 처리 준비 중…")

    try:
        for index, uploaded_file in enumerate(uploaded_files):
            progress.progress(
                index / total_files,
                text=f"({index + 1}/{total_files}) '{uploaded_file.name}' 텍스트 인식·번역 중…",
            )
            processed = process_uploaded_image(
                uploaded_file=uploaded_file,
                min_confidence=MIN_CONFIDENCE,
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
        progress.empty()
        logger.error("Tesseract binary not found on server")
        st.error("서버의 텍스트 인식 엔진에 문제가 있습니다. 잠시 후 다시 시도해 주세요.")
        st.stop()

    progress.empty()

    if not all_results:
        st.warning(
            "업로드한 이미지에서 영어 텍스트를 찾지 못했습니다. "
            "글씨가 선명하게 보이는 스크린샷인지 확인해 주세요."
        )
        st.stop()

    success_line_count = sum(
        1 for result in all_results if result.status == "success"
    )
    if success_line_count:
        st.success(f"총 {success_line_count}개 문장을 번역했습니다.")
    else:
        st.warning("번역할 영어 문장을 찾지 못했거나 번역에 실패했습니다.")

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
