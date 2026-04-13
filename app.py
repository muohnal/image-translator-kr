from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import pytesseract
import streamlit as st
from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


MIN_CONFIDENCE = 40.0
FONT_CANDIDATES = (
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/AppleGothic.ttf",
    "C:/Windows/Fonts/NanumGothic.ttf",
)


@dataclass
class OCRLine:
    """Represents a single OCR line extracted from an image."""

    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int


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


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a readable Korean font for the preview image."""

    for font_path in FONT_CANDIDATES:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def extract_english_lines(image: Image.Image, min_confidence: float) -> list[OCRLine]:
    """Extract English OCR lines from an image above the confidence threshold."""

    data = pytesseract.image_to_data(
        image,
        lang="eng",
        output_type=pytesseract.Output.DICT,
    )

    grouped: dict[tuple[int, int, int], list[dict[str, float | int | str]]] = {}
    for index, raw_text in enumerate(data["text"]):
        text = str(raw_text).strip()
        confidence_raw = str(data["conf"][index]).strip()

        try:
            confidence = float(confidence_raw)
        except ValueError:
            confidence = -1.0

        if not text or confidence < min_confidence:
            continue

        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        grouped.setdefault(key, []).append(
            {
                "text": text,
                "conf": confidence,
                "left": int(data["left"][index]),
                "top": int(data["top"][index]),
                "width": int(data["width"][index]),
                "height": int(data["height"][index]),
            }
        )

    lines: list[OCRLine] = []
    for words in grouped.values():
        ordered_words = sorted(words, key=lambda item: int(item["left"]))
        line_text = " ".join(str(item["text"]) for item in ordered_words)
        left = min(int(item["left"]) for item in ordered_words)
        top = min(int(item["top"]) for item in ordered_words)
        right = max(int(item["left"]) + int(item["width"]) for item in ordered_words)
        bottom = max(int(item["top"]) + int(item["height"]) for item in ordered_words)
        average_conf = sum(float(item["conf"]) for item in ordered_words) / len(ordered_words)

        lines.append(
            OCRLine(
                text=line_text,
                confidence=round(average_conf, 1),
                left=left,
                top=top,
                width=right - left,
                height=bottom - top,
            )
        )

    return sorted(lines, key=lambda item: (item.top, item.left))


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

        try:
            translated_text = translator.translate(line.text)
            error_message = None
            status = "success"
        except Exception as exc:  # pragma: no cover - network/runtime dependency
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


def draw_preview(image: Image.Image, results: Sequence[TranslationResult]) -> Image.Image:
    """Draw OCR boxes and translated labels on a preview image."""

    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)

    for result in results:
        left = result.left
        top = result.top
        width = result.width
        height = result.height
        translated = result.translated_text or "번역 실패"
        font_size = max(18, min(32, height))
        font = load_font(font_size)

        draw.rectangle((left, top, left + width, top + height), outline="red", width=2)

        text_bbox = draw.textbbox((0, 0), translated, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_top = max(0, top - text_height - 10)
        draw.rectangle(
            (left, label_top, left + text_width + 12, label_top + text_height + 8),
            fill="black",
        )
        draw.text((left + 6, label_top + 4), translated, fill="white", font=font)

    return canvas


def build_result_dataframe(results: Sequence[TranslationResult]) -> pd.DataFrame:
    """Convert translation results into a dataframe for display and export."""

    dataframe = pd.DataFrame(
        [
            {
                "파일명": result.file_name,
                "원문": result.source_text,
                "번역문": result.translated_text or "",
                "신뢰도": result.confidence,
                "상태": result.status,
                "오류": result.error_message or "",
            }
            for result in results
        ]
    )
    return dataframe


def process_uploaded_image(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
    min_confidence: float,
) -> tuple[Image.Image, Image.Image, list[TranslationResult]] | None:
    """Process one uploaded image and return source, preview, and translation results."""

    try:
        image = Image.open(uploaded_file)
        image.load()
    except UnidentifiedImageError:
        st.error(f"'{uploaded_file.name}' 파일을 이미지로 읽을 수 없습니다.")
        return None

    lines = extract_english_lines(image, min_confidence=min_confidence)
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
            value=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        )

    return float(min_confidence), tesseract_path.strip()


def render_results(
    uploaded_file_name: str,
    image: Image.Image,
    preview: Image.Image,
    expanded: bool,
) -> None:
    """Render original and preview images for a processed upload."""

    with st.expander(f"이미지 미리보기: {uploaded_file_name}", expanded=expanded):
        left_col, right_col = st.columns(2)
        with left_col:
            st.subheader("원본 이미지")
            st.image(image, width="stretch")
        with right_col:
            st.subheader("번역 미리보기")
            st.image(preview, width="stretch")


def main() -> None:
    """Run the Streamlit OCR translation application."""

    st.set_page_config(page_title="이미지 영어-한국어 번역기", layout="wide")
    st.title("이미지 영어-한국어 번역기")
    st.write(
        "여러 이미지를 업로드하면 영어 텍스트를 OCR로 추출하고 한국어로 번역합니다."
    )

    min_confidence, tesseract_path = configure_sidebar()
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

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

    csv_bytes = dataframe.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="통합 CSV 다운로드",
        data=csv_bytes,
        file_name="image_translation_ko.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
