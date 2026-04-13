from __future__ import annotations

from io import BytesIO
import re
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
ENGLISH_PATTERN = re.compile(r"[A-Za-z]")
KOREAN_PATTERN = re.compile(r"[가-힣]")


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
    """Extract OCR lines from an image above the confidence threshold."""

    data = pytesseract.image_to_data(
        image,
        lang="eng+kor",
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
    """Draw OCR boxes and translated labels with semi-transparent overlays."""

    canvas = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    canvas_width, canvas_height = canvas.size

    for result in results:
        if result.status != "success" or not result.translated_text:
            continue

        left = result.left
        top = result.top
        width = result.width
        height = result.height
        translated = result.translated_text
        font_size = max(16, min(28, height))
        font = load_font(font_size)
        pad = 6

        box_right = min(canvas_width, left + width)
        box_bottom = min(canvas_height, top + height)
        box_left = max(0, left)
        box_top = max(0, top)

        draw.rectangle(
            (box_left, box_top, box_right, box_bottom),
            outline=(255, 96, 96, 220),
            fill=(255, 80, 80, 36),
            width=2,
        )

        text_bbox = draw.textbbox((0, 0), translated, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        label_left = max(0, left - pad)
        label_top = max(0, top - text_height - (pad * 2) - 6)
        label_right = min(canvas_width, label_left + text_width + (pad * 2))
        label_bottom = min(canvas_height, label_top + text_height + (pad * 2))

        if label_right - label_left < text_width + (pad * 2):
            label_left = max(0, label_right - text_width - (pad * 2))

        draw.rectangle(
            (label_left, label_top, label_right, label_bottom),
            fill=(18, 18, 18, 210),
        )
        draw.text(
            (label_left + pad, label_top + pad),
            translated,
            fill=(255, 255, 255, 255),
            font=font,
        )

    composited = Image.alpha_composite(canvas, overlay)
    return composited.convert("RGB")


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


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Convert an image into PNG bytes for downloading."""

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


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

    csv_bytes = dataframe.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="통합 CSV 다운로드",
        data=csv_bytes,
        file_name="image_translation_ko.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
