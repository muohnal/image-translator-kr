from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pandas as pd
import pytesseract
import streamlit as st
from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw, ImageFont


MIN_CONFIDENCE = 40.0
FONT_CANDIDATES = (
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/AppleGothic.ttf",
    "C:/Windows/Fonts/NanumGothic.ttf",
)


@dataclass
class OCRLine:
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in FONT_CANDIDATES:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def extract_english_lines(image: Image.Image, min_confidence: float) -> List[OCRLine]:
    data = pytesseract.image_to_data(
        image,
        lang="eng",
        output_type=pytesseract.Output.DICT,
    )

    grouped: dict[tuple[int, int, int], list[dict[str, float | int | str]]] = {}
    for i, raw_text in enumerate(data["text"]):
        text = str(raw_text).strip()
        confidence_raw = str(data["conf"][i]).strip()
        try:
            confidence = float(confidence_raw)
        except ValueError:
            confidence = -1.0

        if not text or confidence < min_confidence:
            continue

        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        grouped.setdefault(key, []).append(
            {
                "text": text,
                "conf": confidence,
                "left": int(data["left"][i]),
                "top": int(data["top"][i]),
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
            }
        )

    lines: List[OCRLine] = []
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


def translate_lines(lines: Sequence[OCRLine]) -> list[dict[str, str | float | int]]:
    translator = GoogleTranslator(source="en", target="ko")
    rows: list[dict[str, str | float | int]] = []

    for line in lines:
        try:
            translated = translator.translate(line.text)
        except Exception as exc:  # pragma: no cover - network/runtime dependency
            translated = f"Translation failed: {exc}"

        rows.append(
            {
                "source_text": line.text,
                "translated_text": translated,
                "confidence": line.confidence,
                "left": line.left,
                "top": line.top,
                "width": line.width,
                "height": line.height,
            }
        )

    return rows


def draw_preview(image: Image.Image, rows: Sequence[dict[str, str | float | int]]) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)

    for row in rows:
        left = int(row["left"])
        top = int(row["top"])
        width = int(row["width"])
        height = int(row["height"])
        translated = str(row["translated_text"])
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


st.set_page_config(page_title="이미지 영어-한국어 번역기", layout="wide")
st.title("이미지 영어-한국어 번역기")
st.write("여러 이미지를 업로드하면 영어 텍스트를 OCR로 추출하고 한국어로 번역합니다.")

with st.sidebar:
    st.header("설정")
    min_confidence = st.slider("최소 OCR 신뢰도", min_value=0, max_value=100, value=int(MIN_CONFIDENCE))
    tesseract_path = st.text_input(
        "Tesseract 실행 파일 경로(선택)",
        value=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    )
    if tesseract_path.strip():
        pytesseract.pytesseract.tesseract_cmd = tesseract_path.strip()

uploaded_files = st.file_uploader(
    "이미지를 업로드하세요",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("번역할 이미지를 하나 이상 업로드하세요.")
    st.stop()

all_rows: list[dict[str, str | float | int]] = []
processed_count = 0

try:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        lines = extract_english_lines(image, min_confidence=float(min_confidence))

        if not lines:
            st.warning(f"'{uploaded_file.name}' 파일에서 조건에 맞는 영어 텍스트를 찾지 못했습니다.")
            continue

        rows = translate_lines(lines)
        for row in rows:
            row["file_name"] = uploaded_file.name

        preview = draw_preview(image, rows)
        processed_count += 1
        all_rows.extend(rows)

        with st.expander(f"이미지 미리보기: {uploaded_file.name}", expanded=processed_count == 1):
            left_col, right_col = st.columns(2)
            with left_col:
                st.subheader("원본 이미지")
                st.image(image, width="stretch")
            with right_col:
                st.subheader("번역 미리보기")
                st.image(preview, width="stretch")
except pytesseract.TesseractNotFoundError:
    st.error("Tesseract OCR이 설치되어 있지 않거나 실행 파일 경로가 올바르지 않습니다. README를 확인하세요.")
    st.stop()

if not all_rows:
    st.warning("업로드한 모든 이미지에서 번역할 영어 텍스트를 찾지 못했습니다.")
    st.stop()

st.success(f"{processed_count}개 이미지에서 번역 결과를 추출했습니다.")

st.subheader("통합 번역 결과")
df = pd.DataFrame(all_rows)[["file_name", "source_text", "translated_text", "confidence"]]
df = df.rename(
    columns={
        "file_name": "파일명",
        "source_text": "원문",
        "translated_text": "번역문",
        "confidence": "신뢰도",
    }
)
st.dataframe(df, width="stretch", hide_index=True)

csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="통합 CSV 다운로드",
    data=csv_bytes,
    file_name="image_translation_ko.csv",
    mime="text/csv",
)
