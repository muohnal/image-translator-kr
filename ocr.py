from __future__ import annotations

import platform
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


MIN_CONFIDENCE = 20.0


def resolve_tesseract_cmd(tesseract_path: str) -> str:
    """Return the Tesseract command path appropriate for the current runtime."""

    normalized_path = tesseract_path.strip()
    is_windows = platform.system().lower().startswith("win")

    if normalized_path:
        candidate = Path(normalized_path)
        expected_name = "tesseract.exe" if is_windows else "tesseract"
        if candidate.is_file() and candidate.name.lower() == expected_name:
            return str(candidate)
        if is_windows and re.match(r"^[A-Za-z]:\\", normalized_path):
            raise ValueError(
                "지정한 Tesseract 경로가 존재하지 않거나 'tesseract.exe' 파일이 아닙니다."
            )

    detected_tesseract = shutil.which("tesseract")
    if detected_tesseract:
        return detected_tesseract

    return "tesseract"


@dataclass
class OCRLine:
    """Represents a single OCR line extracted from an image."""

    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """Improve OCR readability for screenshots and camera images."""

    grayscale_image = ImageOps.grayscale(image)
    denoised_image = grayscale_image.filter(ImageFilter.MedianFilter(size=3))
    normalized_image = ImageOps.autocontrast(denoised_image)
    contrast_enhancer = ImageEnhance.Contrast(normalized_image)
    contrasted_image = contrast_enhancer.enhance(1.5)
    softened_image = contrasted_image.filter(ImageFilter.GaussianBlur(radius=0.6))
    renormalized_image = ImageOps.autocontrast(softened_image)
    return renormalized_image.point(lambda pixel: 255 if pixel > 128 else 0, mode="L")


def extract_english_lines(image: Image.Image, min_confidence: float) -> list[OCRLine]:
    """Extract OCR lines from an image above the confidence threshold."""

    def run_ocr(target_image: Image.Image) -> dict[str, list[str | int | float]]:
        """Run Tesseract OCR with settings tuned for dense mobile text blocks."""

        return pytesseract.image_to_data(
            target_image,
            lang="eng+kor",
            config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )

    data = run_ocr(image)
    scale_factor = 1.0

    detected_heights = [
        int(data["height"][index])
        for index, raw_text in enumerate(data["text"])
        if str(raw_text).strip()
    ]
    if detected_heights:
        average_height = sum(detected_heights) / len(detected_heights)
        if average_height < 30:
            scale_factor = 2.0
            enlarged_image = image.resize(
                (int(image.width * scale_factor), int(image.height * scale_factor)),
                resample=Image.Resampling.LANCZOS,
            )
            data = run_ocr(enlarged_image)

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
                "left": int(round(int(data["left"][index]) / scale_factor)),
                "top": int(round(int(data["top"][index]) / scale_factor)),
                "width": int(round(int(data["width"][index]) / scale_factor)),
                "height": int(round(int(data["height"][index]) / scale_factor)),
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
