from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from translation import TranslationResult


FONT_CANDIDATES = (
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/AppleGothic.ttf",
    "C:/Windows/Fonts/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a readable Korean font for the preview image."""

    for font_path in FONT_CANDIDATES:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Wrap text into multiple lines so it fits within a target width."""

    def split_long_word(word: str) -> list[str]:
        """Break a single word that alone exceeds max_width into chunks."""

        if draw.textlength(word, font=font) <= max_width:
            return [word]

        chunks: list[str] = []
        current_chunk = ""
        for char in word:
            trial_chunk = current_chunk + char
            if draw.textlength(trial_chunk, font=font) <= max_width or not current_chunk:
                current_chunk = trial_chunk
            else:
                chunks.append(current_chunk)
                current_chunk = char
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    words = text.split()
    if not words:
        return [text]

    lines: list[str] = []
    current_line = ""

    for word in words:
        trial_line = f"{current_line} {word}".strip()
        if draw.textlength(trial_line, font=font) <= max_width:
            current_line = trial_line
            continue

        if current_line:
            lines.append(current_line)

        word_chunks = split_long_word(word)
        for chunk in word_chunks[:-1]:
            lines.append(chunk)
        current_line = word_chunks[-1]

    if current_line:
        lines.append(current_line)
    return lines


def fit_translated_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box_width: int,
    box_height: int,
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, list[str], int, int]:
    """Find a font size and wrapped lines for a translation overlay."""

    min_font_size = 12
    max_font_size = max(min_font_size, min(28, box_height - 4))
    max_text_width = max(20, box_width - 12)
    max_text_height = max(20, box_height - 12)

    best_font = load_font(min_font_size)
    best_lines = [text]
    best_width = 0
    best_height = 0

    for font_size in range(max_font_size, min_font_size - 1, -1):
        font = load_font(font_size)
        lines = wrap_text_to_width(draw, text, font, max_text_width)

        line_heights: list[int] = []
        line_widths: list[int] = []
        for line in lines:
            text_box = draw.textbbox((0, 0), line, font=font)
            line_widths.append(text_box[2] - text_box[0])
            line_heights.append(text_box[3] - text_box[1])

        total_height = sum(line_heights) + max(0, len(lines) - 1) * 4
        total_width = max(line_widths, default=0)

        best_font = font
        best_lines = lines
        best_width = total_width
        best_height = total_height

        if total_width <= max_text_width and total_height <= max_text_height:
            return font, lines, total_width, total_height

    return best_font, best_lines, best_width, best_height


def draw_preview(image: Image.Image, results: Sequence[TranslationResult]) -> Image.Image:
    """Replace source text regions with Google Translate-style overlays."""

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

        box_right = min(canvas_width, left + width)
        box_bottom = min(canvas_height, top + height)
        box_left = max(0, left)
        box_top = max(0, top)
        box_width = max(1, box_right - box_left)
        box_height = max(1, box_bottom - box_top)

        font, wrapped_lines, text_width, text_height = fit_translated_text(
            draw=draw,
            text=translated,
            box_width=box_width,
            box_height=box_height,
        )

        padded_text_height = text_height + 12
        expanded_bottom = min(
            canvas_height,
            box_top + max(box_height, padded_text_height),
        )

        draw.rectangle(
            (box_left, box_top, box_right, expanded_bottom),
            fill=(248, 248, 248, 220),
        )

        start_y = box_top + 6

        current_y = start_y
        for line in wrapped_lines:
            line_box = draw.textbbox((0, 0), line, font=font)
            line_width = line_box[2] - line_box[0]
            line_height = line_box[3] - line_box[1]
            line_x = box_left + max(0, (box_width - line_width) / 2)
            draw.text(
                (line_x, current_y),
                line,
                fill=(30, 30, 30, 255),
                font=font,
            )
            current_y += line_height + 4

    composited = Image.alpha_composite(canvas, overlay)
    return composited.convert("RGB")


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Convert an image into PNG bytes for downloading."""

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()
