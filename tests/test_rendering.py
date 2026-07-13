"""Tests for text wrapping and preview overlay rendering in rendering.py."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rendering
from translation import TranslationResult


def make_result(**overrides) -> TranslationResult:
    defaults = dict(
        file_name="t.png",
        source_text="Hello",
        translated_text="안녕",
        confidence=90.0,
        left=50,
        top=40,
        width=100,
        height=12,
        status="success",
    )
    defaults.update(overrides)
    return TranslationResult(**defaults)


def make_draw() -> ImageDraw.ImageDraw:
    return ImageDraw.Draw(Image.new("RGB", (400, 100)))


def test_wrap_text_splits_on_spaces():
    draw = make_draw()
    font = rendering.load_font(16)
    lines = rendering.wrap_text_to_width(draw, "one two three four five six", font, 60)
    assert len(lines) > 1
    assert " ".join(lines) == "one two three four five six"


def test_wrap_text_breaks_long_word():
    draw = make_draw()
    font = rendering.load_font(16)
    long_word = "x" * 100
    lines = rendering.wrap_text_to_width(draw, long_word, font, 80)
    assert len(lines) > 1
    assert "".join(lines) == long_word


def test_wrap_text_empty_string():
    draw = make_draw()
    font = rendering.load_font(16)
    assert rendering.wrap_text_to_width(draw, "", font, 80) == [""]


def test_fit_translated_text_respects_box():
    draw = make_draw()
    font, lines, width, height = rendering.fit_translated_text(
        draw, "short", box_width=200, box_height=40
    )
    assert width <= 200 - 12
    assert lines == ["short"]


def test_draw_preview_covers_source_text_on_light_background():
    image = Image.new("RGB", (400, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.text((50, 40), "Hello world", fill="black")

    preview = rendering.draw_preview(image, [make_result()])
    pixel = preview.getpixel((55, 45))
    assert sum(pixel[:3]) / 3 > 150, "source text must be covered by a light overlay"


def test_draw_preview_uses_dark_overlay_on_dark_background():
    image = Image.new("RGB", (400, 100), (20, 20, 25))
    draw = ImageDraw.Draw(image)
    draw.text((50, 40), "Hello world", fill="white")

    preview = rendering.draw_preview(image, [make_result()])
    pixel = preview.getpixel((55, 45))
    assert sum(pixel[:3]) / 3 < 100, "overlay must stay dark on dark screenshots"


def test_draw_preview_skips_failed_results():
    image = Image.new("RGB", (400, 100), "white")
    original = image.copy()

    preview = rendering.draw_preview(
        image, [make_result(status="translation_failed", translated_text=None)]
    )
    assert preview.tobytes() == original.convert("RGB").tobytes()


def test_draw_preview_skips_giant_garbage_box():
    image = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.text((50, 40), "Hello", fill="black")
    original = image.copy()

    # OCR garbage: one "line" spanning nearly the whole photo
    giant = make_result(left=0, top=0, width=395, height=195)
    preview = rendering.draw_preview(image, [giant])
    assert preview.tobytes() == original.convert("RGB").tobytes()


def test_image_to_png_bytes_round_trip():
    image = Image.new("RGB", (10, 10), "red")
    data = rendering.image_to_png_bytes(image)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
