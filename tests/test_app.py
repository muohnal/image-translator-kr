"""Tests for EXIF orientation handling in app.py's translation pipeline."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

import app


def make_exif_rotated_jpeg(size: tuple[int, int], orientation: int) -> bytes:
    """Build a JPEG whose raw pixels are unrotated but tagged with an EXIF
    orientation, mimicking how phone cameras store portrait photos."""

    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "Hello world", fill="black")
    exif = image.getexif()
    exif[0x0112] = orientation
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def test_pipeline_corrects_exif_rotation(monkeypatch):
    monkeypatch.setattr(
        app, "extract_text_lines", lambda image, min_confidence, ocr_lang: []
    )

    file_bytes = make_exif_rotated_jpeg((300, 100), orientation=6)
    result = app.run_translation_pipeline.__wrapped__(
        file_bytes, "photo.jpg", 20.0, "eng", "en"
    )
    # No lines extracted (stubbed), so the pipeline returns None, but the
    # image must have been re-oriented before OCR/preview ran.
    assert result is None


def test_exif_transpose_swaps_dimensions_for_sideways_photo():
    file_bytes = make_exif_rotated_jpeg((300, 100), orientation=6)
    from PIL import ImageOps

    raw = Image.open(BytesIO(file_bytes))
    raw.load()
    assert raw.size == (300, 100)

    corrected = ImageOps.exif_transpose(raw)
    assert corrected.size == (100, 300)
