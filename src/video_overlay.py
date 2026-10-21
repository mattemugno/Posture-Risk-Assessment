"""Camera text styled to match the application's Arial-based interface."""

import math
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import base_dir


INK = '#152b23'
MUTED = '#3f5449'
ACCENT = '#14553f'
OUTLINE = '#788f82'


@lru_cache(maxsize=32)
def _font(size, bold=False):
    name = 'LiberationSans-Bold.ttf' if bold else 'LiberationSans-Regular.ttf'
    return ImageFont.truetype(
        os.path.join(base_dir, 'static', 'fonts', name), size,
        layout_engine=ImageFont.Layout.BASIC,
    )


def _paint_region(image, x, y, width, height, painter):
    """Convert only the label region, preserving BGR frames and float masks."""
    frame_height, frame_width = image.shape[:2]
    width, height = min(width, frame_width), min(height, frame_height)
    if width <= 0 or height <= 0:
        return
    x = min(max(0, x), frame_width - width)
    y = min(max(0, y), frame_height - height)
    region = image[y:y + height, x:x + width]
    is_mask = np.issubdtype(image.dtype, np.floating)
    pixels = (np.clip(region, 0, 1) * 255).astype(np.uint8) if is_mask else region
    tile = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
    painter(ImageDraw.Draw(tile), width, height)
    rendered = cv2.cvtColor(np.asarray(tile), cv2.COLOR_RGB2BGR)
    region[:] = rendered / 255 if is_mask else rendered


def draw_angle_label(image, angle, position):
    """Place a compact, high-contrast angle label inside the video boundaries."""
    if angle is None:
        return
    size = max(11, round(image.shape[1] * 0.024))
    font = _font(size, bold=True)
    text = f'{angle:.2f}°'
    padding = max(4, size // 3)
    width = math.ceil(font.getlength(text)) + 2 * padding
    height = size + 2 * padding

    def paint(draw, tile_width, tile_height):
        draw.rounded_rectangle((0, 0, tile_width - 1, tile_height - 1),
                               radius=max(4, size // 3), fill=ACCENT,
                               outline='white', width=1)
        draw.text((padding, padding), text, font=font, fill='white', anchor='lt')

    _paint_region(image, int(position[0]) + 10, int(position[1]) - height - 8,
                  width, height, paint)


def draw_angle_summary(image, l_shldr, r_shldr, trunk_incl):
    """Draw the three original measurements on a light, rounded summary card."""
    rows = (
        ('Left shoulder', l_shldr),
        ('Right shoulder', r_shldr),
        ('Trunk', trunk_incl),
    )
    values = ['—' if angle is None else f'{angle:.2f}°' for _, angle in rows]
    size = max(10, round(image.shape[1] * 0.026))
    padding = max(6, round(size * 0.65))
    regular, bold = _font(size), _font(size, bold=True)
    label_width = max(regular.getlength(label) for label, _ in rows)
    value_width = max(bold.getlength(value) for value in values)
    width = math.ceil(label_width + value_width) + 3 * padding
    row_height = size + max(8, size // 2)
    height = 2 * padding + 3 * row_height
    margin = max(6, round(image.shape[1] * 0.02))

    def paint(draw, tile_width, tile_height):
        draw.rounded_rectangle((0, 0, tile_width - 1, tile_height - 1),
                               radius=max(6, size // 2), fill='white',
                               outline=OUTLINE, width=1)
        for index, ((label, _), value) in enumerate(zip(rows, values)):
            y = padding + index * row_height
            draw.text((padding, y), label, font=regular, fill=MUTED, anchor='lt')
            draw.text((tile_width - padding, y), value, font=bold, fill=INK, anchor='rt')

    _paint_region(image, margin, margin, width, height, paint)
