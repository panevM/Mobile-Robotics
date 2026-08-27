"""Kernel-safe display helpers for notebooks."""

from __future__ import annotations

import math

import numpy as np
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont


CHANNEL_COLORS = {
    "free": (210, 210, 210),
    "wall": (70, 120, 255),
    "unknown": (160, 80, 220),
    "frontier": (255, 190, 40),
    "known_density": (120, 220, 255),
    "free_density": (200, 255, 200),
    "wall_density": (90, 140, 255),
    "frontier_density": (255, 210, 70),
    "agent": (255, 80, 80),
}


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    """Map one 2D channel to an 8-bit grayscale image."""
    array = np.asarray(channel, dtype=np.float32)

    if array.ndim != 2:
        raise ValueError("channel must be 2D")

    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)

    minimum = float(finite.min())
    maximum = float(finite.max())

    if maximum - minimum < 1e-6:
        fill = 255 if maximum > 0 else 0
        return np.full(array.shape, fill, dtype=np.uint8)

    normalized = np.clip((array - minimum) / (maximum - minimum), 0.0, 1.0)
    return (255.0 * normalized).astype(np.uint8)


def _colorize_channel(channel: np.ndarray, name: str) -> Image.Image:
    """Convert one channel to a colored RGB tile."""
    grayscale = _normalize_channel(channel).astype(np.float32) / 255.0
    color = np.asarray(CHANNEL_COLORS.get(name, (235, 235, 235)), dtype=np.float32)
    colored = np.clip(grayscale[..., None] * color[None, None, :], 0.0, 255.0)
    return Image.fromarray(colored.astype(np.uint8), mode="RGB")


def display_channel_grid(
    channels: np.ndarray,
    names: list[str] | tuple[str, ...],
    *,
    columns: int,
    scale: int = 18,
    padding: int = 6,
    title: str | None = None,
) -> None:
    """Display a tensor channel grid without using Matplotlib."""
    channel_arrays = [np.asarray(channel) for channel in channels]
    if not channel_arrays:
        print("No channels to display.")
        return

    font = ImageFont.load_default()
    tile_height, tile_width = channel_arrays[0].shape
    rows = math.ceil(len(channel_arrays) / columns)
    title_height = 18 if title else 0
    label_height = 18
    tile_pixel_width = tile_width * scale
    tile_pixel_height = tile_height * scale
    canvas = Image.new(
        "RGB",
        (
            columns * (tile_pixel_width + padding) + padding,
            title_height
            + rows * (tile_pixel_height + label_height + padding)
            + padding,
        ),
        color=(18, 18, 18),
    )
    draw = ImageDraw.Draw(canvas)

    if title:
        draw.text((padding, padding // 2), title, fill=(235, 235, 235), font=font)

    for index, channel in enumerate(channel_arrays):
        image = _colorize_channel(channel, names[index])
        image = image.resize(
            (tile_pixel_width, tile_pixel_height),
            resample=Image.Resampling.NEAREST,
        )
        x = padding + (index % columns) * (tile_pixel_width + padding)
        y = (
            padding
            + title_height
            + (index // columns) * (tile_pixel_height + label_height + padding)
        )
        canvas.paste(image, (x, y))
        label = f"{index}: {names[index]}"
        draw.text(
            (x, y + tile_pixel_height + 2),
            label,
            fill=(235, 235, 235),
            font=font,
        )
    display(canvas)
