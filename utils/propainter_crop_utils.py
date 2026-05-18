from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter1d

OBJECT_CROP_TARGET_LONG_SIDE = 512
OBJECT_CROP_MAX_PIXELS = 960 * 720


def round_up_to_multiple(value: float | int, multiple: int = 8) -> int:
    return max(multiple, int(math.ceil(float(value) / multiple) * multiple))


def fit_crop_size_to_budget(
    *,
    width: float,
    height: float,
    frame_width: int,
    frame_height: int,
    target_long_side: int = OBJECT_CROP_TARGET_LONG_SIDE,
    max_pixels: int = OBJECT_CROP_MAX_PIXELS,
) -> tuple[int, int]:
    """Scale crop proportionally so max(width, height) equals target_long_side, then clamp."""
    if width <= 0 or height <= 0:
        raise ValueError("Crop size must be positive")

    target_width = float(width)
    target_height = float(height)
    frame_long_side = max(8, max(frame_width, frame_height))
    effective_target = min(target_long_side, frame_long_side)
    long_side = max(target_width, target_height)
    if long_side > 0:
        scale = effective_target / long_side
        target_width *= scale
        target_height *= scale

    if max_pixels > 0 and target_width * target_height > max_pixels:
        scale = math.sqrt(max_pixels / (target_width * target_height))
        target_width *= scale
        target_height *= scale

    target_width = min(target_width, float(frame_width))
    target_height = min(target_height, float(frame_height))
    return round_up_to_multiple(target_width), round_up_to_multiple(target_height)


def smooth_center_trajectory(
    centers: list[tuple[float, float]],
    *,
    sigma: float,
) -> list[tuple[float, float]]:
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0 or len(centers) <= 1:
        return list(centers)

    points = np.asarray(centers, dtype=np.float64)
    smoothed = gaussian_filter1d(points, sigma=sigma, axis=0, mode="nearest")
    return [(float(x), float(y)) for x, y in smoothed]
