from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROPAINTER_ROOT = Path(__file__).resolve().parents[1]
if str(PROPAINTER_ROOT) not in sys.path:
    sys.path.insert(0, str(PROPAINTER_ROOT))

from utils.propainter_crop_utils import fit_crop_size_to_budget, smooth_center_trajectory


def test_smooth_center_trajectory_sigma_zero_returns_input() -> None:
    centers = [(10.0, 20.0), (12.5, 22.5), (15.0, 25.0)]

    assert smooth_center_trajectory(centers, sigma=0.0) == centers


def test_smooth_center_trajectory_preserves_length() -> None:
    centers = [(float(idx), float(idx * 2)) for idx in range(12)]

    assert len(smooth_center_trajectory(centers, sigma=1.5)) == len(centers)


def test_smooth_center_trajectory_reduces_center_noise() -> None:
    base_x = 100.0
    centers = [(base_x + (8.0 if idx % 2 == 0 else -8.0), 50.0) for idx in range(30)]

    smoothed = smooth_center_trajectory(centers, sigma=1.5)

    raw_error = np.mean([abs(x - base_x) for x, _ in centers])
    smoothed_error = np.mean([abs(x - base_x) for x, _ in smoothed])
    assert smoothed_error < raw_error


def test_fit_crop_size_scales_by_long_side_to_target() -> None:
    # Padded union ~219x460 on 1920x1080 → long side 460 → ~243x512
    crop_w, crop_h = fit_crop_size_to_budget(
        width=219.0,
        height=460.0,
        frame_width=1920,
        frame_height=1080,
        target_long_side=512,
    )
    assert crop_h == 512
    assert 240 <= crop_w <= 248


def test_fit_crop_size_does_not_fill_full_frame_height_for_narrow_union() -> None:
    crop_w, crop_h = fit_crop_size_to_budget(
        width=219.0,
        height=460.0,
        frame_width=1920,
        frame_height=1080,
        target_long_side=512,
    )
    assert crop_h < 1080
