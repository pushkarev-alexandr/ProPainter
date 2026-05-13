from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


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
