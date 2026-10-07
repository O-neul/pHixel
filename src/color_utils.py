from __future__ import annotations

import numpy as np


def _prepare_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.shape[-1] != 3:
        raise ValueError(f"RGB input must end with 3 channels, got {arr.shape}")
    finite = np.isfinite(arr)
    if not finite.all():
        raise ValueError("RGB input contains NaN or infinite values")
    if arr.size and arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (0..1 or 0..255) to CIE Lab (D65)."""
    arr = _prepare_rgb(rgb)
    linear = np.where(
        arr <= 0.04045,
        arr / 12.92,
        ((arr + 0.055) / 1.055) ** 2.4,
    )

    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = linear @ matrix.T

    # D65 reference white, normalized to Y=1.
    ref = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
    xyz_n = xyz / ref
    delta = 6.0 / 29.0
    f = np.where(
        xyz_n > delta**3,
        np.cbrt(xyz_n),
        xyz_n / (3 * delta**2) + 4.0 / 29.0,
    )

    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)
