#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
37_background_correction_features.py

Background-based colour corrections that need no reference strips.

Every correction uses only the pad and background of the same photo. All are
computed from the mean sRGB values of the pad and background ROIs
(captures_final_features.csv; images were ICC-converted to sRGB), so the
comparison is like-for-like.

Corrections (feature = Lab of the corrected pad colour, D65 white)
------------------------------------------------------------------
pad          no correction
subtract     Lab(pad) - Lab(background)               (current "Contrast")
wb_linear    linear-RGB white balance (von Kries in camera RGB):
             pad_lin * (grey / bg_lin), grey = neutral with the mean
             background luminance
bradford     Bradford chromatic adaptation in XYZ from the background white
             to a D65 white with the mean background luminance
ratio_gamma  gamma-encoded RGB ratio pad / bg (close to Nelis et al. 2019),
             rescaled by the mean background grey level

The output table has the same id columns as the factorial table so that
35_new_phone_offset_calibration.R can read it with --region <name>.

Outputs
-------
outputs/background_correction/
    corrected_features.csv
    variance_shares.csv     balanced decomposition of each corrected feature
    notes.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = PROJECT_ROOT / "outputs" / "background_correction"
CORRECTIONS = ["pad", "subtract", "wb_linear", "bradford", "ratio_gamma"]

D65 = np.array([0.95047, 1.00000, 1.08883])
SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
BRADFORD = np.array([
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
])


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    c = np.clip(rgb / 255.0, 0.0, 1.0)
    return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)


def linear_to_xyz(linear: np.ndarray) -> np.ndarray:
    return linear @ SRGB_TO_XYZ.T


def xyz_to_lab(xyz: np.ndarray, white: np.ndarray = D65) -> np.ndarray:
    t = xyz / white
    delta = 6 / 29
    f = np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4 / 29)
    return np.column_stack([
        116 * f[:, 1] - 16,
        500 * (f[:, 0] - f[:, 1]),
        200 * (f[:, 1] - f[:, 2]),
    ])


def gamma_to_lab(rgb_gamma: np.ndarray) -> np.ndarray:
    return xyz_to_lab(linear_to_xyz(srgb_to_linear(rgb_gamma)))


def balanced_shares(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Standardized trace shares of pH, phone, light and interactions."""
    values = frame[columns].to_numpy(float)
    values = (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)
    data = pd.DataFrame(values, columns=columns)
    data["ph"] = frame["ph_nominal"].to_numpy()
    data["phone"] = frame["phone"].to_numpy()
    data["light"] = frame["light"].to_numpy()

    grand = data[columns].mean()
    total = float(((data[columns] - grand) ** 2).to_numpy().sum())
    effects = {}
    main = {}
    for factor in ["ph", "phone", "light"]:
        means = data.groupby(factor)[columns].transform("mean") - grand
        main[factor] = means
        effects[factor] = float((means ** 2).to_numpy().sum())
    for a, b in [("ph", "phone"), ("ph", "light"), ("phone", "light")]:
        cell = data.groupby([a, b])[columns].transform("mean") - grand
        interaction = cell - main[a] - main[b]
        effects[f"{a}_x_{b}"] = float((interaction ** 2).to_numpy().sum())
    effects["residual"] = total - sum(effects.values())
    return {f"share_{key}": value / total for key, value in effects.items()}


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(PROJECT_ROOT / "data/raw/captures/captures_final_features.csv",
                           encoding="utf-8-sig")
    table = pd.read_csv(PROJECT_ROOT / "outputs/factorial_analysis/analysis_factorial_table.csv",
                        encoding="utf-8-sig")
    rgb_columns = [f"{c}_{roi}_mean" for roi in ["pad", "bg"] for c in "RGB"]
    merged = table[["image_id", "strip_id", "phone", "light", "ph_nominal"]].merge(
        features[["capture_id"] + rgb_columns], left_on="image_id", right_on="capture_id",
        validate="one_to_one")

    pad_rgb = merged[["R_pad_mean", "G_pad_mean", "B_pad_mean"]].to_numpy(float)
    bg_rgb = merged[["R_bg_mean", "G_bg_mean", "B_bg_mean"]].to_numpy(float)
    pad_lin = srgb_to_linear(pad_rgb)
    bg_lin = srgb_to_linear(bg_rgb)
    pad_xyz = linear_to_xyz(pad_lin)
    bg_xyz = linear_to_xyz(bg_lin)

    lab = {}
    lab["pad"] = xyz_to_lab(pad_xyz)
    lab["subtract"] = lab["pad"] - xyz_to_lab(bg_xyz)

    # Linear-RGB white balance: background -> neutral grey of mean luminance.
    mean_bg_y = float(bg_xyz[:, 1].mean())
    grey_lin = np.full(3, mean_bg_y)            # neutral in linear sRGB has R = G = B = Y
    lab["wb_linear"] = xyz_to_lab(linear_to_xyz(pad_lin * (grey_lin / bg_lin)))

    # Bradford adaptation: background white -> D65 white scaled to mean background Y.
    target_white = D65 * mean_bg_y
    inverse = np.linalg.inv(BRADFORD)
    adapted = np.empty_like(pad_xyz)
    for i in range(len(pad_xyz)):
        gain = (BRADFORD @ target_white) / (BRADFORD @ bg_xyz[i])
        adapted[i] = inverse @ (gain * (BRADFORD @ pad_xyz[i]))
    lab["bradford"] = xyz_to_lab(adapted)

    # Gamma-encoded RGB ratio, rescaled by the mean background grey level.
    grey_gamma = float(bg_rgb.mean())
    lab["ratio_gamma"] = gamma_to_lab(np.clip(pad_rgb / bg_rgb * grey_gamma, 0, 255))

    output = merged[["image_id", "strip_id", "phone", "light", "ph_nominal"]].copy()
    share_rows = []
    for name in CORRECTIONS:
        for index, channel in enumerate(["L", "a", "b"]):
            output[f"{name}_{channel}"] = lab[name][:, index]
        shares = balanced_shares(output, [f"{name}_{c}" for c in ["L", "a", "b"]])
        share_rows.append({"correction": name, **shares})

    output.to_csv(TARGET / "corrected_features.csv", index=False, encoding="utf-8-sig")
    shares = pd.DataFrame(share_rows)
    shares.to_csv(TARGET / "variance_shares.csv", index=False, encoding="utf-8-sig")

    check = np.abs(output["subtract_L"].to_numpy() - table["contrast_L"].to_numpy()).max()
    clipped = int((merged[["R_pad_mean", "G_pad_mean", "B_pad_mean"]] < 1).any(axis=1).sum())
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        notes = [
            "Background-based corrections (script 37)",
            "=" * 80,
            f"images: {len(output)} | pad channels below 1 (possible clipping): {clipped}",
            f"max |subtract_L - table contrast_L| (mean-RGB vs saved Lab): {check:.2f}",
            "",
            "Standardized variance shares of each corrected feature",
            shares.to_string(index=False),
        ]
    (TARGET / "notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
