#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
26_plot_gam_curves.py

Figures for the GAM outputs of 26_gam_phone_specific_curves.R.

Inputs
------
outputs/gam_curves/
    phone_curves.csv
    difference_curves.csv
    variance_components.csv
    fitted_residuals.csv
outputs/factorial_analysis/analysis_factorial_table.csv  (strip means)

Outputs
-------
outputs/gam_curves/figures/
    phone_curves_<region>.png
    shape_differences_<region>.png
    total_differences_<region>.png
    k_sensitivity_shape_<region>.png
    strip_variance_reduction.png
    residual_diagnostics_<region>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGIONS = ["contrast", "pad", "bg"]
CHANNELS = ["L", "a", "b"]

REGION_LABELS = {
    "pad": "Pad",
    "bg": "Background",
    "contrast": "Contrast (Pad - Background)",
}
CHANNEL_LABELS = {"L": "L*", "a": "a*", "b": "b*"}

# Okabe-Ito, colour-blind safe.
PHONE_COLORS = {
    "galaxy_s24plus": "#0072B2",
    "iphone13pro": "#E69F00",
    "iphone16pro": "#009E73",
}
PHONE_LABELS = {
    "galaxy_s24plus": "Galaxy S24+",
    "iphone13pro": "iPhone 13 Pro",
    "iphone16pro": "iPhone 16 Pro",
}
SPEC_STYLES = {
    ("tp", 6): ("#999999", ":"),
    ("tp", 8): ("#999999", "--"),
    ("tp", 10): ("#000000", "-"),
    ("tp", 12): ("#CC79A7", "--"),
    ("cr", 10): ("#D55E00", "-."),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GAM curve outputs.")
    parser.add_argument("--gam-dir", default="outputs/gam_curves")
    parser.add_argument(
        "--input",
        default="outputs/factorial_analysis/analysis_factorial_table.csv",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--save-pdf", action="store_true")
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def save_figure(figure: plt.Figure, path: Path, dpi: int, save_pdf: bool) -> None:
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def comparison_label(text: str) -> str:
    second, first = text.split(" minus ")
    return f"{PHONE_LABELS.get(second, second)} - {PHONE_LABELS.get(first, first)}"


def plot_phone_curves(
    curves: pd.DataFrame,
    strip_means: pd.DataFrame,
    region: str,
    path: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    for axis, channel in zip(axes, CHANNELS):
        subset = curves[curves["channel"] == channel]

        for phone, color in PHONE_COLORS.items():
            line = subset[subset["phone"] == phone]
            axis.fill_between(
                line["ph"], line["lower"], line["upper"],
                color=color, alpha=0.18, linewidth=0,
            )
            axis.plot(
                line["ph"], line["fit"],
                color=color, linewidth=1.8, label=PHONE_LABELS[phone],
            )
            points = strip_means[strip_means["phone"] == phone]
            axis.scatter(
                points["ph_nominal"], points[f"{region}_{channel}"],
                color=color, s=16, zorder=3, edgecolor="white", linewidth=0.4,
            )

        axis.set_xlabel("pH")
        axis.set_ylabel(CHANNEL_LABELS[channel])
        axis.set_xticks(range(1, 14))
        axis.grid(alpha=0.25)

    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"{REGION_LABELS[region]}: phone-specific GAM curves "
        "(M2, tp k=10; points = strip means over 5 lights; band = pointwise 95%)",
        fontsize=10,
    )
    figure.tight_layout()
    save_figure(figure, path, dpi, save_pdf)


def plot_differences(
    differences: pd.DataFrame,
    component: str,
    region: str,
    path: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    comparisons = sorted(differences["comparison"].unique())
    figure, axes = plt.subplots(
        len(CHANNELS), len(comparisons),
        figsize=(4.2 * len(comparisons), 3.0 * len(CHANNELS)),
        sharex=True,
    )

    for row, channel in enumerate(CHANNELS):
        for column, comparison in enumerate(comparisons):
            axis = axes[row, column]
            line = differences[
                (differences["channel"] == channel)
                & (differences["comparison"] == comparison)
                & (differences["component"] == component)
            ]
            axis.fill_between(
                line["ph"], line["simultaneous_lower"], line["simultaneous_upper"],
                color="#56B4E9", alpha=0.25, linewidth=0, label="simultaneous 95%",
            )
            axis.fill_between(
                line["ph"], line["lower"], line["upper"],
                color="#0072B2", alpha=0.30, linewidth=0, label="pointwise 95%",
            )
            axis.plot(line["ph"], line["fit"], color="#000000", linewidth=1.5)
            axis.axhline(0.0, color="#D55E00", linewidth=0.9, linestyle="--")

            excluded = (line["simultaneous_lower"] > 0) | (line["simultaneous_upper"] < 0)
            if excluded.any():
                axis.scatter(
                    line.loc[excluded, "ph"],
                    np.full(int(excluded.sum()), axis.get_ylim()[0]),
                    marker="|", color="#D55E00", s=30, clip_on=False,
                )

            if row == 0:
                axis.set_title(comparison_label(comparison), fontsize=9)
            if column == 0:
                axis.set_ylabel(f"Δ {CHANNEL_LABELS[channel]}")
            if row == len(CHANNELS) - 1:
                axis.set_xlabel("pH")
            axis.set_xticks(range(1, 14))
            axis.grid(alpha=0.25)

    axes[0, 0].legend(frameon=False, fontsize=7, loc="best")
    what = (
        "pH-dependent part only (phone offset removed)"
        if component == "shape"
        else "total difference (offset + pH-dependent part)"
    )
    figure.suptitle(
        f"{REGION_LABELS[region]}: pairwise phone differences, {what}\n"
        "orange ticks = pH where the simultaneous 95% band excludes 0",
        fontsize=10,
    )
    figure.tight_layout()
    save_figure(figure, path, dpi, save_pdf)


def plot_k_sensitivity(
    differences: pd.DataFrame,
    region: str,
    path: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    comparisons = sorted(differences["comparison"].unique())
    figure, axes = plt.subplots(
        len(CHANNELS), len(comparisons),
        figsize=(4.2 * len(comparisons), 3.0 * len(CHANNELS)),
        sharex=True,
    )

    for row, channel in enumerate(CHANNELS):
        for column, comparison in enumerate(comparisons):
            axis = axes[row, column]
            subset = differences[
                (differences["channel"] == channel)
                & (differences["comparison"] == comparison)
                & (differences["component"] == "shape")
            ]

            for (basis, k), (color, style) in SPEC_STYLES.items():
                line = subset[(subset["basis"] == basis) & (subset["k"] == k)]
                if line.empty:
                    continue
                axis.plot(
                    line["ph"], line["fit"],
                    color=color, linestyle=style, linewidth=1.4,
                    label=f"{basis} k={k}",
                )

            axis.axhline(0.0, color="#888888", linewidth=0.8)
            if row == 0:
                axis.set_title(comparison_label(comparison), fontsize=9)
            if column == 0:
                axis.set_ylabel(f"Δ {CHANNEL_LABELS[channel]} (shape)")
            if row == len(CHANNELS) - 1:
                axis.set_xlabel("pH")
            axis.set_xticks(range(1, 14))
            axis.grid(alpha=0.25)

    axes[0, 0].legend(frameon=False, fontsize=7)
    figure.suptitle(
        f"{REGION_LABELS[region]}: basis / k sensitivity of the pH-dependent "
        "phone differences",
        fontsize=10,
    )
    figure.tight_layout()
    save_figure(figure, path, dpi, save_pdf)


def plot_variance_reduction(
    variance: pd.DataFrame,
    path: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 4.0))
    labels = []
    positions = []

    for index, (region, channel) in enumerate(
        (region, channel) for region in REGIONS for channel in CHANNELS
    ):
        subset = variance[
            (variance["region"] == region) & (variance["channel"] == channel)
        ]
        primary = subset[subset["primary_spec"]]
        x = index + (index // 3) * 0.6
        positions.append(x)
        labels.append(f"{REGION_LABELS[region].split(' ')[0]}\n{CHANNEL_LABELS[channel]}")

        axis.bar(
            x,
            100 * float(primary["strip_var_reduction_M1_to_M2"].iloc[0]),
            color="#0072B2", width=0.7,
        )
        others = subset[~subset["primary_spec"]]
        axis.scatter(
            np.full(len(others), x),
            100 * others["strip_var_reduction_M1_to_M2"],
            color="#000000", s=12, zorder=3,
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylabel("Strip variance absorbed by\nphone-specific curves (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.set_title(
        "Reduction in strip random-effect variance, M1 -> M2 "
        "(bar = tp k=10; dots = k/basis sensitivity)",
        fontsize=10,
    )
    figure.tight_layout()
    save_figure(figure, path, dpi, save_pdf)


def plot_residual_diagnostics(
    fitted: pd.DataFrame,
    region: str,
    path: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 7.0))
    normal = NormalDist()

    for column, channel in enumerate(CHANNELS):
        subset = fitted[fitted["channel"] == channel]
        residual = subset["residual"].to_numpy()

        axis = axes[0, column]
        for phone, color in PHONE_COLORS.items():
            mask = subset["phone"] == phone
            axis.scatter(
                subset.loc[mask, "fitted"], subset.loc[mask, "residual"],
                color=color, s=12, alpha=0.8, label=PHONE_LABELS[phone],
            )
        axis.axhline(0.0, color="#888888", linewidth=0.8)
        axis.set_xlabel(f"fitted {CHANNEL_LABELS[channel]}")
        axis.set_ylabel("residual")
        axis.grid(alpha=0.25)

        axis = axes[1, column]
        ordered = np.sort(residual)
        n = len(ordered)
        theoretical = np.array(
            [normal.inv_cdf((i + 0.5) / n) for i in range(n)]
        )
        axis.scatter(theoretical, ordered, s=10, color="#0072B2")
        slope = np.std(ordered, ddof=1)
        axis.plot(theoretical, theoretical * slope + np.mean(ordered),
                  color="#D55E00", linewidth=1.0)
        axis.set_xlabel("normal quantile")
        axis.set_ylabel(f"residual {CHANNEL_LABELS[channel]}")
        axis.grid(alpha=0.25)

    axes[0, 0].legend(frameon=False, fontsize=7)
    figure.suptitle(
        f"{REGION_LABELS[region]}: M2 residual diagnostics (tp k=10)",
        fontsize=10,
    )
    figure.tight_layout()
    save_figure(figure, path, dpi, save_pdf)


def main() -> None:
    args = parse_args()
    gam_dir = resolve(args.gam_dir)
    figure_dir = gam_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    curves = pd.read_csv(gam_dir / "phone_curves.csv")
    differences = pd.read_csv(gam_dir / "difference_curves.csv")
    variance = pd.read_csv(gam_dir / "variance_components.csv")
    fitted = pd.read_csv(gam_dir / "fitted_residuals.csv")
    table = pd.read_csv(resolve(args.input), encoding="utf-8-sig")

    value_columns = [f"{r}_{c}" for r in REGIONS for c in CHANNELS]
    strip_means = (
        table.groupby(["phone", "ph_nominal"], as_index=False)[value_columns].mean()
    )

    for frame in (curves, differences, variance, fitted):
        frame["primary_spec"] = frame["primary_spec"].astype(str).str.upper() == "TRUE"

    written = []

    for region in REGIONS:
        region_curves = curves[(curves["region"] == region) & curves["primary_spec"]]
        region_differences = differences[differences["region"] == region]
        primary_differences = region_differences[region_differences["primary_spec"]]

        targets = {
            f"phone_curves_{region}.png": lambda p: plot_phone_curves(
                region_curves, strip_means, region, p, args.dpi, args.save_pdf),
            f"shape_differences_{region}.png": lambda p: plot_differences(
                primary_differences, "shape", region, p, args.dpi, args.save_pdf),
            f"total_differences_{region}.png": lambda p: plot_differences(
                primary_differences, "total", region, p, args.dpi, args.save_pdf),
            f"k_sensitivity_shape_{region}.png": lambda p: plot_k_sensitivity(
                region_differences, region, p, args.dpi, args.save_pdf),
            f"residual_diagnostics_{region}.png": lambda p: plot_residual_diagnostics(
                fitted[fitted["region"] == region], region, p, args.dpi, args.save_pdf),
        }

        for name, draw in targets.items():
            draw(figure_dir / name)
            written.append(name)

    plot_variance_reduction(
        variance, figure_dir / "strip_variance_reduction.png", args.dpi, args.save_pdf
    )
    written.append("strip_variance_reduction.png")

    print(f"Wrote {len(written)} figures to {figure_dir}")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
