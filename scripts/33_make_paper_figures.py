#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
33_make_paper_figures.py

Build the main paper figures in one consistent style from the result files of
scripts 15-32, and collect supplementary figures with paper numbering.

Main figures (outputs/paper_figures/main/)
-------------------------------------------
Fig1_design_roi           design (39 strips x 5 lights) and pad / background ROIs
Fig2_decomposition_asca   effect composition and ASCA pH vs pH x phone directions
Fig3_gam_curves           phone-specific GAM curves (Contrast)
Fig4_nuisance_mechanism   which nuisance moves the pH reading, and by how much
Fig5_calibration          how much calibration a new phone needs
Fig6_validation           validation unit, GAM vs RCS strip holdout, per-strip gains
Fig7_simulation           simulation S1-S3

S18 (pH-dependent phone differences) and S19 (inverse pH error by pH) are
generated here as supplementary figures.

Supplementary figures (outputs/paper_figures/supplementary/) are copies of
existing figures with S-numbering; see figure_index.csv.

Each main figure is saved as PNG (300 dpi) and PDF.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs"

PHONES = ["galaxy_s24plus", "iphone13pro", "iphone16pro"]
PHONE_LABELS = {
    "galaxy_s24plus": "Galaxy S24+",
    "iphone13pro": "iPhone 13 Pro",
    "iphone16pro": "iPhone 16 Pro",
}
PHONE_COLORS = {
    "galaxy_s24plus": "#0072B2",
    "iphone13pro": "#E69F00",
    "iphone16pro": "#009E73",
}
CHANNELS = ["L", "a", "b"]
CHANNEL_LABELS = {"L": "L*", "a": "a*", "b": "b*"}
REGION_LABELS = {"pad": "Pad", "bg": "Background", "contrast": "Contrast"}
MODEL_COLORS = {"M0": "#999999", "M1": "#0072B2", "M2": "#D55E00"}
MODEL_LABELS = {
    "M0": "M0 shared curve",
    "M1": "M1 + phone offset",
    "M2": "M2 phone-specific curves",
}

FULL_WIDTH = 7.2   # inches (about 18 cm)


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.3,
        "savefig.dpi": 300,
    })


def panel(axis: plt.Axes, letter: str, x: float = -0.16, y: float = 1.06) -> None:
    axis.text(x, y, letter, transform=axis.transAxes, fontsize=11,
              fontweight="bold", va="bottom", ha="left")


def save(figure: plt.Figure, directory: Path, stem: str) -> list[str]:
    figure.savefig(directory / f"{stem}.png", bbox_inches="tight")
    figure.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)
    return [f"{stem}.png", f"{stem}.pdf"]


def is_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper() == "TRUE"


# ---------------------------------------------------------------------------
# Figure 1: design and ROI
# ---------------------------------------------------------------------------

def figure1(directory: Path) -> list[str]:
    figure = plt.figure(figsize=(FULL_WIDTH, 3.3))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.08)

    axis = figure.add_subplot(grid[0])
    axis.set_xlim(0, 16.2)
    axis.set_ylim(-1.6, 4.3)
    axis.axis("off")
    for row, phone in enumerate(PHONES):
        y = 2.6 - row * 1.15
        axis.text(2.75, y + 0.35, PHONE_LABELS[phone], ha="right", va="center", fontsize=7)
        for ph in range(1, 14):
            axis.add_patch(Rectangle((ph + 2.0, y), 0.8, 0.7, facecolor=PHONE_COLORS[phone],
                                     alpha=0.25 + 0.04 * ph, edgecolor="#333333", linewidth=0.4))
    for ph in range(1, 14):
        axis.text(ph + 2.4, 3.45, str(ph), ha="center", va="bottom", fontsize=6.5)
    axis.text(9.4, 3.95, "pH", ha="center", fontsize=7.5)
    axis.text(0.0, -0.45, "Each box = one physical strip (39 strips = experimental unit)",
              fontsize=7)
    axis.text(0.0, -0.95, "Each strip photographed under 5 lights (L01-L05) -> 195 images",
              fontsize=7)
    axis.text(0.0, -1.45, "pH, phone, pH x phone: between-strip | light terms: within-strip",
              fontsize=7, color="#555555")
    panel(axis, "A", x=-0.02, y=0.98)

    table = pd.read_csv(PROJECT_ROOT / "data/raw/captures/captures_final_features.csv",
                        encoding="utf-8-sig")
    crops = []
    for phone in PHONES:
        row = table[(table["phone"] == phone) & (table["ph_value"] == 6)
                    & (table["light"] == "L01")].iloc[0]
        image = Image.open(PROJECT_ROOT / "data/raw/captures/images" /
                           row["image_path_declared"]).convert("RGB")
        center_x = (int(row["left25_bg_x1"]) + int(row["left25_pad_x2"])) // 2
        center_y = (int(row["left25_pad_y1"]) + int(row["left25_pad_y2"])) // 2
        half_width = (int(row["left25_pad_x2"]) - int(row["left25_bg_x1"])) // 2 + 120
        half_height = int(half_width * 1.35)
        x1, x2 = center_x - half_width, center_x + half_width
        y1, y2 = center_y - half_height, center_y + half_height
        crop = image.crop((x1, y1, x2, y2))
        draw = ImageDraw.Draw(crop)
        width = max(4, crop.width // 90)
        draw.rectangle([row["left25_pad_x1"] - x1, row["left25_pad_y1"] - y1,
                        row["left25_pad_x2"] - x1, row["left25_pad_y2"] - y1],
                       outline=(213, 94, 0), width=width)
        draw.rectangle([row["left25_bg_x1"] - x1, row["left25_bg_y1"] - y1,
                        row["left25_bg_x2"] - x1, row["left25_bg_y2"] - y1],
                       outline=(0, 114, 178), width=width)
        crops.append(crop.resize((450, 608)))

    inner = grid[1].subgridspec(1, 3, wspace=0.05)
    for index, (phone, crop) in enumerate(zip(PHONES, crops)):
        image_axis = figure.add_subplot(inner[index])
        image_axis.imshow(crop)
        image_axis.set_xticks([])
        image_axis.set_yticks([])
        for spine in image_axis.spines.values():
            spine.set_visible(False)
        image_axis.set_title(PHONE_LABELS[phone], fontsize=7)
        if index == 0:
            panel(image_axis, "B", x=-0.25, y=1.06)
    figure.text(0.56, 0.02,
                "pH 6, light L01. Orange: pad ROI; blue: background ROI "
                "(same size, shifted 2.5 pad widths left).  Contrast = Pad - Background (Lab)",
                fontsize=6.5)
    return save(figure, directory, "Fig1_design_roi")


# ---------------------------------------------------------------------------
# Figure 2: decomposition and ASCA directions
# ---------------------------------------------------------------------------

def figure2(directory: Path) -> list[str]:
    composition = pd.read_csv(OUT / "multivariate_lab/effect_composition.csv", encoding="utf-8-sig")
    composition = composition[composition["scaling"] == "standardized"].set_index("region")
    loadings = pd.read_csv(OUT / "asca_permutation/loading_bootstrap.csv", encoding="utf-8-sig")
    angles = pd.read_csv(OUT / "asca_permutation/direction_angle_bootstrap.csv",
                         encoding="utf-8-sig")
    tests = pd.read_csv(OUT / "asca_permutation/permutation_tests.csv", encoding="utf-8-sig")

    figure, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.6),
                                gridspec_kw={"width_ratios": [1.35, 1, 1], "wspace": 0.6})

    axis = axes[0]
    parts = [
        ("pH_signal_share", "pH", "#0072B2"),
        ("nuisance_main_share", "phone + light (+ phone x light)", "#999999"),
        ("pH_dependent_interaction_share", "pH x phone, pH x light", "#D55E00"),
        ("residual_share", "residual", "#E0E0E0"),
    ]
    regions = ["bg", "pad", "contrast"]
    left = np.zeros(len(regions))
    for column, label, color in parts:
        values = composition.loc[regions, column].to_numpy()
        axis.barh(range(len(regions)), values, left=left, color=color, label=label,
                  edgecolor="white", linewidth=0.5)
        for index, value in enumerate(values):
            if value > 0.06:
                axis.text(left[index] + value / 2, index, f"{100 * value:.0f}%",
                          ha="center", va="center", fontsize=6.5,
                          color="white" if color != "#E0E0E0" else "black")
        left += values
    axis.set_yticks(range(len(regions)))
    axis.set_yticklabels([REGION_LABELS[r] for r in regions])
    axis.set_xlim(0, 1)
    axis.set_xlabel("share of standardized Lab variation")
    axis.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    bg_p = tests[(tests["region"] == "bg") & (tests["scaling"] == "standardized")
                 & (tests["term"] == "pH")].set_index("scheme")["p_value"]
    axis.set_title(f"Background pH effect: strip-level p = {bg_p['restricted']:.2f}\n"
                   f"(image-level permutation p = {bg_p['unrestricted']:.4f})",
                   fontsize=6.5, color="#555555")
    panel(axis, "A", x=-0.3)

    for axis, scaling, letter in [(axes[1], "standardized", "B"), (axes[2], "raw", "C")]:
        x = np.arange(3)
        for offset, term, color, label in [(-0.18, "pH", "#0072B2", "pH"),
                                           (0.18, "pH_x_phone", "#D55E00", "pH x phone")]:
            part = loadings[(loadings["region"] == "contrast") & (loadings["scaling"] == scaling)
                            & (loadings["term"] == term)].set_index("channel").loc[CHANNELS]
            axis.bar(x + offset, part["loading"], width=0.34, color=color, label=label)
            axis.errorbar(x + offset, part["loading"],
                          yerr=[part["loading"] - part["ci_lower_95"],
                                part["ci_upper_95"] - part["loading"]],
                          fmt="none", ecolor="black", capsize=2, linewidth=0.7)
        angle = angles[(angles["region"] == "contrast") & (angles["scaling"] == scaling)].iloc[0]
        axis.axhline(0, color="#888888", linewidth=0.6)
        axis.set_ylim(-0.75, 1.0)
        axis.set_xticks(x)
        axis.set_xticklabels([CHANNEL_LABELS[c] for c in CHANNELS])
        axis.set_title(f"Contrast PC1, {scaling}\nangle {angle['angle_degrees']:.1f}° "
                       f"[{angle['ci_lower_95']:.1f}, {angle['ci_upper_95']:.1f}]", fontsize=7)
        axis.set_ylabel("PC1 loading")
        panel(axis, letter, x=-0.42, y=1.12)
    axes[1].legend(frameon=False, fontsize=6.5, loc="lower left")
    return save(figure, directory, "Fig2_decomposition_asca")


# ---------------------------------------------------------------------------
# Figures 3-4: GAM curves and differences
# ---------------------------------------------------------------------------

def figure3(directory: Path) -> list[str]:
    curves = pd.read_csv(OUT / "gam_curves/phone_curves.csv")
    curves = curves[is_true(curves["primary_spec"]) & (curves["region"] == "contrast")]
    table = pd.read_csv(OUT / "factorial_analysis/analysis_factorial_table.csv",
                        encoding="utf-8-sig")
    means = table.groupby(["phone", "ph_nominal"], as_index=False)[
        [f"contrast_{c}" for c in CHANNELS]].mean()
    variance = pd.read_csv(OUT / "gam_curves/variance_components.csv")
    variance = variance[is_true(variance["primary_spec"]) & (variance["region"] == "contrast")]
    variance = variance.set_index("channel")

    figure, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.4), gridspec_kw={"wspace": 0.35})
    for axis, channel, letter in zip(axes, CHANNELS, "ABC"):
        for phone in PHONES:
            line = curves[(curves["channel"] == channel) & (curves["phone"] == phone)]
            axis.fill_between(line["ph"], line["lower"], line["upper"],
                              color=PHONE_COLORS[phone], alpha=0.18, linewidth=0)
            axis.plot(line["ph"], line["fit"], color=PHONE_COLORS[phone],
                      label=PHONE_LABELS[phone])
            points = means[means["phone"] == phone]
            axis.scatter(points["ph_nominal"], points[f"contrast_{channel}"], s=8,
                         color=PHONE_COLORS[phone], edgecolor="white", linewidth=0.3, zorder=3)
        reduction = 100 * variance.loc[channel, "strip_var_reduction_M1_to_M2"]
        axis.set_title(f"Contrast {CHANNEL_LABELS[channel]}\n"
                       f"strip variance absorbed by M2: {reduction:.0f}%", fontsize=7.5)
        axis.set_xlabel("pH")
        axis.set_ylabel(CHANNEL_LABELS[channel])
        axis.set_xticks(range(1, 14, 2))
        panel(axis, letter, x=-0.28)
    axes[0].legend(frameon=False, fontsize=6.5)
    return save(figure, directory, "Fig3_gam_curves")


def figure4(directory: Path) -> list[str]:
    differences = pd.read_csv(OUT / "gam_curves/difference_curves.csv")
    differences = differences[is_true(differences["primary_spec"])
                              & (differences["region"] == "contrast")
                              & (differences["component"] == "shape")]
    comparisons = [
        ("iphone13pro minus galaxy_s24plus", "iPhone 13 Pro - Galaxy S24+"),
        ("iphone16pro minus galaxy_s24plus", "iPhone 16 Pro - Galaxy S24+"),
        ("iphone16pro minus iphone13pro", "iPhone 16 Pro - iPhone 13 Pro"),
    ]
    figure, axes = plt.subplots(3, 3, figsize=(FULL_WIDTH, 5.4), sharex=True,
                                gridspec_kw={"hspace": 0.25, "wspace": 0.3})
    for row, channel in enumerate(CHANNELS):
        for column, (comparison, title) in enumerate(comparisons):
            axis = axes[row, column]
            line = differences[(differences["channel"] == channel)
                               & (differences["comparison"] == comparison)]
            axis.fill_between(line["ph"], line["simultaneous_lower"], line["simultaneous_upper"],
                              color="#56B4E9", alpha=0.3, linewidth=0, label="simultaneous 95%")
            axis.fill_between(line["ph"], line["lower"], line["upper"],
                              color="#0072B2", alpha=0.35, linewidth=0, label="pointwise 95%")
            axis.plot(line["ph"], line["fit"], color="black", linewidth=1.1)
            axis.axhline(0, color="#D55E00", linestyle="--", linewidth=0.7)
            excluded = (line["simultaneous_lower"] > 0) | (line["simultaneous_upper"] < 0)
            if excluded.any():
                bottom = axis.get_ylim()[0]
                axis.plot(line.loc[excluded, "ph"], np.full(int(excluded.sum()), bottom),
                          "|", color="#D55E00", markersize=4)
            if row == 0:
                axis.set_title(title, fontsize=7.5)
            if column == 0:
                axis.set_ylabel(f"Δ {CHANNEL_LABELS[channel]} (shape)")
            if row == 2:
                axis.set_xlabel("pH")
            axis.set_xticks(range(1, 14, 2))
    axes[0, 0].legend(frameon=False, fontsize=6, loc="upper right")
    panel(axes[0, 0], "", x=0)
    figure.text(0.5, -0.01,
                "pH-dependent part of each phone difference (phone offset removed; identified up "
                "to a constant). Orange ticks: simultaneous band excludes 0.",
                ha="center", fontsize=6.5)
    return save(figure, directory, "S18_gam_differences")


# ---------------------------------------------------------------------------
# Figure 5: validation
# ---------------------------------------------------------------------------

def figure5(directory: Path) -> list[str]:
    cell = pd.read_csv(OUT / "model_structure_comparison/cv_summary.csv", encoding="utf-8-sig")
    cell = cell[(cell["scheme"] == "leave_one_cell_out") & (cell["region"] == "contrast")]
    cell = cell.set_index("model")["mean_deltaE76"]
    strip = pd.read_csv(OUT / "gam_strip_holdout/summary_by_model.csv", encoding="utf-8-sig")
    strip = strip[is_true(strip["primary_spec"]) & (strip["region"] == "contrast")]
    strip = strip.set_index("model")["mean_deltaE76"]
    consistency = pd.read_csv(OUT / "gam_strip_holdout/gam_rcs_consistency.csv",
                              encoding="utf-8-sig")
    consistency = consistency[consistency["region"] == "contrast"]
    strips = pd.read_csv(OUT / "gam_strip_holdout/strip_metrics.csv", encoding="utf-8-sig")
    strips = strips[is_true(strips["primary_spec"]) & (strips["region"] == "contrast")]
    wide = strips.pivot_table(index=["strip_id", "phone", "ph_nominal"], columns="model",
                              values="mean_deltaE76").reset_index()
    wide["difference"] = wide["M2"] - wide["M1"]

    figure = plt.figure(figsize=(FULL_WIDTH, 5.2))
    grid = figure.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[0.9, 1.25, 1],
                           hspace=0.55, wspace=0.75)

    axis = figure.add_subplot(grid[0, 0])
    bars = [
        ("image-level\nholdout", cell["additive_nuisance"],
         cell["ph_dependent_interactions"]),
        ("strip-level\nholdout", strip["M1"], strip["M2"]),
    ]
    x = np.arange(len(bars))
    axis.bar(x - 0.18, [b[1] for b in bars], width=0.34, color="#0072B2", label="additive")
    axis.bar(x + 0.18, [b[2] for b in bars], width=0.34, color="#D55E00",
             label="phone-specific")
    for xi, (_, additive, specific) in zip(x, bars):
        axis.text(xi, max(additive, specific) + 0.3, f"gain {additive - specific:.1f}",
                  ha="center", fontsize=6.5)
    axis.set_xticks(x)
    axis.set_xticklabels([b[0] for b in bars], fontsize=6.5)
    axis.set_ylabel("mean ΔE76")
    axis.set_ylim(0, 9.5)
    axis.legend(frameon=False, fontsize=6, loc="upper left")
    axis.set_title("Validation unit", fontsize=7.5)
    panel(axis, "A", x=-0.35)

    rows = [("GAM", "tp k=10"), ("GAM", "tp k=6"), ("GAM", "tp k=12"), ("GAM", "cr k=10"),
            ("RCS", "knots=3"), ("RCS", "knots=4"), ("RCS", "knots=5"),
            ("RCS ridge", "knots=3"), ("RCS ridge", "knots=4"), ("RCS ridge", "knots=5")]
    colors = {"GAM": "#000000", "RCS": "#999999", "RCS ridge": "#D55E00"}
    for column, (subset, title, letter) in enumerate([
            ("interior_ph_2_12", "pH 2-12", "B"), ("boundary_ph_1_13", "pH 1, 13", "C")]):
        axis = figure.add_subplot(grid[0, 1 + column])
        for y, (method, variant) in enumerate(rows[::-1]):
            hit = consistency[(consistency["method"] == method) & (consistency["variant"] == variant)
                              & (consistency["subset"] == subset)].iloc[0]
            primary = variant == "tp k=10"
            axis.errorbar(hit["mean_difference"], y,
                          xerr=[[hit["mean_difference"] - hit["ci_lower"]],
                                [hit["ci_upper"] - hit["mean_difference"]]],
                          fmt="o" if primary else "s", color=colors[method],
                          markersize=4 if primary else 3,
                          markerfacecolor=colors[method] if primary else "white",
                          capsize=1.5, linewidth=0.8)
        axis.axvline(0, color="#0072B2", linestyle="--", linewidth=0.7)
        axis.set_yticks(range(len(rows)))
        axis.set_yticklabels([f"{m}: {v}" for m, v in rows[::-1]] if column == 0 else [],
                             fontsize=6)
        axis.set_xlabel("ΔE76 (specific - additive)", fontsize=7)
        axis.set_title(f"Strip holdout, {title}", fontsize=7.5)
        panel(axis, letter, x=-0.12 if column else -0.95)

    axis = figure.add_subplot(grid[1, :])
    offsets = {"galaxy_s24plus": -0.15, "iphone13pro": 0.0, "iphone16pro": 0.15}
    for phone in PHONES:
        part = wide[wide["phone"] == phone]
        axis.scatter(part["ph_nominal"] + offsets[phone], part["difference"], s=14,
                     color=PHONE_COLORS[phone], label=PHONE_LABELS[phone], zorder=3)
    mean = wide.groupby("ph_nominal")["difference"].mean()
    axis.plot(mean.index, mean.values, color="black", linewidth=1.0, label="mean")
    axis.axhline(0, color="#888888", linewidth=0.6)
    for edge in (1, 13):
        axis.axvspan(edge - 0.5, edge + 0.5, color="#D55E00", alpha=0.07)
    axis.set_xticks(range(1, 14))
    axis.set_xlim(0.5, 13.5)
    axis.set_xlabel("pH of held-out strip (shaded: extrapolation at range boundary)")
    axis.set_ylabel("ΔE76: M2 - M1")
    axis.legend(frameon=False, fontsize=6.5, ncol=4, loc="upper center")
    axis.set_title("Per-strip gain of phone-specific GAM curves (Contrast, tp k=10)", fontsize=7.5)
    panel(axis, "D", x=-0.06)
    return save(figure, directory, "Fig6_validation")


# ---------------------------------------------------------------------------
# Figure 6: inverse pH prediction
# ---------------------------------------------------------------------------

def figure6(directory: Path) -> list[str]:
    strips = pd.read_csv(OUT / "gam_inverse_ph/strip_metrics.csv", encoding="utf-8-sig")
    strips = strips[is_true(strips["primary_spec"])]
    comparisons = pd.read_csv(OUT / "gam_inverse_ph/subset_comparisons.csv", encoding="utf-8-sig")
    comparisons = comparisons[is_true(comparisons["primary_spec"])
                              & (comparisons["metric"] == "mean_abs_error_ph")]
    curves = pd.read_csv(OUT / "gam_curves/phone_curves.csv")
    curves = curves[is_true(curves["primary_spec"]) & (curves["region"] == "contrast")]
    speeds = []
    for _, group in curves.groupby("phone"):
        wide = group.pivot_table(index="ph", columns="channel", values="fit").sort_index()
        derivative = np.gradient(wide[CHANNELS].to_numpy(), wide.index.to_numpy(), axis=0)
        speeds.append(pd.Series(np.linalg.norm(derivative, axis=1), index=wide.index))
    gradient = pd.concat(speeds, axis=1).mean(axis=1)

    figure, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.7),
                                gridspec_kw={"width_ratios": [1.3, 1], "wspace": 1.05})
    axis = axes[0]
    part = strips[strips["region"] == "contrast"]
    for model in ["M0", "M1", "M2"]:
        line = part[part["model"] == model].groupby("true_ph")["mean_abs_error_ph"].mean()
        axis.plot(line.index, line.values, marker="o", markersize=2.5,
                  color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    twin = axis.twinx()
    twin.plot(gradient.index, gradient.values, color="#009E73", linestyle=":", linewidth=1.2)
    twin.set_ylabel("|dLab/dpH|", color="#009E73")
    twin.spines["right"].set_visible(True)
    twin.tick_params(axis="y", colors="#009E73")
    axis.set_xlabel("true pH of held-out strip")
    axis.set_ylabel("mean |pH error|")
    axis.set_xticks(range(1, 14))
    axis.legend(frameon=False, fontsize=6.5, loc="upper left")
    axis.set_title("Contrast: inverse pH error by pH (dotted: colour gradient)", fontsize=7.5)
    panel(axis, "A", x=-0.14)

    axis = axes[1]
    labels, y = [], 0
    for region in ["contrast", "pad"]:
        for comparison, color in [("M1 minus M0", "#0072B2"), ("M2 minus M1", "#D55E00")]:
            for subset, marker in [("interior_ph_2_12", "o"), ("boundary_ph_1_13", "^")]:
                hit = comparisons[(comparisons["region"] == region)
                                  & (comparisons["comparison"] == comparison)
                                  & (comparisons["subset"] == subset)].iloc[0]
                axis.errorbar(hit["mean_difference"], y,
                              xerr=[[hit["mean_difference"] - hit["bootstrap_ci_lower"]],
                                    [hit["bootstrap_ci_upper"] - hit["mean_difference"]]],
                              fmt=marker, color=color, markersize=4, capsize=1.5, linewidth=0.8)
                labels.append(f"{REGION_LABELS[region]} {comparison.replace(' minus ', '-')}, "
                              f"{'2-12' if subset.startswith('interior') else '1,13'}")
                y += 1
    axis.axvline(0, color="#888888", linestyle="--", linewidth=0.7)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=6)
    axis.invert_yaxis()
    axis.set_xlabel("difference in mean |pH error|\n(< 0 favours the first model)")
    axis.set_title("Paired strip differences, 95% CI", fontsize=7.5)
    panel(axis, "B", x=-0.7)
    return save(figure, directory, "S19_inverse_ph")



# ---------------------------------------------------------------------------
# Figure 4: which nuisance moves the pH reading (scripts 34-35)
# ---------------------------------------------------------------------------

def figure_mechanism(directory: Path) -> list[str]:
    shifts = pd.read_csv(OUT / "ph_relevant_nuisance/component_shift_by_ph.csv")
    shifts = shifts[shifts["region"] == "contrast"]
    summary = pd.read_csv(OUT / "ph_relevant_nuisance/component_summary.csv")
    summary = summary[(summary["region"] == "contrast")
                      & (summary["subset"] == "interior_ph_2_12")]
    check = pd.read_csv(OUT / "calibration_mechanism/offset_prediction_check.csv",
                        encoding="utf-8-sig")
    check = check[check["region"] == "contrast"].set_index("phone").loc[PHONES]

    figure, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.7),
                                gridspec_kw={"width_ratios": [1.25, 1.1, 0.9], "wspace": 0.5})

    axis = axes[0]
    for phone in PHONES:
        rows = shifts[(shifts["phone"] == phone) & (shifts["component"] == "phone_offset")]
        axis.plot(rows["ph"], rows["shift_median"].abs(), color=PHONE_COLORS[phone],
                  label=f"{PHONE_LABELS[phone]} offset")
        rows = shifts[(shifts["phone"] == phone) & (shifts["component"] == "phone_shape")]
        axis.plot(rows["ph"], rows["shift_median"].abs(), color=PHONE_COLORS[phone],
                  linestyle=":", linewidth=1.0)
    noise = shifts[shifts["component"] == "strip_noise"]
    axis.plot(noise["ph"], noise["shift_median"], color="black", linestyle="--", linewidth=1.0,
              label="strip-to-strip noise (SD)")
    axis.set_xlabel("pH")
    axis.set_ylabel("pH reading error caused (|pH|)")
    axis.set_ylim(0, 1.6)
    axis.set_xticks(range(1, 14, 2))
    axis.legend(frameon=False, fontsize=5.8, loc="upper left")
    axis.set_title("Effect of each nuisance along pH\n(solid: phone offset, dotted: curve shape)",
                   fontsize=7.5)
    panel(axis, "A", x=-0.22)

    axis = axes[1]
    components = [("phone_offset", "phone\noffset", "#0072B2"),
                  ("phone_shape", "curve\nshape", "#56B4E9"),
                  ("light", "light\n(largest)", "#999999")]
    width = 0.25
    for index, phone in enumerate(PHONES):
        for c_index, (component, _, color) in enumerate(components):
            part = summary[(summary["phone"] == phone) & (summary["component"] == component)]
            value = part["mean_abs_shift_median"].max()
            axis.bar(index + (c_index - 1) * width, value, width=width, color=color,
                     label=components[c_index][1].replace("\n", " ") if index == 0 else None)
    strip = summary[summary["component"] == "strip_noise"]["mean_abs_shift_median"].iloc[0]
    axis.axhline(strip, color="black", linestyle="--", linewidth=0.9)
    axis.text(2.45, strip + 0.02, "strip noise", ha="right", fontsize=6)
    axis.set_xticks(range(len(PHONES)))
    axis.set_xticklabels([PHONE_LABELS[p].replace(" ", "\n", 1) for p in PHONES], fontsize=6.5)
    axis.set_ylabel("mean pH error caused, pH 2-12")
    axis.legend(frameon=False, fontsize=6, loc="upper right")
    axis.set_title("Size of each nuisance\nin pH units", fontsize=7.5)
    panel(axis, "B", x=-0.25)

    axis = axes[2]
    x = np.arange(len(PHONES))
    axis.bar(x - 0.18, check["predicted_mean_abs_shift"], width=0.34, color="#999999",
             yerr=[check["predicted_mean_abs_shift"] - check["predicted_lower"],
                   check["predicted_upper"] - check["predicted_mean_abs_shift"]],
             capsize=2, label="predicted")
    axis.bar(x + 0.18, check["observed_mean_bias_new_phone"].abs(), width=0.34,
             color="#D55E00", label="observed")
    axis.set_xticks(x)
    axis.set_xticklabels([PHONE_LABELS[p].replace(" ", "\n", 1) for p in PHONES], fontsize=6.5)
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("|pH bias|, pH 2-12")
    axis.legend(frameon=False, fontsize=6, loc="upper right")
    axis.set_title("Offset: predicted vs observed\non an uncalibrated new phone", fontsize=7.5)
    panel(axis, "C", x=-0.32)
    return save(figure, directory, "Fig4_nuisance_mechanism")


# ---------------------------------------------------------------------------
# Figure 5: how much calibration is needed (scripts 28, 35)
# ---------------------------------------------------------------------------

def figure_calibration(directory: Path) -> list[str]:
    ladder = pd.read_csv(OUT / "calibration_mechanism/calibration_ladder.csv", encoding="utf-8-sig")
    comparisons = pd.read_csv(OUT / "gam_inverse_ph/subset_comparisons.csv", encoding="utf-8-sig")
    comparisons = comparisons[is_true(comparisons["primary_spec"])
                              & (comparisons["metric"] == "mean_abs_error_ph")
                              & (comparisons["subset"] == "interior_ph_2_12")]

    figure = plt.figure(figsize=(FULL_WIDTH, 4.8))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.15, 0.85], hspace=0.75, wspace=0.4)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]),
            figure.add_subplot(grid[1, :])]

    for axis, metric, label, letter in [
            (axes[0], "mean_abs_error_ph", "mean |pH error|, pH 2-12", "A"),
            (axes[1], "exact_accuracy", "exact pH accuracy, pH 2-12", "B")]:
        for region, color, marker in [("contrast", "#0072B2", "o"), ("pad", "#999999", "s")]:
            rows = ladder[ladder["region"] == region].sort_values("reference_strips")
            shown = rows[rows["reference_strips"] <= 3]
            if metric == "mean_abs_error_ph":
                axis.errorbar(shown["reference_strips"], shown[metric],
                              yerr=[shown[metric] - shown["q10"], shown["q90"] - shown[metric]],
                              color=color, marker=marker, markersize=3.5, capsize=2,
                              label="Pad - Background (Contrast)" if region == "contrast"
                              else "Pad only")
            else:
                axis.plot(shown["reference_strips"], shown[metric], color=color, marker=marker,
                          markersize=3.5)
            full = rows[rows["reference_strips"] == 12].iloc[0][metric]
            axis.axhline(full, color=color, linestyle=":", linewidth=0.9)
        axis.set_xticks([0, 1, 2, 3])
        axis.set_xlabel("reference strips photographed\nwith the new phone")
        axis.set_ylabel(label)
        panel(axis, letter, x=-0.3)
    axes[0].legend(frameon=False, fontsize=6, loc="upper right")
    axes[0].set_title("New phone (never seen), light unknown\ndotted: offset from all 12 strips",
                      fontsize=7.5)
    axes[1].set_title("Exact pH accuracy", fontsize=7.5)

    axis = axes[2]
    labels = []
    for y, (region, comparison, color) in enumerate([
            ("contrast", "M1 minus M0", "#0072B2"), ("contrast", "M2 minus M1", "#D55E00"),
            ("pad", "M1 minus M0", "#0072B2"), ("pad", "M2 minus M1", "#D55E00")]):
        hit = comparisons[(comparisons["region"] == region)
                          & (comparisons["comparison"] == comparison)].iloc[0]
        axis.errorbar(hit["mean_difference"], y,
                      xerr=[[hit["mean_difference"] - hit["bootstrap_ci_lower"]],
                            [hit["bootstrap_ci_upper"] - hit["mean_difference"]]],
                      fmt="o", color=color, markersize=4, capsize=2)
        step = "+ phone offset" if comparison.startswith("M1") else "+ phone curve shape"
        labels.append(f"{REGION_LABELS[region]}: {step}")
    axis.axvline(0, color="#888888", linestyle="--", linewidth=0.7)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=6.3)
    axis.invert_yaxis()
    axis.set_xlabel("change in mean |pH error| (< 0 = better), 95% CI")
    axis.set_title("Added value of each calibration step when the phone is already "
                   "characterized (leave-one-strip-out)", fontsize=7.5)
    panel(axis, "C", x=-0.28, y=1.1)
    return save(figure, directory, "Fig5_calibration")

# ---------------------------------------------------------------------------
# Figure 7: simulation
# ---------------------------------------------------------------------------

def figure7(directory: Path) -> list[str]:
    s1 = pd.read_csv(OUT / "simulation/s1/s1_rejection_rates.csv", encoding="utf-8-sig")
    s2 = pd.read_csv(OUT / "simulation/s2/s2_summary.csv", encoding="utf-8-sig")
    s3 = pd.read_csv(OUT / "simulation/s3/loso_dataset_differences.csv", encoding="utf-8-sig")
    consistency = pd.read_csv(OUT / "gam_strip_holdout/gam_rcs_consistency.csv",
                              encoding="utf-8-sig")
    variance = pd.read_csv(OUT / "gam_curves/variance_components.csv")
    variance = variance[is_true(variance["primary_spec"]) & (variance["region"] == "contrast")]

    figure = plt.figure(figsize=(FULL_WIDTH, 5.3))
    grid = figure.add_gridspec(2, 3, hspace=0.6, wspace=0.45)

    axis = figure.add_subplot(grid[0, 0])
    null = s1[(s1["study"] == "S1a") & (s1["tau"] == 0)]
    for test, color, marker, label in [("restricted_pH", "#0072B2", "o", "strip-level"),
                                       ("unrestricted_pH", "#D55E00", "s", "image-level")]:
        rows = null[null["test"] == test].sort_values("r")
        axis.errorbar(rows["r"], rows["rejection_rate"], yerr=1.96 * rows["mc_se"],
                      color=color, marker=marker, markersize=3, capsize=1.5, label=label)
    axis.axhline(0.05, color="#888888", linestyle="--", linewidth=0.7)
    axis.set_xlabel("strip SD / image SD")
    axis.set_ylabel("type I error")
    axis.set_title("S1: pH test, no pH effect", fontsize=7.5)
    axis.legend(frameon=False, fontsize=6.5)
    panel(axis, "A", x=-0.32)

    axis = figure.add_subplot(grid[0, 1])
    rows = s1[s1["study"] == "S1b"].sort_values("r")
    axis.errorbar(rows["r"], rows["rejection_rate"], yerr=1.96 * rows["mc_se"], color="black",
                  marker="^", markersize=3, capsize=1.5)
    axis.axhline(0.05, color="#888888", linestyle="--", linewidth=0.7)
    axis.axvspan(4.5, 6.3, color="#D55E00", alpha=0.1)
    axis.text(5.4, 0.5, "real\nContrast", ha="center", fontsize=6, color="#D55E00")
    axis.set_xlabel("strip SD / image SD")
    axis.set_ylabel("rejection rate")
    axis.set_title("S1: ASCA pH x phone test,\nno phone-specific curves", fontsize=7.5)
    panel(axis, "B", x=-0.32)

    axis = figure.add_subplot(grid[0, 2])
    part = s2[(s2["channel"] == "a") & np.isclose(s2["strip_multiplier"], 1.0)].sort_values("A")
    axis.errorbar(part["A"], part["reject_m2_any_holm"], yerr=1.96 * part["reject_m2_any_holm_mcse"],
                  color="#0072B2", marker="o", markersize=3, capsize=1.5, label="GAM with strip RE")
    axis.errorbar(part["A"], part["reject_naive_any_holm"],
                  yerr=1.96 * part["reject_naive_any_holm_mcse"], color="#D55E00", marker="s",
                  markersize=3, capsize=1.5, label="GAM without strip RE")
    axis.axhline(0.05, color="#888888", linestyle="--", linewidth=0.7)
    axis.set_xlabel("phone-specific curve size A (1 = real)")
    axis.set_ylabel("rejection rate")
    axis.set_title("S2: test of phone-specific curves (a*)", fontsize=7.5)
    axis.legend(frameon=False, fontsize=6)
    panel(axis, "C", x=-0.32)

    axis = figure.add_subplot(grid[1, 0])
    axis.plot(part["A"], part["variance_reduction_mean"], color="#0072B2", marker="o", markersize=3)
    axis.fill_between(part["A"], part["variance_reduction_q05"], part["variance_reduction_q95"],
                      color="#0072B2", alpha=0.15, linewidth=0)
    real = float(variance.set_index("channel").loc["a", "strip_var_reduction_M1_to_M2"])
    axis.axhline(real, color="#D55E00", linestyle=":", linewidth=1.0)
    axis.text(0.02, real + 0.03, f"real a*: {100 * real:.0f}%", color="#D55E00", fontsize=6)
    axis.set_xlabel("phone-specific curve size A")
    axis.set_ylabel("strip variance absorbed")
    axis.set_title("S2: strip variance absorbed", fontsize=7.5)
    panel(axis, "D", x=-0.32)

    for column, (subset, title, letter) in enumerate([
            ("interior_ph_2_12", "S3: strip holdout, pH 2-12 (A = 1)", "E"),
            ("boundary_ph_1_13", "S3: strip holdout, pH 1, 13 (A = 1)", "F")]):
        axis = figure.add_subplot(grid[1, 1 + column])
        entries = [("GAM", "tp k=10", "M2 minus M1", "GAM"),
                   ("RCS", "knots=3", "rcs_phone minus rcs_additive", "RCS"),
                   ("RCS", "knots=4", "rcs_phone minus rcs_additive", "RCS"),
                   ("RCS", "knots=4", "rcs_phone_ridge minus rcs_additive", "RCS ridge")]
        data, labels, real_values = [], [], []
        for method, variant, comparison, real_method in entries:
            rows = s3[(s3["A"] == 1) & (s3["method"] == method) & (s3["variant"] == variant)
                      & (s3["comparison"] == comparison) & (s3["subset"] == subset)]
            data.append(rows["mean_difference"].to_numpy())
            labels.append(f"{real_method}\n{variant}")
            hit = consistency[(consistency["region"] == "contrast")
                              & (consistency["method"] == real_method)
                              & (consistency["variant"] == variant)
                              & (consistency["subset"] == subset)]
            real_values.append(float(hit["mean_difference"].iloc[0]))
        positions = np.arange(len(data))
        axis.boxplot(data, positions=positions, widths=0.55, showfliers=False,
                     medianprops={"color": "black"})
        axis.scatter(positions, real_values, color="#D55E00", marker="D", s=14, zorder=3,
                     label="real data")
        axis.axhline(0, color="#0072B2", linestyle="--", linewidth=0.7)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, fontsize=5.8)
        axis.set_ylabel("ΔE76 (specific - additive)")
        axis.set_title(title, fontsize=7.5)
        if column == 0:
            axis.legend(frameon=False, fontsize=6, loc="upper left")
        panel(axis, letter, x=-0.32)

    return save(figure, directory, "Fig7_simulation")


# ---------------------------------------------------------------------------
# Supplementary figures (copies with S-numbering)
# ---------------------------------------------------------------------------

SUPPLEMENTARY = [
    ("S1", "background_roi_check/roi_montage_L01_ph2_6_12.jpg",
     "Pad and background ROIs for all phones at pH 2, 6, 12"),
    ("S2", "multivariate_lab/figures/effect_composition_standardized.png",
     "Effect composition (script 15 version)"),
    ("S3", "asca_permutation/figures/restricted_vs_unrestricted_main_effects.png",
     "ASCA main-effect tests: strip-restricted vs image-level permutation"),
    ("S4", "asca_permutation/figures/pc1_loading_intervals.png",
     "ASCA PC1 loadings for all regions and scalings"),
    ("S5", "gam_curves/figures/phone_curves_pad.png", "GAM curves, Pad"),
    ("S6", "gam_curves/figures/phone_curves_bg.png", "GAM curves, Background"),
    ("S7", "gam_curves/figures/k_sensitivity_shape_contrast.png",
     "Basis / k sensitivity of the phone differences"),
    ("S8", "gam_curves/figures/total_differences_contrast.png",
     "Total phone differences (offset + pH-dependent part)"),
    ("S9", "gam_curves/figures/residual_diagnostics_contrast.png", "GAM residual diagnostics"),
    ("S10", "gam_curves/figures/strip_variance_reduction.png",
     "Strip variance absorbed, all regions and specifications"),
    ("S11", "gam_strip_holdout/figures/gam_rcs_consistency_pad.png", "GAM vs RCS, Pad"),
    ("S12", "gam_inverse_ph/figures/estimated_vs_true_contrast.png",
     "Estimated vs true pH, Contrast"),
    ("S13", "gam_inverse_ph/figures/deltaE_vs_ph_gain_contrast.png",
     "Colour gain vs pH gain per strip"),
    ("S14", "simulation/s1/figures/s1_rejection_rates.png", "Simulation S1, all scenarios"),
    ("S15", "simulation/figures/s2_identification.png", "Simulation S2, all cells"),
    ("S16", "simulation/figures/s3_loso_differences.png", "Simulation S2-LOSO / S3, all models"),
]


def augmentation_figure(directory: Path) -> list[str]:
    strict = pd.read_csv(OUT / "hybrid_augmentation_pilot_strict_lopo/method_summary.csv",
                         encoding="utf-8-sig").set_index("method")
    order = ["baseline", "colorjitter", "global_empirical", "hybrid_empirical"]
    figure, axis = plt.subplots(figsize=(3.6, 2.4))
    axis.bar(range(len(order)), strict.loc[order, "macro_f1_mean"],
             yerr=strict.loc[order, "macro_f1_std"], color="#999999", capsize=2)
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels(["baseline", "ColorJitter", "global\nempirical", "hybrid\nempirical"],
                         fontsize=6.5)
    axis.set_ylim(0.7, 1.0)
    axis.set_ylabel("macro-F1 (held-out phone)")
    axis.set_title("Strict target-free LOPO CNN pilot (18 runs each)", fontsize=7.5)
    return save(figure, directory, "S17_augmentation_pilot")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper figures.")
    parser.add_argument("--output-dir", default="outputs/paper_figures")
    args = parser.parse_args()

    set_style()
    root = (PROJECT_ROOT / args.output_dir).resolve()
    main_dir = root / "main"
    supplementary_dir = root / "supplementary"
    main_dir.mkdir(parents=True, exist_ok=True)
    supplementary_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    main_builders = [figure1, figure2, figure3, figure_mechanism, figure_calibration,
                     figure5, figure7]
    for number, builder in enumerate(main_builders, start=1):
        files = builder(main_dir)
        index_rows.append({"figure": f"Figure {number}", "file": files[0], "source": builder.__name__})

    for label, source, caption in SUPPLEMENTARY:
        path = OUT / source
        if not path.exists():
            index_rows.append({"figure": f"Figure {label}", "file": "MISSING", "source": source})
            continue
        target = supplementary_dir / f"{label}_{path.name}"
        shutil.copy2(path, target)
        index_rows.append({"figure": f"Figure {label}", "file": target.name, "source": source,
                           "caption": caption})

    files = augmentation_figure(supplementary_dir)
    index_rows.append({"figure": "Figure S17", "file": files[0],
                       "source": "hybrid_augmentation_pilot_strict_lopo/method_summary.csv",
                       "caption": "Initial augmentation experiment (motivation)"})
    for label, builder, caption in [
            ("S18", figure4, "pH-dependent phone differences with simultaneous bands"),
            ("S19", figure6, "Inverse pH error by pH, known phone (LOSO)")]:
        files = builder(supplementary_dir)
        index_rows.append({"figure": f"Figure {label}", "file": files[0],
                           "source": builder.__name__, "caption": caption})

    pd.DataFrame(index_rows).to_csv(root / "figure_index.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(index_rows)[["figure", "file"]].to_string(index=False))


if __name__ == "__main__":
    main()
