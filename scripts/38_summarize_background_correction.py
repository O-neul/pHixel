#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
38_summarize_background_correction.py

Summary and figure for the background-based corrections (script 37) evaluated
with the new-phone calibration procedure (script 35, light unknown).

Question: can the background (white paper) in the same photo replace the
reference strips needed to remove a new phone's offset?

Outputs
-------
outputs/background_correction/
    summary.csv
    summary_notes.txt
    figures/Fig_background_correction.png / .pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = PROJECT_ROOT / "outputs" / "background_correction"
CORRECTIONS = ["pad", "subtract", "wb_linear", "bradford", "ratio_gamma"]
LABELS = {
    "pad": "pad\nonly",
    "subtract": "subtract\n(current)",
    "wb_linear": "WB\nlinear",
    "bradford": "Bradford",
    "ratio_gamma": "RGB\nratio",
}
PHONES = ["galaxy_s24plus", "iphone13pro", "iphone16pro"]
PHONE_LABELS = {"galaxy_s24plus": "Galaxy S24+", "iphone13pro": "iPhone 13 Pro",
                "iphone16pro": "iPhone 16 Pro"}
PHONE_COLORS = {"galaxy_s24plus": "#0072B2", "iphone13pro": "#E69F00",
                "iphone16pro": "#009E73"}


def panel(axis: plt.Axes, letter: str, x: float = -0.2) -> None:
    axis.text(x, 1.06, letter, transform=axis.transAxes, fontsize=11, fontweight="bold",
              va="bottom")


def main() -> None:
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300})
    (TARGET / "figures").mkdir(parents=True, exist_ok=True)
    shares = pd.read_csv(TARGET / "variance_shares.csv", encoding="utf-8-sig").set_index("correction")

    rows, bias_rows = [], []
    for correction in CORRECTIONS:
        frame = pd.read_csv(TARGET / f"calibration/calibration_results_{correction}.csv")
        frame = frame[frame["light_mode"] == "unknown"]
        by_k = frame.groupby("k")
        mae = by_k["mean_abs_error_interior"].mean()
        exact = by_k["exact_accuracy_interior"].mean()
        rows.append({
            "correction": correction,
            "phone_variance_share": shares.loc[correction, "share_phone"],
            "error_no_reference": mae[0], "error_1_reference": mae[1],
            "error_2_reference": mae[2], "error_3_reference": mae[3],
            "error_all_reference": mae[12],
            "exact_no_reference": exact[0], "exact_3_reference": exact[3],
        })
        for _, row in frame[frame["k"] == 0].iterrows():
            bias_rows.append({"correction": correction, "phone": row["target_phone"],
                              "bias": row["mean_bias_interior"]})
    summary = pd.DataFrame(rows)
    bias = pd.DataFrame(bias_rows)
    summary.to_csv(TARGET / "summary.csv", index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.45})
    x = np.arange(len(CORRECTIONS))

    axis = axes[0]
    axis.bar(x, summary["phone_variance_share"] * 100, color="#999999")
    axis.set_xticks(x)
    axis.set_xticklabels([LABELS[c] for c in CORRECTIONS], fontsize=5.8, rotation=0)
    axis.set_ylabel("phone share of colour variation (%)")
    axis.set_title("How much phone effect\nremains after correction", fontsize=7.5)
    panel(axis, "A")

    axis = axes[1]
    axis.bar(x, summary["error_no_reference"], color="#D55E00", label="no reference strip")
    axis.plot(x, summary["error_3_reference"], "o", color="#0072B2", label="3 reference strips")
    axis.plot(x, summary["error_all_reference"], "_", color="black", markersize=12,
              label="all 12 strips")
    axis.set_xticks(x)
    axis.set_xticklabels([LABELS[c] for c in CORRECTIONS], fontsize=5.8)
    axis.set_ylabel("mean |pH error|, pH 2-12")
    axis.set_ylim(0, 1.0)
    axis.legend(frameon=False, fontsize=6)
    axis.set_title("New phone, light unknown", fontsize=7.5)
    panel(axis, "B")

    axis = axes[2]
    width = 0.25
    for index, phone in enumerate(PHONES):
        values = bias[bias["phone"] == phone].set_index("correction").loc[CORRECTIONS, "bias"]
        axis.bar(x + (index - 1) * width, values, width=width, color=PHONE_COLORS[phone],
                 label=PHONE_LABELS[phone])
    axis.axhline(0, color="#888888", linewidth=0.7)
    axis.set_xticks(x)
    axis.set_xticklabels([LABELS[c] for c in CORRECTIONS], fontsize=5.8)
    axis.set_ylabel("mean pH bias without reference")
    axis.legend(frameon=False, fontsize=6)
    axis.set_title("Phone offset left in the pH reading", fontsize=7.5)
    panel(axis, "C")

    for suffix in ("png", "pdf"):
        figure.savefig(TARGET / "figures" / f"Fig_background_correction.{suffix}",
                       bbox_inches="tight")
    plt.close(figure)

    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        notes = [
            "Background-based corrections: summary (script 38)",
            "=" * 80,
            summary.to_string(index=False),
            "",
            "Mean pH bias on an uncalibrated new phone",
            bias.pivot(index="correction", columns="phone", values="bias").loc[CORRECTIONS].to_string(),
        ]
    (TARGET / "summary_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
