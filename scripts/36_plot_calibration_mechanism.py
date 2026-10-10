#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
36_plot_calibration_mechanism.py

Figures and summary tables linking the pH-relevant nuisance decomposition
(script 34) with the new-phone offset calibration (script 35) and the light
known / unknown inverse prediction (script 28).

Outputs
-------
outputs/calibration_mechanism/
    calibration_ladder.csv      pH error by correction step (light unknown, new phone)
    offset_prediction_check.csv predicted pH shift of the phone offset (script 34)
                                vs observed uncalibrated bias on a new phone (script 35)
    component_summary_table.csv mean |pH shift| by nuisance component
    notes.txt
    figures/
        Fig_calibration_mechanism.png / .pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs"
TARGET = OUT / "calibration_mechanism"

PHONES = ["galaxy_s24plus", "iphone13pro", "iphone16pro"]
PHONE_LABELS = {"galaxy_s24plus": "Galaxy S24+", "iphone13pro": "iPhone 13 Pro",
                "iphone16pro": "iPhone 16 Pro"}
PHONE_COLORS = {"galaxy_s24plus": "#0072B2", "iphone13pro": "#E69F00",
                "iphone16pro": "#009E73"}


def panel(axis: plt.Axes, letter: str, x: float = -0.2, y: float = 1.06) -> None:
    axis.text(x, y, letter, transform=axis.transAxes, fontsize=11, fontweight="bold",
              va="bottom")


def main() -> None:
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300})
    (TARGET / "figures").mkdir(parents=True, exist_ok=True)

    calibration = {
        region: pd.read_csv(OUT / f"new_phone_calibration/calibration_results_{region}.csv")
        for region in ["contrast", "pad"]
    }
    components = pd.read_csv(OUT / "ph_relevant_nuisance/component_summary.csv")
    shifts = pd.read_csv(OUT / "ph_relevant_nuisance/component_shift_by_ph.csv")

    # Ladder: new phone, light unknown.
    ladder_rows = []
    for region, frame in calibration.items():
        part = frame[frame["light_mode"] == "unknown"]
        for k, group in part.groupby("k"):
            ladder_rows.append({
                "region": region, "reference_strips": int(k),
                "n_reference_sets": len(group),
                "mean_abs_error_ph": group["mean_abs_error_interior"].mean(),
                "q10": group["mean_abs_error_interior"].quantile(0.1),
                "q90": group["mean_abs_error_interior"].quantile(0.9),
                "exact_accuracy": group["exact_accuracy_interior"].mean(),
                "within1_accuracy": group["within1_accuracy_interior"].mean(),
            })
    ladder = pd.DataFrame(ladder_rows)
    full = ladder.set_index(["region", "reference_strips"])["mean_abs_error_ph"]
    ladder["share_of_full_gain"] = [
        (full[(r, 0)] - e) / (full[(r, 0)] - full[(r, 12)])
        for r, e in zip(ladder["region"], ladder["mean_abs_error_ph"])
    ]
    ladder.to_csv(TARGET / "calibration_ladder.csv", index=False, encoding="utf-8-sig")

    # Mechanism check: predicted offset pH shift vs observed uncalibrated bias.
    check_rows = []
    for region, frame in calibration.items():
        observed = frame[(frame["k"] == 0) & (frame["light_mode"] == "unknown")]
        for _, row in observed.iterrows():
            predicted = components[(components["region"] == region)
                                   & (components["phone"] == row["target_phone"])
                                   & (components["component"] == "phone_offset")
                                   & (components["subset"] == "interior_ph_2_12")].iloc[0]
            check_rows.append({
                "region": region, "phone": row["target_phone"],
                "predicted_mean_abs_shift": predicted["mean_abs_shift_median"],
                "predicted_lower": predicted["mean_abs_shift_lower"],
                "predicted_upper": predicted["mean_abs_shift_upper"],
                "observed_mean_bias_new_phone": row["mean_bias_interior"],
                "observed_mean_abs_error_new_phone": row["mean_abs_error_interior"],
            })
    check = pd.DataFrame(check_rows)
    check.to_csv(TARGET / "offset_prediction_check.csv", index=False, encoding="utf-8-sig")

    interior = components[components["subset"] == "interior_ph_2_12"]
    light_max = (interior[interior["component"] == "light"]
                 .groupby(["region", "phone"])["mean_abs_shift_median"].max()
                 .rename("light_max"))
    table = (interior[interior["component"] != "light"]
             .pivot_table(index=["region", "phone"], columns="component",
                          values="mean_abs_shift_median")
             .join(light_max))
    table.to_csv(TARGET / "component_summary_table.csv", encoding="utf-8-sig")

    # ------------------------------------------------------------------ figure
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.5})

    axis = axes[0]
    contrast_shift = shifts[shifts["region"] == "contrast"]
    for phone in PHONES:
        rows = contrast_shift[(contrast_shift["phone"] == phone)
                              & (contrast_shift["component"] == "phone_offset")]
        axis.plot(rows["ph"], rows["shift_median"].abs(), color=PHONE_COLORS[phone],
                  label=f"offset, {PHONE_LABELS[phone]}")
        rows = contrast_shift[(contrast_shift["phone"] == phone)
                              & (contrast_shift["component"] == "phone_shape")]
        axis.plot(rows["ph"], rows["shift_median"].abs(), color=PHONE_COLORS[phone],
                  linestyle=":", linewidth=1.0)
    noise = contrast_shift[contrast_shift["component"] == "strip_noise"]
    axis.plot(noise["ph"], noise["shift_median"], color="black", linestyle="--",
              linewidth=1.0, label="strip noise (SD)")
    axis.set_xlabel("pH")
    axis.set_ylabel("|pH shift| caused by component")
    axis.set_ylim(0, 2.0)
    axis.set_xticks(range(1, 14, 2))
    axis.legend(frameon=False, fontsize=5.5, loc="upper left")
    axis.set_title("Contrast: pH-relevant nuisance\n(solid offset, dotted shape)", fontsize=7.5)
    panel(axis, "A")

    axis = axes[1]
    x = np.arange(len(PHONES))
    part = check[check["region"] == "contrast"].set_index("phone").loc[PHONES]
    axis.bar(x - 0.18, part["predicted_mean_abs_shift"], width=0.34, color="#999999",
             yerr=[part["predicted_mean_abs_shift"] - part["predicted_lower"],
                   part["predicted_upper"] - part["predicted_mean_abs_shift"]],
             capsize=2, label="predicted from colour geometry")
    axis.bar(x + 0.18, part["observed_mean_bias_new_phone"].abs(), width=0.34,
             color="#D55E00", label="observed bias, new phone")
    axis.set_xticks(x)
    axis.set_xticklabels([PHONE_LABELS[p].replace(" ", "\n", 1) for p in PHONES], fontsize=6.5)
    axis.set_ylabel("|pH shift|, pH 2-12")
    axis.set_ylim(0, 1.05)
    axis.legend(frameon=False, fontsize=6, loc="upper right")
    axis.set_title("Phone offset: predicted vs observed", fontsize=7.5)
    panel(axis, "B")

    axis = axes[2]
    for region, color, marker in [("contrast", "#0072B2", "o"), ("pad", "#999999", "s")]:
        rows = ladder[ladder["region"] == region].sort_values("reference_strips")
        shown = rows[rows["reference_strips"] <= 3]
        axis.errorbar(shown["reference_strips"], shown["mean_abs_error_ph"],
                      yerr=[shown["mean_abs_error_ph"] - shown["q10"],
                            shown["q90"] - shown["mean_abs_error_ph"]],
                      color=color, marker=marker, capsize=2, label=region.capitalize())
        all_strips = rows[rows["reference_strips"] == 12].iloc[0]
        axis.axhline(all_strips["mean_abs_error_ph"], color=color, linestyle=":", linewidth=0.9)
    axis.set_xticks([0, 1, 2, 3])
    axis.set_xlabel("reference strips on the new phone")
    axis.set_ylabel("mean |pH error|, pH 2-12")
    axis.legend(frameon=False, fontsize=6.5)
    axis.set_title("New phone, light unknown\n(dotted: offset from all 12 strips)", fontsize=7.5)
    panel(axis, "C")

    for suffix in ("png", "pdf"):
        figure.savefig(TARGET / "figures" / f"Fig_calibration_mechanism.{suffix}",
                       bbox_inches="tight")
    plt.close(figure)

    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        notes = [
            "Calibration mechanism summary (script 36)",
            "=" * 80,
            "",
            "New phone, light unknown: pH 2-12 error by number of reference strips",
            ladder.to_string(index=False),
            "",
            "Phone offset: predicted pH shift (script 34) vs observed bias without calibration (script 35)",
            check.to_string(index=False),
            "",
            "Mean |pH shift| by nuisance component, pH 2-12 (script 34; noise rows = SD)",
            table.to_string(),
        ]
    (TARGET / "notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
