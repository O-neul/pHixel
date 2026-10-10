#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
27_summarize_gam_strip_holdout.py

Summaries, paired strip-level bootstrap and GAM-RCS consistency for the
leave-one-strip-out GAM predictions of 27_gam_leave_one_strip_out.R.

Conventions match script 19 so the two analyses are directly comparable:
- image error: Delta E76 from the three Lab channel predictions
- strip error: mean Delta E76 over the five light images of the strip
- paired difference: numerator model minus denominator model, per strip
  (negative favours the numerator)
- bootstrap: strips resampled with replacement within each phone,
  3000 replicates, seed 42
- subsets: all_ph, boundary_ph_1_13, interior_ph_2_12, strict_interior_ph_3_11

Model correspondence:
    GAM M1 (phone offset)          <-> rcs_additive
    GAM M2 (phone-specific curves) <-> rcs_phone / rcs_phone_ridge

Outputs
-------
outputs/gam_strip_holdout/
    strip_metrics.csv
    summary_by_model.csv
    summary_by_ph.csv
    subset_comparisons.csv
    gam_rcs_consistency.csv
    analysis_notes.txt
    figures/
        gam_rcs_consistency_<region>.png
        difference_by_ph_<region>.png
        sorted_strip_differences_<region>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGIONS = ["contrast", "pad", "bg"]
REGION_LABELS = {
    "pad": "Pad",
    "bg": "Background",
    "contrast": "Contrast (Pad - Background)",
}
COMPARISONS = [
    ("M2", "M1"),
    ("M2", "M0"),
    ("M1", "M0"),
]
SUBSETS = {
    "all_ph": set(range(1, 14)),
    "boundary_ph_1_13": {1, 13},
    "interior_ph_2_12": set(range(2, 13)),
    "strict_interior_ph_3_11": set(range(3, 12)),
}
SUBSET_LABELS = {
    "all_ph": "all pH",
    "boundary_ph_1_13": "pH 1, 13",
    "interior_ph_2_12": "pH 2-12",
    "strict_interior_ph_3_11": "pH 3-11",
}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GAM LOSO predictions.")
    parser.add_argument("--holdout-dir", default="outputs/gam_strip_holdout")
    parser.add_argument(
        "--rcs-subsets",
        default="outputs/boundary_stabilized_models/subset_comparisons.csv",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
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


def spec_label(basis: str, k: int) -> str:
    return f"{basis} k={k}"


def bootstrap_difference(
    frame: pd.DataFrame,
    value_column: str,
    reps: int,
    seed: int,
) -> tuple[float, float, float, float]:
    """Identical to script 19: strips resampled within each phone."""
    observed = float(frame[value_column].mean())
    rng = np.random.default_rng(seed)
    groups = [
        group[value_column].to_numpy(float)
        for _, group in frame.groupby("phone", sort=True)
    ]
    values = np.empty(reps)

    for r in range(reps):
        sample = [group[rng.integers(0, len(group), size=len(group))] for group in groups]
        values[r] = np.mean(np.concatenate(sample))

    return (
        observed,
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        float(np.mean(values < 0)),
    )


def strip_metrics_from(predictions: pd.DataFrame) -> pd.DataFrame:
    error = predictions[["predicted_L", "predicted_a", "predicted_b"]].to_numpy() - \
        predictions[["actual_L", "actual_a", "actual_b"]].to_numpy()
    predictions = predictions.copy()
    predictions["deltaE76"] = np.linalg.norm(error, axis=1)

    return (
        predictions.groupby(
            ["basis", "k", "primary_spec", "region", "model",
             "strip_id", "phone", "ph_nominal"],
            as_index=False,
        )
        .agg(
            n_predictions=("deltaE76", "size"),
            mean_deltaE76=("deltaE76", "mean"),
            median_deltaE76=("deltaE76", "median"),
        )
    ), predictions


def paired_comparisons(
    strip_metrics: pd.DataFrame,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    keys = ["basis", "k", "primary_spec", "region", "strip_id", "phone", "ph_nominal"]

    for numerator, denominator in COMPARISONS:
        merged = strip_metrics[strip_metrics["model"] == numerator].merge(
            strip_metrics[strip_metrics["model"] == denominator],
            on=keys,
            suffixes=("_num", "_den"),
            validate="one_to_one",
        )
        merged["difference"] = merged["mean_deltaE76_num"] - merged["mean_deltaE76_den"]

        for (basis, k, primary, region), group in merged.groupby(
            ["basis", "k", "primary_spec", "region"], sort=True
        ):
            for subset, ph_set in SUBSETS.items():
                part = group[group["ph_nominal"].round().astype(int).isin(ph_set)]
                mean, lower, upper, below = bootstrap_difference(
                    part, "difference", reps, seed
                )
                rows.append({
                    "basis": basis,
                    "k": int(k),
                    "primary_spec": bool(primary),
                    "region": region,
                    "subset": subset,
                    "numerator_model": numerator,
                    "denominator_model": denominator,
                    "comparison": f"{numerator} minus {denominator}",
                    "n_strips": int(len(part)),
                    "mean_difference": mean,
                    "median_difference": float(part["difference"].median()),
                    "proportion_numerator_better": float((part["difference"] < 0).mean()),
                    "bootstrap_ci_lower": lower,
                    "bootstrap_ci_upper": upper,
                    "bootstrap_probability_below_zero": below,
                })

    return pd.DataFrame(rows)


def consistency_table(gam: pd.DataFrame, rcs: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    gam_part = gam[gam["comparison"] == "M2 minus M1"]

    for _, row in gam_part.iterrows():
        rows.append({
            "method": "GAM",
            "variant": spec_label(row["basis"], row["k"]),
            "primary": bool(row["primary_spec"]),
            "region": row["region"],
            "subset": row["subset"],
            "comparison": "M2 minus M1",
            "mean_difference": row["mean_difference"],
            "ci_lower": row["bootstrap_ci_lower"],
            "ci_upper": row["bootstrap_ci_upper"],
            "proportion_better": row["proportion_numerator_better"],
            "ci_excludes_zero": bool(
                row["bootstrap_ci_upper"] < 0 or row["bootstrap_ci_lower"] > 0
            ),
        })

    if rcs is not None:
        wanted = rcs[
            rcs["denominator_model"].eq("rcs_additive")
            & rcs["numerator_model"].isin(["rcs_phone", "rcs_phone_ridge"])
        ]
        for _, row in wanted.iterrows():
            rows.append({
                "method": "RCS" if row["numerator_model"] == "rcs_phone" else "RCS ridge",
                "variant": f"knots={int(row['knot_count'])}",
                "primary": bool(
                    row["numerator_model"] == "rcs_phone_ridge" and row["knot_count"] == 4
                ),
                "region": row["region"],
                "subset": row["subset"],
                "comparison": f"{row['numerator_model']} minus rcs_additive",
                "mean_difference": row["mean_difference"],
                "ci_lower": row["bootstrap_ci_lower"],
                "ci_upper": row["bootstrap_ci_upper"],
                "proportion_better": row["proportion_numerator_better"],
                "ci_excludes_zero": bool(
                    row["bootstrap_ci_upper"] < 0 or row["bootstrap_ci_lower"] > 0
                ),
            })

    return pd.DataFrame(rows)


def plot_consistency(
    table: pd.DataFrame,
    region: str,
    path: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    subsets = ["interior_ph_2_12", "strict_interior_ph_3_11", "boundary_ph_1_13", "all_ph"]
    method_colors = {"GAM": "#000000", "RCS": "#999999", "RCS ridge": "#D55E00"}
    figure, axes = plt.subplots(1, len(subsets), figsize=(15.0, 5.2), sharey=True)
    part = table[table["region"] == region]
    order = (
        part[["method", "variant"]]
        .drop_duplicates()
        .sort_values(["method", "variant"], key=lambda s: s.map(str))
        .itertuples(index=False)
    )
    labels = [f"{m}: {v}" for m, v in order]
    y_positions = {label: i for i, label in enumerate(labels[::-1])}

    for axis, subset in zip(axes, subsets):
        rows = part[part["subset"] == subset]
        for _, row in rows.iterrows():
            label = f"{row['method']}: {row['variant']}"
            y = y_positions[label]
            color = method_colors[row["method"]]
            axis.errorbar(
                row["mean_difference"], y,
                xerr=[[row["mean_difference"] - row["ci_lower"]],
                      [row["ci_upper"] - row["mean_difference"]]],
                fmt="o" if row["primary"] else "s",
                color=color,
                markersize=7 if row["primary"] else 4,
                markerfacecolor=color if row["primary"] else "white",
                capsize=2, linewidth=1.2,
            )
        axis.axvline(0.0, color="#0072B2", linewidth=0.9, linestyle="--")
        axis.set_title(SUBSET_LABELS[subset], fontsize=10)
        axis.set_xlabel("ΔE76 difference\n(phone-specific - additive)")
        axis.grid(axis="x", alpha=0.25)

    axes[0].set_yticks(list(y_positions.values()))
    axes[0].set_yticklabels(list(y_positions.keys()), fontsize=8)
    figure.suptitle(
        f"{REGION_LABELS[region]}: strip-holdout gain of phone-specific pH structure, "
        "GAM vs RCS (filled = primary spec; 95% strip bootstrap CI; < 0 favours phone-specific)",
        fontsize=10,
    )
    figure.tight_layout()
    save_figure(figure, path, dpi, save_pdf)


def plot_difference_by_ph(
    strip_differences: pd.DataFrame,
    region: str,
    path: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 4.2))
    part = strip_differences[strip_differences["region"] == region]

    for phone, color in PHONE_COLORS.items():
        rows = part[part["phone"] == phone]
        axis.scatter(
            rows["ph_nominal"] + {"galaxy_s24plus": -0.15, "iphone13pro": 0.0,
                                  "iphone16pro": 0.15}[phone],
            rows["difference"], color=color, s=28, label=PHONE_LABELS[phone], zorder=3,
        )

    means = part.groupby("ph_nominal")["difference"].mean()
    axis.plot(means.index, means.values, color="#000000", linewidth=1.4, label="mean over phones")
    axis.axhline(0.0, color="#888888", linewidth=0.8)
    axis.axvspan(0.5, 1.5, color="#D55E00", alpha=0.08)
    axis.axvspan(12.5, 13.5, color="#D55E00", alpha=0.08)
    axis.set_xticks(range(1, 14))
    axis.set_xlim(0.5, 13.5)
    axis.set_xlabel("held-out strip pH (shaded = extrapolation at the range boundary)")
    axis.set_ylabel("strip ΔE76: M2 - M1")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    axis.set_title(
        f"{REGION_LABELS[region]}: per-strip LOSO difference (GAM tp k=10)", fontsize=10
    )
    figure.tight_layout()
    save_figure(figure, path, dpi, save_pdf)


def plot_sorted_differences(
    strip_differences: pd.DataFrame,
    region: str,
    path: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    part = strip_differences[strip_differences["region"] == region].sort_values("difference")
    figure, axis = plt.subplots(figsize=(10.0, 4.0))
    colors = [PHONE_COLORS[p] for p in part["phone"]]
    x = np.arange(len(part))
    axis.bar(x, part["difference"], color=colors)
    axis.axhline(0.0, color="#888888", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels([f"{int(v)}" for v in part["ph_nominal"]], fontsize=7)
    axis.set_xlabel("held-out strip (label = pH; colour = phone)")
    axis.set_ylabel("strip ΔE76: M2 - M1")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PHONE_COLORS.values()]
    axis.legend(handles, list(PHONE_LABELS.values()), frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    axis.set_title(
        f"{REGION_LABELS[region]}: sorted per-strip differences (GAM tp k=10)", fontsize=10
    )
    figure.tight_layout()
    save_figure(figure, path, dpi, save_pdf)


def main() -> None:
    args = parse_args()
    holdout_dir = resolve(args.holdout_dir)
    figure_dir = holdout_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(holdout_dir / "loso_predictions.csv")
    predictions["primary_spec"] = predictions["primary_spec"].astype(str).str.upper() == "TRUE"

    strip_metrics, image_rows = strip_metrics_from(predictions)
    strip_metrics.to_csv(holdout_dir / "strip_metrics.csv", index=False, encoding="utf-8-sig")

    summary = (
        image_rows.groupby(["basis", "k", "primary_spec", "region", "model"], as_index=False)
        .agg(mean_deltaE76=("deltaE76", "mean"),
             median_deltaE76=("deltaE76", "median"),
             q90_deltaE76=("deltaE76", lambda s: float(s.quantile(0.9))))
    )
    summary.to_csv(holdout_dir / "summary_by_model.csv", index=False, encoding="utf-8-sig")

    by_ph = (
        strip_metrics.groupby(["basis", "k", "region", "model", "ph_nominal"], as_index=False)
        ["mean_deltaE76"].mean()
    )
    by_ph.to_csv(holdout_dir / "summary_by_ph.csv", index=False, encoding="utf-8-sig")

    comparisons = paired_comparisons(strip_metrics, args.bootstrap_reps, args.seed)
    comparisons.to_csv(holdout_dir / "subset_comparisons.csv", index=False, encoding="utf-8-sig")

    rcs_path = resolve(args.rcs_subsets)
    rcs = pd.read_csv(rcs_path, encoding="utf-8-sig") if rcs_path.exists() else None
    consistency = consistency_table(comparisons, rcs)
    consistency.to_csv(holdout_dir / "gam_rcs_consistency.csv", index=False, encoding="utf-8-sig")

    primary = strip_metrics[strip_metrics["primary_spec"]]
    keys = ["region", "strip_id", "phone", "ph_nominal"]
    strip_differences = primary[primary["model"] == "M2"].merge(
        primary[primary["model"] == "M1"], on=keys, suffixes=("_m2", "_m1")
    )
    strip_differences["difference"] = (
        strip_differences["mean_deltaE76_m2"] - strip_differences["mean_deltaE76_m1"]
    )

    for region in REGIONS:
        plot_consistency(consistency, region,
                         figure_dir / f"gam_rcs_consistency_{region}.png",
                         args.dpi, args.save_pdf)
        plot_difference_by_ph(strip_differences, region,
                              figure_dir / f"difference_by_ph_{region}.png",
                              args.dpi, args.save_pdf)
        plot_sorted_differences(strip_differences, region,
                                figure_dir / f"sorted_strip_differences_{region}.png",
                                args.dpi, args.save_pdf)

    view_columns = ["region", "basis", "k", "subset", "comparison", "mean_difference",
                    "bootstrap_ci_lower", "bootstrap_ci_upper", "proportion_numerator_better"]
    primary_view = comparisons[comparisons["primary_spec"]][view_columns]
    sensitivity_view = comparisons[
        (comparisons["comparison"] == "M2 minus M1")
        & comparisons["subset"].isin(["interior_ph_2_12", "boundary_ph_1_13"])
    ][view_columns]

    with pd.option_context("display.width", 220, "display.max_columns", 20,
                           "display.float_format", "{:.4g}".format):
        notes = [
            "GAM leave-one-strip-out summary (script 27)",
            "=" * 80,
            f"bootstrap: {args.bootstrap_reps} reps, seed {args.seed}, strips resampled within phone",
            f"RCS reference: {rcs_path if rcs is not None else 'not found'}",
            "",
            "Mean image Delta E76 by model",
            summary.to_string(index=False),
            "",
            "Primary spec (tp k=10) paired comparisons",
            primary_view.to_string(index=False),
            "",
            "M2 minus M1 across specs (interior and boundary)",
            sensitivity_view.to_string(index=False),
            "",
            "Notes",
            "- Negative differences favour the numerator model.",
            "- pH 1 and 13 strips are extrapolations for the held-out phone curve.",
            "- Predictions exclude the strip random effect (new strip).",
        ]

    (holdout_dir / "analysis_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
