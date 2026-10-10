#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
28_summarize_inverse_ph.py

Summaries, paired strip-level bootstrap and figures for the inverse pH
predictions of 28_gam_inverse_ph_prediction.R.

Strip-level metrics (five light images per strip):
- mean absolute pH error
- exact accuracy: estimated pH rounded to the nearest integer equals true pH
- within-1 accuracy: |estimated - true| <= 1
- binary accuracy: acidic (< 7) vs alkaline (> 7); true pH 7 strips excluded

Paired comparisons and bootstrap follow scripts 19 and 27:
numerator minus denominator per strip; strips resampled within phone;
3000 replicates; seed 42; subsets all / pH 1,13 / pH 2-12 / pH 3-11.
Negative differences favour the numerator for error; positive differences
favour the numerator for accuracy.

The colour gradient |d Lab / d pH| of the script 26 Contrast curves is used
to explain how a Delta E gain translates into a pH gain:
    pH error ~ Lab error / colour gradient.

Outputs
-------
outputs/gam_inverse_ph/
    strip_metrics.csv
    summary_by_model.csv
    summary_by_ph.csv
    subset_comparisons.csv
    deltaE_vs_ph_gain.csv
    analysis_notes.txt
    figures/
        estimated_vs_true_<region>.png
        abs_error_by_ph.png
        ph_gain_forest.png
        deltaE_vs_ph_gain_contrast.png
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
MODELS = ["M0", "M1", "M2"]
REGION_LABELS = {
    "pad": "Pad",
    "bg": "Background (negative control)",
    "contrast": "Contrast (Pad - Background)",
}
MODEL_LABELS = {
    "M0": "M0: shared curve",
    "M1": "M1: + phone offset",
    "M2": "M2: phone-specific curves",
}
MODEL_COLORS = {"M0": "#999999", "M1": "#0072B2", "M2": "#D55E00"}
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
COMPARISONS = [("M2", "M1"), ("M1", "M0"), ("M2", "M0")]
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
METRICS = {
    "mean_abs_error_ph": "lower",
    "exact_accuracy": "higher",
    "within1_accuracy": "higher",
    "binary_accuracy": "higher",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize inverse pH prediction.")
    parser.add_argument("--inverse-dir", default="outputs/gam_inverse_ph")
    parser.add_argument("--holdout-dir", default="outputs/gam_strip_holdout")
    parser.add_argument("--curve-dir", default="outputs/gam_curves")
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


def bootstrap_difference(
    frame: pd.DataFrame,
    value_column: str,
    reps: int,
    seed: int,
) -> tuple[float, float, float]:
    """Same resampling as scripts 19 and 27: strips within phone."""
    values_by_phone = [
        group[value_column].dropna().to_numpy(float)
        for _, group in frame.groupby("phone", sort=True)
    ]
    values_by_phone = [v for v in values_by_phone if len(v) > 0]
    observed = float(np.mean(np.concatenate(values_by_phone)))
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)

    for r in range(reps):
        sample = [v[rng.integers(0, len(v), size=len(v))] for v in values_by_phone]
        draws[r] = np.mean(np.concatenate(sample))

    return observed, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def add_image_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["exact_correct"] = (
        np.floor(frame["estimated_ph"] + 0.5) == frame["true_ph"]
    ).astype(float)
    frame["within1_correct"] = (frame["abs_error_ph"] <= 1.0 + 1e-9).astype(float)
    binary = np.where(frame["true_ph"] == 7, np.nan,
                      ((frame["estimated_ph"] < 7) == (frame["true_ph"] < 7))
                      & (frame["estimated_ph"] != 7))
    frame["binary_correct"] = pd.to_numeric(pd.Series(binary, index=frame.index),
                                            errors="coerce")
    return frame


def strip_metrics_from(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            ["basis", "k", "primary_spec", "region", "model",
             "strip_id", "phone", "true_ph"],
            as_index=False,
        )
        .agg(
            mean_abs_error_ph=("abs_error_ph", "mean"),
            mean_error_ph=("error_ph", "mean"),
            exact_accuracy=("exact_correct", "mean"),
            within1_accuracy=("within1_correct", "mean"),
            binary_accuracy=("binary_correct", "mean"),
            mean_min_deltaE76=("min_deltaE76", "mean"),
        )
    )


def paired_comparisons(strips: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rows = []
    keys = ["basis", "k", "primary_spec", "region", "strip_id", "phone", "true_ph"]

    for numerator, denominator in COMPARISONS:
        merged = strips[strips["model"] == numerator].merge(
            strips[strips["model"] == denominator],
            on=keys, suffixes=("_num", "_den"), validate="one_to_one",
        )

        for metric, better in METRICS.items():
            merged["difference"] = merged[f"{metric}_num"] - merged[f"{metric}_den"]

            for (basis, k, primary, region), group in merged.groupby(
                ["basis", "k", "primary_spec", "region"], sort=True
            ):
                for subset, ph_set in SUBSETS.items():
                    part = group[group["true_ph"].round().astype(int).isin(ph_set)]
                    part = part.dropna(subset=["difference"])
                    if part.empty:
                        continue
                    mean, lower, upper = bootstrap_difference(part, "difference", reps, seed)
                    improved = part["difference"] < 0 if better == "lower" else part["difference"] > 0
                    rows.append({
                        "basis": basis,
                        "k": int(k),
                        "primary_spec": bool(primary),
                        "region": region,
                        "subset": subset,
                        "metric": metric,
                        "better_direction": better,
                        "comparison": f"{numerator} minus {denominator}",
                        "n_strips": int(len(part)),
                        "mean_numerator": float(part[f"{metric}_num"].mean()),
                        "mean_denominator": float(part[f"{metric}_den"].mean()),
                        "mean_difference": mean,
                        "bootstrap_ci_lower": lower,
                        "bootstrap_ci_upper": upper,
                        "proportion_numerator_better": float(improved.mean()),
                        "proportion_equal": float((part["difference"] == 0).mean()),
                    })

    return pd.DataFrame(rows)


def colour_gradient(curve_dir: Path) -> pd.DataFrame | None:
    path = curve_dir / "phone_curves.csv"
    if not path.exists():
        return None

    curves = pd.read_csv(path)
    curves["primary_spec"] = curves["primary_spec"].astype(str).str.upper() == "TRUE"
    curves = curves[curves["primary_spec"] & (curves["region"] == "contrast")]
    rows = []

    for phone, group in curves.groupby("phone"):
        wide = group.pivot_table(index="ph", columns="channel", values="fit").sort_index()
        derivative = np.gradient(wide[["L", "a", "b"]].to_numpy(), wide.index.to_numpy(), axis=0)
        speed = np.linalg.norm(derivative, axis=1)
        rows.append(pd.DataFrame({"phone": phone, "ph": wide.index, "gradient": speed}))

    return pd.concat(rows, ignore_index=True)


def plot_estimated_vs_true(frame: pd.DataFrame, region: str, path: Path,
                           dpi: int, save_pdf: bool) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), sharex=True, sharey=True)
    rng = np.random.default_rng(0)

    for axis, model in zip(axes, MODELS):
        part = frame[frame["model"] == model]
        for phone, color in PHONE_COLORS.items():
            rows = part[part["phone"] == phone]
            axis.scatter(
                rows["true_ph"] + rng.uniform(-0.18, 0.18, len(rows)),
                rows["estimated_ph"], s=12, alpha=0.7, color=color,
                label=PHONE_LABELS[phone],
            )
        axis.plot([1, 13], [1, 13], color="#000000", linewidth=0.8)
        mae = part["abs_error_ph"].mean()
        axis.set_title(f"{MODEL_LABELS[model]}\nmean |error| = {mae:.2f} pH", fontsize=9)
        axis.set_xlabel("true pH (jittered)")
        axis.set_xticks(range(1, 14))
        axis.set_yticks(range(1, 14))
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("estimated pH")
    axes[0].legend(frameon=False, fontsize=7)
    figure.suptitle(
        f"{REGION_LABELS[region]}: inverse pH prediction of held-out strips (GAM tp k=10)",
        fontsize=10,
    )
    figure.tight_layout()
    save_figure(figure, path, dpi, save_pdf)


def plot_abs_error_by_ph(strips: pd.DataFrame, gradient: pd.DataFrame | None,
                         path: Path, dpi: int, save_pdf: bool) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    for axis, region in zip(axes, REGIONS):
        part = strips[strips["region"] == region]
        for model in MODELS:
            line = part[part["model"] == model].groupby("true_ph")["mean_abs_error_ph"].mean()
            axis.plot(line.index, line.values, marker="o", markersize=3,
                      color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        axis.set_title(REGION_LABELS[region], fontsize=10)
        axis.set_xlabel("true pH")
        axis.set_xticks(range(1, 14))
        axis.grid(alpha=0.25)

        if region == "contrast" and gradient is not None:
            twin = axis.twinx()
            mean_gradient = gradient.groupby("ph")["gradient"].mean()
            twin.plot(mean_gradient.index, mean_gradient.values, color="#009E73",
                      linestyle=":", linewidth=1.5, label="|dLab/dpH| (right axis)")
            twin.set_ylabel("colour gradient |dLab/dpH|", color="#009E73")
            twin.legend(frameon=False, fontsize=7, loc="upper right")

    axes[0].set_ylabel("mean |pH error| (strip mean over 5 lights)")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    figure.suptitle("Inverse pH prediction error by true pH (GAM tp k=10)", fontsize=10)
    figure.tight_layout()
    save_figure(figure, path, dpi, save_pdf)


def plot_gain_forest(comparisons: pd.DataFrame, path: Path, dpi: int, save_pdf: bool) -> None:
    part = comparisons[
        (comparisons["metric"] == "mean_abs_error_ph")
        & (comparisons["region"].isin(["contrast", "pad"]))
        & (comparisons["subset"] != "all_ph")
    ]
    comparisons_order = ["M2 minus M1", "M1 minus M0", "M2 minus M0"]
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 5.0), sharey=True)

    for axis, subset in zip(axes, ["interior_ph_2_12", "strict_interior_ph_3_11", "boundary_ph_1_13"]):
        rows = part[part["subset"] == subset].copy()
        rows["label"] = (rows["region"] + " | " + rows["comparison"] + " | "
                         + rows["basis"] + " k=" + rows["k"].astype(str))
        rows["order"] = (rows["region"].map({"contrast": 0, "pad": 1}) * 100
                         + rows["comparison"].map({c: i for i, c in enumerate(comparisons_order)}) * 10
                         + (~rows["primary_spec"]).astype(int))
        rows = rows.sort_values(["order", "k"], ascending=False).reset_index(drop=True)

        for y, row in rows.iterrows():
            color = {"M2 minus M1": "#D55E00", "M1 minus M0": "#0072B2",
                     "M2 minus M0": "#000000"}[row["comparison"]]
            axis.errorbar(
                row["mean_difference"], y,
                xerr=[[row["mean_difference"] - row["bootstrap_ci_lower"]],
                      [row["bootstrap_ci_upper"] - row["mean_difference"]]],
                fmt="o" if row["primary_spec"] else "s", color=color,
                markersize=6 if row["primary_spec"] else 3,
                markerfacecolor=color if row["primary_spec"] else "white",
                capsize=2, linewidth=1.0,
            )
        axis.axvline(0.0, color="#888888", linestyle="--", linewidth=0.9)
        axis.set_title(SUBSET_LABELS[subset], fontsize=10)
        axis.set_xlabel("difference in mean |pH error|")
        axis.grid(axis="x", alpha=0.25)
        axis.set_yticks(range(len(rows)))
        axis.set_yticklabels(rows["label"], fontsize=6)

    figure.suptitle(
        "Inverse pH error: paired strip differences (< 0 favours the first model; "
        "filled = tp k=10; 95% strip bootstrap CI)",
        fontsize=10,
    )
    figure.tight_layout()
    save_figure(figure, path, dpi, save_pdf)


def plot_deltae_vs_ph_gain(link: pd.DataFrame, path: Path, dpi: int, save_pdf: bool) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 5.0))

    for phone, color in PHONE_COLORS.items():
        rows = link[link["phone"] == phone]
        boundary = rows["true_ph"].isin([1, 13])
        axis.scatter(rows.loc[~boundary, "deltaE_gain"], rows.loc[~boundary, "ph_gain"],
                     color=color, s=26, label=PHONE_LABELS[phone])
        axis.scatter(rows.loc[boundary, "deltaE_gain"], rows.loc[boundary, "ph_gain"],
                     color=color, s=40, marker="x")

    axis.axhline(0.0, color="#888888", linewidth=0.8)
    axis.axvline(0.0, color="#888888", linewidth=0.8)
    axis.set_xlabel("forward ΔE76 difference, M2 - M1 (script 27)")
    axis.set_ylabel("inverse |pH error| difference, M2 - M1")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8, title="x = pH 1 or 13", title_fontsize=7)
    axis.set_title("Contrast: does a colour gain become a pH gain? (per strip, tp k=10)",
                   fontsize=9)
    figure.tight_layout()
    save_figure(figure, path, dpi, save_pdf)


def main() -> None:
    args = parse_args()
    inverse_dir = resolve(args.inverse_dir)
    figure_dir = inverse_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(inverse_dir / "inverse_predictions.csv")
    frame["primary_spec"] = frame["primary_spec"].astype(str).str.upper() == "TRUE"
    frame = add_image_metrics(frame)

    strips = strip_metrics_from(frame)
    strips.to_csv(inverse_dir / "strip_metrics.csv", index=False, encoding="utf-8-sig")

    summary = (
        frame.assign(interior=frame["true_ph"].between(2, 12))
        .groupby(["basis", "k", "primary_spec", "region", "model", "interior"], as_index=False)
        .agg(mean_abs_error_ph=("abs_error_ph", "mean"),
             median_abs_error_ph=("abs_error_ph", "median"),
             q90_abs_error_ph=("abs_error_ph", lambda s: float(s.quantile(0.9))),
             exact_accuracy=("exact_correct", "mean"),
             within1_accuracy=("within1_correct", "mean"),
             binary_accuracy=("binary_correct", "mean"))
    )
    summary.to_csv(inverse_dir / "summary_by_model.csv", index=False, encoding="utf-8-sig")

    by_ph = (
        strips.groupby(["basis", "k", "region", "model", "true_ph"], as_index=False)
        [["mean_abs_error_ph", "mean_error_ph", "exact_accuracy"]].mean()
    )
    by_ph.to_csv(inverse_dir / "summary_by_ph.csv", index=False, encoding="utf-8-sig")

    comparisons = paired_comparisons(strips, args.bootstrap_reps, args.seed)
    comparisons.to_csv(inverse_dir / "subset_comparisons.csv", index=False, encoding="utf-8-sig")

    primary_frame = frame[frame["primary_spec"]]
    primary_strips = strips[strips["primary_spec"]]
    gradient = colour_gradient(resolve(args.curve_dir))

    for region in REGIONS:
        plot_estimated_vs_true(primary_frame[primary_frame["region"] == region], region,
                               figure_dir / f"estimated_vs_true_{region}.png",
                               args.dpi, args.save_pdf)
    plot_abs_error_by_ph(primary_strips, gradient, figure_dir / "abs_error_by_ph.png",
                         args.dpi, args.save_pdf)
    plot_gain_forest(comparisons, figure_dir / "ph_gain_forest.png", args.dpi, args.save_pdf)

    link = None
    holdout_path = resolve(args.holdout_dir) / "strip_metrics.csv"
    if holdout_path.exists():
        holdout = pd.read_csv(holdout_path, encoding="utf-8-sig")
        holdout["primary_spec"] = holdout["primary_spec"].astype(str).str.upper() == "TRUE"
        forward = holdout[holdout["primary_spec"] & (holdout["region"] == "contrast")]
        forward = forward.pivot_table(index=["strip_id", "phone", "ph_nominal"],
                                      columns="model", values="mean_deltaE76").reset_index()
        forward["deltaE_gain"] = forward["M2"] - forward["M1"]
        inverse = primary_strips[primary_strips["region"] == "contrast"].pivot_table(
            index=["strip_id", "phone", "true_ph"], columns="model",
            values="mean_abs_error_ph").reset_index()
        inverse["ph_gain"] = inverse["M2"] - inverse["M1"]
        link = forward[["strip_id", "deltaE_gain"]].merge(
            inverse[["strip_id", "phone", "true_ph", "ph_gain"]], on="strip_id")
        link.to_csv(inverse_dir / "deltaE_vs_ph_gain.csv", index=False, encoding="utf-8-sig")
        plot_deltae_vs_ph_gain(link, figure_dir / "deltaE_vs_ph_gain_contrast.png",
                               args.dpi, args.save_pdf)

    view = ["region", "subset", "metric", "comparison", "mean_numerator", "mean_denominator",
            "mean_difference", "bootstrap_ci_lower", "bootstrap_ci_upper",
            "proportion_numerator_better", "proportion_equal"]
    primary_view = comparisons[
        comparisons["primary_spec"] & comparisons["subset"].isin(
            ["interior_ph_2_12", "boundary_ph_1_13", "all_ph"])
    ][view]
    sensitivity_view = comparisons[
        (comparisons["comparison"] == "M2 minus M1")
        & (comparisons["metric"] == "mean_abs_error_ph")
        & (comparisons["subset"] == "interior_ph_2_12")
    ][["region", "basis", "k", "mean_difference", "bootstrap_ci_lower", "bootstrap_ci_upper",
       "proportion_numerator_better"]]

    with pd.option_context("display.width", 240, "display.max_columns", 20,
                           "display.max_rows", 500,
                           "display.float_format", "{:.4g}".format):
        notes = [
            "GAM inverse pH prediction summary (script 28)",
            "=" * 80,
            f"bootstrap: {args.bootstrap_reps} reps, seed {args.seed}, strips resampled within phone",
            "",
            "Image-level summary, primary spec (interior = pH 2-12)",
            summary[summary["primary_spec"]].to_string(index=False),
            "",
            "Primary-spec paired strip comparisons",
            primary_view.to_string(index=False),
            "",
            "M2 minus M1, mean |pH error|, pH 2-12, all specs",
            sensitivity_view.to_string(index=False),
        ]
        if link is not None:
            interior = link[~link["true_ph"].isin([1, 13])]
            notes += [
                "",
                "Contrast per-strip link between forward Delta E gain and inverse pH gain",
                f"  correlation (pH 2-12): {interior['deltaE_gain'].corr(interior['ph_gain']):.3f}",
            ]
        notes += [
            "",
            "Notes",
            "- Estimated pH is restricted to [1, 13]; boundary errors cannot overshoot.",
            "- M1 and M2 assume the phone model is known (device calibration scenario).",
            "- Background is a negative control without pH signal.",
        ]

    (inverse_dir / "analysis_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
