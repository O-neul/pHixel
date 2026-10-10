#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
32_summarize_simulation.py

Summaries and figures for simulations S2 (script 30) and the LOSO
simulations used by S2 and S3 (scripts 31). S1 is summarized by script 29.

S2: identification of phone-specific GAM curves
    - rejection rate of the phone-specific curve terms (2 terms per data set)
        any_unadjusted: min p < 0.05   (nominal family-wise rate ~ 0.0975)
        any_holm:       min p < 0.025  (Holm / Bonferroni for 2 terms, ~ 0.05)
      for M2 with strip random effect and for M2 without it (naive)
    - strip variance reduction 1 - var_s(M2) / var_s(M1)
    - curve RMSE, pointwise and simultaneous 95% band coverage

S2-LOSO and S3 (A = 0: no phone-specific curves; A = 1: real-size smooth curves)
    Per data set: mean strip Delta E76 difference (phone-specific minus
    additive) over pH 2-12 and over pH 1/13, and whether the script 27
    strip bootstrap CI (strips resampled within phone) excludes zero.

Real-data values (scripts 27 and 19) are overlaid for comparison.

Outputs
-------
outputs/simulation/
    s2/s2_summary.csv
    s3/loso_dataset_differences.csv
    s3/loso_summary.csv
    simulation_notes.txt
    figures/s2_identification.png
    figures/s3_loso_differences.png
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
INTERIOR = set(range(2, 13))
BOUNDARY = {1, 13}
PAIRS = [
    ("GAM", "tp k=10", "M2", "M1"),
    ("RCS", "knots=3", "rcs_phone", "rcs_additive"),
    ("RCS", "knots=4", "rcs_phone", "rcs_additive"),
    ("RCS", "knots=5", "rcs_phone", "rcs_additive"),
    ("RCS", "knots=3", "rcs_phone_ridge", "rcs_additive"),
    ("RCS", "knots=4", "rcs_phone_ridge", "rcs_additive"),
    ("RCS", "knots=5", "rcs_phone_ridge", "rcs_additive"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize simulations S2 and S3.")
    parser.add_argument("--sim-dir", default="outputs/simulation")
    parser.add_argument(
        "--real-consistency",
        default="outputs/gam_strip_holdout/gam_rcs_consistency.csv",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def mc_se(rate: float, n: int) -> float:
    return float(np.sqrt(rate * (1 - rate) / n)) if n > 0 else np.nan


def summarize_s2(replicates: pd.DataFrame) -> pd.DataFrame:
    frame = replicates.copy()
    frame["min_p_m2"] = frame[["p_m2_iphone13pro", "p_m2_iphone16pro"]].min(axis=1)
    frame["min_p_naive"] = frame[["p_naive_iphone13pro", "p_naive_iphone16pro"]].min(axis=1)
    frame["variance_reduction"] = 1 - frame["strip_var_M2"] / frame["strip_var_M1"]
    rows = []

    for (channel, A, m), group in frame.groupby(["channel", "A", "strip_multiplier"]):
        n = len(group)
        row = {"channel": channel, "A": A, "strip_multiplier": m, "n": n}
        for label, column in [("m2", "min_p_m2"), ("naive", "min_p_naive")]:
            for rule, threshold in [("any_unadjusted", 0.05), ("any_holm", 0.025)]:
                rate = float((group[column] < threshold).mean())
                row[f"reject_{label}_{rule}"] = rate
                row[f"reject_{label}_{rule}_mcse"] = mc_se(rate, n)
        for phone in ["iphone13pro", "iphone16pro"]:
            row[f"reject_m2_{phone}"] = float((group[f"p_m2_{phone}"] < 0.05).mean())
            row[f"rmse_{phone}"] = float(group[f"rmse_{phone}"].mean())
            row[f"pointwise_cov_{phone}"] = float(group[f"pointwise_cov_{phone}"].mean())
            row[f"simultaneous_cov_{phone}"] = float(group[f"simultaneous_cov_{phone}"].mean())
        row["variance_reduction_mean"] = float(group["variance_reduction"].mean())
        row["variance_reduction_median"] = float(group["variance_reduction"].median())
        row["variance_reduction_q05"] = float(group["variance_reduction"].quantile(0.05))
        row["variance_reduction_q95"] = float(group["variance_reduction"].quantile(0.95))
        rows.append(row)

    return pd.DataFrame(rows)


def bootstrap_ci(values_by_phone: list[np.ndarray], reps: int, rng: np.random.Generator):
    draws = np.empty(reps)
    for r in range(reps):
        sample = [v[rng.integers(0, len(v), size=len(v))] for v in values_by_phone]
        draws[r] = np.mean(np.concatenate(sample))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def dataset_differences(metrics: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for method, variant, numerator, denominator in PAIRS:
        part = metrics[(metrics["method"] == method) & (metrics["variant"] == variant)]
        if part.empty:
            continue
        wide = part.pivot_table(
            index=["A", "dataset", "strip_id", "phone", "ph_nominal"],
            columns="model", values="mean_deltaE76",
        ).reset_index()
        if numerator not in wide or denominator not in wide:
            continue
        wide["difference"] = wide[numerator] - wide[denominator]

        for (A, dataset), group in wide.groupby(["A", "dataset"]):
            for subset, ph_set in [("interior_ph_2_12", INTERIOR), ("boundary_ph_1_13", BOUNDARY)]:
                sub = group[group["ph_nominal"].round().astype(int).isin(ph_set)]
                values = [g["difference"].to_numpy(float) for _, g in sub.groupby("phone")]
                lower, upper = bootstrap_ci(values, reps, rng)
                rows.append({
                    "A": A, "dataset": dataset, "method": method, "variant": variant,
                    "comparison": f"{numerator} minus {denominator}",
                    "subset": subset,
                    "mean_difference": float(sub["difference"].mean()),
                    "ci_lower": lower, "ci_upper": upper,
                    "ci_below_zero": upper < 0, "ci_above_zero": lower > 0,
                })

    return pd.DataFrame(rows)


def summarize_loso(differences: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in differences.groupby(["A", "method", "variant", "comparison", "subset"]):
        n = len(group)
        below = float(group["ci_below_zero"].mean())
        above = float(group["ci_above_zero"].mean())
        rows.append({
            "A": keys[0], "method": keys[1], "variant": keys[2], "comparison": keys[3],
            "subset": keys[4], "n_datasets": n,
            "mean_difference": float(group["mean_difference"].mean()),
            "median_difference": float(group["mean_difference"].median()),
            "q05_difference": float(group["mean_difference"].quantile(0.05)),
            "q95_difference": float(group["mean_difference"].quantile(0.95)),
            "share_ci_below_zero": below, "share_ci_below_zero_mcse": mc_se(below, n),
            "share_ci_above_zero": above, "share_ci_above_zero_mcse": mc_se(above, n),
        })
    return pd.DataFrame(rows)


def plot_s2(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    a_part = summary[summary["channel"] == "a"]
    colors = {0.5: "#56B4E9", 1.0: "#0072B2", 2.0: "#000000"}
    figure, axes = plt.subplots(1, 4, figsize=(17.0, 4.0))

    for m, color in colors.items():
        rows = a_part[np.isclose(a_part["strip_multiplier"], m)].sort_values("A")
        axes[0].errorbar(rows["A"], rows["reject_m2_any_holm"],
                         yerr=1.96 * rows["reject_m2_any_holm_mcse"],
                         color=color, marker="o", capsize=2, label=f"M2, strip SD x{m:g}")
        axes[0].errorbar(rows["A"], rows["reject_naive_any_holm"],
                         yerr=1.96 * rows["reject_naive_any_holm_mcse"],
                         color=color, marker="s", linestyle="--", capsize=2,
                         label=f"no strip RE, x{m:g}")
        axes[1].plot(rows["A"], rows["variance_reduction_mean"], color=color, marker="o",
                     label=f"strip SD x{m:g}")
        axes[1].fill_between(rows["A"], rows["variance_reduction_q05"],
                             rows["variance_reduction_q95"], color=color, alpha=0.12)
        axes[2].plot(rows["A"], rows["simultaneous_cov_iphone13pro"], color=color, marker="o",
                     label=f"iPhone 13 Pro, x{m:g}")
        axes[2].plot(rows["A"], rows["simultaneous_cov_iphone16pro"], color=color, marker="^",
                     linestyle=":", label=f"iPhone 16 Pro, x{m:g}")
        axes[3].plot(rows["A"], rows["pointwise_cov_iphone13pro"], color=color, marker="o")
        axes[3].plot(rows["A"], rows["pointwise_cov_iphone16pro"], color=color, marker="^",
                     linestyle=":")

    axes[0].axhline(0.05, color="#D55E00", linestyle="--", linewidth=0.9)
    axes[0].set_title("Rejection of phone-specific curves\n(Holm over 2 phones; A = 0 is the null)",
                      fontsize=9)
    axes[0].set_ylabel("rate")
    axes[1].axhline(0.0, color="#888888", linewidth=0.8)
    axes[1].set_title("Strip variance absorbed by M2\n(mean, 5-95% band)", fontsize=9)
    axes[2].axhline(0.95, color="#D55E00", linestyle="--", linewidth=0.9)
    axes[2].set_title("Simultaneous 95% band covers the true curve", fontsize=9)
    axes[3].axhline(0.95, color="#D55E00", linestyle="--", linewidth=0.9)
    axes[3].set_title("Pointwise 95% band coverage (mean over grid)", fontsize=9)

    for axis in axes:
        axis.set_xlabel("A (phone-specific curve size; 1 = real data)")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=6)
    axes[1].legend(frameon=False, fontsize=7)
    axes[2].legend(frameon=False, fontsize=6)
    figure.suptitle("S2: GAM identification of phone-specific pH curves (a*, 500 data sets per cell)",
                    fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_loso(differences: pd.DataFrame, real: pd.DataFrame | None, path: Path, dpi: int) -> None:
    labels = []
    for method, variant, numerator, _ in PAIRS:
        name = "GAM" if method == "GAM" else ("RCS ridge" if "ridge" in numerator else "RCS")
        labels.append((method, variant, numerator, f"{name}: {variant}"))

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 8.0))
    for row, A in enumerate([0, 1]):
        for column, subset in enumerate(["interior_ph_2_12", "boundary_ph_1_13"]):
            axis = axes[row, column]
            data, names, real_points = [], [], []
            for method, variant, numerator, label in labels:
                part = differences[
                    (differences["A"] == A) & (differences["method"] == method)
                    & (differences["variant"] == variant)
                    & (differences["comparison"].str.startswith(numerator + " "))
                    & (differences["subset"] == subset)
                ]
                if part.empty:
                    continue
                data.append(part["mean_difference"].to_numpy())
                names.append(label)
                if real is not None:
                    method_name = label.split(":")[0]
                    hit = real[(real["region"] == "contrast") & (real["subset"] == subset)
                               & (real["method"] == method_name)
                               & (real["variant"] == variant)]
                    real_points.append(float(hit["mean_difference"].iloc[0]) if len(hit) else np.nan)
            positions = np.arange(len(data))
            axis.boxplot(data, positions=positions, vert=False, widths=0.6, showfliers=False)
            if A == 1 and real_points:
                axis.scatter(real_points, positions, color="#D55E00", marker="D", zorder=3,
                             label="real data")
                axis.legend(frameon=False, fontsize=7, loc="lower right")
            axis.axvline(0.0, color="#0072B2", linestyle="--", linewidth=0.9)
            axis.set_yticks(positions)
            axis.set_yticklabels(names, fontsize=8)
            axis.set_title(
                f"{'A = 0 (no phone-specific curves)' if A == 0 else 'A = 1 (real-size smooth curves)'}"
                f" | {'pH 2-12' if subset.startswith('interior') else 'pH 1, 13'}",
                fontsize=9,
            )
            axis.set_xlabel("per-data-set mean strip ΔE76 difference (phone-specific - additive)")
            axis.grid(axis="x", alpha=0.25)

    figure.suptitle("S2/S3: leave-one-strip-out gain of phone-specific models on simulated data "
                    "(200 data sets each; < 0 favours phone-specific)", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    sim_dir = resolve(args.sim_dir)
    (sim_dir / "figures").mkdir(parents=True, exist_ok=True)
    notes = ["Simulation S2 / S3 summary (script 32)", "=" * 80]

    replicates_path = sim_dir / "s2" / "s2_replicates.csv"
    if replicates_path.exists():
        s2 = summarize_s2(pd.read_csv(replicates_path))
        s2.to_csv(sim_dir / "s2" / "s2_summary.csv", index=False, encoding="utf-8-sig")
        plot_s2(s2, sim_dir / "figures" / "s2_identification.png", args.dpi)
        view = ["channel", "A", "strip_multiplier", "n",
                "reject_m2_any_unadjusted", "reject_m2_any_holm",
                "reject_naive_any_unadjusted", "reject_naive_any_holm",
                "variance_reduction_mean", "simultaneous_cov_iphone13pro",
                "simultaneous_cov_iphone16pro", "pointwise_cov_iphone13pro",
                "pointwise_cov_iphone16pro"]
        with pd.option_context("display.width", 250, "display.max_columns", 30,
                               "display.float_format", "{:.3f}".format):
            notes += ["", "S2 identification (rejection rates, variance reduction, coverage)",
                      s2[view].to_string(index=False)]

    metric_frames = []
    for name in ["gam_strip_metrics.csv", "rcs_strip_metrics.csv"]:
        path = sim_dir / "s3" / name
        if path.exists():
            metric_frames.append(pd.read_csv(path, encoding="utf-8-sig"))

    if metric_frames:
        metrics = pd.concat(metric_frames, ignore_index=True)
        differences = dataset_differences(metrics, args.bootstrap_reps, args.seed)
        differences.to_csv(sim_dir / "s3" / "loso_dataset_differences.csv",
                           index=False, encoding="utf-8-sig")
        loso = summarize_loso(differences)
        loso.to_csv(sim_dir / "s3" / "loso_summary.csv", index=False, encoding="utf-8-sig")

        real_path = resolve(args.real_consistency)
        real = pd.read_csv(real_path, encoding="utf-8-sig") if real_path.exists() else None
        plot_loso(differences, real, sim_dir / "figures" / "s3_loso_differences.png", args.dpi)

        with pd.option_context("display.width", 250, "display.max_columns", 30,
                               "display.float_format", "{:.3f}".format):
            notes += ["", "S2-LOSO / S3 (per-data-set strip bootstrap; share of data sets whose CI "
                      "is below / above zero)",
                      loso.drop(columns=["share_ci_below_zero_mcse", "share_ci_above_zero_mcse"])
                      .to_string(index=False)]

    (sim_dir / "simulation_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
