# -*- coding: utf-8 -*-
"""
13_plot_color_profiles.py

Purpose
-------
Create exploratory and paper-style color profile plots from
analysis_factorial_table.csv.

Outputs
-------
outputs/color_profiles/
    exploratory/
    paper/

Main plot families
------------------
1) Mean profiles by phone for:
   - pad_L, pad_a, pad_b
   - bg_L, bg_a, bg_b
   - contrast_L, contrast_a, contrast_b

2) Interaction residual plots:
   residual = mean(phone,pH) - phone_mean - pH_mean + grand_mean

3) Light-dispersion plots across pH for each Lab triplet region:
   - RMS distance to centroid
   - Mean pairwise DeltaE
   - Max pairwise DeltaE

4) Pairwise light-separation plots across pH:
   - For each light pair, mean DeltaE across phones
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REGION_FEATURES = {
    "pad": ["pad_L", "pad_a", "pad_b"],
    "bg": ["bg_L", "bg_a", "bg_b"],
    "contrast": ["contrast_L", "contrast_a", "contrast_b"],
}

REGION_TITLES = {
    "pad": "Pad",
    "bg": "Background",
    "contrast": "Pad - Background Contrast",
}

AXIS_LABELS = {
    "pad_L": "CIE L*",
    "pad_a": "CIE a*",
    "pad_b": "CIE b*",
    "bg_L": "CIE L*",
    "bg_a": "CIE a*",
    "bg_b": "CIE b*",
    "contrast_L": "CIE L*",
    "contrast_a": "CIE a*",
    "contrast_b": "CIE b*",
}

FEATURE_TITLES = {
    "pad_L": "Pad CIE L* by Phone",
    "pad_a": "Pad CIE a* by Phone",
    "pad_b": "Pad CIE b* by Phone",
    "bg_L": "Background CIE L* by Phone",
    "bg_a": "Background CIE a* by Phone",
    "bg_b": "Background CIE b* by Phone",
    "contrast_L": "Pad - Background Contrast CIE L* by Phone",
    "contrast_a": "Pad - Background Contrast CIE a* by Phone",
    "contrast_b": "Pad - Background Contrast CIE b* by Phone",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-table",
        type=str,
        default="./outputs/factorial_analysis/analysis_factorial_table.csv",
        help="Path to analysis_factorial_table.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/color_profiles",
        help="Output directory",
    )
    parser.add_argument(
        "--plot-style",
        type=str,
        choices=["explore", "paper", "both"],
        default="both",
        help="Generate exploratory plots, paper plots, or both",
    )
    parser.add_argument(
        "--regions",
        nargs="*",
        default=["pad", "bg", "contrast"],
        choices=["pad", "bg", "contrast"],
        help="Regions to plot",
    )
    parser.add_argument(
        "--mark-ph",
        nargs="*",
        type=float,
        default=[],
        help="Optional pH values to mark with vertical lines (e.g. --mark-ph 8 12)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Saved figure DPI",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def sort_ph(values):
    vals = sorted(pd.to_numeric(pd.Series(values), errors="coerce").dropna().unique())
    return vals


def format_ph_tick(v):
    if float(v).is_integer():
        return str(int(v))
    return str(v)


def light_pairs(lights):
    return list(itertools.combinations(sorted(lights), 2))


def lab_distance(v1, v2):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))


def add_mark_lines(ax, mark_ph):
    for ph in mark_ph:
        ax.axvline(ph, linewidth=1.0, alpha=0.5)


def style_to_subdir(style: str) -> str:
    return "exploratory" if style == "explore" else "paper"


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"phone_label", "light_label", "ph_nominal"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["ph_nominal"] = pd.to_numeric(out["ph_nominal"], errors="coerce")
    out = out.dropna(subset=["ph_nominal"]).copy()
    return out


def profile_summary(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    summary = (
        df.groupby(["phone_label", "ph_nominal"], as_index=False)[feature]
        .agg(mean="mean", min="min", max="max")
        .sort_values(["phone_label", "ph_nominal"])
    )
    return summary


def plot_profile(df, feature, output_path: Path, style: str, mark_ph=None, dpi=220):
    mark_ph = mark_ph or []
    summary = profile_summary(df, feature)
    phones = sorted(summary["phone_label"].unique())
    ph_vals = sort_ph(summary["ph_nominal"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    if style == "explore":
        for phone in phones:
            sub = summary[summary["phone_label"] == phone].sort_values("ph_nominal")
            x = sub["ph_nominal"].to_numpy(dtype=float)
            y = sub["mean"].to_numpy(dtype=float)
            y_min = sub["min"].to_numpy(dtype=float)
            y_max = sub["max"].to_numpy(dtype=float)
            yerr = np.vstack([y - y_min, y_max - y])

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linewidth=1.8,
                capsize=4,
                label=phone,
            )

        ax.set_title(f"{FEATURE_TITLES[feature]}\nMean across five lights; uncertainty=range")
    else:
        # raw points + range band + mean line
        phones_in_order = sorted(df["phone_label"].unique())
        phone_to_offset = {}
        offsets = np.linspace(-0.12, 0.12, num=max(len(phones_in_order), 1))
        for phone, off in zip(phones_in_order, offsets):
            phone_to_offset[phone] = off

        for phone in phones:
            raw = df[df["phone_label"] == phone].copy()
            raw = raw.sort_values(["ph_nominal", "light_label"])
            summary_phone = summary[summary["phone_label"] == phone].sort_values("ph_nominal")

            x = summary_phone["ph_nominal"].to_numpy(dtype=float)
            y = summary_phone["mean"].to_numpy(dtype=float)
            y_min = summary_phone["min"].to_numpy(dtype=float)
            y_max = summary_phone["max"].to_numpy(dtype=float)

            line = ax.plot(x, y, marker="o", linewidth=2.2, label=phone)[0]
            color = line.get_color()

            ax.fill_between(x, y_min, y_max, alpha=0.12, color=color)

            for ph in ph_vals:
                raw_sub = raw[raw["ph_nominal"] == ph]
                if raw_sub.empty:
                    continue

                raw_y = raw_sub[feature].to_numpy(dtype=float)
                light_count = len(raw_y)
                jit = np.linspace(-0.05, 0.05, num=light_count) if light_count > 1 else np.array([0.0])
                x_raw = np.full(light_count, ph + phone_to_offset[phone]) + jit

                ax.scatter(
                    x_raw,
                    raw_y,
                    alpha=0.45,
                    s=24,
                    color=color,
                    edgecolors="none",
                )

        ax.set_title(f"{FEATURE_TITLES[feature]}")

    add_mark_lines(ax, mark_ph)
    ax.set_xlabel("Nominal pH")
    ax.set_ylabel(AXIS_LABELS[feature])
    ax.set_xticks(ph_vals)
    ax.set_xticklabels([format_ph_tick(v) for v in ph_vals])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def compute_interaction_residual(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    mean_table = (
        df.groupby(["phone_label", "ph_nominal"], as_index=False)[feature]
        .mean()
        .rename(columns={feature: "mean_value"})
    )

    grand_mean = mean_table["mean_value"].mean()
    phone_means = mean_table.groupby("phone_label")["mean_value"].mean().to_dict()
    ph_means = mean_table.groupby("ph_nominal")["mean_value"].mean().to_dict()

    mean_table["interaction_residual"] = mean_table.apply(
        lambda row: row["mean_value"] - phone_means[row["phone_label"]] - ph_means[row["ph_nominal"]] + grand_mean,
        axis=1,
    )
    return mean_table


def plot_interaction_residual(df, feature, output_path: Path, mark_ph=None, dpi=220):
    mark_ph = mark_ph or []
    res_df = compute_interaction_residual(df, feature)
    phones = sorted(res_df["phone_label"].unique())
    ph_vals = sort_ph(res_df["ph_nominal"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for phone in phones:
        sub = res_df[res_df["phone_label"] == phone].sort_values("ph_nominal")
        x = sub["ph_nominal"].to_numpy(dtype=float)
        y = sub["interaction_residual"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.0, label=phone)

    add_mark_lines(ax, mark_ph)
    ax.axhline(0.0, linewidth=1.2, alpha=0.7)

    title_base = FEATURE_TITLES[feature].replace(" by Phone", "")
    ax.set_title(
        f"{title_base}: pH x Phone Interaction Residual\n"
        f"Zero indicates an additive, parallel profile"
    )
    ax.set_xlabel("Nominal pH")
    ax.set_ylabel("Interaction residual")
    ax.set_xticks(ph_vals)
    ax.set_xticklabels([format_ph_tick(v) for v in ph_vals])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def compute_dispersion_stats(df: pd.DataFrame, region: str) -> pd.DataFrame:
    feature_cols = REGION_FEATURES[region]
    rows = []

    grouped = df.groupby(["phone_label", "ph_nominal"])
    for (phone, ph), sub in grouped:
        vals = sub[feature_cols].to_numpy(dtype=float)
        if vals.shape[0] < 2:
            continue

        centroid = vals.mean(axis=0)
        sq_dists = np.sum((vals - centroid) ** 2, axis=1)
        rms_dist = float(np.sqrt(np.mean(sq_dists)))

        pair_dists = []
        for i, j in itertools.combinations(range(len(vals)), 2):
            pair_dists.append(float(np.sqrt(np.sum((vals[i] - vals[j]) ** 2))))

        rows.append(
            {
                "phone_label": phone,
                "ph_nominal": ph,
                "rms_distance_to_centroid": rms_dist,
                "mean_pairwise_deltaE": float(np.mean(pair_dists)),
                "max_pairwise_deltaE": float(np.max(pair_dists)),
            }
        )

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return stats_df

    out = (
        stats_df.groupby("ph_nominal", as_index=False)[
            ["rms_distance_to_centroid", "mean_pairwise_deltaE", "max_pairwise_deltaE"]
        ]
        .mean()
        .sort_values("ph_nominal")
    )
    return out


def plot_dispersion(df, region, output_path: Path, mark_ph=None, dpi=220):
    mark_ph = mark_ph or []
    stats_df = compute_dispersion_stats(df, region)
    if stats_df.empty:
        return

    x = stats_df["ph_nominal"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, stats_df["rms_distance_to_centroid"], marker="o", linewidth=2.0, label="RMS distance to pH centroid")
    ax.plot(x, stats_df["mean_pairwise_deltaE"], marker="o", linewidth=2.0, label="Mean pairwise ΔE")
    ax.plot(x, stats_df["max_pairwise_deltaE"], marker="o", linewidth=2.0, label="Maximum pairwise ΔE")

    add_mark_lines(ax, mark_ph)
    ax.set_title(f"{REGION_TITLES[region]} Light Lab Dispersion Across pH")
    ax.set_xlabel("Nominal pH")
    ax.set_ylabel("Lab separation")
    ax.set_xticks(x)
    ax.set_xticklabels([format_ph_tick(v) for v in x])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def compute_pairwise_light_separation(df: pd.DataFrame, region: str) -> pd.DataFrame:
    feature_cols = REGION_FEATURES[region]
    lights = sorted(df["light_label"].unique())
    pairs = light_pairs(lights)

    rows = []

    grouped = df.groupby(["phone_label", "ph_nominal"])
    for (phone, ph), sub in grouped:
        sub = sub.set_index("light_label")
        available_lights = set(sub.index)
        for l1, l2 in pairs:
            if l1 not in available_lights or l2 not in available_lights:
                continue

            v1 = sub.loc[l1, feature_cols].to_numpy(dtype=float)
            v2 = sub.loc[l2, feature_cols].to_numpy(dtype=float)

            rows.append(
                {
                    "phone_label": phone,
                    "ph_nominal": ph,
                    "pair": f"{l1} vs {l2}",
                    "deltaE": lab_distance(v1, v2),
                }
            )

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        return pair_df

    out = (
        pair_df.groupby(["ph_nominal", "pair"], as_index=False)["deltaE"]
        .mean()
        .sort_values(["pair", "ph_nominal"])
    )
    return out


def plot_pairwise_light_separation(df, region, output_path: Path, mark_ph=None, dpi=220):
    mark_ph = mark_ph or []
    pair_df = compute_pairwise_light_separation(df, region)
    if pair_df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 7))

    pairs = sorted(pair_df["pair"].unique())
    ph_vals = sort_ph(pair_df["ph_nominal"].unique())

    for pair in pairs:
        sub = pair_df[pair_df["pair"] == pair].sort_values("ph_nominal")
        ax.plot(
            sub["ph_nominal"].to_numpy(dtype=float),
            sub["deltaE"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.8,
            label=pair,
        )

    add_mark_lines(ax, mark_ph)
    ax.set_title(f"{REGION_TITLES[region]} Pairwise Light Separation Across pH")
    ax.set_xlabel("Nominal pH")
    ax.set_ylabel("CIE76 ΔE between Lab centroids")
    ax.set_xticks(ph_vals)
    ax.set_xticklabels([format_ph_tick(v) for v in ph_vals])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    input_table = Path(args.input_table)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    df = pd.read_csv(input_table)
    df = prepare_data(df)

    if args.plot_style == "both":
        styles = ["explore", "paper"]
    else:
        styles = [args.plot_style]

    for style in styles:
        style_dir = output_dir / style_to_subdir(style)
        ensure_dir(style_dir)

        for region in args.regions:
            # 1) profiles
            for feature in REGION_FEATURES[region]:
                plot_profile(
                    df=df,
                    feature=feature,
                    output_path=style_dir / f"{style}_{feature}_by_phone.png",
                    style=style,
                    mark_ph=args.mark_ph,
                    dpi=args.dpi,
                )

                plot_interaction_residual(
                    df=df,
                    feature=feature,
                    output_path=style_dir / f"{style}_{feature}_interaction_residual.png",
                    mark_ph=args.mark_ph,
                    dpi=args.dpi,
                )

            # 2) dispersion
            plot_dispersion(
                df=df,
                region=region,
                output_path=style_dir / f"{style}_{region}_light_dispersion_across_ph.png",
                mark_ph=args.mark_ph,
                dpi=args.dpi,
            )

            # 3) pairwise light separation
            plot_pairwise_light_separation(
                df=df,
                region=region,
                output_path=style_dir / f"{style}_{region}_pairwise_light_separation_across_ph.png",
                mark_ph=args.mark_ph,
                dpi=args.dpi,
            )

    print("Done.")
    print(f"Input table: {input_table}")
    print(f"Output dir: {output_dir}")
    print(f"Plot style: {args.plot_style}")
    print(f"Regions: {args.regions}")


if __name__ == "__main__":
    main()

"""
python .\scripts\13_plot_color_profiles.py `
  --input-table ".\outputs\factorial_analysis\analysis_factorial_table.csv" `
  --output-dir ".\outputs\color_profiles" `
  --plot-style both `
  --mark-ph 8 12
"""