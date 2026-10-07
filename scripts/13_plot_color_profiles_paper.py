#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
13_plot_color_profiles_paper.py

Create exploratory and paper-style figures from the fixed factorial table.

The script accepts either:
- phone / light / ph_nominal
or
- phone_label / light_label / ph_value

It standardizes them internally to phone, light, and ph_nominal.

Outputs
-------
outputs/color_profiles_v2/
├─ exploratory/
├─ paper/
├─ interaction_residuals/
├─ lab_separation/
├─ tables/
├─ figure_manifest.csv
└─ plot_config.json

Paper-style phone profiles contain:
- five light-specific raw observations as semi-transparent points
- phone-specific mean profile
- observed minimum-to-maximum range across lights

The range is descriptive and is not a confidence interval.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGIONS = ["pad", "bg", "contrast"]
COMPONENTS = ["L", "a", "b"]

REGION_LABELS = {
    "pad": "Pad",
    "bg": "Background",
    "contrast": "Pad - Background Contrast",
}

COMPONENT_LABELS = {
    "L": "CIE L*",
    "a": "CIE a*",
    "b": "CIE b*",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/factorial_analysis/analysis_factorial_table.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/color_profiles_v2",
    )
    parser.add_argument(
        "--plot-style",
        choices=["explore", "paper", "both"],
        default="both",
    )
    parser.add_argument(
        "--mark-ph",
        nargs="?",
        const="",
        default="",
        help=(
            "Optional comma-separated pH markers, e.g. 8,12. "
            "The option may be omitted entirely for no markers."
        ),
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--save-pdf", action="store_true")
    return parser.parse_args()


def resolve_existing(path_text: str) -> Path:
    path = Path(path_text)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([PROJECT_ROOT / path, Path.cwd() / path])
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Input not found: {path_text}")


def resolve_output(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def find_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    description: str,
) -> str:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise ValueError(
        f"Could not find {description}. Tried {list(candidates)}. "
        f"Available columns: {list(frame.columns)}"
    )


def parse_float_list(raw: str) -> list[float]:
    text = raw.strip()
    if not text:
        return []
    return [
        float(item.strip())
        for item in text.split(",")
        if item.strip()
    ]


def validate_and_standardize(frame: pd.DataFrame) -> pd.DataFrame:
    phone_col = find_column(
        frame,
        ["phone", "phone_label", "phone_id", "phone_model"],
        "phone column",
    )
    light_col = find_column(
        frame,
        ["light", "light_label", "light_id"],
        "light column",
    )
    ph_col = find_column(
        frame,
        ["ph_nominal", "ph_value", "ph"],
        "pH column",
    )

    required_features = [
        f"{region}_{component}"
        for region in REGIONS
        for component in COMPONENTS
    ]
    missing = sorted(set(required_features) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing Lab feature columns: {missing}")

    result = frame.copy()
    result["phone"] = result[phone_col].astype(str)
    result["light"] = result[light_col].astype(str)
    result["ph_nominal"] = pd.to_numeric(
        result[ph_col], errors="raise"
    ).astype(float)

    for column in required_features:
        result[column] = pd.to_numeric(
            result[column], errors="raise"
        ).astype(float)

    if "image_id" not in result.columns:
        result["image_id"] = [
            f"row_{index:03d}" for index in range(len(result))
        ]

    duplicate_cells = result.duplicated(
        ["phone", "light", "ph_nominal"],
        keep=False,
    )
    if duplicate_cells.any():
        examples = result.loc[
            duplicate_cells,
            ["phone", "light", "ph_nominal"],
        ].head(10).to_dict("records")
        raise ValueError(
            "Duplicate phone x light x pH cells found. "
            f"Examples: {examples}"
        )

    expected_rows = (
        result["phone"].nunique()
        * result["light"].nunique()
        * result["ph_nominal"].nunique()
    )
    if len(result) != expected_rows:
        raise ValueError(
            "Incomplete factorial grid: "
            f"rows={len(result)}, expected={expected_rows}."
        )

    return result.sort_values(
        ["ph_nominal", "phone", "light"],
        kind="stable",
    ).reset_index(drop=True)


def add_ph_markers(axis: plt.Axes, values: Sequence[float]) -> None:
    for value in values:
        axis.axvline(value, linewidth=1.0, alpha=0.35)


def finalize_axis(
    axis: plt.Axes,
    ph_levels: Sequence[float],
    title: str,
    ylabel: str,
    mark_ph: Sequence[float],
) -> None:
    axis.set_title(title)
    axis.set_xlabel("Nominal pH")
    axis.set_ylabel(ylabel)
    axis.set_xticks(ph_levels)
    axis.grid(True, alpha=0.25)
    add_ph_markers(axis, mark_ph)
    axis.legend(frameon=False)
    axis.figure.tight_layout()


def save_figure(
    figure: plt.Figure,
    stem: Path,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    paths = [png_path]

    if save_pdf:
        pdf_path = stem.with_suffix(".pdf")
        figure.savefig(pdf_path, bbox_inches="tight")
        paths.append(pdf_path)

    plt.close(figure)
    return paths


def phone_summary(
    frame: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    summary = (
        frame.groupby(["phone", "ph_nominal"], as_index=False)
        .agg(
            n=(value_column, "size"),
            mean=(value_column, "mean"),
            minimum=(value_column, "min"),
            maximum=(value_column, "max"),
            sd=(value_column, "std"),
        )
    )
    return summary.sort_values(["phone", "ph_nominal"])


def plot_exploratory_profile(
    frame: pd.DataFrame,
    region: str,
    component: str,
    mark_ph: Sequence[float],
    stem: Path,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    value_column = f"{region}_{component}"
    summary = phone_summary(frame, value_column)
    ph_levels = sorted(frame["ph_nominal"].unique())

    figure, axis = plt.subplots(figsize=(8.5, 5.4))

    for phone, group in summary.groupby("phone", sort=True):
        group = group.sort_values("ph_nominal")
        mean = group["mean"].to_numpy(dtype=float)
        lower = mean - group["minimum"].to_numpy(dtype=float)
        upper = group["maximum"].to_numpy(dtype=float) - mean

        axis.errorbar(
            group["ph_nominal"].to_numpy(dtype=float),
            mean,
            yerr=np.vstack([lower, upper]),
            marker="o",
            linewidth=1.7,
            capsize=3,
            label=str(phone),
        )

    finalize_axis(
        axis,
        ph_levels,
        (
            f"{REGION_LABELS[region]} {COMPONENT_LABELS[component]} "
            "by Phone\n"
            "Mean across five lights; bars show observed range"
        ),
        COMPONENT_LABELS[component],
        mark_ph,
    )
    return save_figure(figure, stem, dpi, save_pdf)


def plot_paper_profile(
    frame: pd.DataFrame,
    region: str,
    component: str,
    mark_ph: Sequence[float],
    stem: Path,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    value_column = f"{region}_{component}"
    summary = phone_summary(frame, value_column)
    ph_levels = sorted(frame["ph_nominal"].unique())
    phones = sorted(frame["phone"].unique())

    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    offsets = np.linspace(-0.11, 0.11, num=max(1, len(phones)))

    for offset, phone in zip(offsets, phones):
        raw = frame.loc[frame["phone"] == phone].copy()
        mean_group = summary.loc[
            summary["phone"] == phone
        ].sort_values("ph_nominal")

        line = axis.plot(
            mean_group["ph_nominal"].to_numpy(dtype=float),
            mean_group["mean"].to_numpy(dtype=float),
            marker="o",
            linewidth=2.0,
            label=str(phone),
        )[0]
        line_color = line.get_color()

        axis.fill_between(
            mean_group["ph_nominal"].to_numpy(dtype=float),
            mean_group["minimum"].to_numpy(dtype=float),
            mean_group["maximum"].to_numpy(dtype=float),
            alpha=0.10,
            color=line_color,
        )

        for ph_value, raw_group in raw.groupby(
            "ph_nominal", sort=True
        ):
            values = raw_group[value_column].to_numpy(dtype=float)
            jitter = (
                np.linspace(-0.035, 0.035, len(values))
                if len(values) > 1
                else np.array([0.0])
            )
            axis.scatter(
                np.full(len(values), float(ph_value) + offset) + jitter,
                values,
                s=20,
                alpha=0.35,
                color=line_color,
                edgecolors="none",
            )

    finalize_axis(
        axis,
        ph_levels,
        f"{REGION_LABELS[region]} {COMPONENT_LABELS[component]} by Phone",
        COMPONENT_LABELS[component],
        mark_ph,
    )
    return save_figure(figure, stem, dpi, save_pdf)


def additive_interaction_residuals(
    frame: pd.DataFrame,
    domain_column: str,
    value_column: str,
) -> pd.DataFrame:
    cell_means = (
        frame.groupby(
            [domain_column, "ph_nominal"],
            as_index=False,
        )[value_column]
        .mean()
    )
    grand_mean = float(cell_means[value_column].mean())
    cell_means["domain_mean"] = (
        cell_means.groupby(domain_column)[value_column]
        .transform("mean")
    )
    cell_means["ph_mean"] = (
        cell_means.groupby("ph_nominal")[value_column]
        .transform("mean")
    )
    cell_means["interaction_residual"] = (
        cell_means[value_column]
        - cell_means["domain_mean"]
        - cell_means["ph_mean"]
        + grand_mean
    )
    return cell_means


def plot_phone_interaction(
    frame: pd.DataFrame,
    region: str,
    component: str,
    mark_ph: Sequence[float],
    stem: Path,
    dpi: int,
    save_pdf: bool,
) -> tuple[list[Path], pd.DataFrame]:
    value_column = f"{region}_{component}"
    residuals = additive_interaction_residuals(
        frame,
        "phone",
        value_column,
    )
    ph_levels = sorted(frame["ph_nominal"].unique())

    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    for phone, group in residuals.groupby("phone", sort=True):
        group = group.sort_values("ph_nominal")
        axis.plot(
            group["ph_nominal"].to_numpy(dtype=float),
            group["interaction_residual"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.8,
            label=str(phone),
        )

    axis.axhline(0.0, linewidth=1.0, alpha=0.6)
    finalize_axis(
        axis,
        ph_levels,
        (
            f"{REGION_LABELS[region]} {COMPONENT_LABELS[component]}: "
            "pH x Phone Interaction Residual\n"
            "Zero represents an additive parallel profile"
        ),
        "Interaction residual",
        mark_ph,
    )
    return save_figure(figure, stem, dpi, save_pdf), residuals


def lab_centroids(
    frame: pd.DataFrame,
    domain_column: str,
    region: str,
) -> pd.DataFrame:
    columns = [
        f"{region}_L",
        f"{region}_a",
        f"{region}_b",
    ]
    return (
        frame.groupby(
            ["ph_nominal", domain_column],
            as_index=False,
        )[columns]
        .mean()
        .rename(
            columns={
                columns[0]: "L",
                columns[1]: "a",
                columns[2]: "b",
            }
        )
    )


def pairwise_delta_e(
    centroids: pd.DataFrame,
    domain_column: str,
    region: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for ph_value, group in centroids.groupby(
        "ph_nominal", sort=True
    ):
        records = group.sort_values(domain_column).to_dict("records")
        for first, second in itertools.combinations(records, 2):
            vector_1 = np.array(
                [first["L"], first["a"], first["b"]],
                dtype=float,
            )
            vector_2 = np.array(
                [second["L"], second["a"], second["b"]],
                dtype=float,
            )
            rows.append({
                "region": region,
                "domain_type": domain_column,
                "ph_nominal": float(ph_value),
                "domain_1": str(first[domain_column]),
                "domain_2": str(second[domain_column]),
                "pair": (
                    f"{first[domain_column]} vs "
                    f"{second[domain_column]}"
                ),
                "deltaE76": float(
                    np.linalg.norm(vector_1 - vector_2)
                ),
            })
    return pd.DataFrame(rows)


def domain_dispersion(
    centroids: pd.DataFrame,
    pairwise: pd.DataFrame,
    domain_column: str,
    region: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for ph_value, group in centroids.groupby(
        "ph_nominal", sort=True
    ):
        vectors = group[["L", "a", "b"]].to_numpy(dtype=float)
        center = vectors.mean(axis=0)
        distances = np.linalg.norm(vectors - center, axis=1)
        pair_group = pairwise.loc[
            np.isclose(pairwise["ph_nominal"], ph_value)
        ]

        rows.append({
            "region": region,
            "domain_type": domain_column,
            "ph_nominal": float(ph_value),
            "centroid_rms_distance": float(
                np.sqrt(np.mean(np.square(distances)))
            ),
            "mean_pairwise_deltaE76": float(
                pair_group["deltaE76"].mean()
            ),
            "maximum_pairwise_deltaE76": float(
                pair_group["deltaE76"].max()
            ),
        })
    return pd.DataFrame(rows)


def plot_pairwise_separation(
    pairwise: pd.DataFrame,
    domain_column: str,
    region: str,
    mark_ph: Sequence[float],
    stem: Path,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(9.0, 5.8))
    ph_levels = sorted(pairwise["ph_nominal"].unique())

    for pair, group in pairwise.groupby("pair", sort=True):
        group = group.sort_values("ph_nominal")
        axis.plot(
            group["ph_nominal"].to_numpy(dtype=float),
            group["deltaE76"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.5,
            label=str(pair),
        )

    label = "Phone" if domain_column == "phone" else "Light"
    finalize_axis(
        axis,
        ph_levels,
        (
            f"{REGION_LABELS[region]} Pairwise {label} "
            "Separation Across pH"
        ),
        "CIE76 Delta E between Lab centroids",
        mark_ph,
    )
    return save_figure(figure, stem, dpi, save_pdf)


def plot_dispersion(
    dispersion: pd.DataFrame,
    domain_column: str,
    region: str,
    mark_ph: Sequence[float],
    stem: Path,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    ph_levels = sorted(dispersion["ph_nominal"].unique())

    metric_labels = {
        "centroid_rms_distance": "RMS distance to pH centroid",
        "mean_pairwise_deltaE76": "Mean pairwise Delta E",
        "maximum_pairwise_deltaE76": "Maximum pairwise Delta E",
    }

    for metric, label in metric_labels.items():
        axis.plot(
            dispersion["ph_nominal"].to_numpy(dtype=float),
            dispersion[metric].to_numpy(dtype=float),
            marker="o",
            linewidth=1.7,
            label=label,
        )

    domain_label = (
        "Phone" if domain_column == "phone" else "Light"
    )
    finalize_axis(
        axis,
        ph_levels,
        (
            f"{REGION_LABELS[region]} {domain_label} "
            "Lab Dispersion Across pH"
        ),
        "Lab separation",
        mark_ph,
    )
    return save_figure(figure, stem, dpi, save_pdf)


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0.")

    input_path = resolve_existing(args.input)
    output_dir = resolve_output(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = validate_and_standardize(pd.read_csv(input_path))
    mark_ph = parse_float_list(args.mark_ph)

    explore_dir = output_dir / "exploratory"
    paper_dir = output_dir / "paper"
    interaction_dir = output_dir / "interaction_residuals"
    separation_dir = output_dir / "lab_separation"
    tables_dir = output_dir / "tables"

    for directory in [
        explore_dir,
        paper_dir,
        interaction_dir,
        separation_dir,
        tables_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    styles = (
        ["explore", "paper"]
        if args.plot_style == "both"
        else [args.plot_style]
    )

    manifest_rows: list[dict[str, Any]] = []
    phone_summary_frames: list[pd.DataFrame] = []
    residual_frames: list[pd.DataFrame] = []
    pairwise_frames: list[pd.DataFrame] = []
    dispersion_frames: list[pd.DataFrame] = []

    for region in REGIONS:
        for component in COMPONENTS:
            value_column = f"{region}_{component}"
            summary = phone_summary(frame, value_column)
            summary["region"] = region
            summary["component"] = component
            phone_summary_frames.append(summary)

            if "explore" in styles:
                paths = plot_exploratory_profile(
                    frame,
                    region,
                    component,
                    mark_ph,
                    explore_dir / f"{region}_{component}_by_phone",
                    args.dpi,
                    args.save_pdf,
                )
                for path in paths:
                    manifest_rows.append({
                        "category": "exploratory_profile",
                        "region": region,
                        "component": component,
                        "path": str(path),
                    })

            if "paper" in styles:
                paths = plot_paper_profile(
                    frame,
                    region,
                    component,
                    mark_ph,
                    paper_dir / f"{region}_{component}_by_phone",
                    args.dpi,
                    args.save_pdf,
                )
                for path in paths:
                    manifest_rows.append({
                        "category": "paper_profile",
                        "region": region,
                        "component": component,
                        "path": str(path),
                    })

            paths, residuals = plot_phone_interaction(
                frame,
                region,
                component,
                mark_ph,
                interaction_dir
                / f"{region}_{component}_ph_by_phone_residual",
                args.dpi,
                args.save_pdf,
            )
            residuals["region"] = region
            residuals["component"] = component
            residual_frames.append(residuals)

            for path in paths:
                manifest_rows.append({
                    "category": "phone_interaction_residual",
                    "region": region,
                    "component": component,
                    "path": str(path),
                })

        for domain_column in ["phone", "light"]:
            centroids = lab_centroids(
                frame, domain_column, region
            )
            pairwise = pairwise_delta_e(
                centroids, domain_column, region
            )
            dispersion = domain_dispersion(
                centroids,
                pairwise,
                domain_column,
                region,
            )
            pairwise_frames.append(pairwise)
            dispersion_frames.append(dispersion)

            pair_paths = plot_pairwise_separation(
                pairwise,
                domain_column,
                region,
                mark_ph,
                separation_dir
                / f"{region}_{domain_column}_pairwise_deltaE",
                args.dpi,
                args.save_pdf,
            )
            dispersion_paths = plot_dispersion(
                dispersion,
                domain_column,
                region,
                mark_ph,
                separation_dir
                / f"{region}_{domain_column}_dispersion",
                args.dpi,
                args.save_pdf,
            )

            for path in [*pair_paths, *dispersion_paths]:
                manifest_rows.append({
                    "category": "lab_separation",
                    "region": region,
                    "component": "",
                    "path": str(path),
                })

    pd.concat(
        phone_summary_frames,
        ignore_index=True,
    ).to_csv(
        tables_dir / "phone_profile_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(
        residual_frames,
        ignore_index=True,
    ).to_csv(
        tables_dir / "phone_interaction_residuals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(
        pairwise_frames,
        ignore_index=True,
    ).to_csv(
        tables_dir / "pairwise_deltaE_by_ph.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(
        dispersion_frames,
        ignore_index=True,
    ).to_csv(
        tables_dir / "domain_dispersion_by_ph.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(manifest_rows).to_csv(
        output_dir / "figure_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "input": str(input_path),
        "output": str(output_dir),
        "rows": int(len(frame)),
        "phones": sorted(frame["phone"].unique().tolist()),
        "lights": sorted(frame["light"].unique().tolist()),
        "ph_levels": sorted(
            frame["ph_nominal"].unique().tolist()
        ),
        "plot_style": args.plot_style,
        "mark_ph": mark_ph,
        "note": (
            "Paper profile ranges describe the five observed lighting "
            "conditions and are not confidence intervals."
        ),
    }
    with (output_dir / "plot_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    print("=" * 88)
    print("13. Color profiles")
    print("=" * 88)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Rows: {len(frame)}")
    print(f"Phones: {config['phones']}")
    print(f"Lights: {config['lights']}")
    print(f"Plot style: {args.plot_style}")
    print(f"pH markers: {mark_ph}")
    print("\nPaper figures:")
    print(f"  {paper_dir}")
    print("Interaction figures:")
    print(f"  {interaction_dir}")
    print("Lab separation figures:")
    print(f"  {separation_dir}")


if __name__ == "__main__":
    main()
