#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
22_build_hybrid_augmentation_library.py

Build a fixed empirical augmentation library from the 195-image capture
factorial table.

The library separates:
1) global additive background nuisance
2) pH-independent phone contrast shift
3) pH-dependent phone trajectory shape, estimated only for pH 2-12

The pH-dependent trajectory is disabled at pH 1 and 13 because analyses 18-19
showed unstable boundary behavior.

This script only builds the augmentation library. It does NOT train a classifier
and does NOT inspect target-domain performance.

No statsmodels is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/factorial_analysis/analysis_factorial_table.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hybrid_augmentation_library",
    )
    parser.add_argument(
        "--knot-count",
        type=int,
        default=4,
        help="Total restricted-cubic-spline knot count.",
    )
    parser.add_argument(
        "--trajectory-min-ph",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--trajectory-max-ph",
        type=float,
        default=12.0,
    )
    return parser.parse_args()


def resolve_existing(text: str) -> Path:
    path = Path(text)
    candidates = [path]
    if not path.is_absolute():
        candidates += [PROJECT_ROOT / path, Path.cwd() / path]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(text)


def resolve_output(text: str) -> Path:
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def find_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    label: str,
) -> str:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise ValueError(
        f"Could not find {label}. Tried: {list(candidates)}"
    )


def prepare_table(frame: pd.DataFrame) -> pd.DataFrame:
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

    required = [
        "bg_L", "bg_a", "bg_b",
        "contrast_L", "contrast_a", "contrast_b",
    ]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required Lab columns: {missing}")

    result = frame.copy()
    result["phone"] = result[phone_col].astype(str)
    result["light"] = result[light_col].astype(str)
    result["ph_nominal"] = pd.to_numeric(
        result[ph_col],
        errors="raise",
    ).astype(float)

    for column in required:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        ).astype(float)

    if result.duplicated(
        ["phone", "light", "ph_nominal"],
        keep=False,
    ).any():
        raise ValueError(
            "Duplicate phone x light x pH cells were found."
        )

    expected = (
        result["phone"].nunique()
        * result["light"].nunique()
        * result["ph_nominal"].nunique()
    )
    if len(result) != expected:
        raise ValueError(
            f"Incomplete factorial design: rows={len(result)}, "
            f"expected={expected}."
        )

    return result.sort_values(
        ["phone", "light", "ph_nominal"],
        kind="stable",
    ).reset_index(drop=True)


def restricted_cubic_spline_basis(
    x: np.ndarray,
    x_min: float,
    x_max: float,
    knot_count: int,
):
    """Natural/restricted cubic spline basis with linear tails."""
    if knot_count < 3:
        raise ValueError("knot_count must be >= 3.")

    x = np.asarray(x, dtype=float)
    span = float(x_max - x_min)
    if span <= 0:
        raise ValueError("Invalid spline range.")

    z = (x - x_min) / span
    knots_original = np.linspace(
        x_min,
        x_max,
        knot_count,
        dtype=float,
    )
    knots = (knots_original - x_min) / span

    def cube_plus(v):
        return np.maximum(v, 0.0) ** 3

    last = knots[-1]
    penultimate = knots[-2]

    columns = [np.ones_like(z), z]
    names = ["Intercept", "rcs_linear"]

    for index in range(knot_count - 2):
        knot = knots[index]
        term = (
            (
                cube_plus(z - knot)
                - cube_plus(z - last)
            )
            / (last - knot)
            - (
                cube_plus(z - penultimate)
                - cube_plus(z - last)
            )
            / (last - penultimate)
        )
        columns.append(term)
        names.append(f"rcs_nonlinear_{index + 1}")

    return np.column_stack(columns), names, knots_original


def fit_phone_curves(
    frame: pd.DataFrame,
    knot_count: int,
    min_ph: float,
    max_ph: float,
) -> pd.DataFrame:
    """
    Fit phone-specific smooth contrast trajectories to light-averaged data.

    The fitted curve is evaluated only for integer/observed pH values in the
    internal range. We center each phone curve over pH so the trajectory library
    represents pH-dependent SHAPE rather than the phone's global contrast shift.
    """
    internal = frame.loc[
        (frame["ph_nominal"] >= min_ph)
        & (frame["ph_nominal"] <= max_ph)
    ].copy()

    means = (
        internal.groupby(
            ["phone", "ph_nominal"],
            as_index=False,
        )[
            ["contrast_L", "contrast_a", "contrast_b"]
        ]
        .mean()
    )

    ph_grid = sorted(
        means["ph_nominal"].unique().tolist()
    )
    x_grid = np.asarray(ph_grid, dtype=float)
    design_grid, _, knots = restricted_cubic_spline_basis(
        x_grid,
        min_ph,
        max_ph,
        knot_count,
    )

    rows = []

    for phone, group in means.groupby("phone", sort=True):
        group = group.sort_values("ph_nominal")
        if group["ph_nominal"].tolist() != ph_grid:
            raise ValueError(
                f"Phone {phone} does not contain the complete internal pH grid."
            )

        y = group[
            ["contrast_L", "contrast_a", "contrast_b"]
        ].to_numpy(dtype=float)

        beta, _, _, _ = np.linalg.lstsq(
            design_grid,
            y,
            rcond=None,
        )
        fitted = design_grid @ beta
        centered = fitted - fitted.mean(axis=0, keepdims=True)

        for index, ph_value in enumerate(ph_grid):
            rows.append({
                "phone": phone,
                "ph_nominal": float(ph_value),
                "fitted_contrast_L": float(fitted[index, 0]),
                "fitted_contrast_a": float(fitted[index, 1]),
                "fitted_contrast_b": float(fitted[index, 2]),
                "centered_shape_L": float(centered[index, 0]),
                "centered_shape_a": float(centered[index, 1]),
                "centered_shape_b": float(centered[index, 2]),
                "knot_count": int(knot_count),
                "knots": ",".join(f"{value:g}" for value in knots),
            })

    return pd.DataFrame(rows)


def directed_pairwise_deltas(
    condition_table: pd.DataFrame,
    key_columns: list[str],
    value_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    rows = []

    records = condition_table.reset_index(drop=True)

    for source_index, source in records.iterrows():
        for target_index, target in records.iterrows():
            if source_index == target_index:
                continue

            row = {
                "source_id": "|".join(
                    str(source[column])
                    for column in key_columns
                ),
                "target_id": "|".join(
                    str(target[column])
                    for column in key_columns
                ),
            }

            for column in key_columns:
                row[f"source_{column}"] = source[column]
                row[f"target_{column}"] = target[column]

            delta_vector = []

            for channel_name, value_column in zip(
                ["L", "a", "b"],
                value_columns,
            ):
                delta = float(
                    target[value_column] - source[value_column]
                )
                row[f"delta_{channel_name}"] = delta
                delta_vector.append(delta)

            row["deltaE76"] = float(
                np.linalg.norm(delta_vector)
            )
            row["delta_type"] = prefix
            rows.append(row)

    return pd.DataFrame(rows)


def trajectory_pairwise_deltas(
    phone_curves: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    phones = sorted(
        phone_curves["phone"].unique().tolist()
    )
    ph_values = sorted(
        phone_curves["ph_nominal"].unique().tolist()
    )

    indexed = phone_curves.set_index(
        ["phone", "ph_nominal"]
    )

    for ph_value in ph_values:
        for source_phone in phones:
            for target_phone in phones:
                if source_phone == target_phone:
                    continue

                source = indexed.loc[
                    (source_phone, ph_value)
                ]
                target = indexed.loc[
                    (target_phone, ph_value)
                ]

                delta = np.array([
                    target["centered_shape_L"]
                    - source["centered_shape_L"],
                    target["centered_shape_a"]
                    - source["centered_shape_a"],
                    target["centered_shape_b"]
                    - source["centered_shape_b"],
                ], dtype=float)

                rows.append({
                    "ph_nominal": float(ph_value),
                    "source_phone": source_phone,
                    "target_phone": target_phone,
                    "delta_L": float(delta[0]),
                    "delta_a": float(delta[1]),
                    "delta_b": float(delta[2]),
                    "deltaE76": float(
                        np.linalg.norm(delta)
                    ),
                    "delta_type": "pH_dependent_phone_shape",
                })

    return pd.DataFrame(rows)


def summarize_deltas(
    table: pd.DataFrame,
    name: str,
) -> dict:
    values = table["deltaE76"].to_numpy(dtype=float)
    return {
        "name": name,
        "n_vectors": int(len(values)),
        "mean_deltaE76": float(np.mean(values)),
        "median_deltaE76": float(np.median(values)),
        "q90_deltaE76": float(np.quantile(values, 0.90)),
        "max_deltaE76": float(np.max(values)),
    }


def main():
    args = parse_args()

    if args.trajectory_min_ph >= args.trajectory_max_ph:
        raise ValueError(
            "trajectory-min-ph must be below trajectory-max-ph."
        )

    input_path = resolve_existing(args.input)
    output_dir = resolve_output(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = prepare_table(
        pd.read_csv(input_path)
    )

    bg_condition_means = (
        frame.groupby(
            ["phone", "light"],
            as_index=False,
        )[
            ["bg_L", "bg_a", "bg_b"]
        ]
        .mean()
    )

    bg_pairwise = directed_pairwise_deltas(
        bg_condition_means,
        ["phone", "light"],
        ["bg_L", "bg_a", "bg_b"],
        "global_background",
    )

    contrast_phone_means = (
        frame.groupby(
            ["phone"],
            as_index=False,
        )[
            ["contrast_L", "contrast_a", "contrast_b"]
        ]
        .mean()
    )

    contrast_global_pairwise = directed_pairwise_deltas(
        contrast_phone_means,
        ["phone"],
        ["contrast_L", "contrast_a", "contrast_b"],
        "global_phone_contrast",
    )

    curves = fit_phone_curves(
        frame,
        args.knot_count,
        args.trajectory_min_ph,
        args.trajectory_max_ph,
    )

    trajectory_pairwise = trajectory_pairwise_deltas(
        curves
    )

    summary = pd.DataFrame([
        summarize_deltas(
            bg_pairwise,
            "global_background",
        ),
        summarize_deltas(
            contrast_global_pairwise,
            "global_phone_contrast",
        ),
        summarize_deltas(
            trajectory_pairwise,
            "pH_dependent_phone_shape",
        ),
    ])

    bg_condition_means.to_csv(
        output_dir / "background_condition_means.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bg_pairwise.to_csv(
        output_dir / "background_pairwise_deltas.csv",
        index=False,
        encoding="utf-8-sig",
    )
    contrast_phone_means.to_csv(
        output_dir / "contrast_phone_means.csv",
        index=False,
        encoding="utf-8-sig",
    )
    contrast_global_pairwise.to_csv(
        output_dir / "contrast_global_pairwise_deltas.csv",
        index=False,
        encoding="utf-8-sig",
    )
    curves.to_csv(
        output_dir / "contrast_phone_rcs_curves.csv",
        index=False,
        encoding="utf-8-sig",
    )
    trajectory_pairwise.to_csv(
        output_dir / "contrast_trajectory_pairwise_deltas.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        output_dir / "library_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    library = {
        "version": 1,
        "source_input": str(input_path),
        "knot_count": int(args.knot_count),
        "trajectory_min_ph": float(
            args.trajectory_min_ph
        ),
        "trajectory_max_ph": float(
            args.trajectory_max_ph
        ),
        "boundary_policy": (
            "pH-dependent trajectory disabled outside "
            f"[{args.trajectory_min_ph:g}, {args.trajectory_max_ph:g}]"
        ),
        "background_pairwise_deltas": (
            bg_pairwise[
                ["delta_L", "delta_a", "delta_b"]
            ]
            .to_numpy(dtype=float)
            .tolist()
        ),
        "global_contrast_pairwise_deltas": (
            contrast_global_pairwise[
                ["delta_L", "delta_a", "delta_b"]
            ]
            .to_numpy(dtype=float)
            .tolist()
        ),
        "trajectory_pairwise_deltas_by_ph": {
            f"{float(ph_value):g}": (
                group[
                    ["delta_L", "delta_a", "delta_b"]
                ]
                .to_numpy(dtype=float)
                .tolist()
            )
            for ph_value, group in trajectory_pairwise.groupby(
                "ph_nominal",
                sort=True,
            )
        },
        "summary": summary.to_dict("records"),
        "interpretation": {
            "background": (
                "pH-independent whole-crop additive nuisance candidate"
            ),
            "global_contrast": (
                "pH-independent ROI-relative phone shift"
            ),
            "trajectory": (
                "pH-dependent ROI-relative phone trajectory shape; "
                "internal pH only"
            ),
        },
    }

    with (output_dir / "augmentation_library.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            library,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    audit = {
        "rows": int(len(frame)),
        "phones": sorted(
            frame["phone"].unique().tolist()
        ),
        "lights": sorted(
            frame["light"].unique().tolist()
        ),
        "ph_levels": sorted(
            frame["ph_nominal"].unique().tolist()
        ),
        "trajectory_fit_range": [
            float(args.trajectory_min_ph),
            float(args.trajectory_max_ph),
        ],
        "critical_note": (
            "This library is fixed before classifier evaluation. "
            "Do not tune it after inspecting pilot test performance."
        ),
    }

    with (output_dir / "analysis_audit.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            audit,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 100)
    print("22. Hybrid empirical augmentation library")
    print("=" * 100)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print("\nLibrary summary")
    print(summary.to_string(index=False))
    print("\nBoundary policy")
    print(library["boundary_policy"])
    print("\nMain file")
    print(output_dir / "augmentation_library.json")


if __name__ == "__main__":
    main()
