#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
17_leave_one_strip_out_spline_models.py

Leave-one-strip-out validation of smooth additive and non-additive pH response
models for smartphone colorimetry.

This script does not use statsmodels.

Research question
-----------------
When all five light observations from one phone x pH strip are removed, can a
smooth phone-specific pH trajectory inferred from the remaining pH levels
predict the held-out strip better than a global additive nuisance model?

Cross-validation unit
---------------------
One strip_id = one phone x pH unit observed under five lights.

Each fold removes all five images belonging to one strip. There are 39 folds.

Models
------
1. spline_additive
   spline(pH) + phone + light + phone x light

2. phone_spline
   spline_additive + phone x spline(pH)

3. full_spline_interactions
   phone_spline + light x spline(pH)

Spline basis
------------
A cubic B-spline basis is defined from the fixed experimental pH range. Knot
locations use only the design grid, not response values. The first basis
column is dropped to avoid intercept collinearity.

Sensitivity analysis
--------------------
The default basis counts are 4, 5, and 6. For a cubic spline:

- basis_count=4: no internal knot
- basis_count=5: one internal knot
- basis_count=6: two internal knots

The primary setting is basis_count=5 unless changed.

Interpretation scope
--------------------
This is stronger than holding out one image because no image from the same
strip remains in training.

However, there is still only one physical strip per phone x pH cell. Therefore
the analysis cannot directly verify repeatability for a second independent
strip at the same phone x pH combination.

Outputs
-------
outputs/strip_holdout_spline/
├─ cv_predictions.csv
├─ fold_metrics.csv
├─ cv_summary.csv
├─ metrics_by_ph.csv
├─ metrics_by_phone.csv
├─ paired_model_comparison.csv
├─ spline_sensitivity_summary.csv
├─ design_rank_audit.csv
├─ analysis_audit.json
├─ analysis_notes.txt
└─ figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.interpolate import BSpline
except ImportError as exc:
    raise ImportError(
        "scipy is required for the B-spline basis. "
        "Install it with: python -m pip install scipy"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGION_COLUMNS = {
    "pad": ["pad_L", "pad_a", "pad_b"],
    "bg": ["bg_L", "bg_a", "bg_b"],
    "contrast": ["contrast_L", "contrast_a", "contrast_b"],
}

CHANNEL_NAMES = ["L", "a", "b"]

MODEL_TERMS = {
    "spline_additive": [
        "intercept",
        "spline",
        "phone",
        "light",
        "phone_x_light",
    ],
    "phone_spline": [
        "intercept",
        "spline",
        "phone",
        "light",
        "phone_x_light",
        "phone_x_spline",
    ],
    "full_spline_interactions": [
        "intercept",
        "spline",
        "phone",
        "light",
        "phone_x_light",
        "phone_x_spline",
        "light_x_spline",
    ],
}

MODEL_ORDER = [
    "spline_additive",
    "phone_spline",
    "full_spline_interactions",
]

PAIRWISE_COMPARISONS = [
    (
        "phone_spline",
        "spline_additive",
        "phone_spline minus spline_additive",
    ),
    (
        "full_spline_interactions",
        "spline_additive",
        "full_spline_interactions minus spline_additive",
    ),
    (
        "full_spline_interactions",
        "phone_spline",
        "full_spline_interactions minus phone_spline",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leave-one-strip-out comparison of additive and smooth "
            "pH-dependent interaction models."
        )
    )
    parser.add_argument(
        "--input",
        default=(
            "outputs/factorial_analysis/"
            "analysis_factorial_table.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/strip_holdout_spline",
    )
    parser.add_argument(
        "--basis-counts",
        nargs="*",
        type=int,
        default=[4, 5, 6],
    )
    parser.add_argument(
        "--primary-basis-count",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=MODEL_ORDER,
        default=MODEL_ORDER,
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=3000,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--save-pdf", action="store_true")
    return parser.parse_args()


def resolve_existing(path_text: str) -> Path:
    path = Path(path_text)
    candidates = [path]

    if not path.is_absolute():
        candidates.extend([
            PROJECT_ROOT / path,
            Path.cwd() / path,
        ])

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
    lower = {
        str(column).lower(): str(column)
        for column in frame.columns
    }

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

        if candidate.lower() in lower:
            return lower[candidate.lower()]

    raise ValueError(
        f"Could not find {description}. Tried {list(candidates)}. "
        f"Available columns: {list(frame.columns)}"
    )


def prepare_data(frame: pd.DataFrame) -> pd.DataFrame:
    phone_column = find_column(
        frame,
        ["phone", "phone_label", "phone_id", "phone_model"],
        "phone column",
    )
    light_column = find_column(
        frame,
        ["light", "light_label", "light_id"],
        "light column",
    )
    ph_column = find_column(
        frame,
        ["ph_nominal", "ph_value", "ph"],
        "pH column",
    )

    required_features = [
        column
        for columns in REGION_COLUMNS.values()
        for column in columns
    ]
    missing = sorted(set(required_features) - set(frame.columns))

    if missing:
        raise ValueError(f"Missing Lab columns: {missing}")

    result = frame.copy()
    result["phone"] = result[phone_column].astype(str)
    result["light"] = result[light_column].astype(str)
    result["ph_nominal"] = pd.to_numeric(
        result[ph_column],
        errors="raise",
    ).astype(float)

    for column in required_features:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        ).astype(float)

    if "image_id" not in result.columns:
        result["image_id"] = [
            f"row_{index:03d}"
            for index in range(len(result))
        ]
    else:
        result["image_id"] = result["image_id"].astype(str)

    if "strip_id" not in result.columns:
        result["strip_id"] = (
            result["phone"].astype(str)
            + "_ph"
            + result["ph_nominal"].map(lambda value: f"{value:g}")
        )
    else:
        result["strip_id"] = result["strip_id"].astype(str)

    duplicate_cells = result.duplicated(
        ["ph_nominal", "phone", "light"],
        keep=False,
    )
    if duplicate_cells.any():
        examples = (
            result.loc[
                duplicate_cells,
                ["ph_nominal", "phone", "light"],
            ]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "Duplicate pH x phone x light cells found. "
            f"Examples: {examples}"
        )

    expected_rows = (
        result["ph_nominal"].nunique()
        * result["phone"].nunique()
        * result["light"].nunique()
    )

    if len(result) != expected_rows:
        raise ValueError(
            "Incomplete factorial design: "
            f"rows={len(result)}, expected={expected_rows}."
        )

    strip_audit = (
        result.groupby("strip_id")
        .agg(
            phone_count=("phone", "nunique"),
            ph_count=("ph_nominal", "nunique"),
            light_count=("light", "nunique"),
            row_count=("image_id", "size"),
        )
    )

    expected_light_count = result["light"].nunique()

    valid = (
        strip_audit["phone_count"].eq(1)
        & strip_audit["ph_count"].eq(1)
        & strip_audit["light_count"].eq(expected_light_count)
        & strip_audit["row_count"].eq(expected_light_count)
    )

    if not valid.all():
        bad = strip_audit.loc[~valid].head(10)
        raise ValueError(
            "strip_id must identify one phone x pH unit repeated under "
            f"all lights. Bad examples:\n{bad}"
        )

    expected_strips = (
        result["phone"].nunique()
        * result["ph_nominal"].nunique()
    )

    if result["strip_id"].nunique() != expected_strips:
        raise ValueError(
            "Unexpected number of strips: "
            f"observed={result['strip_id'].nunique()}, "
            f"expected={expected_strips}."
        )

    return result.sort_values(
        ["phone", "ph_nominal", "light"],
        kind="stable",
    ).reset_index(drop=True)


def make_bspline_knots(
    x_min: float,
    x_max: float,
    basis_count: int,
    degree: int,
) -> np.ndarray:
    if degree < 1:
        raise ValueError("Spline degree must be >= 1.")
    if basis_count < degree + 1:
        raise ValueError(
            f"basis_count={basis_count} is too small for degree={degree}. "
            f"Use at least {degree + 1}."
        )
    if x_max <= x_min:
        raise ValueError("Spline pH range must have positive width.")

    internal_count = basis_count - degree - 1

    if internal_count > 0:
        internal = np.linspace(
            x_min,
            x_max,
            internal_count + 2,
            dtype=float,
        )[1:-1]
    else:
        internal = np.array([], dtype=float)

    return np.concatenate([
        np.repeat(x_min, degree + 1),
        internal,
        np.repeat(x_max, degree + 1),
    ])


def bspline_basis(
    x: np.ndarray,
    x_min: float,
    x_max: float,
    basis_count: int,
    degree: int,
    drop_first: bool = True,
) -> tuple[np.ndarray, list[str]]:
    x = np.asarray(x, dtype=float)

    if np.any(x < x_min) or np.any(x > x_max):
        raise ValueError(
            "All pH values must lie within the fixed spline range."
        )

    knots = make_bspline_knots(
        x_min,
        x_max,
        basis_count,
        degree,
    )

    basis_columns: list[np.ndarray] = []

    for basis_index in range(basis_count):
        coefficients = np.zeros(
            basis_count,
            dtype=float,
        )
        coefficients[basis_index] = 1.0

        spline = BSpline(
            knots,
            coefficients,
            degree,
            extrapolate=False,
        )
        basis_columns.append(spline(x))

    matrix = np.column_stack(basis_columns)
    names = [
        f"spline_basis_{index + 1}"
        for index in range(basis_count)
    ]

    if drop_first:
        matrix = matrix[:, 1:]
        names = names[1:]

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Non-finite values were produced by the spline basis."
        )

    return matrix, names


def treatment_dummies(
    values: pd.Series,
    levels: Sequence[Any],
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    categorical = pd.Categorical(
        values,
        categories=list(levels),
        ordered=True,
    )
    codes = categorical.codes

    if np.any(codes < 0):
        raise ValueError(
            f"{prefix} contains a value outside the fixed level list."
        )

    columns: list[np.ndarray] = []
    names: list[str] = []

    for level_index in range(1, len(levels)):
        columns.append(
            (codes == level_index).astype(float)
        )
        names.append(
            f"{prefix}[{levels[level_index]}]"
        )

    if not columns:
        return (
            np.empty((len(values), 0), dtype=float),
            names,
        )

    return np.column_stack(columns), names


def interaction_block(
    left: np.ndarray,
    left_names: Sequence[str],
    right: np.ndarray,
    right_names: Sequence[str],
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = []
    names: list[str] = []

    for left_index, left_name in enumerate(left_names):
        for right_index, right_name in enumerate(right_names):
            columns.append(
                left[:, left_index]
                * right[:, right_index]
            )
            names.append(
                f"{prefix}:{left_name}:{right_name}"
            )

    if not columns:
        return (
            np.empty((left.shape[0], 0), dtype=float),
            names,
        )

    return np.column_stack(columns), names


def all_design_blocks(
    frame: pd.DataFrame,
    ph_levels: Sequence[float],
    phone_levels: Sequence[str],
    light_levels: Sequence[str],
    basis_count: int,
    degree: int,
) -> dict[str, tuple[np.ndarray, list[str]]]:
    x_min = float(min(ph_levels))
    x_max = float(max(ph_levels))

    spline_matrix, spline_names = bspline_basis(
        frame["ph_nominal"].to_numpy(dtype=float),
        x_min,
        x_max,
        basis_count,
        degree,
        drop_first=True,
    )
    phone_matrix, phone_names = treatment_dummies(
        frame["phone"],
        phone_levels,
        "phone",
    )
    light_matrix, light_names = treatment_dummies(
        frame["light"],
        light_levels,
        "light",
    )

    phone_light = interaction_block(
        phone_matrix,
        phone_names,
        light_matrix,
        light_names,
        "phone_x_light",
    )
    phone_spline = interaction_block(
        phone_matrix,
        phone_names,
        spline_matrix,
        spline_names,
        "phone_x_spline",
    )
    light_spline = interaction_block(
        light_matrix,
        light_names,
        spline_matrix,
        spline_names,
        "light_x_spline",
    )

    return {
        "intercept": (
            np.ones((len(frame), 1), dtype=float),
            ["Intercept"],
        ),
        "spline": (spline_matrix, spline_names),
        "phone": (phone_matrix, phone_names),
        "light": (light_matrix, light_names),
        "phone_x_light": phone_light,
        "phone_x_spline": phone_spline,
        "light_x_spline": light_spline,
    }


def design_matrix(
    frame: pd.DataFrame,
    model_name: str,
    ph_levels: Sequence[float],
    phone_levels: Sequence[str],
    light_levels: Sequence[str],
    basis_count: int,
    degree: int,
) -> tuple[np.ndarray, list[str]]:
    blocks = all_design_blocks(
        frame,
        ph_levels,
        phone_levels,
        light_levels,
        basis_count,
        degree,
    )

    terms = MODEL_TERMS[model_name]
    matrices = [blocks[term][0] for term in terms]
    names = [
        name
        for term in terms
        for name in blocks[term][1]
    ]

    return np.column_stack(matrices), names


def strip_holdout_splits(
    frame: pd.DataFrame,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    all_indices = np.arange(len(frame), dtype=int)
    strip_table = (
        frame[
            ["strip_id", "phone", "ph_nominal"]
        ]
        .drop_duplicates()
        .sort_values(
            ["phone", "ph_nominal", "strip_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    splits: list[
        tuple[str, np.ndarray, np.ndarray]
    ] = []

    for row in strip_table.itertuples(index=False):
        test_mask = (
            frame["strip_id"].astype(str)
            == str(row.strip_id)
        ).to_numpy()

        train_indices = all_indices[~test_mask]
        test_indices = all_indices[test_mask]

        if len(test_indices) != frame["light"].nunique():
            raise ValueError(
                f"Strip {row.strip_id} does not have one image per light."
            )

        if (
            frame.iloc[test_indices]["phone"].nunique() != 1
            or frame.iloc[test_indices]["ph_nominal"].nunique() != 1
        ):
            raise ValueError(
                f"Strip {row.strip_id} spans more than one phone or pH."
            )

        splits.append(
            (
                str(row.strip_id),
                train_indices,
                test_indices,
            )
        )

    tested_indices = np.concatenate(
        [test_indices for _, _, test_indices in splits]
    )

    if sorted(tested_indices.tolist()) != list(range(len(frame))):
        raise ValueError(
            "Strip-holdout folds did not test every image exactly once."
        )

    return splits


def fit_predict_multiresponse(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    response_columns: Sequence[str],
    model_name: str,
    ph_levels: Sequence[float],
    phone_levels: Sequence[str],
    light_levels: Sequence[str],
    basis_count: int,
    degree: int,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    x_train, column_names = design_matrix(
        train_frame,
        model_name,
        ph_levels,
        phone_levels,
        light_levels,
        basis_count,
        degree,
    )
    x_test, _ = design_matrix(
        test_frame,
        model_name,
        ph_levels,
        phone_levels,
        light_levels,
        basis_count,
        degree,
    )

    rank = int(np.linalg.matrix_rank(x_train))
    n_columns = int(x_train.shape[1])

    if rank != n_columns:
        raise ValueError(
            f"Rank-deficient matrix for model={model_name}, "
            f"basis_count={basis_count}: "
            f"rank={rank}, columns={n_columns}."
        )

    y_train = train_frame[
        list(response_columns)
    ].to_numpy(dtype=float)

    coefficients, _, _, _ = np.linalg.lstsq(
        x_train,
        y_train,
        rcond=None,
    )
    predictions = x_test @ coefficients

    training_scale = y_train.std(axis=0, ddof=1)

    if np.any(training_scale <= 0.0):
        raise ValueError(
            "A training Lab channel has zero variance."
        )

    audit = {
        "model": model_name,
        "basis_count": basis_count,
        "degree": degree,
        "n_train": int(len(train_frame)),
        "n_test": int(len(test_frame)),
        "n_columns": n_columns,
        "rank": rank,
        "residual_df": int(len(train_frame) - rank),
        "column_names": " | ".join(column_names),
    }

    return predictions, audit, training_scale


def prediction_rows(
    test_frame: pd.DataFrame,
    predictions: np.ndarray,
    actual: np.ndarray,
    training_scale: np.ndarray,
    region: str,
    model_name: str,
    basis_count: int,
    degree: int,
    fold_id: str,
) -> list[dict[str, Any]]:
    errors = predictions - actual
    standardized_errors = (
        errors / training_scale.reshape(1, -1)
    )

    rows: list[dict[str, Any]] = []

    for row_position, (_, metadata) in enumerate(
        test_frame.iterrows()
    ):
        error = errors[row_position]
        standardized_error = standardized_errors[
            row_position
        ]

        rows.append({
            "fold_id": fold_id,
            "basis_count": basis_count,
            "degree": degree,
            "region": region,
            "model": model_name,
            "image_id": str(metadata["image_id"]),
            "strip_id": str(metadata["strip_id"]),
            "phone": str(metadata["phone"]),
            "light": str(metadata["light"]),
            "ph_nominal": float(
                metadata["ph_nominal"]
            ),
            "actual_L": float(actual[row_position, 0]),
            "actual_a": float(actual[row_position, 1]),
            "actual_b": float(actual[row_position, 2]),
            "predicted_L": float(
                predictions[row_position, 0]
            ),
            "predicted_a": float(
                predictions[row_position, 1]
            ),
            "predicted_b": float(
                predictions[row_position, 2]
            ),
            "error_L": float(error[0]),
            "error_a": float(error[1]),
            "error_b": float(error[2]),
            "squared_error_L": float(
                np.square(error[0])
            ),
            "squared_error_a": float(
                np.square(error[1])
            ),
            "squared_error_b": float(
                np.square(error[2])
            ),
            "deltaE76": float(np.linalg.norm(error)),
            "standardized_squared_error": float(
                np.mean(np.square(standardized_error))
            ),
            "standardized_euclidean_error": float(
                np.linalg.norm(standardized_error)
            ),
            "training_scale_L": float(training_scale[0]),
            "training_scale_a": float(training_scale[1]),
            "training_scale_b": float(training_scale[2]),
        })

    return rows


def summarize_predictions(
    predictions: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for keys, group in predictions.groupby(
        list(group_columns),
        sort=True,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {
            column: value
            for column, value in zip(group_columns, keys)
        }

        row.update({
            "n_predictions": int(len(group)),
            "n_strips": int(
                group["strip_id"].nunique()
            ),
            "rmse_L": float(
                np.sqrt(group["squared_error_L"].mean())
            ),
            "rmse_a": float(
                np.sqrt(group["squared_error_a"].mean())
            ),
            "rmse_b": float(
                np.sqrt(group["squared_error_b"].mean())
            ),
            "mean_deltaE76": float(
                group["deltaE76"].mean()
            ),
            "median_deltaE76": float(
                group["deltaE76"].median()
            ),
            "q90_deltaE76": float(
                group["deltaE76"].quantile(0.90)
            ),
            "standardized_rmse": float(
                np.sqrt(
                    group[
                        "standardized_squared_error"
                    ].mean()
                )
            ),
            "mean_standardized_euclidean_error": float(
                group[
                    "standardized_euclidean_error"
                ].mean()
            ),
        })
        rows.append(row)

    return pd.DataFrame(rows)


def build_fold_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    return summarize_predictions(
        predictions,
        [
            "fold_id",
            "strip_id",
            "phone",
            "ph_nominal",
            "basis_count",
            "degree",
            "region",
            "model",
        ],
    )


def stratified_phone_bootstrap(
    paired_strip_data: pd.DataFrame,
    difference_column: str,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    observed = float(
        paired_strip_data[difference_column].mean()
    )

    if repetitions <= 0:
        return {
            "mean_difference": observed,
            "bootstrap_ci_lower": np.nan,
            "bootstrap_ci_upper": np.nan,
            "bootstrap_probability_difference_below_zero": np.nan,
        }

    rng = np.random.default_rng(seed)
    phone_groups = {
        phone: group.reset_index(drop=True)
        for phone, group in paired_strip_data.groupby(
            "phone",
            sort=True,
        )
    }

    bootstrap_means = np.empty(
        repetitions,
        dtype=float,
    )

    for repetition in range(repetitions):
        sampled_values: list[np.ndarray] = []

        for group in phone_groups.values():
            indices = rng.integers(
                0,
                len(group),
                size=len(group),
            )
            sampled_values.append(
                group.iloc[indices][
                    difference_column
                ].to_numpy(dtype=float)
            )

        bootstrap_means[repetition] = float(
            np.mean(np.concatenate(sampled_values))
        )

    return {
        "mean_difference": observed,
        "bootstrap_ci_lower": float(
            np.quantile(bootstrap_means, 0.025)
        ),
        "bootstrap_ci_upper": float(
            np.quantile(bootstrap_means, 0.975)
        ),
        "bootstrap_probability_difference_below_zero": float(
            np.mean(bootstrap_means < 0.0)
        ),
    }


def paired_model_comparisons(
    fold_metrics: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    metrics = [
        "mean_deltaE76",
        "standardized_rmse",
    ]
    rows: list[dict[str, Any]] = []

    join_columns = [
        "fold_id",
        "strip_id",
        "phone",
        "ph_nominal",
        "basis_count",
        "degree",
        "region",
    ]

    for (
        numerator_model,
        denominator_model,
        comparison_label,
    ) in PAIRWISE_COMPARISONS:
        numerator = fold_metrics.loc[
            fold_metrics["model"] == numerator_model
        ].copy()
        denominator = fold_metrics.loc[
            fold_metrics["model"] == denominator_model
        ].copy()

        if numerator.empty or denominator.empty:
            continue

        merged = numerator.merge(
            denominator,
            on=join_columns,
            how="inner",
            suffixes=("_numerator", "_denominator"),
            validate="one_to_one",
        )

        for (
            basis_count,
            region,
        ), group in merged.groupby(
            ["basis_count", "region"],
            sort=True,
        ):
            for metric in metrics:
                difference_column = (
                    f"{metric}_difference"
                )
                group = group.copy()
                group[difference_column] = (
                    group[f"{metric}_numerator"]
                    - group[f"{metric}_denominator"]
                )

                bootstrap = stratified_phone_bootstrap(
                    group,
                    difference_column,
                    repetitions,
                    seed,
                )

                differences = group[difference_column]

                rows.append({
                    "basis_count": int(basis_count),
                    "degree": int(
                        group["degree"].iloc[0]
                    ),
                    "region": region,
                    "metric": metric,
                    "numerator_model": numerator_model,
                    "denominator_model": denominator_model,
                    "comparison": comparison_label,
                    "n_strips": int(len(group)),
                    "median_difference": float(
                        differences.median()
                    ),
                    "proportion_strips_numerator_better": float(
                        np.mean(differences < 0.0)
                    ),
                    **bootstrap,
                    "interpretation": (
                        "Negative values favor the numerator model."
                    ),
                })

    return pd.DataFrame(rows)


def sensitivity_summary(
    cv_summary: pd.DataFrame,
    paired: pd.DataFrame,
) -> pd.DataFrame:
    comparison = paired.loc[
        (
            paired["numerator_model"]
            == "phone_spline"
        )
        & (
            paired["denominator_model"]
            == "spline_additive"
        )
        & (paired["metric"] == "mean_deltaE76")
    ].copy()

    summary = cv_summary.merge(
        comparison[
            [
                "basis_count",
                "region",
                "mean_difference",
                "bootstrap_ci_lower",
                "bootstrap_ci_upper",
                "proportion_strips_numerator_better",
            ]
        ],
        on=["basis_count", "region"],
        how="left",
    )

    return summary


def save_figure(
    figure: plt.Figure,
    stem: Path,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)

    png_path = stem.with_suffix(".png")
    figure.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    paths = [png_path]

    if save_pdf:
        pdf_path = stem.with_suffix(".pdf")
        figure.savefig(
            pdf_path,
            bbox_inches="tight",
        )
        paths.append(pdf_path)

    plt.close(figure)
    return paths


def plot_primary_model_comparison(
    cv_summary: pd.DataFrame,
    basis_count: int,
    region: str,
    stem: Path,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    subset = cv_summary.loc[
        (cv_summary["basis_count"] == basis_count)
        & (cv_summary["region"] == region)
    ].copy()

    subset["model"] = pd.Categorical(
        subset["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    subset = subset.sort_values("model")

    figure, axis = plt.subplots(figsize=(7.7, 4.9))
    axis.bar(
        subset["model"].astype(str),
        subset["standardized_rmse"].to_numpy(dtype=float),
    )
    axis.set_ylabel(
        "Leave-one-strip-out standardized RMSE"
    )
    axis.set_title(
        f"{region}: Spline Model Comparison "
        f"(basis count={basis_count})"
    )
    axis.tick_params(axis="x", rotation=20)
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()

    return save_figure(
        figure,
        stem,
        dpi,
        save_pdf,
    )


def plot_primary_ph_errors(
    metrics_by_ph: pd.DataFrame,
    basis_count: int,
    region: str,
    stem: Path,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    subset = metrics_by_ph.loc[
        (metrics_by_ph["basis_count"] == basis_count)
        & (metrics_by_ph["region"] == region)
    ].copy()

    figure, axis = plt.subplots(figsize=(8.5, 5.2))

    for model_name, group in subset.groupby(
        "model",
        sort=True,
    ):
        group = group.sort_values("ph_nominal")
        axis.plot(
            group["ph_nominal"].to_numpy(dtype=float),
            group["mean_deltaE76"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.7,
            label=str(model_name),
        )

    axis.set_xlabel("Nominal pH")
    axis.set_ylabel("Mean leave-one-strip-out Delta E")
    axis.set_xticks(
        sorted(subset["ph_nominal"].unique())
    )
    axis.set_title(
        f"{region}: Held-out Strip Error by pH "
        f"(basis count={basis_count})"
    )
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()

    return save_figure(
        figure,
        stem,
        dpi,
        save_pdf,
    )


def plot_basis_sensitivity(
    cv_summary: pd.DataFrame,
    region: str,
    stem: Path,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    subset = cv_summary.loc[
        cv_summary["region"] == region
    ].copy()

    figure, axis = plt.subplots(figsize=(8.0, 5.0))

    for model_name, group in subset.groupby(
        "model",
        sort=True,
    ):
        group = group.sort_values("basis_count")
        axis.plot(
            group["basis_count"].to_numpy(dtype=float),
            group["standardized_rmse"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.7,
            label=str(model_name),
        )

    axis.set_xlabel("Spline basis count")
    axis.set_ylabel(
        "Leave-one-strip-out standardized RMSE"
    )
    axis.set_xticks(
        sorted(subset["basis_count"].unique())
    )
    axis.set_title(
        f"{region}: Spline Flexibility Sensitivity"
    )
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()

    return save_figure(
        figure,
        stem,
        dpi,
        save_pdf,
    )


def plot_paired_primary_difference(
    paired: pd.DataFrame,
    basis_count: int,
    stem: Path,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    subset = paired.loc[
        (
            paired["basis_count"] == basis_count
        )
        & (
            paired["numerator_model"]
            == "phone_spline"
        )
        & (
            paired["denominator_model"]
            == "spline_additive"
        )
        & (paired["metric"] == "mean_deltaE76")
    ].sort_values("region")

    means = subset["mean_difference"].to_numpy(dtype=float)
    lower = (
        means
        - subset["bootstrap_ci_lower"].to_numpy(dtype=float)
    )
    upper = (
        subset["bootstrap_ci_upper"].to_numpy(dtype=float)
        - means
    )
    x = np.arange(len(subset), dtype=float)

    figure, axis = plt.subplots(figsize=(7.2, 4.7))
    axis.errorbar(
        x,
        means,
        yerr=np.vstack([lower, upper]),
        marker="o",
        linestyle="none",
        capsize=4,
    )
    axis.axhline(0.0, linewidth=1.0, alpha=0.6)
    axis.set_xticks(x)
    axis.set_xticklabels(subset["region"])
    axis.set_ylabel(
        "Phone-spline Delta E minus additive Delta E"
    )
    axis.set_title(
        f"Paired Held-out Strip Difference "
        f"(basis count={basis_count})"
    )
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()

    return save_figure(
        figure,
        stem,
        dpi,
        save_pdf,
    )


def main() -> None:
    args = parse_args()

    if args.degree < 1:
        raise ValueError("--degree must be >= 1.")
    if args.bootstrap_reps < 0:
        raise ValueError(
            "--bootstrap-reps must be nonnegative."
        )
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0.")

    basis_counts = sorted(set(args.basis_counts))

    if not basis_counts:
        raise ValueError(
            "At least one --basis-counts value is required."
        )
    if args.primary_basis_count not in basis_counts:
        raise ValueError(
            "--primary-basis-count must be included in "
            "--basis-counts."
        )
    if min(basis_counts) < args.degree + 1:
        raise ValueError(
            "Every basis count must be at least degree + 1."
        )

    input_path = resolve_existing(args.input)
    output_dir = resolve_output(args.output_dir)
    figure_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    frame = prepare_data(pd.read_csv(input_path))

    ph_levels = sorted(
        frame["ph_nominal"].unique().tolist()
    )
    phone_levels = sorted(
        frame["phone"].unique().tolist()
    )
    light_levels = sorted(
        frame["light"].unique().tolist()
    )
    splits = strip_holdout_splits(frame)

    prediction_records: list[dict[str, Any]] = []
    rank_audit_rows: list[dict[str, Any]] = []

    for basis_count in basis_counts:
        for fold_id, train_indices, test_indices in splits:
            train_frame = frame.iloc[
                train_indices
            ].copy()
            test_frame = frame.iloc[
                test_indices
            ].copy()

            heldout_phone = str(
                test_frame["phone"].iloc[0]
            )
            heldout_ph = float(
                test_frame["ph_nominal"].iloc[0]
            )

            for region, response_columns in (
                REGION_COLUMNS.items()
            ):
                actual = test_frame[
                    response_columns
                ].to_numpy(dtype=float)

                for model_name in args.models:
                    (
                        predictions,
                        audit,
                        training_scale,
                    ) = fit_predict_multiresponse(
                        train_frame,
                        test_frame,
                        response_columns,
                        model_name,
                        ph_levels,
                        phone_levels,
                        light_levels,
                        basis_count,
                        args.degree,
                    )

                    rank_audit_rows.append({
                        "fold_id": fold_id,
                        "heldout_phone": heldout_phone,
                        "heldout_ph": heldout_ph,
                        "region": region,
                        **audit,
                    })

                    prediction_records.extend(
                        prediction_rows(
                            test_frame,
                            predictions,
                            actual,
                            training_scale,
                            region,
                            model_name,
                            basis_count,
                            args.degree,
                            fold_id,
                        )
                    )

    predictions = pd.DataFrame(prediction_records)

    expected_predictions = (
        len(frame)
        * len(basis_counts)
        * len(REGION_COLUMNS)
        * len(args.models)
    )

    if len(predictions) != expected_predictions:
        raise ValueError(
            "Unexpected number of prediction rows: "
            f"observed={len(predictions)}, "
            f"expected={expected_predictions}."
        )

    fold_metrics = build_fold_metrics(predictions)

    cv_summary = summarize_predictions(
        predictions,
        [
            "basis_count",
            "degree",
            "region",
            "model",
        ],
    )
    metrics_by_ph = summarize_predictions(
        predictions,
        [
            "basis_count",
            "degree",
            "region",
            "model",
            "ph_nominal",
        ],
    )
    metrics_by_phone = summarize_predictions(
        predictions,
        [
            "basis_count",
            "degree",
            "region",
            "model",
            "phone",
        ],
    )

    paired = paired_model_comparisons(
        fold_metrics,
        args.bootstrap_reps,
        args.seed,
    )
    sensitivity = sensitivity_summary(
        cv_summary,
        paired,
    )
    rank_audit = pd.DataFrame(rank_audit_rows)

    predictions.to_csv(
        output_dir / "cv_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_metrics.to_csv(
        output_dir / "fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv_summary.to_csv(
        output_dir / "cv_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics_by_ph.to_csv(
        output_dir / "metrics_by_ph.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics_by_phone.to_csv(
        output_dir / "metrics_by_phone.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        output_dir / "paired_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensitivity.to_csv(
        output_dir / "spline_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rank_audit.to_csv(
        output_dir / "design_rank_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for region in REGION_COLUMNS:
        plot_primary_model_comparison(
            cv_summary,
            args.primary_basis_count,
            region,
            figure_dir
            / (
                f"{region}_primary_model_comparison_"
                f"basis{args.primary_basis_count}"
            ),
            args.dpi,
            args.save_pdf,
        )
        plot_primary_ph_errors(
            metrics_by_ph,
            args.primary_basis_count,
            region,
            figure_dir
            / (
                f"{region}_error_by_ph_"
                f"basis{args.primary_basis_count}"
            ),
            args.dpi,
            args.save_pdf,
        )
        plot_basis_sensitivity(
            cv_summary,
            region,
            figure_dir
            / f"{region}_basis_sensitivity",
            args.dpi,
            args.save_pdf,
        )

    plot_paired_primary_difference(
        paired,
        args.primary_basis_count,
        figure_dir
        / (
            "phone_spline_minus_additive_"
            f"basis{args.primary_basis_count}"
        ),
        args.dpi,
        args.save_pdf,
    )

    audit = {
        "input": str(input_path),
        "output": str(output_dir),
        "rows": int(len(frame)),
        "n_strips": int(frame["strip_id"].nunique()),
        "n_folds": int(len(splits)),
        "images_per_fold": int(frame["light"].nunique()),
        "basis_counts": basis_counts,
        "primary_basis_count": args.primary_basis_count,
        "degree": args.degree,
        "models": args.models,
        "bootstrap_repetitions": args.bootstrap_reps,
        "seed": args.seed,
        "phones": phone_levels,
        "lights": light_levels,
        "ph_levels": ph_levels,
        "cross_validation": (
            "leave one complete strip_id out; all five lights removed"
        ),
        "primary_comparison": (
            "phone_spline minus spline_additive; "
            "negative error differences favor phone-specific "
            "smooth pH trajectories"
        ),
        "important_scope": (
            "This tests interpolation of an unobserved phone x pH cell "
            "from neighboring pH levels. It does not test a second "
            "independent physical strip at the same phone x pH."
        ),
        "limitations": [
            (
                "Only one physical strip exists per phone x pH cell."
            ),
            (
                "Spline smoothness is a modeling assumption and is "
                "examined with basis-count sensitivity analysis."
            ),
            (
                "Boundary pH values are harder because one-sided "
                "interpolation is required."
            ),
        ],
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

    primary_summary = cv_summary.loc[
        cv_summary["basis_count"]
        == args.primary_basis_count
    ].copy()
    primary_paired = paired.loc[
        paired["basis_count"]
        == args.primary_basis_count
    ].copy()

    notes = [
        "Leave-one-strip-out spline model comparison",
        "=" * 80,
        "",
        (
            "Each fold removes every lighting observation from one "
            "phone x pH strip."
        ),
        "",
        (
            "Primary interpretation: negative phone_spline-minus-additive "
            "differences indicate that a smooth phone-specific pH curve "
            "improves prediction of an unobserved phone x pH cell."
        ),
        "",
        f"Primary basis count: {args.primary_basis_count}",
        "",
        "Primary CV summary",
        "-" * 80,
        primary_summary.to_string(index=False),
        "",
        "Primary paired comparisons",
        "-" * 80,
        primary_paired.to_string(index=False),
        "",
        (
            "This analysis cannot prove repeatability for a second "
            "independent strip at the same phone x pH because no such "
            "replicates were collected."
        ),
    ]

    (output_dir / "analysis_notes.txt").write_text(
        "\n".join(notes),
        encoding="utf-8",
    )

    print("=" * 100)
    print("17. Leave-one-strip-out spline model comparison")
    print("=" * 100)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Rows: {len(frame)}")
    print(f"Strips/folds: {len(splits)}")
    print(f"Images removed per fold: {frame['light'].nunique()}")
    print(f"Basis counts: {basis_counts}")
    print(f"Primary basis count: {args.primary_basis_count}")
    print("\nPrimary cross-validation summary")
    print(primary_summary.to_string(index=False))
    print("\nPrimary paired comparisons")
    print(primary_paired.to_string(index=False))
    print("\nNegative paired differences favor the numerator model.")
    print("\nMain files")
    print(f"  {output_dir / 'cv_summary.csv'}")
    print(f"  {output_dir / 'paired_model_comparison.csv'}")
    print(f"  {output_dir / 'metrics_by_ph.csv'}")
    print(
        f"  {output_dir / 'spline_sensitivity_summary.csv'}"
    )
    print(f"  {figure_dir}")


if __name__ == "__main__":
    main()
