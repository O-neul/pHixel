#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
16_compare_additive_interaction_models.py

Compare additive and pH-dependent interaction models for predicting Lab values
within the observed factorial design.

This script does not use statsmodels.

Research purpose
----------------
The analysis asks whether a global additive nuisance structure is sufficient
to describe observed Lab values, or whether adding pH-dependent phone and light
interactions improves held-out cell prediction.

Models
------
1. main_effects
   pH + phone + light

2. additive_nuisance
   pH + phone + light + phone x light

3. ph_dependent_interactions
   pH + phone + light + phone x light
   + pH x phone + pH x light

Cross-validation schemes
------------------------
1. leave_one_cell_out
   Each pH x phone x light cell is predicted once from the other 194 cells.

2. cyclic_light_block
   Five folds. For every phone x pH strip, one of its five light observations
   is held out. The held-out light is cycled across strips so that every light
   level remains represented in training and every image is predicted once.

Important scope limitation
--------------------------
This is NOT leave-one-phone-out generalization. Phone levels in the test set
are also present in training. The purpose is structural model diagnosis, not
external validation to an unseen phone.

Primary paired comparison
-------------------------
ph_dependent_interactions minus additive_nuisance

Negative error differences favor the pH-dependent interaction model.

Uncertainty
-----------
Paired differences are aggregated to strip level and bootstrapped within each
phone, preserving the fixed three-phone composition.

Outputs
-------
outputs/model_structure_comparison/
├─ cv_predictions.csv
├─ cv_summary.csv
├─ cv_metrics_by_ph.csv
├─ cv_metrics_by_phone.csv
├─ cv_metrics_by_light.csv
├─ paired_model_comparison.csv
├─ full_data_fit_by_channel.csv
├─ design_rank_audit.csv
├─ analysis_audit.json
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGION_COLUMNS = {
    "pad": ["pad_L", "pad_a", "pad_b"],
    "bg": ["bg_L", "bg_a", "bg_b"],
    "contrast": ["contrast_L", "contrast_a", "contrast_b"],
}

CHANNEL_NAMES = ["L", "a", "b"]

MODEL_TERMS = {
    "main_effects": [
        "intercept",
        "pH",
        "phone",
        "light",
    ],
    "additive_nuisance": [
        "intercept",
        "pH",
        "phone",
        "light",
        "phone_x_light",
    ],
    "ph_dependent_interactions": [
        "intercept",
        "pH",
        "phone",
        "light",
        "phone_x_light",
        "pH_x_phone",
        "pH_x_light",
    ],
}

SCHEME_ORDER = [
    "leave_one_cell_out",
    "cyclic_light_block",
]

MODEL_ORDER = [
    "main_effects",
    "additive_nuisance",
    "ph_dependent_interactions",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-validated comparison of additive and interaction models."
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
        default="outputs/model_structure_comparison",
    )
    parser.add_argument(
        "--schemes",
        nargs="*",
        choices=SCHEME_ORDER,
        default=SCHEME_ORDER,
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
        default=2000,
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
        raise ValueError(
            "Duplicate pH x phone x light cells were found."
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

    valid_strips = (
        strip_audit["phone_count"].eq(1)
        & strip_audit["ph_count"].eq(1)
        & strip_audit["light_count"].eq(expected_light_count)
        & strip_audit["row_count"].eq(expected_light_count)
    )

    if not valid_strips.all():
        raise ValueError(
            "strip_id does not identify one phone x pH unit repeated "
            "under all light conditions."
        )

    return result.sort_values(
        ["phone", "ph_nominal", "light"],
        kind="stable",
    ).reset_index(drop=True)


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
            f"{prefix} contains values outside the fixed level list."
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
) -> dict[str, tuple[np.ndarray, list[str]]]:
    ph_matrix, ph_names = treatment_dummies(
        frame["ph_nominal"],
        ph_levels,
        "pH",
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
    ph_phone = interaction_block(
        ph_matrix,
        ph_names,
        phone_matrix,
        phone_names,
        "pH_x_phone",
    )
    ph_light = interaction_block(
        ph_matrix,
        ph_names,
        light_matrix,
        light_names,
        "pH_x_light",
    )

    return {
        "intercept": (
            np.ones((len(frame), 1), dtype=float),
            ["Intercept"],
        ),
        "pH": (ph_matrix, ph_names),
        "phone": (phone_matrix, phone_names),
        "light": (light_matrix, light_names),
        "phone_x_light": phone_light,
        "pH_x_phone": ph_phone,
        "pH_x_light": ph_light,
    }


def design_matrix(
    frame: pd.DataFrame,
    model_name: str,
    ph_levels: Sequence[float],
    phone_levels: Sequence[str],
    light_levels: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    blocks = all_design_blocks(
        frame,
        ph_levels,
        phone_levels,
        light_levels,
    )

    terms = MODEL_TERMS[model_name]
    matrices = [blocks[term][0] for term in terms]
    names = [
        name
        for term in terms
        for name in blocks[term][1]
    ]

    return np.column_stack(matrices), names


def fit_predict_multiresponse(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    response_columns: Sequence[str],
    model_name: str,
    ph_levels: Sequence[float],
    phone_levels: Sequence[str],
    light_levels: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    x_train, column_names = design_matrix(
        train_frame,
        model_name,
        ph_levels,
        phone_levels,
        light_levels,
    )
    x_test, _ = design_matrix(
        test_frame,
        model_name,
        ph_levels,
        phone_levels,
        light_levels,
    )

    rank = int(np.linalg.matrix_rank(x_train))
    n_columns = int(x_train.shape[1])

    if rank != n_columns:
        raise ValueError(
            f"Rank-deficient training matrix for {model_name}: "
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
            "A training response channel has zero variance."
        )

    audit = {
        "model": model_name,
        "n_train": int(len(train_frame)),
        "n_test": int(len(test_frame)),
        "n_columns": n_columns,
        "rank": rank,
        "residual_df": int(len(train_frame) - rank),
        "column_names": " | ".join(column_names),
    }

    return predictions, audit, training_scale


def leave_one_cell_out_splits(
    frame: pd.DataFrame,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    splits: list[tuple[str, np.ndarray, np.ndarray]] = []
    all_indices = np.arange(len(frame), dtype=int)

    for test_index in all_indices:
        test_mask = np.zeros(len(frame), dtype=bool)
        test_mask[test_index] = True
        train_indices = all_indices[~test_mask]
        test_indices = all_indices[test_mask]

        splits.append(
            (
                f"cell_{test_index:03d}",
                train_indices,
                test_indices,
            )
        )

    return splits


def cyclic_light_block_splits(
    frame: pd.DataFrame,
    light_levels: Sequence[str],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """
    Five folds for five lights.

    For strip position s and fold r, hold out:
        light_levels[(s + r) mod number_of_lights]

    Every strip contributes one test image per fold.
    Every image is tested exactly once across all folds.
    """
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

    strip_positions = {
        str(row.strip_id): index
        for index, row in strip_table.iterrows()
    }

    splits: list[
        tuple[str, np.ndarray, np.ndarray]
    ] = []
    all_indices = np.arange(len(frame), dtype=int)

    for fold_index in range(len(light_levels)):
        test_mask = np.zeros(len(frame), dtype=bool)

        for row_index, row in frame.iterrows():
            strip_position = strip_positions[
                str(row["strip_id"])
            ]
            target_light = light_levels[
                (
                    strip_position + fold_index
                ) % len(light_levels)
            ]

            if str(row["light"]) == str(target_light):
                test_mask[row_index] = True

        train_indices = all_indices[~test_mask]
        test_indices = all_indices[test_mask]

        if len(test_indices) != len(strip_table):
            raise ValueError(
                "Cyclic light-block fold does not contain exactly "
                "one test observation per strip."
            )

        train_light_levels = set(
            frame.iloc[train_indices]["light"].astype(str)
        )

        if train_light_levels != set(light_levels):
            raise ValueError(
                "A cyclic fold omitted a light level from training."
            )

        splits.append(
            (
                f"fold_{fold_index + 1}",
                train_indices,
                test_indices,
            )
        )

    tested = np.concatenate(
        [split[2] for split in splits]
    )

    if sorted(tested.tolist()) != list(range(len(frame))):
        raise ValueError(
            "Cyclic light-block folds did not test every row exactly once."
        )

    return splits


def split_collection(
    frame: pd.DataFrame,
    scheme: str,
    light_levels: Sequence[str],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if scheme == "leave_one_cell_out":
        return leave_one_cell_out_splits(frame)

    if scheme == "cyclic_light_block":
        return cyclic_light_block_splits(
            frame,
            light_levels,
        )

    raise ValueError(f"Unknown scheme: {scheme}")


def prediction_rows(
    test_frame: pd.DataFrame,
    predictions: np.ndarray,
    actual: np.ndarray,
    training_scale: np.ndarray,
    region: str,
    model_name: str,
    scheme: str,
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
        standardized_error = (
            standardized_errors[row_position]
        )

        rows.append({
            "scheme": scheme,
            "fold_id": fold_id,
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
            "training_scale_L": float(
                training_scale[0]
            ),
            "training_scale_a": float(
                training_scale[1]
            ),
            "training_scale_b": float(
                training_scale[2]
            ),
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


def full_data_fit_rows(
    frame: pd.DataFrame,
    ph_levels: Sequence[float],
    phone_levels: Sequence[str],
    light_levels: Sequence[str],
    models: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for region, response_columns in REGION_COLUMNS.items():
        y = frame[response_columns].to_numpy(dtype=float)

        for model_name in models:
            x, column_names = design_matrix(
                frame,
                model_name,
                ph_levels,
                phone_levels,
                light_levels,
            )
            rank = int(np.linalg.matrix_rank(x))
            n_columns = int(x.shape[1])

            if rank != n_columns:
                raise ValueError(
                    f"Full-data matrix is rank deficient for "
                    f"{model_name}: rank={rank}, columns={n_columns}."
                )

            coefficients, _, _, _ = np.linalg.lstsq(
                x,
                y,
                rcond=None,
            )
            fitted = x @ coefficients
            residuals = y - fitted
            residual_df = len(frame) - rank

            audit_rows.append({
                "scope": "full_data",
                "scheme": "",
                "fold_id": "",
                "region": region,
                "model": model_name,
                "n_train": int(len(frame)),
                "n_test": 0,
                "n_columns": n_columns,
                "rank": rank,
                "residual_df": int(residual_df),
                "column_names": " | ".join(column_names),
            })

            for channel_index, channel in enumerate(
                CHANNEL_NAMES
            ):
                observed = y[:, channel_index]
                residual = residuals[:, channel_index]
                sse = float(np.sum(np.square(residual)))
                sst = float(
                    np.sum(
                        np.square(
                            observed - observed.mean()
                        )
                    )
                )
                r_squared = (
                    1.0 - sse / sst
                    if sst > 0.0
                    else np.nan
                )
                adjusted_r_squared = (
                    1.0
                    - (1.0 - r_squared)
                    * (len(frame) - 1)
                    / residual_df
                    if residual_df > 0
                    else np.nan
                )
                rmse = (
                    np.sqrt(sse / residual_df)
                    if residual_df > 0
                    else np.nan
                )

                fit_rows.append({
                    "region": region,
                    "model": model_name,
                    "channel": channel,
                    "n": int(len(frame)),
                    "n_parameters": n_columns,
                    "rank": rank,
                    "residual_df": int(residual_df),
                    "sse": sse,
                    "sst": sst,
                    "r_squared": float(r_squared),
                    "adjusted_r_squared": float(
                        adjusted_r_squared
                    ),
                    "residual_rmse": float(rmse),
                })

    return pd.DataFrame(fit_rows), pd.DataFrame(audit_rows)


def stratified_strip_bootstrap(
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
    predictions: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    additive = predictions.loc[
        predictions["model"] == "additive_nuisance"
    ].copy()
    full = predictions.loc[
        predictions["model"]
        == "ph_dependent_interactions"
    ].copy()

    join_columns = [
        "scheme",
        "region",
        "image_id",
        "strip_id",
        "phone",
        "light",
        "ph_nominal",
    ]

    merged = full.merge(
        additive,
        on=join_columns,
        how="inner",
        suffixes=("_full", "_additive"),
        validate="one_to_one",
    )

    expected_rows = len(full)

    if len(merged) != expected_rows:
        raise ValueError(
            "Could not pair all full and additive predictions."
        )

    metrics = [
        "deltaE76",
        "standardized_squared_error",
    ]

    rows: list[dict[str, Any]] = []

    for (scheme, region), group in merged.groupby(
        ["scheme", "region"],
        sort=True,
    ):
        for metric in metrics:
            difference_column = f"{metric}_difference"
            group = group.copy()
            group[difference_column] = (
                group[f"{metric}_full"]
                - group[f"{metric}_additive"]
            )

            strip_level = (
                group.groupby(
                    ["strip_id", "phone"],
                    as_index=False,
                )[difference_column]
                .mean()
            )

            bootstrap = stratified_strip_bootstrap(
                strip_level,
                difference_column,
                repetitions,
                seed,
            )

            row_level_difference = group[
                difference_column
            ]

            rows.append({
                "scheme": scheme,
                "region": region,
                "metric": metric,
                "comparison": (
                    "ph_dependent_interactions "
                    "minus additive_nuisance"
                ),
                "n_images": int(len(group)),
                "n_strips": int(len(strip_level)),
                "mean_image_level_difference": float(
                    row_level_difference.mean()
                ),
                "median_image_level_difference": float(
                    row_level_difference.median()
                ),
                "proportion_images_full_model_better": float(
                    np.mean(row_level_difference < 0.0)
                ),
                **bootstrap,
                "interpretation": (
                    "Negative values favor "
                    "ph_dependent_interactions."
                ),
            })

    return pd.DataFrame(rows)


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


def plot_cv_model_comparison(
    cv_summary: pd.DataFrame,
    scheme: str,
    region: str,
    stem: Path,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    subset = cv_summary.loc[
        (cv_summary["scheme"] == scheme)
        & (cv_summary["region"] == region)
    ].copy()

    subset["model"] = pd.Categorical(
        subset["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    subset = subset.sort_values("model")

    figure, axis = plt.subplots(figsize=(7.8, 4.9))
    axis.bar(
        subset["model"].astype(str),
        subset["standardized_rmse"].to_numpy(dtype=float),
    )
    axis.set_ylabel("Cross-validated standardized RMSE")
    axis.set_title(
        f"{region}: Model Comparison ({scheme})"
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


def plot_ph_errors(
    by_ph: pd.DataFrame,
    scheme: str,
    region: str,
    stem: Path,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    subset = by_ph.loc[
        (by_ph["scheme"] == scheme)
        & (by_ph["region"] == region)
        & (
            by_ph["model"].isin([
                "additive_nuisance",
                "ph_dependent_interactions",
            ])
        )
    ].copy()

    figure, axis = plt.subplots(figsize=(8.4, 5.2))

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
    axis.set_ylabel("Mean cross-validated Delta E")
    axis.set_xticks(
        sorted(subset["ph_nominal"].unique())
    )
    axis.set_title(
        f"{region}: pH-specific Prediction Error ({scheme})"
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


def plot_paired_differences(
    paired: pd.DataFrame,
    scheme: str,
    metric: str,
    stem: Path,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    subset = paired.loc[
        (paired["scheme"] == scheme)
        & (paired["metric"] == metric)
    ].sort_values("region")

    x = np.arange(len(subset), dtype=float)
    means = subset["mean_difference"].to_numpy(dtype=float)
    lower = (
        means
        - subset[
            "bootstrap_ci_lower"
        ].to_numpy(dtype=float)
    )
    upper = (
        subset[
            "bootstrap_ci_upper"
        ].to_numpy(dtype=float)
        - means
    )

    figure, axis = plt.subplots(figsize=(7.4, 4.8))
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
        "Full interaction error minus additive error"
    )
    axis.set_title(
        f"Paired Model Difference: {metric} ({scheme})"
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

    if args.bootstrap_reps < 0:
        raise ValueError(
            "--bootstrap-reps must be nonnegative."
        )
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0.")

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

    (
        full_fit,
        full_rank_audit,
    ) = full_data_fit_rows(
        frame,
        ph_levels,
        phone_levels,
        light_levels,
        args.models,
    )

    prediction_records: list[dict[str, Any]] = []
    cv_rank_audit_rows: list[dict[str, Any]] = []

    for scheme in args.schemes:
        splits = split_collection(
            frame,
            scheme,
            light_levels,
        )

        for fold_id, train_indices, test_indices in splits:
            train_frame = frame.iloc[
                train_indices
            ].copy()
            test_frame = frame.iloc[
                test_indices
            ].copy()

            for region, response_columns in (
                REGION_COLUMNS.items()
            ):
                actual = test_frame[
                    response_columns
                ].to_numpy(dtype=float)

                for model_name in args.models:
                    (
                        predictions,
                        rank_audit,
                        training_scale,
                    ) = fit_predict_multiresponse(
                        train_frame,
                        test_frame,
                        response_columns,
                        model_name,
                        ph_levels,
                        phone_levels,
                        light_levels,
                    )

                    cv_rank_audit_rows.append({
                        "scope": "cross_validation",
                        "scheme": scheme,
                        "fold_id": fold_id,
                        "region": region,
                        **rank_audit,
                    })

                    prediction_records.extend(
                        prediction_rows(
                            test_frame,
                            predictions,
                            actual,
                            training_scale,
                            region,
                            model_name,
                            scheme,
                            fold_id,
                        )
                    )

    predictions = pd.DataFrame(prediction_records)

    expected_predictions = (
        len(frame)
        * len(args.schemes)
        * len(REGION_COLUMNS)
        * len(args.models)
    )

    if len(predictions) != expected_predictions:
        raise ValueError(
            "Unexpected prediction row count: "
            f"observed={len(predictions)}, "
            f"expected={expected_predictions}."
        )

    cv_summary = summarize_predictions(
        predictions,
        ["scheme", "region", "model"],
    )
    by_ph = summarize_predictions(
        predictions,
        [
            "scheme",
            "region",
            "model",
            "ph_nominal",
        ],
    )
    by_phone = summarize_predictions(
        predictions,
        ["scheme", "region", "model", "phone"],
    )
    by_light = summarize_predictions(
        predictions,
        ["scheme", "region", "model", "light"],
    )

    paired = paired_model_comparisons(
        predictions,
        args.bootstrap_reps,
        args.seed,
    )

    rank_audit = pd.concat(
        [
            full_rank_audit,
            pd.DataFrame(cv_rank_audit_rows),
        ],
        ignore_index=True,
        sort=False,
    )

    predictions.to_csv(
        output_dir / "cv_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv_summary.to_csv(
        output_dir / "cv_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_ph.to_csv(
        output_dir / "cv_metrics_by_ph.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_phone.to_csv(
        output_dir / "cv_metrics_by_phone.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_light.to_csv(
        output_dir / "cv_metrics_by_light.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        output_dir / "paired_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    full_fit.to_csv(
        output_dir / "full_data_fit_by_channel.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rank_audit.to_csv(
        output_dir / "design_rank_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for scheme in args.schemes:
        for region in REGION_COLUMNS:
            plot_cv_model_comparison(
                cv_summary,
                scheme,
                region,
                figure_dir
                / f"{scheme}_{region}_model_comparison",
                args.dpi,
                args.save_pdf,
            )
            plot_ph_errors(
                by_ph,
                scheme,
                region,
                figure_dir
                / f"{scheme}_{region}_ph_errors",
                args.dpi,
                args.save_pdf,
            )

        for metric in [
            "deltaE76",
            "standardized_squared_error",
        ]:
            plot_paired_differences(
                paired,
                scheme,
                metric,
                figure_dir
                / f"{scheme}_{metric}_paired_difference",
                args.dpi,
                args.save_pdf,
            )

    audit = {
        "input": str(input_path),
        "output": str(output_dir),
        "rows": int(len(frame)),
        "schemes": args.schemes,
        "models": args.models,
        "bootstrap_repetitions": args.bootstrap_reps,
        "seed": args.seed,
        "phones": phone_levels,
        "lights": light_levels,
        "ph_levels": ph_levels,
        "purpose": (
            "diagnose whether pH-dependent interactions improve "
            "within-design held-out Lab prediction"
        ),
        "not_claimed": (
            "This is not unseen-phone or unseen-factor-level "
            "generalization."
        ),
        "primary_comparison": (
            "ph_dependent_interactions minus additive_nuisance; "
            "negative error differences favor the full interaction model"
        ),
        "bootstrap": (
            "paired errors averaged by strip and resampled within phone"
        ),
        "limitations": [
            (
                "Other light observations from the same strip remain in "
                "training; this is intentional because the analysis tests "
                "factorial structure rather than unseen-strip prediction."
            ),
            (
                "Categorical interaction models cannot directly predict a "
                "phone level absent from training, so LOPO is not used here."
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

    notes = [
        "Additive versus pH-dependent interaction model comparison",
        "=" * 80,
        "",
        (
            "The primary result is paired_model_comparison.csv. "
            "A negative mean difference means that adding pH x phone "
            "and pH x light reduced prediction error."
        ),
        "",
        "Cross-validation summary",
        "-" * 80,
        cv_summary.to_string(index=False),
        "",
        "Paired comparison",
        "-" * 80,
        paired.to_string(index=False),
        "",
        (
            "This analysis is a structural diagnostic within observed "
            "phone, pH, and light levels. Existing LOPO classifier results "
            "remain the external phone-generalization evidence."
        ),
    ]

    (output_dir / "analysis_notes.txt").write_text(
        "\n".join(notes),
        encoding="utf-8",
    )

    print("=" * 96)
    print("16. Additive versus interaction model comparison")
    print("=" * 96)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Rows: {len(frame)}")
    print(f"Schemes: {args.schemes}")
    print(f"Models: {args.models}")
    print("\nCross-validation summary")
    print(cv_summary.to_string(index=False))
    print("\nPaired full-minus-additive comparison")
    print(paired.to_string(index=False))
    print("\nMain files")
    print(f"  {output_dir / 'cv_summary.csv'}")
    print(
        f"  {output_dir / 'paired_model_comparison.csv'}"
    )
    print(f"  {output_dir / 'cv_metrics_by_ph.csv'}")
    print(
        f"  {output_dir / 'full_data_fit_by_channel.csv'}"
    )
    print(f"  {figure_dir}")


if __name__ == "__main__":
    main()
