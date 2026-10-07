#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
24_run_hybrid_augmentation_pilot_strict_lopo.py

Strict target-free LOPO CNN augmentation pilot.

This version fixes a methodological issue in the previous pilot:
the empirical augmentation library is rebuilt separately inside every LOPO
fold using ONLY the classifier-training cells for that fold.

For each fold:
- holdout phone is completely excluded from augmentation-library estimation
- validation light is also excluded from augmentation-library estimation
- pH 7 is excluded, matching the binary classifier training set

Thus, for a 3-phone x 5-light x 13-pH design:
- classifier training rows = 2 phones x 4 lights x 12 pH = 96
- augmentation library rows = the exact same 96 training cells
- target/holdout phone contributes 0 rows to the library
- validation light contributes 0 rows to the library

Methods
-------
baseline
colorjitter
global_empirical
hybrid_empirical

global_empirical:
    whole-crop background delta
    + pH-independent ROI-relative phone contrast delta

hybrid_empirical:
    global_empirical
    + pH-dependent ROI-relative phone trajectory shape for internal pH only

The hybrid library is rebuilt independently for each LOPO fold from source
training data only. No target-phone or validation-light capture statistics are
used in the fold library.

Default pilot size
------------------
- all holdout phones
- 3 validation lights selected deterministically: first / middle / last
- 2 random seeds
- 4 methods

For a 3-phone x 5-light dataset this gives:
3 holdout phones x 3 validation lights x 2 seeds x 4 methods = 72 fits.

This is intentionally a pilot, not a hyperparameter search.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH_CANDIDATES = [
    "model_input_path",
    "local_context_path",
    "local_context_file",
    "context_path",
    "context_file",
    "crop_path",
    "crop_file",
    "preprocessed_path",
    "processed_path",
]

MASK_PATH_CANDIDATES = [
    "roi_mask_path",
    "context_roi_mask_path",
    "mask_path",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="",
        help=(
            "CSV manifest containing precomputed local-context image paths. "
            "If omitted, the script searches under outputs/."
        ),
    )
    parser.add_argument(
        "--image-col",
        default="",
        help="Optional explicit local-context image-path column.",
    )
    parser.add_argument(
        "--mask-col",
        default="",
        help="Optional explicit ROI-mask path column.",
    )
    parser.add_argument(
        "--image-root",
        default="",
        help="Optional root for relative image/mask paths.",
    )
    parser.add_argument(
        "--factorial-table",
        default="outputs/factorial_analysis/analysis_factorial_table.csv",
        help=(
            "Full capture-factorial Lab table. Fold-specific augmentation "
            "libraries are estimated from the exact classifier-training cells."
        ),
    )
    parser.add_argument(
        "--knot-count",
        type=int,
        default=4,
        help="Restricted-cubic-spline knot count for phone trajectories.",
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
    parser.add_argument(
        "--output-dir",
        default="outputs/hybrid_augmentation_pilot_strict_lopo",
    )

    parser.add_argument(
        "--methods",
        nargs="*",
        default=[
            "baseline",
            "colorjitter",
            "global_empirical",
            "hybrid_empirical",
        ],
        choices=[
            "baseline",
            "colorjitter",
            "global_empirical",
            "hybrid_empirical",
        ],
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[42, 123],
    )
    parser.add_argument(
        "--validation-light-count",
        type=int,
        default=3,
        help=(
            "Deterministically select this many validation lights. "
            "For five lights, 3 selects first/middle/last."
        ),
    )
    parser.add_argument(
        "--all-validation-lights",
        action="store_true",
    )

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--head-epochs", type=int, default=4)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--augmentation-prob", type=float, default=0.70)
    parser.add_argument("--background-strength", type=float, default=0.50)
    parser.add_argument("--global-contrast-strength", type=float, default=0.50)
    parser.add_argument("--trajectory-strength", type=float, default=0.50)

    parser.add_argument(
        "--roi-box-frac",
        nargs=4,
        type=float,
        default=[0.52, 0.08, 0.98, 0.92],
        metavar=("X1", "Y1", "X2", "Y2"),
        help=(
            "Fallback ROI rectangle in normalized local-context coordinates "
            "when no ROI mask column is available."
        ),
    )
    parser.add_argument(
        "--mask-feather-px",
        type=float,
        default=16.0,
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Only for debugging; publication pilot should use pretrained weights.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifest/library/folds without training.",
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
        f"Could not find {label}. Tried {list(candidates)}. "
        f"Columns: {list(frame.columns)}"
    )


def candidate_manifest_score(path: Path) -> tuple[int, str]:
    try:
        frame = pd.read_csv(path, nrows=10)
    except Exception:
        return (-1, "")

    columns = {str(c).lower() for c in frame.columns}

    factor_score = 0
    if columns & {"phone", "phone_id", "phone_label", "phone_model"}:
        factor_score += 1
    if columns & {"light", "light_id", "light_label"}:
        factor_score += 1
    if columns & {"ph_nominal", "ph_value", "ph"}:
        factor_score += 1

    image_matches = [
        candidate
        for candidate in IMAGE_PATH_CANDIDATES
        if candidate.lower() in columns
    ]
    if not image_matches:
        return (-1, "")

    score = factor_score * 10 + len(image_matches)

    lowered_path = str(path).lower()
    if "factorial_analysis" in lowered_path:
        score -= 20
    if "capture" in lowered_path:
        score -= 10
    if "context" in lowered_path:
        score += 5
    if "preprocess" in lowered_path:
        score += 3

    return score, image_matches[0]


def auto_find_manifest() -> Path:
    search_root = PROJECT_ROOT / "outputs"
    if not search_root.exists():
        raise FileNotFoundError(
            "Could not auto-find a manifest because outputs/ does not exist."
        )

    candidates = []
    for path in search_root.rglob("*.csv"):
        score, _ = candidate_manifest_score(path)
        if score >= 30:
            candidates.append((score, path))

    if not candidates:
        raise FileNotFoundError(
            "No manifest with phone/light/pH and a local-context image path "
            "column was found under outputs/. Pass --manifest explicitly."
        )

    candidates.sort(
        key=lambda item: (-item[0], len(str(item[1])))
    )
    return candidates[0][1].resolve()


def resolve_file_path(
    value: str,
    manifest_path: Path,
    image_root: Path | None,
) -> Path:
    path = Path(str(value))

    candidates = [path]

    if not path.is_absolute():
        if image_root is not None:
            candidates.append(image_root / path)
        candidates.append(manifest_path.parent / path)
        candidates.append(PROJECT_ROOT / path)

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    return candidates[-1].resolve()


def prepare_manifest(
    frame: pd.DataFrame,
    manifest_path: Path,
    image_col_arg: str,
    mask_col_arg: str,
    image_root: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    phone_col = find_column(
        frame,
        ["phone", "phone_label", "phone_id", "phone_model"],
        "phone",
    )
    light_col = find_column(
        frame,
        ["light", "light_label", "light_id"],
        "light",
    )
    ph_col = find_column(
        frame,
        ["ph_nominal", "ph_value", "ph"],
        "pH",
    )

    if image_col_arg:
        if image_col_arg not in frame.columns:
            raise ValueError(
                f"--image-col {image_col_arg} does not exist."
            )
        image_col = image_col_arg
    else:
        image_col = find_column(
            frame,
            IMAGE_PATH_CANDIDATES,
            "precomputed local-context image path",
        )

    mask_col = None
    if mask_col_arg:
        if mask_col_arg not in frame.columns:
            raise ValueError(
                f"--mask-col {mask_col_arg} does not exist."
            )
        mask_col = mask_col_arg
    else:
        for candidate in MASK_PATH_CANDIDATES:
            if candidate in frame.columns:
                mask_col = candidate
                break

    result = frame.copy()
    result["phone"] = result[phone_col].astype(str)
    result["light"] = result[light_col].astype(str)
    result["ph_nominal"] = pd.to_numeric(
        result[ph_col],
        errors="raise",
    ).astype(float)

    if "image_id" not in result.columns:
        if "original_file" in result.columns:
            result["image_id"] = result["original_file"].astype(str)
        else:
            result["image_id"] = [
                f"row_{index:04d}"
                for index in range(len(result))
            ]
    else:
        result["image_id"] = result["image_id"].astype(str)

    result["resolved_image_path"] = result[image_col].map(
        lambda value: str(
            resolve_file_path(
                value,
                manifest_path,
                image_root,
            )
        )
    )

    missing_images = [
        path
        for path in result["resolved_image_path"].tolist()
        if not Path(path).exists()
    ]
    if missing_images:
        raise FileNotFoundError(
            "Some local-context images do not exist. "
            f"First missing examples: {missing_images[:5]}"
        )

    roi_model_columns = [
        "roi_model_x1",
        "roi_model_y1",
        "roi_model_x2",
        "roi_model_y2",
    ]
    has_roi_model_box = all(
        column in result.columns
        for column in roi_model_columns
    )

    if mask_col is not None:
        result["resolved_mask_path"] = result[mask_col].map(
            lambda value: str(
                resolve_file_path(
                    value,
                    manifest_path,
                    image_root,
                )
            )
        )
        missing_masks = [
            path
            for path in result["resolved_mask_path"].tolist()
            if not Path(path).exists()
        ]
        if missing_masks:
            raise FileNotFoundError(
                "Some ROI masks do not exist. "
                f"First missing examples: {missing_masks[:5]}"
            )
        mask_mode = "mask_file"

    elif has_roi_model_box:
        result["resolved_mask_path"] = ""
        for column in roi_model_columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="raise",
            ).astype(float)
        mask_mode = "manifest_roi_model_box"

    else:
        result["resolved_mask_path"] = ""
        mask_mode = "fallback_right_roi_box"

    if result.duplicated(
        ["phone", "light", "ph_nominal"],
        keep=False,
    ).any():
        raise ValueError(
            "Duplicate phone x light x pH rows found in target manifest."
        )

    audit = {
        "phone_column": phone_col,
        "light_column": light_col,
        "ph_column": ph_col,
        "image_column": image_col,
        "mask_column": mask_col,
        "mask_mode": mask_mode,
    }

    return result, audit


def binary_filter(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[
        frame["ph_nominal"] != 7
    ].copy()

    result["target"] = np.where(
        result["ph_nominal"] < 7,
        0,
        1,
    ).astype(int)

    return result.reset_index(drop=True)


def select_validation_lights(
    light_levels: list[str],
    count: int,
    use_all: bool,
) -> list[str]:
    if use_all or count >= len(light_levels):
        return light_levels

    if count <= 0:
        raise ValueError(
            "validation-light-count must be positive."
        )

    indices = np.linspace(
        0,
        len(light_levels) - 1,
        count,
    )
    indices = np.unique(
        np.round(indices).astype(int)
    )

    return [light_levels[index] for index in indices]


def build_lopo_folds(
    frame: pd.DataFrame,
    validation_lights: list[str],
) -> list[dict[str, Any]]:
    phones = sorted(frame["phone"].unique().tolist())
    folds = []

    for holdout_phone in phones:
        for validation_light in validation_lights:
            test_mask = frame["phone"] == holdout_phone
            remaining = ~test_mask
            val_mask = (
                remaining
                & (frame["light"] == validation_light)
            )
            train_mask = (
                remaining
                & (frame["light"] != validation_light)
            )

            train_idx = np.flatnonzero(
                train_mask.to_numpy()
            )
            val_idx = np.flatnonzero(
                val_mask.to_numpy()
            )
            test_idx = np.flatnonzero(
                test_mask.to_numpy()
            )

            if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
                raise ValueError(
                    "A LOPO fold is empty."
                )

            folds.append({
                "fold_id": (
                    f"holdout_{holdout_phone}"
                    f"__val_{validation_light}"
                ),
                "holdout_phone": holdout_phone,
                "validation_light": validation_light,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
            })

    return folds



def prepare_factorial_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the full capture-factorial Lab table."""
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
        raise ValueError(
            f"Factorial table missing required Lab columns: {missing}"
        )

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
            "Duplicate phone x light x pH cells were found "
            "in the factorial table."
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
        raise ValueError("knot-count must be >= 3.")

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
        names.append(
            f"rcs_nonlinear_{index + 1}"
        )

    return (
        np.column_stack(columns),
        names,
        knots_original,
    )


def fit_phone_curves_training_only(
    frame: pd.DataFrame,
    knot_count: int,
    min_ph: float,
    max_ph: float,
) -> pd.DataFrame:
    """
    Fit phone-specific contrast trajectories to the fold TRAINING data only.

    pH 7 is absent because the strict fold library is built from the same
    binary-task training cells as the classifier.
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
    if len(ph_grid) < knot_count:
        raise ValueError(
            "Not enough internal pH levels to fit the RCS trajectory."
        )

    x_grid = np.asarray(ph_grid, dtype=float)
    design_grid, _, knots = restricted_cubic_spline_basis(
        x_grid,
        min_ph,
        max_ph,
        knot_count,
    )

    phones = sorted(
        means["phone"].unique().tolist()
    )
    rows = []

    for phone in phones:
        group = (
            means.loc[means["phone"] == phone]
            .sort_values("ph_nominal")
        )
        if group["ph_nominal"].tolist() != ph_grid:
            raise ValueError(
                f"Training phone {phone} does not contain the "
                "complete internal binary-task pH grid."
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
        centered = (
            fitted
            - fitted.mean(axis=0, keepdims=True)
        )

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
                "knots": ",".join(
                    f"{value:g}"
                    for value in knots
                ),
            })

    return pd.DataFrame(rows)


def directed_pairwise_deltas(
    condition_table: pd.DataFrame,
    key_columns: list[str],
    value_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    rows = []
    records = condition_table.reset_index(
        drop=True
    )

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
                    target[value_column]
                    - source[value_column]
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
                    "delta_type": (
                        "pH_dependent_phone_shape"
                    ),
                })

    return pd.DataFrame(rows)


def summarize_delta_table(
    table: pd.DataFrame,
    name: str,
) -> dict[str, Any]:
    values = table["deltaE76"].to_numpy(
        dtype=float
    )
    return {
        "name": name,
        "n_vectors": int(len(values)),
        "mean_deltaE76": float(
            np.mean(values)
        ),
        "median_deltaE76": float(
            np.median(values)
        ),
        "q90_deltaE76": float(
            np.quantile(values, 0.90)
        ),
        "max_deltaE76": float(
            np.max(values)
        ),
    }


def build_fold_training_only_library(
    factorial: pd.DataFrame,
    classifier_train_frame: pd.DataFrame,
    holdout_phone: str,
    validation_light: str,
    knot_count: int,
    min_ph: float,
    max_ph: float,
    fold_library_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Build an empirical augmentation library from the exact 96 classifier
    training cells in one LOPO fold.

    This is the key anti-leakage rule:
    - 0 target-phone cells
    - 0 validation-light cells
    - pH 7 excluded
    """
    library_frame = factorial.loc[
        (factorial["phone"] != holdout_phone)
        & (factorial["light"] != validation_light)
        & (factorial["ph_nominal"] != 7)
    ].copy()

    train_keys = {
        (
            str(row.phone),
            str(row.light),
            float(row.ph_nominal),
        )
        for row in classifier_train_frame[
            ["phone", "light", "ph_nominal"]
        ].itertuples(index=False)
    }
    library_keys = {
        (
            str(row.phone),
            str(row.light),
            float(row.ph_nominal),
        )
        for row in library_frame[
            ["phone", "light", "ph_nominal"]
        ].itertuples(index=False)
    }

    if train_keys != library_keys:
        missing_from_library = sorted(
            train_keys - library_keys
        )
        extra_in_library = sorted(
            library_keys - train_keys
        )
        raise ValueError(
            "Classifier-training cells and fold-library cells do not "
            "match exactly. "
            f"Missing: {missing_from_library[:5]}; "
            f"Extra: {extra_in_library[:5]}"
        )

    if (
        library_frame["phone"]
        .eq(holdout_phone)
        .any()
    ):
        raise AssertionError(
            "Target/holdout phone leaked into fold library."
        )
    if (
        library_frame["light"]
        .eq(validation_light)
        .any()
    ):
        raise AssertionError(
            "Validation light leaked into fold library."
        )
    if (
        library_frame["ph_nominal"]
        .eq(7)
        .any()
    ):
        raise AssertionError(
            "pH7 leaked into binary-task fold library."
        )

    bg_condition_means = (
        library_frame.groupby(
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
        library_frame.groupby(
            ["phone"],
            as_index=False,
        )[
            ["contrast_L", "contrast_a", "contrast_b"]
        ]
        .mean()
    )
    contrast_global_pairwise = (
        directed_pairwise_deltas(
            contrast_phone_means,
            ["phone"],
            [
                "contrast_L",
                "contrast_a",
                "contrast_b",
            ],
            "global_phone_contrast",
        )
    )

    curves = fit_phone_curves_training_only(
        library_frame,
        knot_count,
        min_ph,
        max_ph,
    )
    trajectory_pairwise = (
        trajectory_pairwise_deltas(curves)
    )

    summary = pd.DataFrame([
        summarize_delta_table(
            bg_pairwise,
            "global_background",
        ),
        summarize_delta_table(
            contrast_global_pairwise,
            "global_phone_contrast",
        ),
        summarize_delta_table(
            trajectory_pairwise,
            "pH_dependent_phone_shape",
        ),
    ])

    library = {
        "version": 2,
        "library_mode": (
            "strict_fold_training_only"
        ),
        "holdout_phone_excluded": holdout_phone,
        "validation_light_excluded": (
            validation_light
        ),
        "n_source_training_rows": int(
            len(library_frame)
        ),
        "source_phones": sorted(
            library_frame["phone"]
            .unique()
            .tolist()
        ),
        "source_lights": sorted(
            library_frame["light"]
            .unique()
            .tolist()
        ),
        "source_ph_levels": sorted(
            library_frame["ph_nominal"]
            .unique()
            .tolist()
        ),
        "knot_count": int(knot_count),
        "trajectory_min_ph": float(min_ph),
        "trajectory_max_ph": float(max_ph),
        "boundary_policy": (
            "pH-dependent trajectory disabled outside "
            f"[{min_ph:g}, {max_ph:g}]"
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
            for (
                ph_value,
                group,
            ) in trajectory_pairwise.groupby(
                "ph_nominal",
                sort=True,
            )
        },
        "summary": summary.to_dict(
            "records"
        ),
    }

    audit = {
        "library_mode": (
            "strict_fold_training_only"
        ),
        "n_classifier_train_rows": int(
            len(classifier_train_frame)
        ),
        "n_library_rows": int(
            len(library_frame)
        ),
        "exact_training_cell_match": bool(
            train_keys == library_keys
        ),
        "holdout_phone": holdout_phone,
        "holdout_phone_rows_in_library": int(
            (
                library_frame["phone"]
                == holdout_phone
            ).sum()
        ),
        "validation_light": validation_light,
        "validation_light_rows_in_library": int(
            (
                library_frame["light"]
                == validation_light
            ).sum()
        ),
        "pH7_rows_in_library": int(
            (
                library_frame["ph_nominal"]
                == 7
            ).sum()
        ),
        "phones_used": sorted(
            library_frame["phone"]
            .unique()
            .tolist()
        ),
        "lights_used": sorted(
            library_frame["light"]
            .unique()
            .tolist()
        ),
        "ph_levels_used": sorted(
            library_frame["ph_nominal"]
            .unique()
            .tolist()
        ),
        "n_background_vectors": int(
            len(bg_pairwise)
        ),
        "n_global_contrast_vectors": int(
            len(contrast_global_pairwise)
        ),
        "n_trajectory_vectors": int(
            len(trajectory_pairwise)
        ),
        "critical_note": (
            "No target-phone capture statistics and no validation-light "
            "capture statistics are used in this fold library."
        ),
    }

    fold_library_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    library_frame.to_csv(
        fold_library_dir
        / "library_training_cells.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        fold_library_dir
        / "library_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bg_pairwise.to_csv(
        fold_library_dir
        / "background_pairwise_deltas.csv",
        index=False,
        encoding="utf-8-sig",
    )
    contrast_global_pairwise.to_csv(
        fold_library_dir
        / "contrast_global_pairwise_deltas.csv",
        index=False,
        encoding="utf-8-sig",
    )
    curves.to_csv(
        fold_library_dir
        / "contrast_phone_rcs_curves.csv",
        index=False,
        encoding="utf-8-sig",
    )
    trajectory_pairwise.to_csv(
        fold_library_dir
        / "contrast_trajectory_pairwise_deltas.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (
        fold_library_dir
        / "augmentation_library.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            library,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    with (
        fold_library_dir
        / "library_audit.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            audit,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    return library, audit

def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(
        np.asarray(rgb, dtype=np.float64),
        0.0,
        1.0,
    )

    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )

    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])

    xyz = linear @ matrix.T
    white = np.array([
        0.95047,
        1.00000,
        1.08883,
    ])
    xyz = xyz / white

    delta = 6 / 29

    def f(t):
        return np.where(
            t > delta ** 3,
            np.cbrt(t),
            t / (3 * delta ** 2) + 4 / 29,
        )

    fx = f(xyz[..., 0])
    fy = f(xyz[..., 1])
    fz = f(xyz[..., 2])

    lab = np.empty_like(rgb)
    lab[..., 0] = 116 * fy - 16
    lab[..., 1] = 500 * (fx - fy)
    lab[..., 2] = 200 * (fy - fz)
    return lab


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)

    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200

    delta = 6 / 29

    def f_inv(t):
        return np.where(
            t > delta,
            t ** 3,
            3 * delta ** 2 * (t - 4 / 29),
        )

    xyz = np.stack(
        [
            f_inv(fx),
            f_inv(fy),
            f_inv(fz),
        ],
        axis=-1,
    )

    white = np.array([
        0.95047,
        1.00000,
        1.08883,
    ])
    xyz = xyz * white

    matrix_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ])

    linear = xyz @ matrix_inv.T

    rgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055
        * np.power(
            np.maximum(linear, 0.0),
            1 / 2.4,
        )
        - 0.055,
    )

    return np.clip(rgb, 0.0, 1.0)


def load_library(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        library = json.load(handle)

    required = {
        "background_pairwise_deltas",
        "global_contrast_pairwise_deltas",
        "trajectory_pairwise_deltas_by_ph",
        "trajectory_min_ph",
        "trajectory_max_ph",
    }
    missing = sorted(required - set(library))
    if missing:
        raise ValueError(
            f"Augmentation library missing keys: {missing}"
        )

    return library


def soft_rectangle_mask(
    size: int,
    box_fraction: list[float],
    feather_px: float,
) -> np.ndarray:
    x1, y1, x2, y2 = box_fraction

    if not (
        0 <= x1 < x2 <= 1
        and 0 <= y1 < y2 <= 1
    ):
        raise ValueError(
            "roi-box-frac must satisfy 0<=x1<x2<=1 and 0<=y1<y2<=1."
        )

    mask = Image.new(
        "L",
        (size, size),
        color=0,
    )

    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle(
        [
            int(round(x1 * size)),
            int(round(y1 * size)),
            int(round(x2 * size)),
            int(round(y2 * size)),
        ],
        fill=255,
    )

    if feather_px > 0:
        mask = mask.filter(
            ImageFilter.GaussianBlur(
                radius=float(feather_px)
            )
        )

    return (
        np.asarray(mask, dtype=np.float32)
        / 255.0
    )


def load_mask(
    mask_path: str,
    size: int,
    box_fraction: list[float],
    feather_px: float,
) -> np.ndarray:
    if mask_path:
        mask = (
            Image.open(mask_path)
            .convert("L")
            .resize(
                (size, size),
                resample=Image.Resampling.BILINEAR,
            )
        )
        if feather_px > 0:
            mask = mask.filter(
                ImageFilter.GaussianBlur(
                    radius=float(feather_px)
                )
            )
        return (
            np.asarray(mask, dtype=np.float32)
            / 255.0
        )

    return soft_rectangle_mask(
        size,
        box_fraction,
        feather_px,
    )


def mask_from_model_roi_box(
    row: pd.Series,
    original_width: int,
    original_height: int,
    target_size: int,
    feather_px: float,
) -> np.ndarray:
    """Build the ROI mask from manifest roi_model_* coordinates."""
    x1 = float(row["roi_model_x1"])
    y1 = float(row["roi_model_y1"])
    x2 = float(row["roi_model_x2"])
    y2 = float(row["roi_model_y2"])

    scale_x = (
        target_size
        / float(original_width)
    )
    scale_y = (
        target_size
        / float(original_height)
    )

    x1 *= scale_x
    x2 *= scale_x
    y1 *= scale_y
    y2 *= scale_y

    x1 = int(round(
        np.clip(x1, 0, target_size)
    ))
    x2 = int(round(
        np.clip(x2, 0, target_size)
    ))
    y1 = int(round(
        np.clip(y1, 0, target_size)
    ))
    y2 = int(round(
        np.clip(y2, 0, target_size)
    ))

    if not (
        x1 < x2
        and y1 < y2
    ):
        raise ValueError(
            "Invalid roi_model box after scaling: "
            f"({x1}, {y1}, {x2}, {y2})"
        )

    mask = Image.new(
        "L",
        (target_size, target_size),
        color=0,
    )

    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle(
        [x1, y1, x2, y2],
        fill=255,
    )

    if feather_px > 0:
        mask = mask.filter(
            ImageFilter.GaussianBlur(
                radius=float(feather_px)
            )
        )

    return (
        np.asarray(mask, dtype=np.float32)
        / 255.0
    )


class EmpiricalLabAugment:
    def __init__(
        self,
        library: dict[str, Any],
        method: str,
        probability: float,
        background_strength: float,
        global_contrast_strength: float,
        trajectory_strength: float,
    ):
        self.method = method
        self.probability = float(probability)
        self.background_strength = float(
            background_strength
        )
        self.global_contrast_strength = float(
            global_contrast_strength
        )
        self.trajectory_strength = float(
            trajectory_strength
        )

        self.bg = np.asarray(
            library["background_pairwise_deltas"],
            dtype=float,
        )
        self.global_contrast = np.asarray(
            library["global_contrast_pairwise_deltas"],
            dtype=float,
        )
        self.trajectory = {
            str(key): np.asarray(value, dtype=float)
            for key, value in (
                library[
                    "trajectory_pairwise_deltas_by_ph"
                ].items()
            )
        }
        self.min_ph = float(
            library["trajectory_min_ph"]
        )
        self.max_ph = float(
            library["trajectory_max_ph"]
        )

    @staticmethod
    def sample_scaled(
        table: np.ndarray,
        max_strength: float,
    ) -> np.ndarray:
        if len(table) == 0 or max_strength <= 0:
            return np.zeros(3, dtype=float)

        index = np.random.randint(0, len(table))
        amplitude = np.random.uniform(
            0.0,
            max_strength,
        )
        return amplitude * table[index]

    def __call__(
        self,
        image: Image.Image,
        ph_value: float,
        roi_mask: np.ndarray,
    ) -> Image.Image:
        if np.random.random() > self.probability:
            return image

        rgb = (
            np.asarray(image, dtype=np.float64)
            / 255.0
        )
        lab = srgb_to_lab(rgb)

        bg_delta = self.sample_scaled(
            self.bg,
            self.background_strength,
        )
        lab = lab + bg_delta.reshape(1, 1, 3)

        roi_delta = self.sample_scaled(
            self.global_contrast,
            self.global_contrast_strength,
        )

        if self.method == "hybrid_empirical":
            if (
                ph_value >= self.min_ph
                and ph_value <= self.max_ph
            ):
                key = f"{float(ph_value):g}"
                if key in self.trajectory:
                    roi_delta = (
                        roi_delta
                        + self.sample_scaled(
                            self.trajectory[key],
                            self.trajectory_strength,
                        )
                    )

        lab = (
            lab
            + roi_mask[..., None]
            * roi_delta.reshape(1, 1, 3)
        )

        lab[..., 0] = np.clip(
            lab[..., 0],
            0.0,
            100.0,
        )
        lab[..., 1] = np.clip(
            lab[..., 1],
            -128.0,
            127.0,
        )
        lab[..., 2] = np.clip(
            lab[..., 2],
            -128.0,
            127.0,
        )

        output = (
            lab_to_srgb(lab)
            * 255.0
        ).round().astype(np.uint8)

        return Image.fromarray(
            output,
            mode="RGB",
        )


class PilotDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_size: int,
        method: str,
        library: dict[str, Any] | None,
        training: bool,
        args,
    ):
        self.frame = frame.reset_index(drop=True)
        self.image_size = int(image_size)
        self.method = method
        self.training = training
        self.args = args

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.to_tensor = transforms.ToTensor()

        self.colorjitter = transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.20,
            hue=0.05,
        )

        self.empirical = None
        if method in {
            "global_empirical",
            "hybrid_empirical",
        }:
            if library is None:
                raise ValueError(
                    "Empirical method requires augmentation library."
                )
            self.empirical = EmpiricalLabAugment(
                library=library,
                method=method,
                probability=args.augmentation_prob,
                background_strength=args.background_strength,
                global_contrast_strength=args.global_contrast_strength,
                trajectory_strength=args.trajectory_strength,
            )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]

        image = Image.open(
            row["resolved_image_path"]
        ).convert("RGB")

        original_width, original_height = (
            image.size
        )

        image = image.resize(
            (
                self.image_size,
                self.image_size,
            ),
            resample=Image.Resampling.BILINEAR,
        )

        if self.training:
            if self.method == "colorjitter":
                image = self.colorjitter(image)

            elif self.method in {
                "global_empirical",
                "hybrid_empirical",
            }:
                if row["resolved_mask_path"]:
                    mask = load_mask(
                        row["resolved_mask_path"],
                        self.image_size,
                        self.args.roi_box_frac,
                        self.args.mask_feather_px,
                    )
                elif all(
                    column in row.index
                    for column in [
                        "roi_model_x1",
                        "roi_model_y1",
                        "roi_model_x2",
                        "roi_model_y2",
                    ]
                ):
                    mask = mask_from_model_roi_box(
                        row=row,
                        original_width=(
                            original_width
                        ),
                        original_height=(
                            original_height
                        ),
                        target_size=(
                            self.image_size
                        ),
                        feather_px=(
                            self.args.mask_feather_px
                        ),
                    )
                else:
                    mask = load_mask(
                        "",
                        self.image_size,
                        self.args.roi_box_frac,
                        self.args.mask_feather_px,
                    )

                image = self.empirical(
                    image,
                    float(row["ph_nominal"]),
                    mask,
                )

        tensor = self.normalize(
            self.to_tensor(image)
        )

        return (
            tensor,
            int(row["target"]),
            str(row["image_id"]),
        )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )
        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def make_model(
    pretrained: bool,
) -> nn.Module:
    weights = (
        ResNet18_Weights.DEFAULT
        if pretrained
        else None
    )

    model = resnet18(weights=weights)
    model.fc = nn.Linear(
        model.fc.in_features,
        2,
    )
    return model


def freeze_all_except_head(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def unfreeze_layer4_and_head(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def freeze_batchnorm_stats(model):
    for module in model.modules():
        if isinstance(
            module,
            nn.modules.batchnorm._BatchNorm,
        ):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False


def make_loaders(
    train_frame,
    val_frame,
    test_frame,
    method,
    library,
    seed,
    args,
):
    train_dataset = PilotDataset(
        train_frame,
        args.image_size,
        method,
        library,
        training=True,
        args=args,
    )
    val_dataset = PilotDataset(
        val_frame,
        args.image_size,
        method,
        library,
        training=False,
        args=args,
    )
    test_dataset = PilotDataset(
        test_frame,
        args.image_size,
        method,
        library,
        training=False,
        args=args,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    y_true = []
    y_pred = []
    ids = []
    probabilities = []

    with torch.no_grad():
        for images, targets, image_ids in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(
                logits,
                dim=1,
            )[:, 1]
            pred = torch.argmax(
                logits,
                dim=1,
            )

            y_true.extend(
                targets.numpy().tolist()
            )
            y_pred.extend(
                pred.cpu().numpy().tolist()
            )
            probabilities.extend(
                probs.cpu().numpy().tolist()
            )
            ids.extend(list(image_ids))

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }

    prediction_table = pd.DataFrame({
        "image_id": ids,
        "target": y_true,
        "prediction": y_pred,
        "probability_alkaline": probabilities,
    })

    return metrics, prediction_table


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    model.train()
    freeze_batchnorm_stats(model)

    total_loss = 0.0
    total_count = 0

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(
            logits,
            targets,
        )
        loss.backward()
        optimizer.step()

        total_loss += (
            float(loss.item())
            * len(targets)
        )
        total_count += len(targets)

    return total_loss / max(total_count, 1)


def train_one_run(
    train_loader,
    val_loader,
    test_loader,
    device,
    args,
):
    model = make_model(
        pretrained=not args.no_pretrained
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    freeze_all_except_head(model)
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    best_state = copy.deepcopy(
        model.state_dict()
    )
    best_val = -np.inf
    history = []

    for epoch in range(1, args.head_epochs + 1):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        val_metrics, _ = evaluate(
            model,
            val_loader,
            device,
        )

        history.append({
            "stage": "head",
            "epoch": epoch,
            "train_loss": loss,
            **{
                f"val_{key}": value
                for key, value in val_metrics.items()
            },
        })

        if val_metrics["balanced_accuracy"] > best_val:
            best_val = val_metrics[
                "balanced_accuracy"
            ]
            best_state = copy.deepcopy(
                model.state_dict()
            )

    model.load_state_dict(best_state)

    unfreeze_layer4_and_head(model)
    freeze_batchnorm_stats(model)

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=args.finetune_lr,
        weight_decay=args.weight_decay,
    )

    patience_counter = 0

    for epoch in range(1, args.finetune_epochs + 1):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        val_metrics, _ = evaluate(
            model,
            val_loader,
            device,
        )

        history.append({
            "stage": "layer4",
            "epoch": epoch,
            "train_loss": loss,
            **{
                f"val_{key}": value
                for key, value in val_metrics.items()
            },
        })

        current = val_metrics[
            "balanced_accuracy"
        ]

        if current > best_val + 1e-9:
            best_val = current
            best_state = copy.deepcopy(
                model.state_dict()
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            break

    model.load_state_dict(best_state)

    val_metrics, _ = evaluate(
        model,
        val_loader,
        device,
    )
    test_metrics, test_predictions = evaluate(
        model,
        test_loader,
        device,
    )

    return (
        val_metrics,
        test_metrics,
        test_predictions,
        pd.DataFrame(history),
    )


def summarize_runs(run_table: pd.DataFrame):
    method_summary = (
        run_table.groupby(
            "method",
            as_index=False,
        )
        .agg(
            n_runs=("macro_f1", "size"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=(
                "balanced_accuracy",
                "mean",
            ),
            balanced_accuracy_std=(
                "balanced_accuracy",
                "std",
            ),
            accuracy_mean=("accuracy", "mean"),
        )
    )

    by_phone = (
        run_table.groupby(
            ["method", "holdout_phone"],
            as_index=False,
        )
        .agg(
            n_runs=("macro_f1", "size"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=(
                "balanced_accuracy",
                "mean",
            ),
        )
    )

    return method_summary, by_phone


def paired_comparisons(
    run_table: pd.DataFrame,
) -> pd.DataFrame:
    primary = "hybrid_empirical"
    comparators = [
        "baseline",
        "colorjitter",
        "global_empirical",
    ]

    rows = []
    keys = [
        "fold_id",
        "holdout_phone",
        "validation_light",
        "seed",
    ]

    hybrid = run_table.loc[
        run_table["method"] == primary
    ]

    for comparator in comparators:
        other = run_table.loc[
            run_table["method"] == comparator
        ]

        if hybrid.empty or other.empty:
            continue

        merged = hybrid.merge(
            other,
            on=keys,
            how="inner",
            suffixes=("_hybrid", "_comparator"),
            validate="one_to_one",
        )

        for metric in [
            "macro_f1",
            "balanced_accuracy",
        ]:
            difference = (
                merged[f"{metric}_hybrid"]
                - merged[f"{metric}_comparator"]
            )

            rows.append({
                "comparison": (
                    f"hybrid_empirical minus {comparator}"
                ),
                "metric": metric,
                "n_matched_runs": int(len(merged)),
                "mean_difference": float(
                    difference.mean()
                ),
                "median_difference": float(
                    difference.median()
                ),
                "proportion_hybrid_better": float(
                    np.mean(difference > 0)
                ),
                "proportion_equal": float(
                    np.mean(np.isclose(difference, 0))
                ),
            })

    return pd.DataFrame(rows)


def build_decision_summary(
    by_phone: pd.DataFrame,
    method_summary: pd.DataFrame,
) -> pd.DataFrame:
    methods = set(
        method_summary["method"].tolist()
    )

    required = {
        "baseline",
        "global_empirical",
        "hybrid_empirical",
    }
    if not required.issubset(methods):
        return pd.DataFrame([{
            "criterion": "pilot_decision",
            "passed": False,
            "detail": (
                "Cannot evaluate pre-specified decision because "
                "baseline/global_empirical/hybrid_empirical were not all run."
            ),
        }])

    pivot = by_phone.pivot(
        index="holdout_phone",
        columns="method",
        values="macro_f1_mean",
    )

    phone_vs_baseline = (
        pivot["hybrid_empirical"]
        > pivot["baseline"]
    )
    phone_vs_global = (
        pivot["hybrid_empirical"]
        > pivot["global_empirical"]
    )

    overall = method_summary.set_index("method")

    overall_vs_baseline = (
        overall.loc[
            "hybrid_empirical",
            "macro_f1_mean",
        ]
        > overall.loc[
            "baseline",
            "macro_f1_mean",
        ]
    )
    overall_vs_global = (
        overall.loc[
            "hybrid_empirical",
            "macro_f1_mean",
        ]
        > overall.loc[
            "global_empirical",
            "macro_f1_mean",
        ]
    )

    needed_phones = math.ceil(
        2 * len(pivot) / 3
    )

    pass_phone_baseline = int(
        phone_vs_baseline.sum()
    ) >= needed_phones
    pass_phone_global = int(
        phone_vs_global.sum()
    ) >= needed_phones

    primary_pass = bool(
        overall_vs_baseline
        and overall_vs_global
        and pass_phone_baseline
        and pass_phone_global
    )

    rows = [
        {
            "criterion": (
                "overall hybrid macro-F1 > baseline"
            ),
            "passed": bool(
                overall_vs_baseline
            ),
            "detail": (
                f"{overall.loc['hybrid_empirical','macro_f1_mean']:.6f} "
                f"vs {overall.loc['baseline','macro_f1_mean']:.6f}"
            ),
        },
        {
            "criterion": (
                "overall hybrid macro-F1 > global empirical"
            ),
            "passed": bool(
                overall_vs_global
            ),
            "detail": (
                f"{overall.loc['hybrid_empirical','macro_f1_mean']:.6f} "
                f"vs {overall.loc['global_empirical','macro_f1_mean']:.6f}"
            ),
        },
        {
            "criterion": (
                "hybrid beats baseline in at least 2/3 holdout phones"
            ),
            "passed": bool(
                pass_phone_baseline
            ),
            "detail": (
                f"{int(phone_vs_baseline.sum())}/{len(pivot)} phones"
            ),
        },
        {
            "criterion": (
                "hybrid beats global empirical in at least 2/3 holdout phones"
            ),
            "passed": bool(
                pass_phone_global
            ),
            "detail": (
                f"{int(phone_vs_global.sum())}/{len(pivot)} phones"
            ),
        },
        {
            "criterion": "PRIMARY PILOT SUCCESS",
            "passed": primary_pass,
            "detail": (
                "Pre-specified success requires all four criteria above."
            ),
        },
    ]

    if "colorjitter" in methods:
        color_better = (
            overall.loc[
                "hybrid_empirical",
                "macro_f1_mean",
            ]
            > overall.loc[
                "colorjitter",
                "macro_f1_mean",
            ]
        )

        rows.append({
            "criterion": (
                "secondary: hybrid macro-F1 > ColorJitter"
            ),
            "passed": bool(color_better),
            "detail": (
                f"{overall.loc['hybrid_empirical','macro_f1_mean']:.6f} "
                f"vs {overall.loc['colorjitter','macro_f1_mean']:.6f}"
            ),
        })

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    if not args.seeds:
        raise ValueError(
            "At least one seed is required."
        )
    if (
        args.trajectory_min_ph
        >= args.trajectory_max_ph
    ):
        raise ValueError(
            "trajectory-min-ph must be below trajectory-max-ph."
        )

    factorial_path = resolve_existing(
        args.factorial_table
    )
    factorial = prepare_factorial_table(
        pd.read_csv(factorial_path)
    )

    if args.manifest.strip():
        manifest_path = resolve_existing(
            args.manifest
        )
    else:
        manifest_path = auto_find_manifest()

    image_root = (
        resolve_existing(args.image_root)
        if args.image_root.strip()
        else None
    )

    manifest_raw = pd.read_csv(
        manifest_path
    )
    manifest, manifest_audit = (
        prepare_manifest(
            manifest_raw,
            manifest_path,
            args.image_col,
            args.mask_col,
            image_root,
        )
    )
    manifest = binary_filter(
        manifest
    )

    phones = sorted(
        manifest["phone"]
        .unique()
        .tolist()
    )
    lights = sorted(
        manifest["light"]
        .unique()
        .tolist()
    )

    factorial_phones = sorted(
        factorial["phone"]
        .unique()
        .tolist()
    )
    factorial_lights = sorted(
        factorial["light"]
        .unique()
        .tolist()
    )

    if phones != factorial_phones:
        raise ValueError(
            "Phone levels differ between manifest and factorial table. "
            f"Manifest={phones}; factorial={factorial_phones}"
        )
    if lights != factorial_lights:
        raise ValueError(
            "Light levels differ between manifest and factorial table. "
            f"Manifest={lights}; factorial={factorial_lights}"
        )

    validation_lights = (
        select_validation_lights(
            lights,
            args.validation_light_count,
            args.all_validation_lights,
        )
    )

    folds = build_lopo_folds(
        manifest,
        validation_lights,
    )

    output_dir = resolve_output(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_library_root = (
        output_dir
        / "fold_libraries_training_only"
    )
    fold_library_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_audit_rows = []
    fold_libraries = {}

    for fold in folds:
        train_frame = manifest.iloc[
            fold["train_idx"]
        ].copy()

        fold_library_dir = (
            fold_library_root
            / fold["fold_id"]
        )

        library, library_audit = (
            build_fold_training_only_library(
                factorial=factorial,
                classifier_train_frame=(
                    train_frame
                ),
                holdout_phone=(
                    fold["holdout_phone"]
                ),
                validation_light=(
                    fold["validation_light"]
                ),
                knot_count=args.knot_count,
                min_ph=args.trajectory_min_ph,
                max_ph=args.trajectory_max_ph,
                fold_library_dir=(
                    fold_library_dir
                ),
            )
        )
        fold_libraries[
            fold["fold_id"]
        ] = library

        fold_audit_rows.append({
            "fold_id": fold["fold_id"],
            "holdout_phone": (
                fold["holdout_phone"]
            ),
            "validation_light": (
                fold["validation_light"]
            ),
            "n_train": len(
                fold["train_idx"]
            ),
            "n_val": len(
                fold["val_idx"]
            ),
            "n_test": len(
                fold["test_idx"]
            ),
            "library_mode": (
                library_audit[
                    "library_mode"
                ]
            ),
            "n_library_rows": (
                library_audit[
                    "n_library_rows"
                ]
            ),
            "library_exact_train_match": (
                library_audit[
                    "exact_training_cell_match"
                ]
            ),
            "target_phone_rows_in_library": (
                library_audit[
                    "holdout_phone_rows_in_library"
                ]
            ),
            "validation_light_rows_in_library": (
                library_audit[
                    "validation_light_rows_in_library"
                ]
            ),
            "pH7_rows_in_library": (
                library_audit[
                    "pH7_rows_in_library"
                ]
            ),
            "n_bg_vectors": (
                library_audit[
                    "n_background_vectors"
                ]
            ),
            "n_global_contrast_vectors": (
                library_audit[
                    "n_global_contrast_vectors"
                ]
            ),
            "n_trajectory_vectors": (
                library_audit[
                    "n_trajectory_vectors"
                ]
            ),
        })

    fold_audit = pd.DataFrame(
        fold_audit_rows
    )
    fold_audit.to_csv(
        output_dir
        / "fold_and_library_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    critical_checks = [
        bool(
            (
                fold_audit[
                    "n_train"
                ]
                == fold_audit[
                    "n_library_rows"
                ]
            ).all()
        ),
        bool(
            fold_audit[
                "library_exact_train_match"
            ].all()
        ),
        bool(
            (
                fold_audit[
                    "target_phone_rows_in_library"
                ]
                == 0
            ).all()
        ),
        bool(
            (
                fold_audit[
                    "validation_light_rows_in_library"
                ]
                == 0
            ).all()
        ),
        bool(
            (
                fold_audit[
                    "pH7_rows_in_library"
                ]
                == 0
            ).all()
        ),
    ]
    if not all(critical_checks):
        raise AssertionError(
            "Strict training-only library audit failed. "
            "Do not train models."
        )

    config = {
        "manifest": str(
            manifest_path
        ),
        "factorial_table": str(
            factorial_path
        ),
        "library_mode": (
            "strict_fold_training_only"
        ),
        "manifest_audit": (
            manifest_audit
        ),
        "phones": phones,
        "lights": lights,
        "validation_lights": (
            validation_lights
        ),
        "seeds": args.seeds,
        "methods": args.methods,
        "n_folds": len(folds),
        "binary_task": (
            "acidic pH 1-6 vs alkaline pH 8-13; pH7 excluded"
        ),
        "anti_leakage_rule": {
            "fold_library_rows": (
                "exact classifier-training cells only"
            ),
            "holdout_phone_in_library": False,
            "validation_light_in_library": False,
            "pH7_in_library": False,
            "expected_training_rows_per_fold": (
                96
                if (
                    len(phones) == 3
                    and len(lights) == 5
                )
                else None
            ),
        },
        "augmentation": {
            "probability": (
                args.augmentation_prob
            ),
            "background_strength": (
                args.background_strength
            ),
            "global_contrast_strength": (
                args.global_contrast_strength
            ),
            "trajectory_strength": (
                args.trajectory_strength
            ),
            "knot_count": (
                args.knot_count
            ),
            "trajectory_min_ph": (
                args.trajectory_min_ph
            ),
            "trajectory_max_ph": (
                args.trajectory_max_ph
            ),
            "trajectory_boundary_policy": (
                "pH-dependent term disabled "
                "outside the internal range"
            ),
            "roi_box_frac_fallback": (
                args.roi_box_frac
            ),
            "mask_feather_px": (
                args.mask_feather_px
            ),
        },
        "training": {
            "image_size": (
                args.image_size
            ),
            "batch_size": (
                args.batch_size
            ),
            "head_epochs": (
                args.head_epochs
            ),
            "finetune_epochs": (
                args.finetune_epochs
            ),
            "patience": (
                args.patience
            ),
            "head_lr": (
                args.head_lr
            ),
            "finetune_lr": (
                args.finetune_lr
            ),
            "weight_decay": (
                args.weight_decay
            ),
            "pretrained": (
                not args.no_pretrained
            ),
        },
        "decision_rule": (
            "Hybrid macro-F1 must exceed baseline and global empirical "
            "overall, and in at least two-thirds of holdout phones."
        ),
        "important_note": (
            "This strict rerun is a leakage correction, not a "
            "hyperparameter search. Keep all augmentation/training "
            "settings fixed from the previous pilot."
        ),
    }

    with (
        output_dir
        / "pilot_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            config,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 100)
    print(
        "24. STRICT target-free hybrid empirical augmentation LOPO pilot"
    )
    print("=" * 100)
    print(
        f"Manifest: {manifest_path}"
    )
    print(
        f"Factorial Lab table: {factorial_path}"
    )
    print(
        "Library mode: strict_fold_training_only"
    )
    print(
        "Target-phone capture statistics in fold library: 0"
    )
    print(
        "Validation-light capture statistics in fold library: 0"
    )
    print(
        f"Phones: {phones}"
    )
    print(
        f"Lights: {lights}"
    )
    print(
        f"Validation lights used: {validation_lights}"
    )
    print(
        f"Folds: {len(folds)}"
    )
    print(
        f"Seeds: {args.seeds}"
    )
    print(
        f"Methods: {args.methods}"
    )
    print(
        f"Mask mode: {manifest_audit['mask_mode']}"
    )

    print(
        "\nFold + fold-library audit"
    )
    audit_columns = [
        "fold_id",
        "holdout_phone",
        "validation_light",
        "n_train",
        "n_val",
        "n_test",
        "n_library_rows",
        "target_phone_rows_in_library",
        "validation_light_rows_in_library",
        "n_bg_vectors",
        "n_global_contrast_vectors",
        "n_trajectory_vectors",
    ]
    print(
        fold_audit[
            audit_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nSTRICT AUDIT: PASS"
    )
    print(
        "Each augmentation library uses exactly the classifier-training "
        "cells for its fold."
    )

    if args.dry_run:
        print(
            "\nDry run complete. No models were trained."
        )
        return

    device = resolve_device(
        args.device
    )
    print(
        f"\nDevice: {device}"
    )

    run_rows = []
    prediction_frames = []
    history_frames = []

    total_runs = (
        len(folds)
        * len(args.seeds)
        * len(args.methods)
    )
    run_counter = 0

    for fold in folds:
        train_frame = manifest.iloc[
            fold["train_idx"]
        ].copy()
        val_frame = manifest.iloc[
            fold["val_idx"]
        ].copy()
        test_frame = manifest.iloc[
            fold["test_idx"]
        ].copy()

        library = fold_libraries[
            fold["fold_id"]
        ]

        for seed in args.seeds:
            for method in args.methods:
                run_counter += 1

                print(
                    f"\n[{run_counter}/{total_runs}] "
                    f"{fold['fold_id']} | "
                    f"seed={seed} | {method}"
                )

                set_seed(seed)

                (
                    train_loader,
                    val_loader,
                    test_loader,
                ) = make_loaders(
                    train_frame,
                    val_frame,
                    test_frame,
                    method,
                    library,
                    seed,
                    args,
                )

                (
                    val_metrics,
                    test_metrics,
                    predictions,
                    history,
                ) = train_one_run(
                    train_loader,
                    val_loader,
                    test_loader,
                    device,
                    args,
                )

                run_row = {
                    "fold_id": (
                        fold["fold_id"]
                    ),
                    "holdout_phone": (
                        fold[
                            "holdout_phone"
                        ]
                    ),
                    "validation_light": (
                        fold[
                            "validation_light"
                        ]
                    ),
                    "seed": seed,
                    "method": method,
                    "library_mode": (
                        "strict_fold_training_only"
                    ),
                    "n_library_rows": (
                        len(train_frame)
                    ),
                    "target_phone_rows_in_library": 0,
                    "validation_light_rows_in_library": 0,
                    "n_train": (
                        len(train_frame)
                    ),
                    "n_val": (
                        len(val_frame)
                    ),
                    "n_test": (
                        len(test_frame)
                    ),
                    "val_balanced_accuracy": (
                        val_metrics[
                            "balanced_accuracy"
                        ]
                    ),
                    "val_macro_f1": (
                        val_metrics[
                            "macro_f1"
                        ]
                    ),
                    "accuracy": (
                        test_metrics[
                            "accuracy"
                        ]
                    ),
                    "balanced_accuracy": (
                        test_metrics[
                            "balanced_accuracy"
                        ]
                    ),
                    "macro_f1": (
                        test_metrics[
                            "macro_f1"
                        ]
                    ),
                }
                run_rows.append(
                    run_row
                )

                predictions = (
                    predictions.copy()
                )
                for key, value in {
                    "fold_id": (
                        fold["fold_id"]
                    ),
                    "holdout_phone": (
                        fold[
                            "holdout_phone"
                        ]
                    ),
                    "validation_light": (
                        fold[
                            "validation_light"
                        ]
                    ),
                    "seed": seed,
                    "method": method,
                    "library_mode": (
                        "strict_fold_training_only"
                    ),
                }.items():
                    predictions[
                        key
                    ] = value
                prediction_frames.append(
                    predictions
                )

                history = (
                    history.copy()
                )
                for key, value in {
                    "fold_id": (
                        fold["fold_id"]
                    ),
                    "holdout_phone": (
                        fold[
                            "holdout_phone"
                        ]
                    ),
                    "validation_light": (
                        fold[
                            "validation_light"
                        ]
                    ),
                    "seed": seed,
                    "method": method,
                    "library_mode": (
                        "strict_fold_training_only"
                    ),
                }.items():
                    history[key] = value
                history_frames.append(
                    history
                )

                print(
                    "Test macro-F1="
                    f"{test_metrics['macro_f1']:.4f}, "
                    "balanced accuracy="
                    f"{test_metrics['balanced_accuracy']:.4f}"
                )

    run_table = pd.DataFrame(
        run_rows
    )
    predictions_all = pd.concat(
        prediction_frames,
        ignore_index=True,
    )
    history_all = pd.concat(
        history_frames,
        ignore_index=True,
    )

    (
        method_summary,
        by_phone,
    ) = summarize_runs(
        run_table
    )
    paired = paired_comparisons(
        run_table
    )
    decision = build_decision_summary(
        by_phone,
        method_summary,
    )

    run_table.to_csv(
        output_dir
        / "pilot_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions_all.to_csv(
        output_dir
        / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    history_all.to_csv(
        output_dir
        / "training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    method_summary.to_csv(
        output_dir
        / "method_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_phone.to_csv(
        output_dir
        / "by_phone_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        output_dir
        / "paired_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )
    decision.to_csv(
        output_dir
        / "pilot_decision.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n" + "=" * 100
    )
    print(
        "STRICT target-free pilot method summary"
    )
    print(
        "=" * 100
    )
    print(
        method_summary.to_string(
            index=False
        )
    )

    print(
        "\nBy holdout phone"
    )
    print(
        by_phone.to_string(
            index=False
        )
    )

    print(
        "\nMatched comparisons"
    )
    print(
        paired.to_string(
            index=False
        )
    )

    print(
        "\nPre-specified decision"
    )
    print(
        decision.to_string(
            index=False
        )
    )

    print(
        "\nMain files"
    )
    print(
        output_dir
        / "method_summary.csv"
    )
    print(
        output_dir
        / "by_phone_summary.csv"
    )
    print(
        output_dir
        / "paired_comparisons.csv"
    )
    print(
        output_dir
        / "pilot_decision.csv"
    )
    print(
        output_dir
        / "fold_and_library_audit.csv"
    )


if __name__ == "__main__":
    main()
