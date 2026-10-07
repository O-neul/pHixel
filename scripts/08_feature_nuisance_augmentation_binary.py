#!/usr/bin/env python
"""
08_feature_nuisance_augmentation_binary.py

Feature-space acidic-vs-alkaline experiment for one LOPO split.

Methods
-------
1. baseline:
   Original train rows only.

2. gaussian_matched:
   Original rows plus Gaussian perturbations whose covariance is matched to
   empirical nuisance-recombination deltas. This is a generic-noise control.

3. nuisance_recombined:
   Original rows plus train-only empirical nuisance recombinations.

Important design choices
------------------------
- pH 7 is excluded from train, validation, and test.
- The scaler is fitted on original train rows only and shared by all methods.
- C is selected using validation macro-F1 only.
- The test phone is never used for scaling, augmentation, or model selection.
- Pad and background nuisance components are transformed jointly.
- Contrast is recomputed as augmented pad - augmented background.
- Residual effects are excluded by default.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.preprocessing import StandardScaler


LABEL_ORDER = ["acidic", "alkaline"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABEL_ORDER)}

FEATURE_GROUPS: dict[str, list[str]] = {
    "pad_only": ["pad_L", "pad_a", "pad_b"],
    "contrast_only": ["contrast_L", "contrast_a", "contrast_b"],
    "pad_plus_bg": [
        "pad_L", "pad_a", "pad_b",
        "bg_L", "bg_a", "bg_b",
    ],
}

REGION_COLUMNS: dict[str, list[str]] = {
    "pad": ["pad_L", "pad_a", "pad_b"],
    "bg": ["bg_L", "bg_a", "bg_b"],
}


@dataclass
class Metrics:
    loss: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline, matched Gaussian, and train-only empirical "
            "nuisance augmentation on one binary LOPO split."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/capture_manifest.csv"),
        help="Canonical manifest CSV.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        required=True,
        help="One LOPO split CSV.",
    )
    parser.add_argument(
        "--decomposition-dir",
        type=Path,
        required=True,
        help="Binary train-only decomposition directory for the same split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--feature-groups",
        default="pad_only,contrast_only",
        help="Comma-separated subset of: pad_only,contrast_only,pad_plus_bg.",
    )
    parser.add_argument(
        "--methods",
        default="baseline,gaussian_matched,nuisance_recombined",
        help=(
            "Comma-separated subset of: "
            "baseline,gaussian_matched,nuisance_recombined."
        ),
    )
    parser.add_argument(
        "--c-grid",
        default="0.01,0.1,1,10,100",
        help="Comma-separated positive Logistic Regression C values.",
    )
    parser.add_argument(
        "--augmentations-per-row",
        type=int,
        default=4,
        help="Number of synthetic rows per original train row.",
    )
    parser.add_argument(
        "--sampler",
        choices=["recombined", "structured"],
        default="recombined",
        help=(
            "recombined samples phone, light, and interaction components "
            "independently; structured samples an observed target phone-light cell."
        ),
    )
    parser.add_argument(
        "--phone-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--light-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--interaction-scale",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.0,
        help=(
            "Scale for same-pH residual replacement. Keep at 0.0 for the "
            "primary experiment."
        ),
    )
    parser.add_argument(
        "--gaussian-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to matched Gaussian perturbations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--no-lab-clipping",
        action="store_true",
        help="Do not clip augmented Lab channels to conventional bounds.",
    )
    return parser.parse_args()


def infer_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_existing_path(path: Path, project_root: Path) -> Path:
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([project_root / path, Path.cwd() / path])
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Path not found: {path}")


def resolve_output_path(path: Path, project_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


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
        f"Could not identify {description}. Tried {list(candidates)}. "
        f"Available columns: {list(frame.columns)}"
    )


def parse_choices(raw: str, allowed: set[str], description: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"No {description} supplied.")
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown {description}: {unknown}. Allowed: {sorted(allowed)}"
        )
    return list(dict.fromkeys(values))


def parse_c_grid(raw: str) -> list[float]:
    try:
        values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --c-grid: {raw}") from exc
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("--c-grid must contain positive finite values.")
    return sorted(set(values))


def merge_manifest_and_split(
    manifest: pd.DataFrame,
    split_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    manifest_id = find_column(
        manifest, ["image_id", "capture_id"], "manifest image ID"
    )
    split_id = find_column(
        split_frame, ["image_id", "capture_id"], "split image ID"
    )
    split_col = find_column(
        split_frame, ["split", "partition", "subset"], "split column"
    )

    split_min = split_frame[[split_id, split_col]].copy()
    split_min.columns = ["__image_id__", "__split__"]
    split_min["__image_id__"] = split_min["__image_id__"].astype(str)

    if split_min["__image_id__"].duplicated().any():
        raise ValueError("Split file contains duplicate image IDs.")

    merged = manifest.copy()
    merged["__image_id__"] = merged[manifest_id].astype(str)
    merged = merged.merge(
        split_min,
        on="__image_id__",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(split_min):
        present = set(manifest[manifest_id].astype(str))
        missing = [
            value
            for value in split_min["__image_id__"]
            if value not in present
        ]
        raise ValueError(
            f"Only {len(merged)} of {len(split_min)} split rows matched. "
            f"Missing examples: {missing[:10]}"
        )

    merged["__split__"] = (
        merged["__split__"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"validation": "val", "valid": "val"})
    )
    unexpected = sorted(set(merged["__split__"]) - {"train", "val", "test"})
    if unexpected:
        raise ValueError(f"Unexpected split labels: {unexpected}")

    return merged, {
        "manifest_image_id": manifest_id,
        "split_image_id": split_id,
        "split_column": split_col,
    }


def prepare_binary_task(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    ph_col = find_column(
        frame,
        ["ph_nominal", "ph_value", "ph"],
        "nominal pH column",
    )
    result = frame.copy()
    result[ph_col] = pd.to_numeric(result[ph_col], errors="coerce")
    if result[ph_col].isna().any():
        raise ValueError("Non-numeric pH values were found.")

    before = len(result)
    result = result.loc[~np.isclose(result[ph_col], 7.0)].copy()
    result["label_class"] = np.where(
        result[ph_col] < 7.0, "acidic", "alkaline"
    )
    result["label_index"] = (
        result["label_class"].map(LABEL_TO_INDEX).astype(int)
    )

    split_rows = {
        split: int((result["__split__"] == split).sum())
        for split in ["train", "val", "test"]
    }
    label_counts = {}
    for split in ["train", "val", "test"]:
        counts = (
            result.loc[result["__split__"] == split, "label_class"]
            .value_counts()
            .reindex(LABEL_ORDER, fill_value=0)
        )
        label_counts[split] = {
            label: int(counts[label]) for label in LABEL_ORDER
        }

    for split in ["train", "val", "test"]:
        if split_rows[split] == 0:
            raise ValueError(f"Split '{split}' is empty after removing pH 7.")

    if label_counts["train"]["acidic"] != label_counts["train"]["alkaline"]:
        print(
            f"WARNING: train classes are not balanced: {label_counts['train']}",
            file=sys.stderr,
        )

    return result, ph_col, {
        "rows_before_filter": before,
        "excluded_ph7_rows": before - len(result),
        "rows_after_filter": len(result),
        "split_rows": split_rows,
        "label_counts": label_counts,
    }


def require_numeric_features(frame: pd.DataFrame) -> None:
    required = [
        "pad_L", "pad_a", "pad_b",
        "bg_L", "bg_a", "bg_b",
        "contrast_L", "contrast_a", "contrast_b",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Manifest is missing feature columns: {missing}")

    converted = frame[required].apply(pd.to_numeric, errors="coerce")
    if converted.isna().any().any():
        bad = converted.columns[converted.isna().any()].tolist()
        raise ValueError(f"Missing or non-numeric values in: {bad}")
    frame[required] = converted


def vector_columns(prefix: str) -> list[str]:
    return [f"{prefix}_L", f"{prefix}_a", f"{prefix}_b"]


def load_and_audit_library(
    decomposition_dir: Path,
    train_frame: pd.DataFrame,
    split_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    library_path = decomposition_dir / "nuisance_vector_library.csv"
    if not library_path.exists():
        raise FileNotFoundError(f"Missing nuisance library: {library_path}")

    library = pd.read_csv(library_path)
    required = {
        "image_id", "phone_id", "light_id", "ph_nominal", "region",
        *vector_columns("phone_effect"),
        *vector_columns("light_effect"),
        *vector_columns("interaction_effect"),
        *vector_columns("residual"),
    }
    missing = sorted(required - set(library.columns))
    if missing:
        raise ValueError(f"Nuisance library missing columns: {missing}")

    library["image_id"] = library["image_id"].astype(str)
    library["region"] = library["region"].astype(str)
    library["ph_nominal"] = pd.to_numeric(
        library["ph_nominal"], errors="raise"
    )

    train_ids = set(train_frame["__image_id__"].astype(str))
    library_ids = set(library["image_id"].astype(str))
    if train_ids != library_ids:
        raise ValueError(
            "Binary nuisance library image IDs do not exactly match the "
            f"96 filtered train rows. Missing from library: "
            f"{sorted(train_ids - library_ids)[:10]}; extra: "
            f"{sorted(library_ids - train_ids)[:10]}"
        )

    required_regions = {"pad", "bg"}
    if not required_regions.issubset(set(library["region"])):
        raise ValueError(
            f"Library must contain pad and bg regions. Found: "
            f"{sorted(set(library['region']))}"
        )

    per_image_region = library.groupby(["image_id", "region"]).size()
    bad = per_image_region[per_image_region != 1]
    if len(bad):
        raise ValueError(
            "Each train image must have exactly one row per region. "
            f"Bad examples: {bad.head().to_dict()}"
        )

    config_path = decomposition_dir / "decomposition_config.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        config_split = str(config.get("split_file", ""))
        if config_split:
            if Path(config_split).name != split_path.name:
                raise ValueError(
                    "Decomposition config and supplied split do not match: "
                    f"{Path(config_split).name} vs {split_path.name}"
                )

        config_task = config.get("task")
        if config_task not in (None, "binary"):
            raise ValueError(
                f"Expected binary decomposition, found task={config_task!r}"
            )

        training_rows = config.get("training_rows")
        if training_rows is not None and int(training_rows) != len(train_frame):
            raise ValueError(
                f"Config training_rows={training_rows}, but filtered train "
                f"contains {len(train_frame)} rows."
            )

    audit = {
        "library_path": str(library_path),
        "library_rows": len(library),
        "library_images": len(library_ids),
        "regions": sorted(set(library["region"])),
        "config_path": str(config_path) if config_path.exists() else None,
        "config": config,
    }
    return library, audit


def aggregate_component_table(
    library: pd.DataFrame,
    region: str,
    level_columns: list[str],
    effect_prefix: str,
) -> pd.DataFrame:
    columns = vector_columns(effect_prefix)
    subset = library.loc[library["region"] == region].copy()
    grouped = (
        subset.groupby(level_columns, as_index=False)[columns]
        .mean()
        .sort_values(level_columns)
        .reset_index(drop=True)
    )

    # Components should be constant within each level. Record a warning if not.
    spread = (
        subset.groupby(level_columns)[columns]
        .agg(lambda values: float(np.max(values) - np.min(values)))
    )
    if float(spread.to_numpy().max(initial=0.0)) > 1e-8:
        print(
            f"WARNING: {effect_prefix} varied within levels for region={region}; "
            "using level means.",
            file=sys.stderr,
        )
    return grouped


def make_component_tables(
    library: pd.DataFrame,
) -> dict[str, dict[str, pd.DataFrame]]:
    tables: dict[str, dict[str, pd.DataFrame]] = {}
    for region in ["pad", "bg"]:
        tables[region] = {
            "phone": aggregate_component_table(
                library, region, ["phone_id"], "phone_effect"
            ),
            "light": aggregate_component_table(
                library, region, ["light_id"], "light_effect"
            ),
            "interaction": aggregate_component_table(
                library,
                region,
                ["phone_id", "light_id"],
                "interaction_effect",
            ),
        }
    return tables


def lookup_component(
    table: pd.DataFrame,
    conditions: dict[str, Any],
    prefix: str,
) -> np.ndarray:
    mask = np.ones(len(table), dtype=bool)
    for column, value in conditions.items():
        mask &= table[column].astype(str).to_numpy() == str(value)
    matched = table.loc[mask]
    if len(matched) != 1:
        raise ValueError(
            f"Expected exactly one {prefix} component for {conditions}, "
            f"found {len(matched)}."
        )
    return matched[vector_columns(prefix)].iloc[0].to_numpy(dtype=float)


def make_library_index(
    library: pd.DataFrame,
) -> dict[tuple[str, str], pd.Series]:
    return {
        (str(row.image_id), str(row.region)): row
        for _, row in library.iterrows()
    }


def current_effect_vector(
    row: pd.Series,
    phone_scale: float,
    light_scale: float,
    interaction_scale: float,
) -> np.ndarray:
    return (
        phone_scale
        * row[vector_columns("phone_effect")].to_numpy(dtype=float)
        + light_scale
        * row[vector_columns("light_effect")].to_numpy(dtype=float)
        + interaction_scale
        * row[vector_columns("interaction_effect")].to_numpy(dtype=float)
    )


def target_effect_vector(
    tables: dict[str, pd.DataFrame],
    sampled_phone: str,
    sampled_light: str,
    interaction_phone: str,
    interaction_light: str,
    phone_scale: float,
    light_scale: float,
    interaction_scale: float,
) -> np.ndarray:
    phone = lookup_component(
        tables["phone"], {"phone_id": sampled_phone}, "phone_effect"
    )
    light = lookup_component(
        tables["light"], {"light_id": sampled_light}, "light_effect"
    )
    interaction = lookup_component(
        tables["interaction"],
        {
            "phone_id": interaction_phone,
            "light_id": interaction_light,
        },
        "interaction_effect",
    )
    return (
        phone_scale * phone
        + light_scale * light
        + interaction_scale * interaction
    )


def clip_lab(values: np.ndarray) -> tuple[np.ndarray, int]:
    clipped = values.copy()
    before = clipped.copy()
    clipped[0] = np.clip(clipped[0], 0.0, 100.0)
    clipped[1:] = np.clip(clipped[1:], -128.0, 127.0)
    changed = int(np.any(np.abs(clipped - before) > 1e-12))
    return clipped, changed


def select_nonidentity_target(
    rng: np.random.Generator,
    sampler: str,
    phones: list[str],
    lights: list[str],
    pairs: list[tuple[str, str]],
    current_phone: str,
    current_light: str,
) -> tuple[str, str, str, str]:
    for _ in range(100):
        if sampler == "structured":
            eligible = [
                pair for pair in pairs
                if pair != (current_phone, current_light)
            ]
            sampled_phone, sampled_light = eligible[
                int(rng.integers(len(eligible)))
            ]
            interaction_phone, interaction_light = (
                sampled_phone,
                sampled_light,
            )
        else:
            sampled_phone = phones[int(rng.integers(len(phones)))]
            sampled_light = lights[int(rng.integers(len(lights)))]
            interaction_phone, interaction_light = pairs[
                int(rng.integers(len(pairs)))
            ]

        if (
            sampled_phone != current_phone
            or sampled_light != current_light
            or interaction_phone != current_phone
            or interaction_light != current_light
        ):
            return (
                sampled_phone,
                sampled_light,
                interaction_phone,
                interaction_light,
            )
    raise RuntimeError("Could not sample a non-identity nuisance target.")


def feature_vector_from_pad_bg(
    pad: np.ndarray,
    bg: np.ndarray,
    feature_group: str,
) -> np.ndarray:
    contrast = pad - bg
    if feature_group == "pad_only":
        return pad
    if feature_group == "contrast_only":
        return contrast
    if feature_group == "pad_plus_bg":
        return np.concatenate([pad, bg])
    raise ValueError(f"Unsupported feature group: {feature_group}")


def generate_nuisance_augmentation(
    train_frame: pd.DataFrame,
    ph_col: str,
    library: pd.DataFrame,
    feature_group: str,
    augmentations_per_row: int,
    sampler: str,
    phone_scale: float,
    light_scale: float,
    interaction_scale: float,
    residual_scale: float,
    rng: np.random.Generator,
    use_lab_clipping: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    if augmentations_per_row < 1:
        raise ValueError("--augmentations-per-row must be at least 1.")

    tables = make_component_tables(library)
    library_index = make_library_index(library)

    phones = sorted(
        tables["pad"]["phone"]["phone_id"].astype(str).unique().tolist()
    )
    lights = sorted(
        tables["pad"]["light"]["light_id"].astype(str).unique().tolist()
    )
    pairs = [
        (str(row.phone_id), str(row.light_id))
        for _, row in tables["pad"]["interaction"].iterrows()
    ]

    residual_pool: dict[tuple[float, str], list[np.ndarray]] = {}
    if residual_scale != 0.0:
        for _, row in library.iterrows():
            key = (float(row["ph_nominal"]), str(row["region"]))
            residual_pool.setdefault(key, []).append(
                row[vector_columns("residual")].to_numpy(dtype=float)
            )

    augmented_x: list[np.ndarray] = []
    augmented_y: list[int] = []
    metadata: list[dict[str, Any]] = []
    delta_vectors: list[np.ndarray] = []

    for _, source in train_frame.iterrows():
        image_id = str(source["__image_id__"])
        current_phone = str(
            source["phone_id"]
            if "phone_id" in source.index
            else source.get("phone_label")
        )
        current_light = str(
            source["light_id"]
            if "light_id" in source.index
            else source.get("light_label")
        )
        ph_value = float(source[ph_col])

        pad_source = source[REGION_COLUMNS["pad"]].to_numpy(dtype=float)
        bg_source = source[REGION_COLUMNS["bg"]].to_numpy(dtype=float)
        original_feature = feature_vector_from_pad_bg(
            pad_source, bg_source, feature_group
        )

        pad_library_row = library_index[(image_id, "pad")]
        bg_library_row = library_index[(image_id, "bg")]

        pad_current = current_effect_vector(
            pad_library_row,
            phone_scale,
            light_scale,
            interaction_scale,
        )
        bg_current = current_effect_vector(
            bg_library_row,
            phone_scale,
            light_scale,
            interaction_scale,
        )

        for variant in range(augmentations_per_row):
            (
                sampled_phone,
                sampled_light,
                interaction_phone,
                interaction_light,
            ) = select_nonidentity_target(
                rng,
                sampler,
                phones,
                lights,
                pairs,
                current_phone,
                current_light,
            )

            pad_target = target_effect_vector(
                tables["pad"],
                sampled_phone,
                sampled_light,
                interaction_phone,
                interaction_light,
                phone_scale,
                light_scale,
                interaction_scale,
            )
            bg_target = target_effect_vector(
                tables["bg"],
                sampled_phone,
                sampled_light,
                interaction_phone,
                interaction_light,
                phone_scale,
                light_scale,
                interaction_scale,
            )

            delta_pad = pad_target - pad_current
            delta_bg = bg_target - bg_current

            residual_source_pad = np.zeros(3)
            residual_source_bg = np.zeros(3)
            residual_target_pad = np.zeros(3)
            residual_target_bg = np.zeros(3)

            if residual_scale != 0.0:
                residual_source_pad = pad_library_row[
                    vector_columns("residual")
                ].to_numpy(dtype=float)
                residual_source_bg = bg_library_row[
                    vector_columns("residual")
                ].to_numpy(dtype=float)

                pad_candidates = residual_pool[(ph_value, "pad")]
                bg_candidates = residual_pool[(ph_value, "bg")]
                residual_target_pad = pad_candidates[
                    int(rng.integers(len(pad_candidates)))
                ]
                residual_target_bg = bg_candidates[
                    int(rng.integers(len(bg_candidates)))
                ]

                delta_pad += residual_scale * (
                    residual_target_pad - residual_source_pad
                )
                delta_bg += residual_scale * (
                    residual_target_bg - residual_source_bg
                )

            pad_aug = pad_source + delta_pad
            bg_aug = bg_source + delta_bg

            pad_clipped = 0
            bg_clipped = 0
            if use_lab_clipping:
                pad_aug, pad_clipped = clip_lab(pad_aug)
                bg_aug, bg_clipped = clip_lab(bg_aug)

            aug_feature = feature_vector_from_pad_bg(
                pad_aug, bg_aug, feature_group
            )
            delta_feature = aug_feature - original_feature

            augmented_x.append(aug_feature)
            augmented_y.append(int(source["label_index"]))
            delta_vectors.append(delta_feature)

            metadata.append({
                "source_image_id": image_id,
                "source_phone": current_phone,
                "source_light": current_light,
                "ph_nominal": ph_value,
                "label_class": source["label_class"],
                "variant": variant,
                "sampled_phone_component": sampled_phone,
                "sampled_light_component": sampled_light,
                "sampled_interaction_phone": interaction_phone,
                "sampled_interaction_light": interaction_light,
                "delta_norm": float(np.linalg.norm(delta_feature)),
                "delta_pad_EuclideanLab": float(np.linalg.norm(delta_pad)),
                "delta_bg_EuclideanLab": float(np.linalg.norm(delta_bg)),
                "pad_lab_clipped": pad_clipped,
                "bg_lab_clipped": bg_clipped,
            })

    return (
        np.asarray(augmented_x, dtype=float),
        np.asarray(augmented_y, dtype=int),
        pd.DataFrame(metadata),
        np.asarray(delta_vectors, dtype=float),
    )


def generate_matched_gaussian(
    original_x: np.ndarray,
    original_y: np.ndarray,
    empirical_deltas: np.ndarray,
    augmentations_per_row: int,
    gaussian_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    centered = empirical_deltas - empirical_deltas.mean(axis=0, keepdims=True)
    if centered.shape[1] == 1:
        covariance = np.array([[float(np.var(centered, ddof=1))]])
    else:
        covariance = np.cov(centered, rowvar=False)

    covariance = np.atleast_2d(covariance)
    jitter = max(1e-8, float(np.trace(covariance)) * 1e-8)
    covariance = covariance + np.eye(covariance.shape[0]) * jitter
    covariance *= gaussian_scale ** 2

    augmented_x: list[np.ndarray] = []
    augmented_y: list[int] = []
    metadata: list[dict[str, Any]] = []

    for row_index, (source, label) in enumerate(zip(original_x, original_y)):
        for variant in range(augmentations_per_row):
            delta = rng.multivariate_normal(
                mean=np.zeros(source.shape[0]),
                cov=covariance,
                check_valid="warn",
            )
            augmented_x.append(source + delta)
            augmented_y.append(int(label))
            metadata.append({
                "source_row_index": row_index,
                "variant": variant,
                "delta_norm": float(np.linalg.norm(delta)),
            })

    return (
        np.asarray(augmented_x, dtype=float),
        np.asarray(augmented_y, dtype=int),
        pd.DataFrame(metadata),
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probability_alkaline: np.ndarray,
) -> Metrics:
    clipped = np.clip(probability_alkaline, 1e-7, 1 - 1e-7)
    probabilities = np.column_stack([1.0 - clipped, clipped])
    return Metrics(
        loss=float(log_loss(y_true, probabilities, labels=[0, 1])),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        macro_f1=float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        n=int(len(y_true)),
    )


def build_model(c_value: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        solver="liblinear",
        max_iter=5000,
        random_state=seed,
        class_weight=None,
    )


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["val_macro_f1"]),
        float(row["val_balanced_accuracy"]),
        float(row["val_accuracy"]),
        -float(row["C"]),
    )


def evaluate_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    subset: pd.DataFrame,
    features: list[str],
    ph_col: str,
) -> tuple[Metrics, pd.DataFrame]:
    x = subset[features].to_numpy(dtype=float)
    y = subset["label_index"].to_numpy(dtype=int)
    x_scaled = scaler.transform(x)
    prediction = model.predict(x_scaled)
    probability = model.predict_proba(x_scaled)[:, 1]
    metrics = calculate_metrics(y, prediction, probability)

    predictions = pd.DataFrame({
        "image_id": subset["__image_id__"].astype(str).to_numpy(),
        "split": subset["__split__"].astype(str).to_numpy(),
        "ph_nominal": subset[ph_col].to_numpy(),
        "true_index": y,
        "true_label": [LABEL_ORDER[index] for index in y],
        "pred_index": prediction,
        "pred_label": [LABEL_ORDER[index] for index in prediction],
        "probability_acidic": 1.0 - probability,
        "probability_alkaline": probability,
        "correct": prediction == y,
    })
    for possible in [
        "phone_id", "phone_label", "light_id", "light_label"
    ]:
        if possible in subset.columns:
            predictions[possible] = subset[possible].to_numpy()

    return metrics, predictions


def make_per_ph_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ph_value, group in predictions.groupby("ph_nominal"):
        rows.append({
            "ph_nominal": float(ph_value),
            "n": len(group),
            "accuracy": float(group["correct"].mean()),
            "mean_probability_alkaline": float(
                group["probability_alkaline"].mean()
            ),
            "min_probability_alkaline": float(
                group["probability_alkaline"].min()
            ),
            "max_probability_alkaline": float(
                group["probability_alkaline"].max()
            ),
        })
    return pd.DataFrame(rows).sort_values("ph_nominal")


def boundary_summary(predictions: pd.DataFrame) -> dict[str, float]:
    output: dict[str, float] = {}
    for ph_value in [6.0, 8.0, 9.0]:
        subset = predictions.loc[
            np.isclose(predictions["ph_nominal"], ph_value)
        ]
        key = str(int(ph_value))
        if len(subset):
            output[f"ph{key}_accuracy"] = float(subset["correct"].mean())
            output[f"ph{key}_mean_probability_alkaline"] = float(
                subset["probability_alkaline"].mean()
            )
        else:
            output[f"ph{key}_accuracy"] = float("nan")
            output[f"ph{key}_mean_probability_alkaline"] = float("nan")
    return output


def save_confusion_and_report(
    run_dir: Path,
    split_name: str,
    predictions: pd.DataFrame,
) -> None:
    y_true = predictions["true_index"].to_numpy(dtype=int)
    y_pred = predictions["pred_index"].to_numpy(dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in LABEL_ORDER],
        columns=[f"pred_{label}" for label in LABEL_ORDER],
    ).to_csv(run_dir / f"confusion_matrix_{split_name}.csv")

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        run_dir / f"classification_report_{split_name}.csv"
    )


def summarize_delta_metadata(
    feature_group: str,
    method: str,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    values = metadata["delta_norm"].to_numpy(dtype=float)
    row = {
        "feature_group": feature_group,
        "method": method,
        "n_augmented": len(metadata),
        "mean_delta_norm": float(np.mean(values)),
        "median_delta_norm": float(np.median(values)),
        "q95_delta_norm": float(np.quantile(values, 0.95)),
        "max_delta_norm": float(np.max(values)),
    }
    if "pad_lab_clipped" in metadata:
        row["pad_lab_clipped_ratio"] = float(
            metadata["pad_lab_clipped"].mean()
        )
        row["bg_lab_clipped_ratio"] = float(
            metadata["bg_lab_clipped"].mean()
        )
    return row


def main() -> None:
    args = parse_args()
    project_root = infer_project_root()

    manifest_path = resolve_existing_path(args.manifest, project_root)
    split_path = resolve_existing_path(args.split_file, project_root)
    decomposition_dir = resolve_existing_path(
        args.decomposition_dir, project_root
    )
    output_dir = resolve_output_path(args.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_groups = parse_choices(
        args.feature_groups,
        set(FEATURE_GROUPS),
        "feature groups",
    )
    methods = parse_choices(
        args.methods,
        {"baseline", "gaussian_matched", "nuisance_recombined"},
        "methods",
    )
    c_grid = parse_c_grid(args.c_grid)

    for name, value in [
        ("phone_scale", args.phone_scale),
        ("light_scale", args.light_scale),
        ("interaction_scale", args.interaction_scale),
        ("residual_scale", args.residual_scale),
        ("gaussian_scale", args.gaussian_scale),
    ]:
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and >= 0.")

    print(f"Manifest: {manifest_path}")
    print(f"Split: {split_path}")
    print(f"Decomposition: {decomposition_dir}")
    print(f"Output: {output_dir}")
    print(f"Feature groups: {feature_groups}")
    print(f"Methods: {methods}")
    print(f"Sampler: {args.sampler}")
    print(
        "Scales: "
        f"phone={args.phone_scale}, light={args.light_scale}, "
        f"interaction={args.interaction_scale}, "
        f"residual={args.residual_scale}"
    )
    print(
        f"Augmentations per train row: {args.augmentations_per_row}"
    )

    manifest = pd.read_csv(manifest_path)
    split_frame = pd.read_csv(split_path)
    frame, merge_columns = merge_manifest_and_split(
        manifest, split_frame
    )
    frame, ph_col, task_audit = prepare_binary_task(frame)
    require_numeric_features(frame)

    train_frame = frame.loc[frame["__split__"] == "train"].copy()
    val_frame = frame.loc[frame["__split__"] == "val"].copy()
    test_frame = frame.loc[frame["__split__"] == "test"].copy()

    library, library_audit = load_and_audit_library(
        decomposition_dir,
        train_frame,
        split_path,
    )

    print("\nRows and labels:")
    for split_name, subset in [
        ("train", train_frame),
        ("val", val_frame),
        ("test", test_frame),
    ]:
        counts = (
            subset["label_class"]
            .value_counts()
            .reindex(LABEL_ORDER, fill_value=0)
        )
        print(
            f"  {split_name:5s}: n={len(subset):3d} | "
            f"acidic={int(counts['acidic']):2d} "
            f"alkaline={int(counts['alkaline']):2d}"
        )

    audit = {
        "manifest": str(manifest_path),
        "split_file": str(split_path),
        "decomposition_dir": str(decomposition_dir),
        "output_dir": str(output_dir),
        "task": "binary",
        "labels": LABEL_ORDER,
        "ph7_excluded": True,
        "merge_columns": merge_columns,
        "task_audit": task_audit,
        "library_audit": library_audit,
        "feature_groups": {
            name: FEATURE_GROUPS[name] for name in feature_groups
        },
        "methods": methods,
        "c_grid": c_grid,
        "augmentation": {
            "augmentations_per_row": args.augmentations_per_row,
            "sampler": args.sampler,
            "phone_scale": args.phone_scale,
            "light_scale": args.light_scale,
            "interaction_scale": args.interaction_scale,
            "residual_scale": args.residual_scale,
            "gaussian_scale": args.gaussian_scale,
            "lab_clipping": not args.no_lab_clipping,
        },
        "scaling": (
            "One StandardScaler per feature group, fitted on original train "
            "rows only and shared across all methods."
        ),
        "model_selection": (
            "C selected on validation macro-F1, then validation balanced "
            "accuracy, validation accuracy, then smaller C."
        ),
        "test_usage": "Test evaluated only after validation selection.",
        "seed": args.seed,
    }
    with (output_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)

    candidate_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    augmentation_summary_rows: list[dict[str, Any]] = []

    start_time = time.perf_counter()

    for group_index, feature_group in enumerate(feature_groups):
        features = FEATURE_GROUPS[feature_group]

        x_train_original = train_frame[features].to_numpy(dtype=float)
        y_train_original = train_frame["label_index"].to_numpy(dtype=int)

        scaler = StandardScaler()
        scaler.fit(x_train_original)
        x_train_original_scaled = scaler.transform(x_train_original)

        nuisance_rng = np.random.default_rng(
            args.seed + 1000 * (group_index + 1)
        )
        (
            nuisance_x,
            nuisance_y,
            nuisance_metadata,
            empirical_deltas,
        ) = generate_nuisance_augmentation(
            train_frame=train_frame,
            ph_col=ph_col,
            library=library,
            feature_group=feature_group,
            augmentations_per_row=args.augmentations_per_row,
            sampler=args.sampler,
            phone_scale=args.phone_scale,
            light_scale=args.light_scale,
            interaction_scale=args.interaction_scale,
            residual_scale=args.residual_scale,
            rng=nuisance_rng,
            use_lab_clipping=not args.no_lab_clipping,
        )

        gaussian_rng = np.random.default_rng(
            args.seed + 2000 * (group_index + 1)
        )
        (
            gaussian_x,
            gaussian_y,
            gaussian_metadata,
        ) = generate_matched_gaussian(
            original_x=x_train_original,
            original_y=y_train_original,
            empirical_deltas=empirical_deltas,
            augmentations_per_row=args.augmentations_per_row,
            gaussian_scale=args.gaussian_scale,
            rng=gaussian_rng,
        )

        nuisance_metadata.to_csv(
            output_dir / f"nuisance_samples_{feature_group}.csv",
            index=False,
        )
        gaussian_metadata.to_csv(
            output_dir / f"gaussian_samples_{feature_group}.csv",
            index=False,
        )

        augmentation_summary_rows.append(
            summarize_delta_metadata(
                feature_group,
                "nuisance_recombined",
                nuisance_metadata,
            )
        )
        augmentation_summary_rows.append(
            summarize_delta_metadata(
                feature_group,
                "gaussian_matched",
                gaussian_metadata,
            )
        )

        method_train_data = {
            "baseline": (
                x_train_original_scaled,
                y_train_original,
            ),
            "nuisance_recombined": (
                np.vstack([
                    x_train_original_scaled,
                    scaler.transform(nuisance_x),
                ]),
                np.concatenate([y_train_original, nuisance_y]),
            ),
            "gaussian_matched": (
                np.vstack([
                    x_train_original_scaled,
                    scaler.transform(gaussian_x),
                ]),
                np.concatenate([y_train_original, gaussian_y]),
            ),
        }

        val_x_scaled = scaler.transform(
            val_frame[features].to_numpy(dtype=float)
        )
        val_y = val_frame["label_index"].to_numpy(dtype=int)

        for method in methods:
            print("\n" + "=" * 88)
            print(f"Feature group: {feature_group} | Method: {method}")
            print("=" * 88)

            train_x_method, train_y_method = method_train_data[method]
            method_candidates: list[dict[str, Any]] = []

            for c_value in c_grid:
                model = build_model(c_value, args.seed)
                model.fit(train_x_method, train_y_method)

                val_prediction = model.predict(val_x_scaled)
                val_probability = model.predict_proba(val_x_scaled)[:, 1]
                val_metrics = calculate_metrics(
                    val_y,
                    val_prediction,
                    val_probability,
                )

                candidate = {
                    "feature_group": feature_group,
                    "method": method,
                    "C": c_value,
                    "train_rows_used": len(train_y_method),
                    "val_loss": val_metrics.loss,
                    "val_accuracy": val_metrics.accuracy,
                    "val_balanced_accuracy": (
                        val_metrics.balanced_accuracy
                    ),
                    "val_macro_f1": val_metrics.macro_f1,
                }
                candidate_rows.append(candidate)
                method_candidates.append(candidate)

                print(
                    f"C={c_value:<6g} "
                    f"val_f1={val_metrics.macro_f1:.3f} "
                    f"val_bal_acc={val_metrics.balanced_accuracy:.3f} "
                    f"val_acc={val_metrics.accuracy:.3f}"
                )

            selected = max(method_candidates, key=selection_key)
            selected_c = float(selected["C"])
            final_model = build_model(selected_c, args.seed)
            final_model.fit(train_x_method, train_y_method)

            run_dir = output_dir / f"{feature_group}__{method}"
            run_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"scaler": scaler, "model": final_model},
                run_dir / "model.joblib",
            )

            coefficients = pd.DataFrame({
                "feature": features,
                "train_original_mean": scaler.mean_,
                "train_original_std": scaler.scale_,
                "coefficient_standardized": final_model.coef_.reshape(-1),
                "absolute_coefficient_standardized": np.abs(
                    final_model.coef_.reshape(-1)
                ),
                "coefficient_raw_scale": (
                    final_model.coef_.reshape(-1) / scaler.scale_
                ),
            }).sort_values(
                "absolute_coefficient_standardized",
                ascending=False,
            )
            coefficients.to_csv(
                run_dir / "coefficients.csv",
                index=False,
            )

            split_outputs: dict[str, tuple[Metrics, pd.DataFrame]] = {}
            for split_name, subset in [
                ("train", train_frame),
                ("val", val_frame),
                ("test", test_frame),
            ]:
                metrics, predictions = evaluate_model(
                    final_model,
                    scaler,
                    subset,
                    features,
                    ph_col,
                )
                split_outputs[split_name] = (metrics, predictions)

                predictions.to_csv(
                    run_dir / f"predictions_{split_name}.csv",
                    index=False,
                )
                save_confusion_and_report(
                    run_dir,
                    split_name,
                    predictions,
                )

                all_metric_rows.append({
                    "feature_group": feature_group,
                    "method": method,
                    "selected_C": selected_c,
                    "split": split_name,
                    "train_rows_used": len(train_y_method),
                    **asdict(metrics),
                })

            test_metrics, test_predictions = split_outputs["test"]
            per_ph = make_per_ph_summary(test_predictions)
            per_ph.to_csv(
                run_dir / "test_metrics_by_ph.csv",
                index=False,
            )

            boundary = boundary_summary(test_predictions)
            comparison_row = {
                "feature_group": feature_group,
                "method": method,
                "selected_C": selected_c,
                "train_original_rows": len(train_frame),
                "train_augmented_rows": (
                    len(train_y_method) - len(train_frame)
                ),
                "train_total_rows": len(train_y_method),
                "best_val_macro_f1": selected["val_macro_f1"],
                "best_val_balanced_accuracy": (
                    selected["val_balanced_accuracy"]
                ),
                "test_loss": test_metrics.loss,
                "test_accuracy": test_metrics.accuracy,
                "test_balanced_accuracy": (
                    test_metrics.balanced_accuracy
                ),
                "test_macro_f1": test_metrics.macro_f1,
                **boundary,
            }
            comparison_rows.append(comparison_row)

            with (run_dir / "run_config.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {
                        "feature_group": feature_group,
                        "features": features,
                        "method": method,
                        "selected_C": selected_c,
                        "train_original_rows": len(train_frame),
                        "train_total_rows": len(train_y_method),
                        "seed": args.seed,
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )

            print(
                f"Selected C={selected_c:g} | "
                f"test_f1={test_metrics.macro_f1:.3f} "
                f"test_bal_acc={test_metrics.balanced_accuracy:.3f} "
                f"test_acc={test_metrics.accuracy:.3f} | "
                f"pH8_acc={boundary['ph8_accuracy']:.3f}"
            )

    pd.DataFrame(candidate_rows).sort_values(
        ["feature_group", "method", "C"]
    ).to_csv(
        output_dir / "validation_candidates.csv",
        index=False,
    )

    pd.DataFrame(all_metric_rows).sort_values(
        ["feature_group", "method", "split"]
    ).to_csv(
        output_dir / "selected_model_metrics_all_splits.csv",
        index=False,
    )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        [
            "test_macro_f1",
            "test_balanced_accuracy",
            "ph8_accuracy",
        ],
        ascending=False,
    )
    comparison.to_csv(
        output_dir / "test_comparison.csv",
        index=False,
    )

    augmentation_summary = pd.DataFrame(
        augmentation_summary_rows
    ).sort_values(["feature_group", "method"])
    augmentation_summary.to_csv(
        output_dir / "augmentation_summary.csv",
        index=False,
    )

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 88)
    print("Test comparison")
    print("=" * 88)
    display_columns = [
        "feature_group",
        "method",
        "selected_C",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "ph6_accuracy",
        "ph8_accuracy",
        "ph9_accuracy",
    ]
    print(comparison[display_columns].to_string(index=False))
    print(f"\nElapsed: {elapsed:.2f} seconds")
    print(f"Saved to: {output_dir}")
    print(
        "\nThis is an exploratory one-split experiment. Do not tune scales "
        "against the test result and then report the same test result as final."
    )


if __name__ == "__main__":
    main()


"""
$runName = "capture_lopo_holdout_galaxy_s24plus_vallight_L01"

python .\scripts\08_feature_nuisance_augmentation_binary.py `
  --manifest ".\data\processed\capture_manifest.csv" `
  --split-file ".\data\splits\$runName.csv" `
  --decomposition-dir ".\outputs\decomposition_binary\$runName" `
  --output-dir ".\outputs\feature_nuisance_binary_smoke\${runName}_seed42" `
  --feature-groups "pad_only,contrast_only" `
  --methods "baseline,gaussian_matched,nuisance_recombined" `
  --c-grid "0.01,0.1,1,10,100" `
  --augmentations-per-row 4 `
  --sampler recombined `
  --phone-scale 1.0 `
  --light-scale 1.0 `
  --interaction-scale 0.5 `
  --residual-scale 0.0 `
  --gaussian-scale 1.0 `
  --seed 42
"""