#!/usr/bin/env python
"""
07_feature_sanity_check_binary.py

Binary acidic-vs-alkaline sanity check using canonical Lab features.

Key rules
---------
- pH 7 is excluded from train, validation, and test.
- StandardScaler is fitted on train rows only.
- Hyperparameter C is selected using validation macro-F1 only.
- The test split is evaluated once after model selection.
- No class weights are used because the binary task is balanced.
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
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


LABEL_ORDER = ["acidic", "alkaline"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABEL_ORDER)}

FEATURE_GROUPS: dict[str, list[str]] = {
    # Negative-control-like input: background alone should not solve pH reliably.
    "bg_only": ["bg_L", "bg_a", "bg_b"],
    # Reaction area only.
    "pad_only": ["pad_L", "pad_a", "pad_b"],
    # Relative reaction-to-background color.
    "contrast_only": ["contrast_L", "contrast_a", "contrast_b"],
    # Reaction and background, without explicitly duplicated contrast columns.
    "pad_plus_bg": [
        "pad_L", "pad_a", "pad_b",
        "bg_L", "bg_a", "bg_b",
    ],
    # Full canonical Lab representation.
    "all_lab": [
        "pad_L", "pad_a", "pad_b",
        "bg_L", "bg_a", "bg_b",
        "contrast_L", "contrast_a", "contrast_b",
    ],
}


@dataclass
class Metrics:
    loss: float | None
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a train-only-standardized acidic-vs-alkaline sanity check "
            "using canonical Lab features."
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
        help="One LOPO split CSV containing an image identifier and split column.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which results will be written.",
    )
    parser.add_argument(
        "--models",
        default="logistic,linear_svm",
        help="Comma-separated subset of: logistic,linear_svm.",
    )
    parser.add_argument(
        "--feature-groups",
        default="bg_only,pad_only,contrast_only,pad_plus_bg,all_lab",
        help="Comma-separated feature groups.",
    )
    parser.add_argument(
        "--c-grid",
        default="0.01,0.1,1,10,100",
        help="Comma-separated positive C values selected using validation macro-F1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def resolve_existing_path(path: Path, project_root: Path) -> Path:
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([
            project_root / path,
            Path.cwd() / path,
        ])
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"File not found: {path}")


def infer_project_root() -> Path:
    # scripts/07_...py -> project root
    return Path(__file__).resolve().parents[1]


def find_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    description: str,
) -> str:
    exact = {str(col): str(col) for col in frame.columns}
    lower = {str(col).lower(): str(col) for col in frame.columns}

    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    raise ValueError(
        f"Could not identify {description}. "
        f"Tried {list(candidates)}. Available columns: {list(frame.columns)}"
    )


def merge_manifest_and_split(
    manifest: pd.DataFrame,
    split_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    manifest_id = find_column(
        manifest,
        ["image_id", "capture_id"],
        "manifest image identifier",
    )
    split_id = find_column(
        split_frame,
        ["image_id", "capture_id"],
        "split image identifier",
    )
    split_col = find_column(
        split_frame,
        ["split", "partition", "subset"],
        "split label column",
    )

    split_min = split_frame[[split_id, split_col]].copy()
    split_min.columns = ["__image_id__", "__split__"]

    if split_min["__image_id__"].duplicated().any():
        dupes = split_min.loc[
            split_min["__image_id__"].duplicated(keep=False),
            "__image_id__",
        ].tolist()
        raise ValueError(
            "Split file contains duplicate image identifiers. "
            f"Examples: {dupes[:10]}"
        )

    merged = manifest.copy()
    merged["__image_id__"] = merged[manifest_id].astype(str)
    split_min["__image_id__"] = split_min["__image_id__"].astype(str)

    merged = merged.merge(
        split_min,
        on="__image_id__",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(split_min):
        manifest_ids = set(manifest[manifest_id].astype(str))
        missing = [
            image_id
            for image_id in split_min["__image_id__"].astype(str)
            if image_id not in manifest_ids
        ]
        raise ValueError(
            f"Only {len(merged)} of {len(split_min)} split rows matched the manifest. "
            f"Missing examples: {missing[:10]}"
        )

    merged["__split__"] = (
        merged["__split__"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"validation": "val", "valid": "val"})
    )

    allowed = {"train", "val", "test"}
    unexpected = sorted(set(merged["__split__"]) - allowed)
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
        bad = result.loc[result[ph_col].isna(), "__image_id__"].tolist()
        raise ValueError(f"Non-numeric pH values found. Examples: {bad[:10]}")

    before = len(result)
    result = result.loc[~np.isclose(result[ph_col], 7.0)].copy()
    excluded = before - len(result)

    result["label_class"] = np.where(
        result[ph_col] < 7.0,
        "acidic",
        "alkaline",
    )
    result["label_index"] = result["label_class"].map(LABEL_TO_INDEX).astype(int)

    counts = (
        result.groupby(["__split__", "label_class"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=["train", "val", "test"], fill_value=0)
        .reindex(columns=LABEL_ORDER, fill_value=0)
    )

    expected_total = {"train": 96, "val": 24, "test": 60}
    observed_total = {
        split: int((result["__split__"] == split).sum())
        for split in ["train", "val", "test"]
    }

    audit = {
        "rows_before_task_filter": before,
        "excluded_ph7_rows": excluded,
        "rows_after_task_filter": len(result),
        "split_rows": observed_total,
        "label_counts": {
            split: {
                label: int(counts.loc[split, label])
                for label in LABEL_ORDER
            }
            for split in counts.index
        },
        "expected_rows_for_current_design": expected_total,
    }

    for split in ["train", "val", "test"]:
        if observed_total[split] == 0:
            raise ValueError(f"Split '{split}' is empty after removing pH 7.")

    # Warn rather than fail so the script remains usable if the dataset later changes.
    for split, expected in expected_total.items():
        if observed_total[split] != expected:
            print(
                f"WARNING: expected {expected} rows for {split}, "
                f"found {observed_total[split]}.",
                file=sys.stderr,
            )

    train_counts = audit["label_counts"]["train"]
    if train_counts["acidic"] != train_counts["alkaline"]:
        print(
            "WARNING: binary train classes are not balanced. "
            f"Counts: {train_counts}",
            file=sys.stderr,
        )

    return result, ph_col, audit


def require_features(frame: pd.DataFrame, groups: list[str]) -> None:
    missing: list[str] = []
    for group in groups:
        if group not in FEATURE_GROUPS:
            raise ValueError(
                f"Unknown feature group '{group}'. "
                f"Available: {sorted(FEATURE_GROUPS)}"
            )
        for column in FEATURE_GROUPS[group]:
            if column not in frame.columns:
                missing.append(column)
    if missing:
        raise ValueError(
            "Canonical manifest is missing required feature columns: "
            f"{sorted(set(missing))}"
        )

    numeric_cols = sorted({
        column
        for group in groups
        for column in FEATURE_GROUPS[group]
    })
    converted = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if converted.isna().any().any():
        bad_columns = converted.columns[converted.isna().any()].tolist()
        raise ValueError(
            "Missing or non-numeric feature values found in: "
            f"{bad_columns}"
        )
    frame[numeric_cols] = converted


def parse_choices(raw: str, allowed: set[str], name: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"No {name} were supplied.")
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown {name}: {unknown}. Allowed: {sorted(allowed)}"
        )
    # Keep order while removing duplicates.
    return list(dict.fromkeys(values))


def parse_c_grid(raw: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --c-grid: {raw}") from exc

    if not values or any((not math.isfinite(value) or value <= 0) for value in values):
        raise ValueError("--c-grid must contain positive finite values.")
    return sorted(set(values))


def build_pipeline(model_name: str, c_value: float, seed: int) -> Pipeline:
    if model_name == "logistic":
        estimator = LogisticRegression(
            C=c_value,
            solver="liblinear",
            max_iter=5000,
            random_state=seed,
            class_weight=None,
        )
    elif model_name == "linear_svm":
        estimator = SVC(
            C=c_value,
            kernel="linear",
            probability=True,
            random_state=seed,
            class_weight=None,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", estimator),
    ])


def positive_probability(pipeline: Pipeline, x: np.ndarray) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        probability = pipeline.predict_proba(x)
        classes = list(pipeline.named_steps["classifier"].classes_)
        alkaline_index = classes.index(LABEL_TO_INDEX["alkaline"])
        return probability[:, alkaline_index]

    decision = pipeline.decision_function(x)
    return 1.0 / (1.0 + np.exp(-decision))


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probability_alkaline: np.ndarray | None,
) -> Metrics:
    loss_value: float | None = None
    if probability_alkaline is not None:
        clipped = np.clip(probability_alkaline, 1e-7, 1 - 1e-7)
        probabilities = np.column_stack([1.0 - clipped, clipped])
        loss_value = float(log_loss(y_true, probabilities, labels=[0, 1]))

    return Metrics(
        loss=loss_value,
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        n=int(len(y_true)),
    )


def split_xy(
    frame: pd.DataFrame,
    split: str,
    features: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    subset = frame.loc[frame["__split__"] == split].copy()
    x = subset[features].to_numpy(dtype=np.float64)
    y = subset["label_index"].to_numpy(dtype=np.int64)
    return subset, x, y


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    # Maximize validation macro-F1, balanced accuracy, and accuracy.
    # On exact ties, prefer the smaller C (represented as -C).
    return (
        float(row["val_macro_f1"]),
        float(row["val_balanced_accuracy"]),
        float(row["val_accuracy"]),
        -float(row["C"]),
    )


def save_predictions(
    output_path: Path,
    subset: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probability_alkaline: np.ndarray,
    ph_col: str,
) -> None:
    pred_frame = pd.DataFrame({
        "image_id": subset["__image_id__"].astype(str).to_numpy(),
        "split": subset["__split__"].astype(str).to_numpy(),
        "ph_nominal": subset[ph_col].to_numpy(),
        "true_index": y_true,
        "true_label": [LABEL_ORDER[index] for index in y_true],
        "pred_index": y_pred,
        "pred_label": [LABEL_ORDER[index] for index in y_pred],
        "probability_acidic": 1.0 - probability_alkaline,
        "probability_alkaline": probability_alkaline,
        "correct": y_true == y_pred,
    })

    for possible in ["phone_id", "phone_label", "light_id", "light_label"]:
        if possible in subset.columns:
            pred_frame[possible] = subset[possible].to_numpy()

    pred_frame.to_csv(output_path, index=False)


def save_coefficients(
    output_path: Path,
    pipeline: Pipeline,
    features: list[str],
) -> None:
    classifier = pipeline.named_steps["classifier"]
    scaler = pipeline.named_steps["scaler"]

    coef = np.asarray(classifier.coef_).reshape(-1)
    scale = np.asarray(scaler.scale_).reshape(-1)
    mean = np.asarray(scaler.mean_).reshape(-1)

    # Coefficients in standardized feature space and approximate raw feature space.
    raw_coef = coef / scale

    coefficient_frame = pd.DataFrame({
        "feature": features,
        "train_mean": mean,
        "train_std": scale,
        "coefficient_standardized": coef,
        "coefficient_raw_scale": raw_coef,
        "absolute_coefficient_standardized": np.abs(coef),
    }).sort_values(
        "absolute_coefficient_standardized",
        ascending=False,
    )
    coefficient_frame.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    project_root = infer_project_root()

    manifest_path = resolve_existing_path(args.manifest, project_root)
    split_path = resolve_existing_path(args.split_file, project_root)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    models = parse_choices(
        args.models,
        {"logistic", "linear_svm"},
        "models",
    )
    feature_groups = parse_choices(
        args.feature_groups,
        set(FEATURE_GROUPS),
        "feature groups",
    )
    c_grid = parse_c_grid(args.c_grid)

    print(f"Manifest: {manifest_path}")
    print(f"Split: {split_path}")
    print(f"Output: {output_dir}")
    print(f"Models: {models}")
    print(f"Feature groups: {feature_groups}")
    print(f"C grid: {c_grid}")
    print("Task: binary acidic vs alkaline; pH 7 excluded")

    manifest = pd.read_csv(manifest_path)
    split_frame = pd.read_csv(split_path)
    frame, merge_columns = merge_manifest_and_split(manifest, split_frame)
    frame, ph_col, task_audit = prepare_binary_task(frame)
    require_features(frame, feature_groups)

    print("\nRows and labels:")
    for split in ["train", "val", "test"]:
        subset = frame.loc[frame["__split__"] == split]
        counts = subset["label_class"].value_counts().reindex(LABEL_ORDER, fill_value=0)
        print(
            f"  {split:5s}: n={len(subset):3d} | "
            f"acidic={int(counts['acidic']):2d} "
            f"alkaline={int(counts['alkaline']):2d}"
        )

    audit = {
        "manifest": str(manifest_path),
        "split_file": str(split_path),
        "output_dir": str(output_dir),
        "task": "binary",
        "labels": LABEL_ORDER,
        "ph7_excluded": True,
        "class_weights": None,
        "scaling": "StandardScaler fitted separately for each model using train rows only.",
        "model_selection": (
            "C selected using validation macro-F1, then validation balanced "
            "accuracy, validation accuracy, then smaller C."
        ),
        "test_usage": "Test evaluated only after selecting C from validation.",
        "merge_columns": merge_columns,
        "task_audit": task_audit,
        "feature_groups": {
            group: FEATURE_GROUPS[group]
            for group in feature_groups
        },
        "models": models,
        "c_grid": c_grid,
        "seed": args.seed,
    }
    with (output_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)

    candidate_rows: list[dict[str, Any]] = []
    selected_test_rows: list[dict[str, Any]] = []
    all_split_metric_rows: list[dict[str, Any]] = []

    start_all = time.perf_counter()

    for feature_group in feature_groups:
        features = FEATURE_GROUPS[feature_group]
        train_subset, x_train, y_train = split_xy(frame, "train", features)
        val_subset, x_val, y_val = split_xy(frame, "val", features)
        test_subset, x_test, y_test = split_xy(frame, "test", features)

        for model_name in models:
            print("\n" + "=" * 80)
            print(f"Feature group: {feature_group} | Model: {model_name}")
            print("=" * 80)

            model_candidates: list[dict[str, Any]] = []

            for c_value in c_grid:
                pipeline = build_pipeline(model_name, c_value, args.seed)
                pipeline.fit(x_train, y_train)

                val_pred = pipeline.predict(x_val)
                val_prob = positive_probability(pipeline, x_val)
                val_metrics = calculate_metrics(y_val, val_pred, val_prob)

                candidate = {
                    "feature_group": feature_group,
                    "model": model_name,
                    "C": c_value,
                    "val_loss": val_metrics.loss,
                    "val_accuracy": val_metrics.accuracy,
                    "val_balanced_accuracy": val_metrics.balanced_accuracy,
                    "val_macro_f1": val_metrics.macro_f1,
                    "val_n": val_metrics.n,
                }
                candidate_rows.append(candidate)
                model_candidates.append(candidate)

                print(
                    f"C={c_value:<6g} "
                    f"val_f1={val_metrics.macro_f1:.3f} "
                    f"val_bal_acc={val_metrics.balanced_accuracy:.3f} "
                    f"val_acc={val_metrics.accuracy:.3f}"
                )

            selected = max(model_candidates, key=selection_key)
            selected_c = float(selected["C"])

            final_pipeline = build_pipeline(model_name, selected_c, args.seed)
            final_pipeline.fit(x_train, y_train)

            run_dir = output_dir / f"{feature_group}__{model_name}"
            run_dir.mkdir(parents=True, exist_ok=True)

            joblib.dump(final_pipeline, run_dir / "model.joblib")
            save_coefficients(
                run_dir / "coefficients.csv",
                final_pipeline,
                features,
            )

            split_results: dict[str, Metrics] = {}
            for split_name, subset, x_values, y_values in [
                ("train", train_subset, x_train, y_train),
                ("val", val_subset, x_val, y_val),
                ("test", test_subset, x_test, y_test),
            ]:
                prediction = final_pipeline.predict(x_values)
                probability = positive_probability(final_pipeline, x_values)
                metrics = calculate_metrics(y_values, prediction, probability)
                split_results[split_name] = metrics

                save_predictions(
                    run_dir / f"predictions_{split_name}.csv",
                    subset,
                    y_values,
                    prediction,
                    probability,
                    ph_col,
                )

                cm = confusion_matrix(y_values, prediction, labels=[0, 1])
                pd.DataFrame(
                    cm,
                    index=[f"true_{label}" for label in LABEL_ORDER],
                    columns=[f"pred_{label}" for label in LABEL_ORDER],
                ).to_csv(run_dir / f"confusion_matrix_{split_name}.csv")

                report = classification_report(
                    y_values,
                    prediction,
                    labels=[0, 1],
                    target_names=LABEL_ORDER,
                    output_dict=True,
                    zero_division=0,
                )
                pd.DataFrame(report).transpose().to_csv(
                    run_dir / f"classification_report_{split_name}.csv"
                )

                all_split_metric_rows.append({
                    "feature_group": feature_group,
                    "features": ",".join(features),
                    "model": model_name,
                    "selected_C": selected_c,
                    "split": split_name,
                    **asdict(metrics),
                })

            test_metrics = split_results["test"]
            selected_test_rows.append({
                "feature_group": feature_group,
                "features": ",".join(features),
                "n_features": len(features),
                "model": model_name,
                "selected_C": selected_c,
                "best_val_macro_f1": selected["val_macro_f1"],
                "best_val_balanced_accuracy": selected["val_balanced_accuracy"],
                "test_loss": test_metrics.loss,
                "test_accuracy": test_metrics.accuracy,
                "test_balanced_accuracy": test_metrics.balanced_accuracy,
                "test_macro_f1": test_metrics.macro_f1,
                "test_n": test_metrics.n,
            })

            config = {
                "feature_group": feature_group,
                "features": features,
                "model": model_name,
                "selected_C": selected_c,
                "seed": args.seed,
                "labels": LABEL_ORDER,
                "train_only_standardization": True,
                "class_weight": None,
            }
            with (run_dir / "run_config.json").open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(config, handle, ensure_ascii=False, indent=2)

            print(
                f"Selected C={selected_c:g} | "
                f"test_f1={test_metrics.macro_f1:.3f} "
                f"test_bal_acc={test_metrics.balanced_accuracy:.3f} "
                f"test_acc={test_metrics.accuracy:.3f}"
            )

    candidate_frame = pd.DataFrame(candidate_rows).sort_values(
        ["feature_group", "model", "C"]
    )
    candidate_frame.to_csv(
        output_dir / "validation_candidates.csv",
        index=False,
    )

    all_metrics_frame = pd.DataFrame(all_split_metric_rows).sort_values(
        ["feature_group", "model", "split"]
    )
    all_metrics_frame.to_csv(
        output_dir / "selected_model_metrics_all_splits.csv",
        index=False,
    )

    comparison = pd.DataFrame(selected_test_rows).sort_values(
        ["test_macro_f1", "test_balanced_accuracy", "test_accuracy"],
        ascending=False,
    )
    comparison.to_csv(output_dir / "test_comparison.csv", index=False)

    elapsed = time.perf_counter() - start_all

    print("\n" + "=" * 80)
    print("Test comparison")
    print("=" * 80)
    display_columns = [
        "feature_group",
        "model",
        "selected_C",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
    ]
    print(comparison[display_columns].to_string(index=False))
    print(f"\nElapsed: {elapsed:.2f} seconds")
    print(f"Saved to: {output_dir}")

    best = comparison.iloc[0]
    print("\nInterpretation aid:")
    print(
        "  - bg_only high: possible background/class confounding or shortcut."
    )
    print(
        "  - contrast_only high: relative pad-to-background color transfers well."
    )
    print(
        "  - pad_only high but full-image CNN failed: CNN input/training was likely the bottleneck."
    )
    print(
        "  - all groups near 0.5 balanced accuracy: cross-phone color shift remains unresolved."
    )
    print(
        f"  - Current best: {best['feature_group']} + {best['model']} "
        f"(test balanced accuracy={best['test_balanced_accuracy']:.3f}, "
        f"macro-F1={best['test_macro_f1']:.3f})."
    )


if __name__ == "__main__":
    main()


"""
python .\scripts\07_feature_sanity_check_binary.py `
  --manifest ".\data\processed\capture_manifest.csv" `
  --split-file ".\data\splits\capture_lopo_holdout_galaxy_s24plus_vallight_L01.csv" `
  --output-dir ".\outputs\feature_sanity_binary\capture_lopo_holdout_galaxy_s24plus_vallight_L01_seed42" `
  --models "logistic,linear_svm" `
  --feature-groups "bg_only,pad_only,contrast_only,pad_plus_bg,all_lab" `
  --c-grid "0.01,0.1,1,10,100" `
  --seed 42
"""

"""
Get-ChildItem .\data\splits -Filter "capture_lopo_holdout_*.csv" |
ForEach-Object {
    $splitPath = $_.FullName
    $runName = $_.BaseName
    $outputDir = Join-Path `
      ".\outputs\feature_sanity_binary_all" `
      $runName

    Write-Host ""
    Write-Host "Running: $runName"

    python .\scripts\07_feature_sanity_check_binary.py `
      --manifest ".\data\processed\capture_manifest.csv" `
      --split-file $splitPath `
      --output-dir $outputDir `
      --models "logistic,linear_svm" `
      --feature-groups "bg_only,pad_only,contrast_only,pad_plus_bg,all_lab" `
      --c-grid "0.01,0.1,1,10,100" `
      --seed 42

    if ($LASTEXITCODE -ne 0) {
        throw "Failed: $runName"
    }
}
"""