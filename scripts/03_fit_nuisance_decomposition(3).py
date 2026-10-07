#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGION_CHANNELS = {
    "pad": ["pad_L", "pad_a", "pad_b"],
    "bg": ["bg_L", "bg_a", "bg_b"],
    "contrast": ["contrast_L", "contrast_a", "contrast_b"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fit an ANOVA-style nuisance decomposition on TRAIN rows only: "
            "value = grand + pH + phone + light + phone×light + residual."
        )
    )
    p.add_argument("--manifest", default="data/processed/capture_manifest.csv")
    p.add_argument("--split-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--regions",
        nargs="+",
        choices=sorted(REGION_CHANNELS),
        default=["pad", "bg", "contrast"],
    )
    return p.parse_args()


def resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_train(manifest_path: Path, split_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, dtype={"image_id": str, "phone_id": str, "light_id": str})
    split = pd.read_csv(split_path, dtype={"image_id": str, "phone_id": str, "light_id": str})
    if "split" not in split.columns:
        raise ValueError(f"Split file lacks 'split' column: {split_path}")
    keep = split[["image_id", "split"]].drop_duplicates("image_id")
    df = manifest.merge(keep, on="image_id", how="inner")
    train = df[df["split"].eq("train")].copy()
    if train.empty:
        raise ValueError("No training rows after merging manifest and split")
    return train.reset_index(drop=True)


def weighted_center(series: pd.Series, weights: pd.Series | None = None) -> pd.Series:
    if weights is None:
        return series - series.mean()
    w = weights.reindex(series.index).astype(float)
    center = float(np.average(series.to_numpy(float), weights=w.to_numpy(float)))
    return series - center


def fit_one(train: pd.DataFrame, value_col: str) -> Tuple[pd.DataFrame, list[dict], dict]:
    data = train[["image_id", "phone_id", "light_id", "ph_nominal", value_col]].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    if data[value_col].isna().any():
        bad = int(data[value_col].isna().sum())
        raise ValueError(f"{value_col} has {bad} missing/non-numeric training values")

    y = data[value_col].astype(float)
    grand = float(y.mean())

    ph_means = data.groupby("ph_nominal", sort=True)[value_col].mean()
    ph_counts = data.groupby("ph_nominal", sort=True)[value_col].size()
    ph_effect = weighted_center(ph_means, ph_counts)

    phone_means = data.groupby("phone_id", sort=True)[value_col].mean()
    phone_counts = data.groupby("phone_id", sort=True)[value_col].size()
    phone_effect = weighted_center(phone_means, phone_counts)

    light_means = data.groupby("light_id", sort=True)[value_col].mean()
    light_counts = data.groupby("light_id", sort=True)[value_col].size()
    light_effect = weighted_center(light_means, light_counts)

    base = (
        grand
        + data["ph_nominal"].map(ph_effect).astype(float)
        + data["phone_id"].map(phone_effect).astype(float)
        + data["light_id"].map(light_effect).astype(float)
    )
    interaction_raw = (y - base).groupby([data["phone_id"], data["light_id"]]).mean()
    interaction_counts = data.groupby(["phone_id", "light_id"])[value_col].size()
    interaction_effect = weighted_center(interaction_raw, interaction_counts)

    pred = (
        base
        + pd.MultiIndex.from_frame(data[["phone_id", "light_id"]]).map(interaction_effect).astype(float)
    )
    residual = y.to_numpy(float) - np.asarray(pred, dtype=float)

    row = data.copy()
    row["grand"] = grand
    row["ph_effect"] = data["ph_nominal"].map(ph_effect).astype(float)
    row["phone_effect"] = data["phone_id"].map(phone_effect).astype(float)
    row["light_effect"] = data["light_id"].map(light_effect).astype(float)
    row["interaction_effect"] = pd.MultiIndex.from_frame(data[["phone_id", "light_id"]]).map(interaction_effect).astype(float)
    row["fitted"] = np.asarray(pred, dtype=float)
    row["residual"] = residual
    row["value_column"] = value_col

    components: list[dict] = [
        {
            "value_column": value_col,
            "effect_type": "grand",
            "level_1": "all",
            "level_2": "",
            "effect": grand,
            "n_rows": len(data),
        }
    ]
    for level, val in ph_effect.items():
        components.append({
            "value_column": value_col,
            "effect_type": "ph",
            "level_1": level,
            "level_2": "",
            "effect": float(val),
            "n_rows": int(ph_counts.loc[level]),
        })
    for level, val in phone_effect.items():
        components.append({
            "value_column": value_col,
            "effect_type": "phone",
            "level_1": level,
            "level_2": "",
            "effect": float(val),
            "n_rows": int(phone_counts.loc[level]),
        })
    for level, val in light_effect.items():
        components.append({
            "value_column": value_col,
            "effect_type": "light",
            "level_1": level,
            "level_2": "",
            "effect": float(val),
            "n_rows": int(light_counts.loc[level]),
        })
    for (phone, light), val in interaction_effect.items():
        components.append({
            "value_column": value_col,
            "effect_type": "phone_light",
            "level_1": phone,
            "level_2": light,
            "effect": float(val),
            "n_rows": int(interaction_counts.loc[(phone, light)]),
        })

    sse = float(np.sum(residual**2))
    sst = float(np.sum((y.to_numpy(float) - grand) ** 2))
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    summary = {
        "value_column": value_col,
        "n_rows": len(data),
        "grand_mean": grand,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "residual_std": float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0,
        "n_ph": int(data["ph_nominal"].nunique()),
        "n_phones": int(data["phone_id"].nunique()),
        "n_lights": int(data["light_id"].nunique()),
        "n_phone_light_cells": int(data[["phone_id", "light_id"]].drop_duplicates().shape[0]),
    }
    return row, components, summary


def build_vector_library(row_decomp: pd.DataFrame, regions: Iterable[str]) -> pd.DataFrame:
    parts = []
    for region in regions:
        cols = REGION_CHANNELS[region]
        sub = row_decomp[row_decomp["value_column"].isin(cols)].copy()
        if sub.empty:
            continue
        key_cols = ["image_id", "phone_id", "light_id", "ph_nominal"]
        pivot = sub.pivot(index=key_cols, columns="value_column", values=[
            "phone_effect", "light_effect", "interaction_effect", "residual"
        ])
        pivot.columns = [f"{effect}_{channel}" for effect, channel in pivot.columns]
        pivot = pivot.reset_index()
        pivot["region"] = region

        rename = {}
        for effect in ["phone_effect", "light_effect", "interaction_effect", "residual"]:
            for channel_name, short in zip(cols, ["L", "a", "b"]):
                old = f"{effect}_{channel_name}"
                if old in pivot.columns:
                    rename[old] = f"{effect}_{short}"
        pivot = pivot.rename(columns=rename)

        for effect in ["phone_effect", "light_effect", "interaction_effect", "residual"]:
            needed = [f"{effect}_{c}" for c in ["L", "a", "b"]]
            if all(c in pivot.columns for c in needed):
                pivot[f"{effect}_deltaE"] = np.sqrt(sum(pivot[c].astype(float) ** 2 for c in needed))
        parts.append(pivot)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main() -> None:
    args = parse_args()
    manifest_path = resolve(args.manifest)
    split_path = resolve(args.split_file)
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_train(manifest_path, split_path)
    required = [col for region in args.regions for col in REGION_CHANNELS[region]]
    missing = sorted(set(required) - set(train.columns))
    if missing:
        raise ValueError(f"Manifest lacks required feature columns: {missing}")

    row_parts = []
    component_rows: list[dict] = []
    summaries: list[dict] = []
    for region in args.regions:
        for value_col in REGION_CHANNELS[region]:
            row, components, summary = fit_one(train, value_col)
            row["region"] = region
            row_parts.append(row)
            component_rows.extend(components)
            summary["region"] = region
            summaries.append(summary)

    row_decomp = pd.concat(row_parts, ignore_index=True)
    components = pd.DataFrame(component_rows)
    fit_summary = pd.DataFrame(summaries)
    vector_library = build_vector_library(row_decomp, args.regions)

    row_decomp.to_csv(out_dir / "row_decomposition.csv", index=False, encoding="utf-8-sig")
    components.to_csv(out_dir / "component_effects.csv", index=False, encoding="utf-8-sig")
    fit_summary.to_csv(out_dir / "fit_summary.csv", index=False, encoding="utf-8-sig")
    vector_library.to_csv(out_dir / "nuisance_vector_library.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "manifest": str(manifest_path),
        "split_file": str(split_path),
        "training_rows": len(train),
        "training_phones": sorted(train["phone_id"].astype(str).unique().tolist()),
        "training_lights": sorted(train["light_id"].astype(str).unique().tolist()),
        "training_ph_levels": sorted(float(v) for v in train["ph_nominal"].unique()),
        "regions": args.regions,
        "model": "grand + pH + phone + light + phone×light + residual",
        "important": "All effects were estimated from rows labelled train in the supplied split only.",
    }
    with (out_dir / "decomposition_config.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Saved decomposition to:", out_dir)
    print("Training rows:", len(train))
    print("Training phones:", metadata["training_phones"])
    print("Training lights:", metadata["training_lights"])
    print("\nFit summary:")
    print(fit_summary[["region", "value_column", "rmse", "mae", "r2", "residual_std"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""
python .\scripts\03_fit_nuisance_decomposition.py `
  --split-file ".\data\splits\capture_lopo_holdout_galaxy_s24plus_vallight_L01.csv" `
  --output-dir ".\outputs\decomposition\test_decomposition"
"""

"""
Get-ChildItem .\data\splits -Filter "capture_lopo_holdout_*.csv" |
ForEach-Object {
    $splitPath = $_.FullName
    $runName = $_.BaseName
    $outputDir = Join-Path ".\outputs\decomposition" $runName

    Write-Host ""
    Write-Host "Running: $runName"

    python .\scripts\03_fit_nuisance_decomposition.py `
      --split-file $splitPath `
      --output-dir $outputDir

    if ($LASTEXITCODE -ne 0) {
        throw "Failed: $runName"
    }
}
"""