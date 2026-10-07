#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create leakage-safe direct-capture LOPO splits. Test uses one whole phone; "
            "validation uses one light from the remaining phones; train uses the other phones/lights."
        )
    )
    p.add_argument(
        "--manifest",
        default="data/processed/capture_manifest.csv",
        type=str,
    )
    p.add_argument("--output-dir", default="data/splits", type=str)
    p.add_argument(
        "--val-lights",
        nargs="*",
        default=None,
        help="Optional light IDs. Default: create one split for every light.",
    )
    return p.parse_args()


def resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def validate_columns(df: pd.DataFrame) -> None:
    required = {"image_id", "phone_id", "light_id", "ph_nominal", "label_class"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")


def class_counts(df: pd.DataFrame) -> dict:
    return {
        str(k): int(v)
        for k, v in df["label_class"].value_counts().sort_index().items()
    }


def main() -> None:
    args = parse_args()
    manifest_path = resolve(args.manifest)
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path, dtype={"image_id": str, "phone_id": str, "light_id": str})
    validate_columns(df)
    df["phone_id"] = df["phone_id"].astype(str)
    df["light_id"] = df["light_id"].astype(str)

    phones = sorted(df["phone_id"].unique().tolist())
    all_lights = sorted(df["light_id"].unique().tolist())
    val_lights = all_lights if args.val_lights is None else [str(v) for v in args.val_lights]

    unknown = sorted(set(val_lights) - set(all_lights))
    if unknown:
        raise ValueError(f"Unknown validation light IDs: {unknown}. Available: {all_lights}")

    summary_rows = []
    for holdout_phone in phones:
        source_mask = df["phone_id"].ne(holdout_phone)
        test_mask = df["phone_id"].eq(holdout_phone)

        for val_light in val_lights:
            split = pd.Series("", index=df.index, dtype=object)
            split.loc[test_mask] = "test"
            split.loc[source_mask & df["light_id"].eq(val_light)] = "val"
            split.loc[source_mask & df["light_id"].ne(val_light)] = "train"

            split_df = df[
                ["image_id", "image_path", "phone_id", "light_id", "ph_nominal", "label_class"]
            ].copy()
            split_df["split"] = split
            split_df["holdout_phone"] = holdout_phone
            split_df["validation_light"] = val_light

            # Leakage checks.
            train = split_df[split_df["split"].eq("train")]
            val = split_df[split_df["split"].eq("val")]
            test = split_df[split_df["split"].eq("test")]
            if train.empty or val.empty or test.empty:
                raise ValueError(
                    f"Empty split for holdout_phone={holdout_phone}, val_light={val_light}: "
                    f"train={len(train)}, val={len(val)}, test={len(test)}"
                )
            if holdout_phone in set(train["phone_id"]) | set(val["phone_id"]):
                raise AssertionError("Held-out phone leaked into train/val")
            if val_light in set(train["light_id"]):
                raise AssertionError("Validation light leaked into train")
            if set(train["image_id"]) & set(test["image_id"]):
                raise AssertionError("Train/test image leakage")

            safe_phone = holdout_phone.replace(" ", "_").replace("/", "-")
            safe_light = val_light.replace(" ", "_").replace("/", "-")
            filename = f"capture_lopo_holdout_{safe_phone}_vallight_{safe_light}.csv"
            split_df.to_csv(out_dir / filename, index=False, encoding="utf-8-sig")

            summary_rows.append(
                {
                    "split_file": filename,
                    "holdout_phone": holdout_phone,
                    "validation_light": val_light,
                    "train_rows": len(train),
                    "val_rows": len(val),
                    "test_rows": len(test),
                    "train_phones": ",".join(sorted(train["phone_id"].unique())),
                    "train_lights": ",".join(sorted(train["light_id"].unique())),
                    "train_class_counts": json.dumps(class_counts(train), ensure_ascii=False),
                    "val_class_counts": json.dumps(class_counts(val), ensure_ascii=False),
                    "test_class_counts": json.dumps(class_counts(test), ensure_ascii=False),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "capture_lopo_split_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Created {len(summary_rows)} LOPO split files in {out_dir}")
    print(summary[["holdout_phone", "validation_light", "train_rows", "val_rows", "test_rows"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""
python .\scripts\02_make_lopo_splits.py
"""