#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.color_utils import srgb_to_lab


def norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


ALIASES: Dict[str, list[str]] = {
    "image_id": ["image_id", "sample_id", "capture_id"],
    "image_path": ["image_path", "original_file", "filename", "file_name", "image_file", "file"],
    "phone_id": ["phone_id", "device_id", "camera_id"],
    "phone_label": ["phone_model", "phone_label", "phone", "device_model", "device"],
    "light_id": ["light_id", "illumination_id", "lighting_id"],
    "light_label": ["light_label", "light", "illumination", "lighting"],
    "ph_nominal": ["ph_nominal", "label_ph", "ph", "p_h", "ph_value"],
    "strip_id": ["strip_id", "strip", "test_strip_id"],
    "pad_L": ["pad_l", "pad_lab_l", "reaction_l", "reaction_pad_l", "roi_l", "roi_lab_l", "patch_l"],
    "pad_a": ["pad_a", "pad_lab_a", "reaction_a", "reaction_pad_a", "roi_a", "roi_lab_a", "patch_a"],
    "pad_b": ["pad_b", "pad_lab_b", "reaction_b", "reaction_pad_b", "roi_b", "roi_lab_b", "patch_b"],
    "pad_R": ["pad_r", "pad_rgb_r", "reaction_r", "reaction_pad_r", "roi_r", "roi_rgb_r", "patch_r"],
    "pad_G": ["pad_g", "pad_rgb_g", "reaction_g", "reaction_pad_g", "roi_g", "roi_rgb_g", "patch_g"],
    "pad_B": ["pad_b_rgb", "pad_rgb_b", "reaction_blue", "reaction_pad_blue", "roi_blue", "roi_rgb_b", "patch_blue"],
    "bg_L": ["bg_l", "background_l", "background_lab_l", "bg_lab_l"],
    "bg_a": ["bg_a", "background_a", "background_lab_a", "bg_lab_a"],
    "bg_b": ["bg_b", "background_b_lab", "background_lab_b", "bg_lab_b"],
    "bg_R": ["bg_r", "background_r", "background_rgb_r"],
    "bg_G": ["bg_g", "background_g", "background_rgb_g"],
    "bg_B": ["bg_blue", "background_blue", "background_rgb_b", "bg_rgb_b"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert direct-capture feature CSV to a canonical manifest."
    )
    parser.add_argument("--features", required=True, type=str)
    parser.add_argument("--image-dir", required=True, type=str)
    parser.add_argument(
        "--output",
        default="data/processed/capture_manifest.csv",
        type=str,
    )
    parser.add_argument(
        "--audit-output",
        default="outputs/audit/capture_manifest_audit.json",
        type=str,
    )
    parser.add_argument(
        "--column-map",
        default="",
        help="Optional JSON mapping canonical column names to source CSV columns.",
    )
    parser.add_argument("--neutral-ph", type=float, default=7.0)
    parser.add_argument("--allow-missing-images", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def load_override(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("column-map JSON must be an object")
    return {str(k): str(v) for k, v in raw.items() if str(v).strip()}


def find_column(columns: Iterable[str], canonical: str, override: Dict[str, str]) -> Optional[str]:
    cols = list(columns)
    if canonical in override:
        requested = override[canonical]
        if requested not in cols:
            raise KeyError(f"column-map specifies missing column {requested!r} for {canonical}")
        return requested

    normalized = {norm_name(c): c for c in cols}
    for alias in ALIASES.get(canonical, [canonical]):
        if norm_name(alias) in normalized:
            return normalized[norm_name(alias)]

    # Conservative fallback for metadata fields only. Color channel names such as
    # pad_b (Lab b*) and pad_B (RGB blue) normalize to the same token, so they
    # must be resolved only through the explicit alias lists or column-map.
    if canonical.startswith(("pad_", "bg_")):
        return None
    target = norm_name(canonical)
    for c in cols:
        n = norm_name(c)
        if n == target or n.endswith(target):
            return c
    return None


def numeric_series(df: pd.DataFrame, source: Optional[str], name: str) -> pd.Series:
    if source is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    out = pd.to_numeric(df[source], errors="coerce")
    if out.isna().all():
        raise ValueError(f"Column {source!r}, mapped to {name}, contains no numeric values")
    return out


def recursive_image_index(image_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    duplicates: set[str] = set()
    for p in image_dir.rglob("*"):
        if not p.is_file():
            continue
        key = p.name.lower()
        if key in index:
            duplicates.add(key)
        else:
            index[key] = p
    for key in duplicates:
        index.pop(key, None)
    return index


def resolve_image_path(raw_value: str, image_dir: Path, index: Dict[str, Path]) -> Path:
    text = str(raw_value).strip()
    if not text:
        return Path("")
    p = Path(text)
    if p.is_absolute() and p.exists():
        return p
    direct = image_dir / p
    if direct.exists():
        return direct
    by_name = index.get(p.name.lower())
    if by_name is not None:
        return by_name
    return direct


def assign_class(ph: float, neutral: float) -> str:
    if not np.isfinite(ph):
        return ""
    if np.isclose(ph, neutral):
        return "neutral"
    return "acidic" if ph < neutral else "alkaline"


def main() -> None:
    args = parse_args()
    root = PROJECT_ROOT
    features_path = resolve(root, args.features)
    image_dir = resolve(root, args.image_dir)
    output_path = resolve(root, args.output)
    audit_path = resolve(root, args.audit_output)
    map_path = resolve(root, args.column_map) if args.column_map else None

    if not features_path.exists():
        raise FileNotFoundError(features_path)
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)

    df = pd.read_csv(features_path, keep_default_na=False)
    if df.empty:
        raise ValueError("Feature CSV is empty")

    override = load_override(map_path)
    mapping = {key: find_column(df.columns, key, override) for key in ALIASES}

    required_condition_groups = [
        ("image_path",),
        ("phone_id", "phone_label"),
        ("light_id", "light_label"),
        ("ph_nominal",),
    ]
    for group in required_condition_groups:
        if not any(mapping.get(k) for k in group):
            raise ValueError(
                f"Could not identify any of {group}. Edit config/column_map.example.json "
                "and pass it with --column-map.\n"
                f"Available columns: {list(df.columns)}"
            )

    out = pd.DataFrame(index=df.index)
    image_source = mapping["image_path"]
    assert image_source is not None
    image_index = recursive_image_index(image_dir)
    resolved_images = [resolve_image_path(v, image_dir, image_index) for v in df[image_source]]
    out["image_path"] = [
        p.resolve().relative_to(root.resolve()).as_posix()
        if p and p.exists() and root.resolve() in p.resolve().parents
        else str(p)
        for p in resolved_images
    ]

    if mapping["image_id"]:
        out["image_id"] = df[mapping["image_id"]].astype(str).str.strip()
    else:
        out["image_id"] = [Path(str(v)).stem for v in df[image_source]]

    phone_id_src = mapping["phone_id"] or mapping["phone_label"]
    phone_label_src = mapping["phone_label"] or mapping["phone_id"]
    light_id_src = mapping["light_id"] or mapping["light_label"]
    light_label_src = mapping["light_label"] or mapping["light_id"]
    assert phone_id_src and phone_label_src and light_id_src and light_label_src

    out["phone_id"] = df[phone_id_src].astype(str).str.strip()
    out["phone_label"] = df[phone_label_src].astype(str).str.strip()
    out["light_id"] = df[light_id_src].astype(str).str.strip()
    out["light_label"] = df[light_label_src].astype(str).str.strip()
    out["ph_nominal"] = numeric_series(df, mapping["ph_nominal"], "ph_nominal")
    out["label_class"] = [assign_class(v, args.neutral_ph) for v in out["ph_nominal"]]
    out["strip_id"] = (
        df[mapping["strip_id"]].astype(str).str.strip()
        if mapping["strip_id"]
        else [f"{p}_ph{int(ph):02d}" for p, ph in zip(out["phone_id"], out["ph_nominal"])]
    )

    # Prefer Lab columns; derive from RGB when necessary.
    for region in ["pad", "bg"]:
        lab_sources = [mapping[f"{region}_{c}"] for c in ["L", "a", "b"]]
        if all(lab_sources):
            for c, src in zip(["L", "a", "b"], lab_sources):
                out[f"{region}_{c}"] = numeric_series(df, src, f"{region}_{c}")
            continue

        rgb_sources = [mapping[f"{region}_{c}"] for c in ["R", "G", "B"]]
        if not all(rgb_sources):
            raise ValueError(
                f"Could not identify complete {region} Lab or RGB columns. "
                "Use --column-map to specify them exactly.\n"
                f"Detected mapping: {mapping}"
            )
        rgb = np.column_stack(
            [numeric_series(df, src, f"{region}_{c}").to_numpy() for c, src in zip(["R", "G", "B"], rgb_sources)]
        )
        lab = srgb_to_lab(rgb)
        out[[f"{region}_L", f"{region}_a", f"{region}_b"]] = lab

    for c in ["L", "a", "b"]:
        out[f"contrast_{c}"] = out[f"pad_{c}"] - out[f"bg_{c}"]

    out["source_feature_file"] = features_path.name
    out["preprocess_status"] = "ok"

    duplicate_condition = out.duplicated(["phone_id", "light_id", "ph_nominal"], keep=False)
    duplicate_image_id = out["image_id"].duplicated(keep=False)
    missing_images = [
        path for path in out["image_path"] if not resolve(root, str(path)).exists()
    ]

    condition_counts = {
        "rows": int(len(out)),
        "phones": int(out["phone_id"].nunique()),
        "lights": int(out["light_id"].nunique()),
        "ph_levels": int(out["ph_nominal"].nunique()),
        "expected_full_factorial_rows": int(
            out["phone_id"].nunique()
            * out["light_id"].nunique()
            * out["ph_nominal"].nunique()
        ),
    }
    audit = {
        "source_features": str(features_path),
        "source_image_dir": str(image_dir),
        "detected_column_mapping": mapping,
        "condition_counts": condition_counts,
        "duplicate_condition_rows": int(duplicate_condition.sum()),
        "duplicate_image_id_rows": int(duplicate_image_id.sum()),
        "missing_image_count": len(missing_images),
        "missing_images_first_20": missing_images[:20],
        "phone_levels": sorted(out["phone_id"].astype(str).unique().tolist()),
        "light_levels": sorted(out["light_id"].astype(str).unique().tolist()),
        "ph_levels": sorted(float(v) for v in out["ph_nominal"].dropna().unique()),
    }

    problems = []
    if duplicate_condition.any():
        problems.append("duplicate phone/light/pH combinations")
    if duplicate_image_id.any():
        problems.append("duplicate image_id values")
    if condition_counts["rows"] != condition_counts["expected_full_factorial_rows"]:
        problems.append("dataset is not a complete phone × light × pH factorial")
    if missing_images and not args.allow_missing_images:
        problems.append("missing image files")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    print("Saved canonical manifest:", output_path)
    print("Saved audit:", audit_path)
    print("Counts:", condition_counts)
    print("Detected columns:")
    for k, v in mapping.items():
        if v:
            print(f"  {k:14s} <- {v}")
    if problems:
        print("\nDATA CHECK WARNINGS:")
        for item in problems:
            print(" -", item)
        if missing_images and not args.allow_missing_images:
            raise SystemExit(2)
    else:
        print("Dataset audit passed.")


if __name__ == "__main__":
    main()

"""
python .\scripts\01_build_canonical_manifest.py `
  --features ".\data\raw\captures\captures_final_features.csv" `
  --image-dir ".\data\raw\captures\images" `
  --column-map ".\config\column_map.json"
"""