#!/usr/bin/env python
"""
12_build_factorial_analysis_table.py

Build the final balanced-factorial analysis table for the smartphone pH
colorimetry study and audit the experimental design.

Expected design
---------------
- 3 phones
- 13 pH levels
- 5 lights
- one image in every phone x pH x light cell
- 195 images in total
- one strip for each phone x pH combination
- each strip repeated under all five lights

Primary outputs
---------------
- analysis_factorial_table.csv
- analysis_factorial_table_long.csv
- design_cell_counts.csv
- strip_repeat_audit.csv
- factor_level_counts.csv
- feature_quality_summary.csv
- design_term_df.csv
- data_dictionary.csv
- design_audit.json

Lab extraction
--------------
The script first looks for saved pad/background Lab columns. If complete Lab
columns are unavailable, it falls back to converting saved mean RGB values to
CIE Lab (D65). This fallback is explicitly recorded because Lab(mean RGB) is
not generally identical to mean(pixelwise Lab).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


COLUMN_ALIASES: dict[str, list[str]] = {
    "image_id": [
        "image_id",
        "capture_id",
        "sample_id",
    ],
    "strip_id": [
        "strip_id",
        "strip",
        "test_strip_id",
    ],
    "phone": [
        "phone",
        "phone_id",
        "phone_label",
        "phone_model",
    ],
    "light": [
        "light",
        "light_id",
        "light_label",
        "lighting",
    ],
    "ph_nominal": [
        "ph_nominal",
        "ph_value",
        "ph",
        "pH",
    ],
    "image_path": [
        "image_path",
        "image_path_declared",
        "resolved_image_path",
        "original_file",
    ],
    # Direct pad Lab
    "pad_L": [
        "pad_L",
        "L_pad",
        "pad_L_mean",
        "L_pad_mean",
        "pad_lab_L",
        "pad_mean_L",
        "roi_L",
        "L_roi_mean",
    ],
    "pad_a": [
        "pad_a",
        "a_pad",
        "pad_a_mean",
        "a_pad_mean",
        "pad_lab_a",
        "pad_mean_a",
        "roi_a",
        "a_roi_mean",
    ],
    "pad_b": [
        "pad_b",
        "b_pad",
        "pad_b_mean",
        "b_pad_mean",
        "pad_lab_b",
        "pad_mean_b",
        "roi_b",
        "b_roi_mean",
    ],
    # Direct background Lab
    "bg_L": [
        "bg_L",
        "L_bg",
        "bg_L_mean",
        "L_bg_mean",
        "background_L",
        "background_L_mean",
        "L_background_mean",
        "bg_lab_L",
    ],
    "bg_a": [
        "bg_a",
        "a_bg",
        "bg_a_mean",
        "a_bg_mean",
        "background_a",
        "background_a_mean",
        "a_background_mean",
        "bg_lab_a",
    ],
    "bg_b": [
        "bg_b",
        "b_bg",
        "bg_b_mean",
        "b_bg_mean",
        "background_b",
        "background_b_mean",
        "b_background_mean",
        "bg_lab_b",
    ],
    # Pad RGB fallback
    "pad_R": [
        "pad_R",
        "R_pad",
        "pad_R_mean",
        "R_pad_mean",
        "roi_R",
        "R_roi_mean",
    ],
    "pad_G": [
        "pad_G",
        "G_pad",
        "pad_G_mean",
        "G_pad_mean",
        "roi_G",
        "G_roi_mean",
    ],
    "pad_B": [
        "pad_B",
        "B_pad",
        "pad_B_mean",
        "B_pad_mean",
        "roi_B",
        "B_roi_mean",
    ],
    # Background RGB fallback
    "bg_R": [
        "bg_R",
        "R_bg",
        "bg_R_mean",
        "R_bg_mean",
        "background_R",
        "background_R_mean",
        "R_background_mean",
    ],
    "bg_G": [
        "bg_G",
        "G_bg",
        "bg_G_mean",
        "G_bg_mean",
        "background_G",
        "background_G_mean",
        "G_background_mean",
    ],
    "bg_B": [
        "bg_B",
        "B_bg",
        "bg_B_mean",
        "B_bg_mean",
        "background_B",
        "background_B_mean",
        "B_background_mean",
    ],
}


@dataclass
class AuditItem:
    name: str
    status: str
    observed: Any
    expected: Any
    details: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the final Pad/Background/Contrast Lab table and audit the "
            "3 x 13 x 5 balanced factorial design."
        )
    )
    parser.add_argument(
        "--input",
        default="data/processed/capture_manifest.csv",
        help=(
            "Feature table or canonical manifest containing metadata and "
            "Pad/Background Lab or RGB means."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/factorial_analysis",
    )
    parser.add_argument("--expected-rows", type=int, default=195)
    parser.add_argument("--expected-phones", type=int, default=3)
    parser.add_argument("--expected-lights", type=int, default=5)
    parser.add_argument(
        "--expected-ph",
        default="1:13",
        help=(
            "Expected pH levels. Use start:end for integer sequences or a "
            "comma-separated list, e.g. 1:13 or 1,2,3."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Return a non-zero exit code when any critical design audit fails. "
            "Outputs are still written before exiting."
        ),
    )
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
    raise FileNotFoundError(f"Input file not found: {path_text}")


def resolve_output(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def parse_expected_ph(raw: str) -> list[float]:
    text = raw.strip()
    if ":" in text:
        parts = [part.strip() for part in text.split(":")]
        if len(parts) != 2:
            raise ValueError(
                "--expected-ph range must have the form start:end."
            )
        start = int(parts[0])
        end = int(parts[1])
        step = 1 if end >= start else -1
        return [float(value) for value in range(start, end + step, step)]

    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--expected-ph must contain at least one value.")
    return [float(value) for value in values]


def find_column(
    frame: pd.DataFrame,
    canonical_name: str,
    required: bool = True,
) -> str | None:
    aliases = COLUMN_ALIASES[canonical_name]
    exact = {str(column): str(column) for column in frame.columns}
    lower = {str(column).lower(): str(column) for column in frame.columns}

    for alias in aliases:
        if alias in exact:
            return exact[alias]
        if alias.lower() in lower:
            return lower[alias.lower()]

    if required:
        raise ValueError(
            f"Could not find {canonical_name!r}. Tried aliases {aliases}. "
            f"Available columns: {list(frame.columns)}"
        )
    return None


def numeric_series(
    frame: pd.DataFrame,
    column: str,
    logical_name: str,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    bad = int(values.isna().sum())
    if bad:
        examples = frame.loc[values.isna(), column].head(5).tolist()
        raise ValueError(
            f"{logical_name} contains {bad} non-numeric/missing values. "
            f"Examples: {examples}"
        )
    return values.astype(float)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an N x 3 sRGB array to CIE Lab under D65."""
    values = np.asarray(rgb, dtype=np.float64)

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("RGB array has no finite values.")

    maximum = float(np.max(finite))
    minimum = float(np.min(finite))

    if minimum < 0:
        raise ValueError(
            f"RGB values must be nonnegative; observed minimum={minimum}."
        )

    if maximum <= 1.5:
        rgb_01 = values
        scale_name = "0_to_1"
    elif maximum <= 255.0 + 1e-9:
        rgb_01 = values / 255.0
        scale_name = "0_to_255"
    else:
        raise ValueError(
            f"RGB values exceed 255; observed maximum={maximum}."
        )

    if np.any((rgb_01 < 0.0) | (rgb_01 > 1.0)):
        raise ValueError("RGB values fall outside the inferred valid range.")

    linear = np.where(
        rgb_01 <= 0.04045,
        rgb_01 / 12.92,
        ((rgb_01 + 0.055) / 1.055) ** 2.4,
    )
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = linear @ matrix.T
    white = np.array([0.95047, 1.0, 1.08883])
    scaled = xyz / white

    delta = 6.0 / 29.0
    f = np.where(
        scaled > delta**3,
        np.cbrt(scaled),
        scaled / (3.0 * delta**2) + 4.0 / 29.0,
    )

    lab = np.stack(
        [
            116.0 * f[:, 1] - 16.0,
            500.0 * (f[:, 0] - f[:, 1]),
            200.0 * (f[:, 1] - f[:, 2]),
        ],
        axis=1,
    )
    return lab, scale_name


def extract_region_lab(
    frame: pd.DataFrame,
    region: str,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    if region not in {"pad", "bg"}:
        raise ValueError(f"Unknown region: {region}")

    warnings: list[str] = []
    lab_keys = [f"{region}_L", f"{region}_a", f"{region}_b"]
    rgb_keys = [f"{region}_R", f"{region}_G", f"{region}_B"]

    lab_columns = {
        key: find_column(frame, key, required=False)
        for key in lab_keys
    }
    rgb_columns = {
        key: find_column(frame, key, required=False)
        for key in rgb_keys
    }

    all_lab = all(column is not None for column in lab_columns.values())
    any_lab = any(column is not None for column in lab_columns.values())
    all_rgb = all(column is not None for column in rgb_columns.values())

    result = pd.DataFrame(index=frame.index)
    row_source = pd.Series(index=frame.index, dtype="object")
    rgb_scale = None

    if all_lab:
        for key in lab_keys:
            result[key] = pd.to_numeric(
                frame[lab_columns[key]], errors="coerce"
            )

        missing_mask = result[lab_keys].isna().any(axis=1)

        if missing_mask.any():
            if not all_rgb:
                count = int(missing_mask.sum())
                raise ValueError(
                    f"{region} Lab has {count} incomplete rows and complete "
                    f"{region} RGB fallback columns are unavailable."
                )

            rgb = np.column_stack([
                numeric_series(frame, rgb_columns[key], key).to_numpy()
                for key in rgb_keys
            ])
            converted, rgb_scale = srgb_to_lab(rgb)
            for index, key in enumerate(lab_keys):
                result.loc[missing_mask, key] = converted[missing_mask, index]

            row_source.loc[~missing_mask] = "saved_lab"
            row_source.loc[missing_mask] = "rgb_fallback"
            warnings.append(
                f"{region}: filled {int(missing_mask.sum())} rows from mean RGB."
            )
            overall_source = "mixed_saved_lab_and_rgb_fallback"
        else:
            row_source[:] = "saved_lab"
            overall_source = "saved_lab"

    elif all_rgb:
        rgb = np.column_stack([
            numeric_series(frame, rgb_columns[key], key).to_numpy()
            for key in rgb_keys
        ])
        converted, rgb_scale = srgb_to_lab(rgb)

        for index, key in enumerate(lab_keys):
            result[key] = converted[:, index]

        row_source[:] = "computed_from_mean_rgb"
        overall_source = "computed_from_mean_rgb"

        if any_lab:
            warnings.append(
                f"{region}: incomplete saved Lab columns were ignored because "
                "all RGB means were available."
            )
        warnings.append(
            f"{region}: Lab was computed from mean RGB. Lab(mean RGB) is an "
            "approximation to mean(pixelwise Lab)."
        )
    else:
        missing_lab = [
            key for key, column in lab_columns.items() if column is None
        ]
        missing_rgb = [
            key for key, column in rgb_columns.items() if column is None
        ]
        raise ValueError(
            f"Cannot construct {region} Lab. Missing direct Lab columns "
            f"{missing_lab}; missing RGB fallback columns {missing_rgb}."
        )

    if result[lab_keys].isna().any().any():
        raise ValueError(f"{region} Lab still contains missing values.")

    result[f"{region}_lab_source"] = row_source.astype(str)

    metadata = {
        "region": region,
        "overall_source": overall_source,
        "lab_columns": lab_columns,
        "rgb_columns": rgb_columns,
        "rgb_scale": rgb_scale,
        "source_counts": {
            str(key): int(value)
            for key, value in row_source.value_counts().to_dict().items()
        },
    }
    return result, metadata, warnings


def sorted_levels(values: Sequence[Any]) -> list[str]:
    unique = [str(value) for value in pd.unique(pd.Series(values).astype(str))]
    return sorted(unique)


def treatment_dummy_matrix(
    series: pd.Series,
    levels: Sequence[Any],
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    categorical = pd.Categorical(series, categories=list(levels), ordered=True)
    codes = categorical.codes
    if np.any(codes < 0):
        raise ValueError(f"{prefix} contains values outside supplied levels.")

    columns: list[np.ndarray] = []
    names: list[str] = []

    for level_index in range(1, len(levels)):
        columns.append((codes == level_index).astype(float))
        names.append(f"{prefix}[{levels[level_index]}]")

    if columns:
        return np.column_stack(columns), names
    return np.empty((len(series), 0), dtype=float), names


def interaction_matrix(
    left: np.ndarray,
    left_names: Sequence[str],
    right: np.ndarray,
    right_names: Sequence[str],
    term_name: str,
) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = []
    names: list[str] = []

    for left_index, left_name in enumerate(left_names):
        for right_index, right_name in enumerate(right_names):
            columns.append(
                left[:, left_index] * right[:, right_index]
            )
            names.append(f"{term_name}:{left_name}:{right_name}")

    if columns:
        return np.column_stack(columns), names
    return np.empty((left.shape[0], 0), dtype=float), names


def build_design_matrix_audit(
    frame: pd.DataFrame,
    expected_ph: Sequence[float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    phone_levels = sorted_levels(frame["phone"])
    light_levels = sorted_levels(frame["light"])
    ph_levels = [float(value) for value in expected_ph]

    ph_matrix, ph_names = treatment_dummy_matrix(
        frame["ph_nominal"].astype(float),
        ph_levels,
        "ph",
    )
    phone_matrix, phone_names = treatment_dummy_matrix(
        frame["phone"].astype(str),
        phone_levels,
        "phone",
    )
    light_matrix, light_names = treatment_dummy_matrix(
        frame["light"].astype(str),
        light_levels,
        "light",
    )

    ph_phone, ph_phone_names = interaction_matrix(
        ph_matrix,
        ph_names,
        phone_matrix,
        phone_names,
        "ph_phone",
    )
    ph_light, ph_light_names = interaction_matrix(
        ph_matrix,
        ph_names,
        light_matrix,
        light_names,
        "ph_light",
    )
    phone_light, phone_light_names = interaction_matrix(
        phone_matrix,
        phone_names,
        light_matrix,
        light_names,
        "phone_light",
    )

    blocks = [
        ("Intercept", np.ones((len(frame), 1)), ["Intercept"]),
        ("pH", ph_matrix, ph_names),
        ("Phone", phone_matrix, phone_names),
        ("Light", light_matrix, light_names),
        ("pH:Phone", ph_phone, ph_phone_names),
        ("pH:Light", ph_light, ph_light_names),
        ("Phone:Light", phone_light, phone_light_names),
    ]

    matrix = np.column_stack([block[1] for block in blocks])
    column_names = [
        name
        for _, _, names in blocks
        for name in names
    ]

    rank = int(np.linalg.matrix_rank(matrix))
    n_columns = int(matrix.shape[1])
    residual_df = int(matrix.shape[0] - rank)

    term_rows = []
    for term, block, names in blocks:
        term_rows.append({
            "term": term,
            "degrees_of_freedom": int(block.shape[1]),
            "design_columns": " | ".join(names),
        })

    return {
        "n_rows": int(matrix.shape[0]),
        "n_columns": n_columns,
        "rank": rank,
        "full_column_rank": rank == n_columns,
        "residual_degrees_of_freedom": residual_df,
        "column_names": column_names,
    }, pd.DataFrame(term_rows)


def feature_quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "pad_L", "pad_a", "pad_b",
        "bg_L", "bg_a", "bg_b",
        "contrast_L", "contrast_a", "contrast_b",
        "pad_deltaE_from_bg",
    ]

    rows = []
    for column in feature_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        rows.append({
            "feature": column,
            "n": int(values.notna().sum()),
            "missing": int(values.isna().sum()),
            "minimum": float(values.min()),
            "q05": float(values.quantile(0.05)),
            "median": float(values.median()),
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)),
            "q95": float(values.quantile(0.95)),
            "maximum": float(values.max()),
        })
    return pd.DataFrame(rows)


def make_data_dictionary() -> pd.DataFrame:
    rows = [
        ("image_id", "Unique image/capture identifier", "identifier"),
        ("strip_id", "Physical strip identifier", "identifier"),
        ("phone", "Smartphone model/domain", "factor"),
        ("light", "Lighting condition", "factor"),
        ("ph_nominal", "Nominal pH level, treated as a factor in factorial analysis", "factor"),
        ("image_path", "Source image path when available", "metadata"),
        ("pad_L", "Reaction/strip ROI CIE Lab lightness", "response"),
        ("pad_a", "Reaction/strip ROI CIE Lab green-red coordinate", "response"),
        ("pad_b", "Reaction/strip ROI CIE Lab blue-yellow coordinate", "response"),
        ("bg_L", "Local/background CIE Lab lightness", "response"),
        ("bg_a", "Local/background CIE Lab green-red coordinate", "response"),
        ("bg_b", "Local/background CIE Lab blue-yellow coordinate", "response"),
        ("contrast_L", "pad_L minus bg_L", "derived response"),
        ("contrast_a", "pad_a minus bg_a", "derived response"),
        ("contrast_b", "pad_b minus bg_b", "derived response"),
        ("pad_deltaE_from_bg", "Euclidean CIE76 distance between Pad and Background Lab", "derived response"),
        ("pad_lab_source", "Whether Pad Lab was saved directly or computed from mean RGB", "provenance"),
        ("bg_lab_source", "Whether Background Lab was saved directly or computed from mean RGB", "provenance"),
    ]
    return pd.DataFrame(rows, columns=["column", "description", "role"])


def add_audit(
    items: list[AuditItem],
    name: str,
    observed: Any,
    expected: Any,
    passed: bool,
    details: str = "",
    severity: str = "FAIL",
) -> None:
    items.append(
        AuditItem(
            name=name,
            status="PASS" if passed else severity,
            observed=observed,
            expected=expected,
            details=details,
        )
    )


def main() -> None:
    args = parse_args()
    input_path = resolve_existing(args.input)
    output_dir = resolve_output(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_ph = parse_expected_ph(args.expected_ph)
    expected_ph_set = set(expected_ph)

    raw = pd.read_csv(input_path)
    if raw.empty:
        raise ValueError(f"Input table is empty: {input_path}")

    column_map = {
        "image_id": find_column(raw, "image_id"),
        "strip_id": find_column(raw, "strip_id"),
        "phone": find_column(raw, "phone"),
        "light": find_column(raw, "light"),
        "ph_nominal": find_column(raw, "ph_nominal"),
        "image_path": find_column(raw, "image_path", required=False),
    }

    table = pd.DataFrame({
        "image_id": raw[column_map["image_id"]].astype(str).str.strip(),
        "strip_id": raw[column_map["strip_id"]].astype(str).str.strip(),
        "phone": raw[column_map["phone"]].astype(str).str.strip(),
        "light": raw[column_map["light"]].astype(str).str.strip(),
        "ph_nominal": numeric_series(
            raw,
            column_map["ph_nominal"],
            "ph_nominal",
        ),
    })

    if column_map["image_path"] is not None:
        table["image_path"] = raw[column_map["image_path"]].astype(str)
    else:
        table["image_path"] = ""

    pad_lab, pad_metadata, pad_warnings = extract_region_lab(raw, "pad")
    bg_lab, bg_metadata, bg_warnings = extract_region_lab(raw, "bg")
    table = pd.concat([table, pad_lab, bg_lab], axis=1)

    table["contrast_L"] = table["pad_L"] - table["bg_L"]
    table["contrast_a"] = table["pad_a"] - table["bg_a"]
    table["contrast_b"] = table["pad_b"] - table["bg_b"]
    table["pad_deltaE_from_bg"] = np.sqrt(
        table["contrast_L"] ** 2
        + table["contrast_a"] ** 2
        + table["contrast_b"] ** 2
    )

    table = table.sort_values(
        ["phone", "ph_nominal", "light", "image_id"],
        kind="stable",
    ).reset_index(drop=True)

    audit_items: list[AuditItem] = []
    warnings = [*pad_warnings, *bg_warnings]

    # Basic dimensions and factor levels.
    add_audit(
        audit_items,
        "row_count",
        len(table),
        args.expected_rows,
        len(table) == args.expected_rows,
    )
    add_audit(
        audit_items,
        "unique_image_id_count",
        table["image_id"].nunique(),
        len(table),
        table["image_id"].nunique() == len(table),
        details="Each image_id must identify exactly one row.",
    )
    add_audit(
        audit_items,
        "nonempty_strip_id",
        int((table["strip_id"].str.len() > 0).sum()),
        len(table),
        bool((table["strip_id"].str.len() > 0).all()),
    )
    add_audit(
        audit_items,
        "phone_level_count",
        table["phone"].nunique(),
        args.expected_phones,
        table["phone"].nunique() == args.expected_phones,
        details=", ".join(sorted_levels(table["phone"])),
    )
    add_audit(
        audit_items,
        "light_level_count",
        table["light"].nunique(),
        args.expected_lights,
        table["light"].nunique() == args.expected_lights,
        details=", ".join(sorted_levels(table["light"])),
    )

    observed_ph_set = set(table["ph_nominal"].astype(float).unique())
    add_audit(
        audit_items,
        "ph_levels",
        sorted(observed_ph_set),
        sorted(expected_ph_set),
        observed_ph_set == expected_ph_set,
    )

    # Missingness and plausible Lab bounds.
    core_columns = [
        "image_id", "strip_id", "phone", "light", "ph_nominal",
        "pad_L", "pad_a", "pad_b",
        "bg_L", "bg_a", "bg_b",
        "contrast_L", "contrast_a", "contrast_b",
    ]
    missing_total = int(table[core_columns].isna().sum().sum())
    add_audit(
        audit_items,
        "core_missing_values",
        missing_total,
        0,
        missing_total == 0,
    )

    l_values = pd.concat([table["pad_L"], table["bg_L"]])
    lab_l_valid = bool(((l_values >= 0.0) & (l_values <= 100.0)).all())
    add_audit(
        audit_items,
        "Lab_L_range",
        [float(l_values.min()), float(l_values.max())],
        "[0, 100]",
        lab_l_valid,
    )

    ab_values = pd.concat([
        table["pad_a"], table["pad_b"],
        table["bg_a"], table["bg_b"],
    ])
    lab_ab_valid = bool(
        ((ab_values >= -160.0) & (ab_values <= 160.0)).all()
    )
    add_audit(
        audit_items,
        "Lab_ab_broad_range",
        [float(ab_values.min()), float(ab_values.max())],
        "[-160, 160]",
        lab_ab_valid,
        details="Broad plausibility bound, not a substantive clipping rule.",
    )

    # Full phone x pH x light grid.
    phones = sorted_levels(table["phone"])
    lights = sorted_levels(table["light"])
    expected_grid = pd.MultiIndex.from_product(
        [phones, expected_ph, lights],
        names=["phone", "ph_nominal", "light"],
    ).to_frame(index=False)

    observed_cells = (
        table.groupby(
            ["phone", "ph_nominal", "light"],
            as_index=False,
            dropna=False,
        )
        .size()
        .rename(columns={"size": "observed_count"})
    )
    cell_counts = expected_grid.merge(
        observed_cells,
        on=["phone", "ph_nominal", "light"],
        how="left",
    )
    cell_counts["observed_count"] = (
        cell_counts["observed_count"].fillna(0).astype(int)
    )
    cell_counts["expected_count"] = 1
    cell_counts["cell_status"] = np.select(
        [
            cell_counts["observed_count"].eq(1),
            cell_counts["observed_count"].eq(0),
        ],
        ["PASS", "MISSING"],
        default="DUPLICATE",
    )
    cell_counts.to_csv(
        output_dir / "design_cell_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    bad_cells = cell_counts.loc[
        cell_counts["observed_count"] != 1
    ]
    add_audit(
        audit_items,
        "balanced_phone_ph_light_cells",
        int((cell_counts["observed_count"] == 1).sum()),
        len(cell_counts),
        bad_cells.empty,
        details=(
            "Every phone x pH x light cell must contain exactly one image. "
            f"Bad cells={len(bad_cells)}."
        ),
    )

    observed_outside_grid = table.merge(
        expected_grid,
        on=["phone", "ph_nominal", "light"],
        how="left",
        indicator=True,
    )
    outside_count = int(
        (observed_outside_grid["_merge"] == "left_only").sum()
    )
    add_audit(
        audit_items,
        "rows_outside_expected_grid",
        outside_count,
        0,
        outside_count == 0,
    )

    # phone x pH strip structure.
    phone_ph_audit = (
        table.groupby(["phone", "ph_nominal"], as_index=False)
        .agg(
            row_count=("image_id", "size"),
            unique_strip_count=("strip_id", "nunique"),
            unique_light_count=("light", "nunique"),
            strip_ids=("strip_id", lambda values: " | ".join(
                sorted(set(map(str, values)))
            )),
            light_ids=("light", lambda values: " | ".join(
                sorted(set(map(str, values)))
            )),
        )
    )
    phone_ph_audit["expected_rows"] = args.expected_lights
    phone_ph_audit["expected_unique_strips"] = 1
    phone_ph_audit["expected_unique_lights"] = args.expected_lights
    phone_ph_audit["status"] = np.where(
        (
            phone_ph_audit["row_count"].eq(args.expected_lights)
            & phone_ph_audit["unique_strip_count"].eq(1)
            & phone_ph_audit["unique_light_count"].eq(args.expected_lights)
        ),
        "PASS",
        "FAIL",
    )
    phone_ph_audit.to_csv(
        output_dir / "phone_ph_strip_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    expected_phone_ph_cells = args.expected_phones * len(expected_ph)
    good_phone_ph = int((phone_ph_audit["status"] == "PASS").sum())
    add_audit(
        audit_items,
        "one_strip_repeated_across_lights_per_phone_ph",
        good_phone_ph,
        expected_phone_ph_cells,
        (
            len(phone_ph_audit) == expected_phone_ph_cells
            and good_phone_ph == expected_phone_ph_cells
        ),
        details=(
            "Each phone x pH combination should use one strip observed under "
            "all light levels."
        ),
    )

    # strip-level mapping and repeat audit.
    strip_audit = (
        table.groupby("strip_id", as_index=False)
        .agg(
            row_count=("image_id", "size"),
            unique_phone_count=("phone", "nunique"),
            unique_ph_count=("ph_nominal", "nunique"),
            unique_light_count=("light", "nunique"),
            phones=("phone", lambda values: " | ".join(
                sorted(set(map(str, values)))
            )),
            ph_values=("ph_nominal", lambda values: " | ".join(
                f"{value:g}" for value in sorted(set(map(float, values)))
            )),
            lights=("light", lambda values: " | ".join(
                sorted(set(map(str, values)))
            )),
        )
    )
    strip_audit["expected_rows"] = args.expected_lights
    strip_audit["status"] = np.where(
        (
            strip_audit["row_count"].eq(args.expected_lights)
            & strip_audit["unique_phone_count"].eq(1)
            & strip_audit["unique_ph_count"].eq(1)
            & strip_audit["unique_light_count"].eq(args.expected_lights)
        ),
        "PASS",
        "FAIL",
    )
    strip_audit.to_csv(
        output_dir / "strip_repeat_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    expected_strips = args.expected_phones * len(expected_ph)
    good_strips = int((strip_audit["status"] == "PASS").sum())
    add_audit(
        audit_items,
        "strip_repeat_structure",
        {
            "unique_strips": int(len(strip_audit)),
            "passing_strips": good_strips,
        },
        {
            "unique_strips": expected_strips,
            "passing_strips": expected_strips,
        },
        (
            len(strip_audit) == expected_strips
            and good_strips == expected_strips
        ),
    )

    # Factor counts.
    factor_rows: list[dict[str, Any]] = []
    for factor in ["phone", "light", "ph_nominal"]:
        counts = table[factor].value_counts(dropna=False, sort=False)
        for level, count in counts.items():
            factor_rows.append({
                "factor": factor,
                "level": level,
                "row_count": int(count),
            })
    factor_level_counts = pd.DataFrame(factor_rows)
    factor_level_counts.to_csv(
        output_dir / "factor_level_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Design matrix rank for the planned two-way factorial model.
    design_matrix_audit, term_df = build_design_matrix_audit(
        table,
        expected_ph,
    )
    term_df.to_csv(
        output_dir / "design_term_df.csv",
        index=False,
        encoding="utf-8-sig",
    )
    add_audit(
        audit_items,
        "two_way_factorial_design_matrix_full_rank",
        {
            "rank": design_matrix_audit["rank"],
            "columns": design_matrix_audit["n_columns"],
            "residual_df": (
                design_matrix_audit["residual_degrees_of_freedom"]
            ),
        },
        {
            "rank_equals_columns": True,
            "residual_df_positive": True,
        },
        (
            design_matrix_audit["full_column_rank"]
            and design_matrix_audit[
                "residual_degrees_of_freedom"
            ] > 0
        ),
        details=(
            "Planned model: pH + Phone + Light + pH:Phone + pH:Light "
            "+ Phone:Light."
        ),
    )

    # Output analysis tables.
    table.to_csv(
        output_dir / "analysis_factorial_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    long_frames = []
    for region in ["pad", "bg", "contrast"]:
        region_frame = table[
            [
                "image_id",
                "strip_id",
                "phone",
                "light",
                "ph_nominal",
                "image_path",
            ]
        ].copy()
        region_frame["region"] = region
        region_frame["L"] = table[f"{region}_L"]
        region_frame["a"] = table[f"{region}_a"]
        region_frame["b"] = table[f"{region}_b"]
        region_frame["deltaE_from_bg"] = (
            table["pad_deltaE_from_bg"]
            if region == "pad"
            else np.nan
        )
        region_frame["lab_source"] = (
            table["pad_lab_source"]
            if region == "pad"
            else (
                table["bg_lab_source"]
                if region == "bg"
                else (
                    table["pad_lab_source"].astype(str)
                    + " minus "
                    + table["bg_lab_source"].astype(str)
                )
            )
        )
        long_frames.append(region_frame)

    long_table = pd.concat(long_frames, ignore_index=True)
    long_table.to_csv(
        output_dir / "analysis_factorial_table_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    quality = feature_quality_summary(table)
    quality.to_csv(
        output_dir / "feature_quality_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    make_data_dictionary().to_csv(
        output_dir / "data_dictionary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit_frame = pd.DataFrame([
        {
            "audit_item": item.name,
            "status": item.status,
            "observed": json.dumps(
                item.observed,
                ensure_ascii=False,
                default=str,
            ),
            "expected": json.dumps(
                item.expected,
                ensure_ascii=False,
                default=str,
            ),
            "details": item.details,
        }
        for item in audit_items
    ])
    audit_frame.to_csv(
        output_dir / "design_audit_items.csv",
        index=False,
        encoding="utf-8-sig",
    )

    failed_items = [
        item.name for item in audit_items if item.status == "FAIL"
    ]
    warned_items = [
        item.name for item in audit_items if item.status == "WARN"
    ]
    overall_status = "FAIL" if failed_items else (
        "WARN" if warnings or warned_items else "PASS"
    )

    audit_json = {
        "overall_status": overall_status,
        "input_file": str(input_path),
        "output_directory": str(output_dir),
        "input_rows": int(len(raw)),
        "analysis_rows": int(len(table)),
        "expected_design": {
            "rows": args.expected_rows,
            "phones": args.expected_phones,
            "lights": args.expected_lights,
            "ph_levels": expected_ph,
            "phone_ph_light_cell_count": (
                args.expected_phones
                * len(expected_ph)
                * args.expected_lights
            ),
            "phone_ph_strip_count": (
                args.expected_phones * len(expected_ph)
            ),
        },
        "observed_levels": {
            "phones": phones,
            "lights": lights,
            "ph_levels": sorted(observed_ph_set),
        },
        "column_map": column_map,
        "lab_extraction": {
            "pad": pad_metadata,
            "background": bg_metadata,
        },
        "warnings": warnings,
        "failed_audit_items": failed_items,
        "warned_audit_items": warned_items,
        "design_matrix": design_matrix_audit,
        "audit_items": [
            {
                "name": item.name,
                "status": item.status,
                "observed": item.observed,
                "expected": item.expected,
                "details": item.details,
            }
            for item in audit_items
        ],
        "important_limitations": [
            (
                "Each phone x pH cell contains one physical strip. Therefore "
                "the empirical pH x phone profile cannot be separated from "
                "strip-specific variation within that cell."
            ),
            (
                "There is one observation per phone x pH x light cell. The "
                "three-way interaction and pure measurement error are not "
                "separately estimable and remain in the residual term of the "
                "planned two-way model."
            ),
            (
                "If Lab was computed from mean RGB, it is an approximation to "
                "the mean of pixelwise Lab values."
            ),
        ],
    }
    with (output_dir / "design_audit.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            audit_json,
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print("=" * 88)
    print("Factorial analysis table build")
    print("=" * 88)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Rows: {len(table)}")
    print(f"Phones: {phones}")
    print(f"Lights: {lights}")
    print(f"pH levels: {sorted(observed_ph_set)}")
    print(
        "Pad Lab source:",
        pad_metadata["overall_source"],
        pad_metadata["source_counts"],
    )
    print(
        "Background Lab source:",
        bg_metadata["overall_source"],
        bg_metadata["source_counts"],
    )
    print(
        "Design matrix:",
        f"rank={design_matrix_audit['rank']}/"
        f"{design_matrix_audit['n_columns']}, "
        f"residual df="
        f"{design_matrix_audit['residual_degrees_of_freedom']}",
    )
    print(f"Overall audit status: {overall_status}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if failed_items:
        print("\nFailed audit items:")
        for item in failed_items:
            print(f"  - {item}")

    print("\nMain files:")
    print(f"  {output_dir / 'analysis_factorial_table.csv'}")
    print(f"  {output_dir / 'analysis_factorial_table_long.csv'}")
    print(f"  {output_dir / 'design_audit.json'}")
    print(f"  {output_dir / 'design_audit_items.csv'}")

    if args.strict and failed_items:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
