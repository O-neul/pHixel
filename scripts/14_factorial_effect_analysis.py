#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
14_factorial_effect_analysis.py

Balanced three-factor ANOVA for the smartphone pH colorimetry study.

This version does NOT use statsmodels.

Design
------
A = pH       : 13 levels
B = phone    : 3 levels
C = light    : 5 levels

There is exactly one observation in every A x B x C cell.

Fitted two-way model
--------------------
Y = grand mean
  + pH
  + phone
  + light
  + pH x phone
  + pH x light
  + phone x light
  + residual

With one observation per three-way cell, the residual contains:

- pH x phone x light interaction
- pure measurement/extraction error

These two parts cannot be separated.

Because the design is complete and balanced, sums of squares are computed
directly from marginal means. This is the standard orthogonal factorial ANOVA
decomposition and does not require formula software.

Dependencies
------------
- numpy
- pandas
- scipy only for F-distribution p-values

Outputs
-------
outputs/factorial_effects/
├─ anova_long.csv
├─ feature_effect_summary.csv
├─ effect_composition.csv
├─ region_effect_summary.csv
├─ model_fit_summary.csv
├─ paper_effect_size_table.csv
├─ analysis_audit.json
└─ analysis_notes.txt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from scipy.stats import f as f_distribution
except ImportError as exc:
    raise ImportError(
        "scipy is required only for ANOVA p-values. "
        "Install it with: python -m pip install scipy"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FEATURES = [
    "pad_L",
    "pad_a",
    "pad_b",
    "bg_L",
    "bg_a",
    "bg_b",
    "contrast_L",
    "contrast_a",
    "contrast_b",
]

TERM_ORDER = [
    "pH",
    "phone",
    "light",
    "pH_x_phone",
    "pH_x_light",
    "phone_x_light",
    "residual",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute balanced three-factor ANOVA without statsmodels."
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
        default="outputs/factorial_effects",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=DEFAULT_FEATURES,
    )
    parser.add_argument(
        "--negative-tolerance",
        type=float,
        default=1e-8,
        help=(
            "Numerical tolerance for a slightly negative residual SS."
        ),
    )
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
        f"Could not find {description}. "
        f"Tried {list(candidates)}. "
        f"Available columns: {list(frame.columns)}"
    )


def prepare_data(
    frame: pd.DataFrame,
    requested_features: list[str],
) -> pd.DataFrame:
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

    missing_features = sorted(
        set(requested_features) - set(frame.columns)
    )
    if missing_features:
        raise ValueError(
            f"Missing requested features: {missing_features}"
        )

    result = frame.copy()
    result["phone"] = result[phone_column].astype(str)
    result["light"] = result[light_column].astype(str)
    result["ph_nominal"] = pd.to_numeric(
        result[ph_column],
        errors="raise",
    ).astype(float)

    for feature in requested_features:
        result[feature] = pd.to_numeric(
            result[feature],
            errors="raise",
        ).astype(float)

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

    ph_levels = sorted(
        result["ph_nominal"].unique().tolist()
    )
    phone_levels = sorted(
        result["phone"].unique().tolist()
    )
    light_levels = sorted(
        result["light"].unique().tolist()
    )

    expected_rows = (
        len(ph_levels)
        * len(phone_levels)
        * len(light_levels)
    )

    if len(result) != expected_rows:
        raise ValueError(
            "Incomplete factorial design: "
            f"rows={len(result)}, expected={expected_rows}."
        )

    expected_index = pd.MultiIndex.from_product(
        [ph_levels, phone_levels, light_levels],
        names=["ph_nominal", "phone", "light"],
    )
    observed_index = pd.MultiIndex.from_frame(
        result[["ph_nominal", "phone", "light"]]
    )

    missing_cells = expected_index.difference(observed_index)
    extra_cells = observed_index.difference(expected_index)

    if len(missing_cells) or len(extra_cells):
        raise ValueError(
            "Factorial cell audit failed. "
            f"Missing cells={len(missing_cells)}, "
            f"extra cells={len(extra_cells)}."
        )

    return result.sort_values(
        ["ph_nominal", "phone", "light"],
        kind="stable",
    ).reset_index(drop=True)


def significance_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""

    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"

    return ""


def factorial_array(
    frame: pd.DataFrame,
    feature: str,
) -> tuple[
    np.ndarray,
    list[float],
    list[str],
    list[str],
]:
    ph_levels = sorted(
        frame["ph_nominal"].unique().tolist()
    )
    phone_levels = sorted(
        frame["phone"].unique().tolist()
    )
    light_levels = sorted(
        frame["light"].unique().tolist()
    )

    indexed = frame.set_index(
        ["ph_nominal", "phone", "light"]
    )

    values = np.empty(
        (
            len(ph_levels),
            len(phone_levels),
            len(light_levels),
        ),
        dtype=float,
    )

    for ph_index, ph_value in enumerate(ph_levels):
        for phone_index, phone_value in enumerate(
            phone_levels
        ):
            for light_index, light_value in enumerate(
                light_levels
            ):
                value = indexed.loc[
                    (ph_value, phone_value, light_value),
                    feature,
                ]

                if isinstance(value, pd.Series):
                    raise ValueError(
                        "Duplicate factorial cell encountered while "
                        f"constructing {feature}."
                    )

                values[
                    ph_index,
                    phone_index,
                    light_index,
                ] = float(value)

    return values, ph_levels, phone_levels, light_levels


def balanced_anova_components(
    values: np.ndarray,
    negative_tolerance: float,
) -> dict[str, dict[str, float]]:
    """
    Direct orthogonal ANOVA decomposition for a complete A x B x C design.

    Axis 0 = pH
    Axis 1 = phone
    Axis 2 = light
    """
    if values.ndim != 3:
        raise ValueError("Expected a three-dimensional factorial array.")

    a, b, c = values.shape
    grand = float(values.mean())

    mean_a = values.mean(axis=(1, 2))
    mean_b = values.mean(axis=(0, 2))
    mean_c = values.mean(axis=(0, 1))

    mean_ab = values.mean(axis=2)
    mean_ac = values.mean(axis=1)
    mean_bc = values.mean(axis=0)

    ss_total = float(
        np.sum(np.square(values - grand))
    )

    ss_a = float(
        b * c
        * np.sum(np.square(mean_a - grand))
    )
    ss_b = float(
        a * c
        * np.sum(np.square(mean_b - grand))
    )
    ss_c = float(
        a * b
        * np.sum(np.square(mean_c - grand))
    )

    ab_effect = (
        mean_ab
        - mean_a[:, None]
        - mean_b[None, :]
        + grand
    )
    ac_effect = (
        mean_ac
        - mean_a[:, None]
        - mean_c[None, :]
        + grand
    )
    bc_effect = (
        mean_bc
        - mean_b[:, None]
        - mean_c[None, :]
        + grand
    )

    ss_ab = float(c * np.sum(np.square(ab_effect)))
    ss_ac = float(b * np.sum(np.square(ac_effect)))
    ss_bc = float(a * np.sum(np.square(bc_effect)))

    explained_ss = (
        ss_a
        + ss_b
        + ss_c
        + ss_ab
        + ss_ac
        + ss_bc
    )
    ss_residual = float(ss_total - explained_ss)

    scale = max(1.0, abs(ss_total))
    tolerance = negative_tolerance * scale

    if ss_residual < -tolerance:
        raise ValueError(
            "Residual sum of squares is materially negative. "
            f"SST={ss_total}, explained={explained_ss}, "
            f"residual={ss_residual}."
        )

    if ss_residual < 0.0:
        ss_residual = 0.0

    df = {
        "pH": a - 1,
        "phone": b - 1,
        "light": c - 1,
        "pH_x_phone": (a - 1) * (b - 1),
        "pH_x_light": (a - 1) * (c - 1),
        "phone_x_light": (b - 1) * (c - 1),
        "residual": (a - 1) * (b - 1) * (c - 1),
    }

    ss = {
        "pH": ss_a,
        "phone": ss_b,
        "light": ss_c,
        "pH_x_phone": ss_ab,
        "pH_x_light": ss_ac,
        "phone_x_light": ss_bc,
        "residual": ss_residual,
    }

    residual_df = df["residual"]
    residual_ms = (
        ss_residual / residual_df
        if residual_df > 0
        else float("nan")
    )

    output: dict[str, dict[str, float]] = {}

    for term in TERM_ORDER:
        term_df = int(df[term])
        term_ss = float(ss[term])
        term_ms = (
            term_ss / term_df
            if term_df > 0
            else float("nan")
        )

        if term == "residual":
            f_value = float("nan")
            p_value = float("nan")
            partial_eta_sq = float("nan")
        elif residual_ms > 0.0:
            f_value = float(term_ms / residual_ms)
            p_value = float(
                f_distribution.sf(
                    f_value,
                    term_df,
                    residual_df,
                )
            )
            partial_eta_sq = float(
                term_ss / (term_ss + ss_residual)
            )
        else:
            f_value = float("inf") if term_ms > 0 else float("nan")
            p_value = 0.0 if term_ms > 0 else float("nan")
            partial_eta_sq = (
                1.0 if term_ss > 0 else float("nan")
            )

        eta_sq = (
            float(term_ss / ss_total)
            if ss_total > 0.0
            else float("nan")
        )

        output[term] = {
            "sum_sq": term_ss,
            "df": term_df,
            "mean_sq": term_ms,
            "F": f_value,
            "p_value": p_value,
            "eta_sq": eta_sq,
            "partial_eta_sq": partial_eta_sq,
        }

    output["_model"] = {
        "grand_mean": grand,
        "corrected_total_ss": ss_total,
        "explained_ss": explained_ss,
        "residual_ss": ss_residual,
        "n": int(a * b * c),
        "model_df": int(
            sum(
                df[term]
                for term in TERM_ORDER
                if term != "residual"
            )
        ),
        "residual_df": int(residual_df),
    }
    return output


def analyze_feature(
    frame: pd.DataFrame,
    feature: str,
    negative_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    (
        values,
        ph_levels,
        phone_levels,
        light_levels,
    ) = factorial_array(frame, feature)

    analysis = balanced_anova_components(
        values,
        negative_tolerance,
    )
    model_info = analysis.pop("_model")

    rows: list[dict[str, Any]] = []

    for term in TERM_ORDER:
        term_result = analysis[term]
        rows.append({
            "feature": feature,
            "term": term,
            "sum_sq": term_result["sum_sq"],
            "df": term_result["df"],
            "mean_sq": term_result["mean_sq"],
            "F": term_result["F"],
            "p_value": term_result["p_value"],
            "significance": significance_label(
                term_result["p_value"]
            ),
            "eta_sq": term_result["eta_sq"],
            "partial_eta_sq": (
                term_result["partial_eta_sq"]
            ),
        })

    residual_df = int(model_info["residual_df"])
    residual_ss = float(model_info["residual_ss"])
    total_ss = float(model_info["corrected_total_ss"])
    n = int(model_info["n"])
    model_df = int(model_info["model_df"])

    r_squared = (
        1.0 - residual_ss / total_ss
        if total_ss > 0.0
        else float("nan")
    )
    adjusted_r_squared = (
        1.0
        - (1.0 - r_squared)
        * (n - 1)
        / residual_df
        if residual_df > 0 and np.isfinite(r_squared)
        else float("nan")
    )
    rmse = (
        math.sqrt(residual_ss / residual_df)
        if residual_df > 0
        else float("nan")
    )

    model_row = {
        "feature": feature,
        "n": n,
        "ph_levels": len(ph_levels),
        "phone_levels": len(phone_levels),
        "light_levels": len(light_levels),
        "model_df": model_df,
        "residual_df": residual_df,
        "grand_mean": model_info["grand_mean"],
        "corrected_total_ss": total_ss,
        "explained_ss": model_info["explained_ss"],
        "residual_ss": residual_ss,
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted_r_squared,
        "rmse": rmse,
    }

    return pd.DataFrame(rows), model_row


def value_for_term(
    table: pd.DataFrame,
    term: str,
    column: str,
) -> float:
    matched = table.loc[
        table["term"] == term,
        column,
    ]

    if matched.empty:
        return float("nan")

    return float(matched.iloc[0])


def build_feature_summaries(
    anova_long: pd.DataFrame,
    model_fit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []

    effect_terms = [
        "pH",
        "phone",
        "light",
        "pH_x_phone",
        "pH_x_light",
        "phone_x_light",
    ]

    for feature, group in anova_long.groupby(
        "feature",
        sort=True,
    ):
        effects = group.loc[
            group["term"].isin(effect_terms)
        ].copy()

        ranked = effects.sort_values(
            "partial_eta_sq",
            ascending=False,
        )

        model_row = model_fit.loc[
            model_fit["feature"] == feature
        ].iloc[0]

        eta = {
            term: value_for_term(
                group,
                term,
                "eta_sq",
            )
            for term in TERM_ORDER
        }
        partial = {
            term: value_for_term(
                group,
                term,
                "partial_eta_sq",
            )
            for term in effect_terms
        }
        p_values = {
            term: value_for_term(
                group,
                term,
                "p_value",
            )
            for term in effect_terms
        }

        pH_signal_ratio = eta["pH"]
        nuisance_main_ratio = (
            eta["phone"]
            + eta["light"]
            + eta["phone_x_light"]
        )
        ph_dependent_interaction_ratio = (
            eta["pH_x_phone"]
            + eta["pH_x_light"]
        )
        residual_ratio = eta["residual"]

        summary_rows.append({
            "feature": feature,
            "top_effect": str(ranked.iloc[0]["term"]),
            "top_partial_eta_sq": float(
                ranked.iloc[0]["partial_eta_sq"]
            ),
            "pH_eta_sq": eta["pH"],
            "phone_eta_sq": eta["phone"],
            "light_eta_sq": eta["light"],
            "pH_x_phone_eta_sq": eta["pH_x_phone"],
            "pH_x_light_eta_sq": eta["pH_x_light"],
            "phone_x_light_eta_sq": eta["phone_x_light"],
            "residual_eta_sq": eta["residual"],
            "pH_partial_eta_sq": partial["pH"],
            "phone_partial_eta_sq": partial["phone"],
            "light_partial_eta_sq": partial["light"],
            "pH_x_phone_partial_eta_sq": (
                partial["pH_x_phone"]
            ),
            "pH_x_light_partial_eta_sq": (
                partial["pH_x_light"]
            ),
            "phone_x_light_partial_eta_sq": (
                partial["phone_x_light"]
            ),
            "pH_p_value": p_values["pH"],
            "phone_p_value": p_values["phone"],
            "light_p_value": p_values["light"],
            "pH_x_phone_p_value": p_values["pH_x_phone"],
            "pH_x_light_p_value": p_values["pH_x_light"],
            "phone_x_light_p_value": (
                p_values["phone_x_light"]
            ),
            "r_squared": float(model_row["r_squared"]),
            "adjusted_r_squared": float(
                model_row["adjusted_r_squared"]
            ),
            "rmse": float(model_row["rmse"]),
        })

        composition_rows.append({
            "feature": feature,
            "pH_signal_ratio": pH_signal_ratio,
            "nuisance_main_ratio": nuisance_main_ratio,
            "pH_dependent_interaction_ratio": (
                ph_dependent_interaction_ratio
            ),
            "residual_ratio": residual_ratio,
            "explained_ratio_sum": (
                pH_signal_ratio
                + nuisance_main_ratio
                + ph_dependent_interaction_ratio
            ),
            "all_ratio_sum": (
                pH_signal_ratio
                + nuisance_main_ratio
                + ph_dependent_interaction_ratio
                + residual_ratio
            ),
        })

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(composition_rows),
    )


def feature_region(feature: str) -> str:
    if feature.startswith("pad_"):
        return "pad"
    if feature.startswith("bg_"):
        return "bg"
    if feature.startswith("contrast_"):
        return "contrast"

    return "other"


def build_region_summary(
    composition: pd.DataFrame,
) -> pd.DataFrame:
    working = composition.copy()
    working["region"] = working["feature"].map(
        feature_region
    )

    return (
        working.groupby(
            "region",
            as_index=False,
        )
        .agg(
            n_features=("feature", "size"),
            mean_pH_signal_ratio=(
                "pH_signal_ratio",
                "mean",
            ),
            mean_nuisance_main_ratio=(
                "nuisance_main_ratio",
                "mean",
            ),
            mean_pH_dependent_interaction_ratio=(
                "pH_dependent_interaction_ratio",
                "mean",
            ),
            mean_residual_ratio=(
                "residual_ratio",
                "mean",
            ),
            min_pH_signal_ratio=(
                "pH_signal_ratio",
                "min",
            ),
            max_pH_signal_ratio=(
                "pH_signal_ratio",
                "max",
            ),
        )
    )


def main() -> None:
    args = parse_args()

    if args.negative_tolerance < 0:
        raise ValueError(
            "--negative-tolerance must be nonnegative."
        )

    input_path = resolve_existing(args.input)
    output_dir = resolve_output(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = prepare_data(
        pd.read_csv(input_path),
        args.features,
    )

    anova_frames: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []

    for feature in args.features:
        anova, model_row = analyze_feature(
            frame,
            feature,
            args.negative_tolerance,
        )
        anova_frames.append(anova)
        model_rows.append(model_row)

    anova_long = pd.concat(
        anova_frames,
        ignore_index=True,
    )
    model_fit = pd.DataFrame(model_rows)

    (
        feature_summary,
        composition,
    ) = build_feature_summaries(
        anova_long,
        model_fit,
    )
    region_summary = build_region_summary(composition)

    paper_table = feature_summary[
        [
            "feature",
            "pH_eta_sq",
            "phone_eta_sq",
            "light_eta_sq",
            "pH_x_phone_eta_sq",
            "pH_x_light_eta_sq",
            "phone_x_light_eta_sq",
            "residual_eta_sq",
            "pH_x_phone_partial_eta_sq",
            "pH_x_light_partial_eta_sq",
            "r_squared",
        ]
    ].copy()

    numeric_columns = [
        column
        for column in paper_table.columns
        if column != "feature"
    ]
    paper_table[numeric_columns] = (
        paper_table[numeric_columns].round(4)
    )

    anova_long.to_csv(
        output_dir / "anova_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    model_fit.to_csv(
        output_dir / "model_fit_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_summary.to_csv(
        output_dir / "feature_effect_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    composition.to_csv(
        output_dir / "effect_composition.csv",
        index=False,
        encoding="utf-8-sig",
    )
    region_summary.to_csv(
        output_dir / "region_effect_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paper_table.to_csv(
        output_dir / "paper_effect_size_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit = {
        "implementation": (
            "direct balanced factorial ANOVA; no statsmodels"
        ),
        "input": str(input_path),
        "output": str(output_dir),
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
        "features": args.features,
        "model": (
            "Y = pH + phone + light + pH:phone + "
            "pH:light + phone:light + residual"
        ),
        "residual_interpretation": (
            "pH x phone x light interaction plus "
            "measurement/extraction error"
        ),
        "expected_degrees_of_freedom": {
            "pH": 12,
            "phone": 2,
            "light": 4,
            "pH_x_phone": 24,
            "pH_x_light": 48,
            "phone_x_light": 8,
            "residual": 96,
            "corrected_total": 194,
        },
        "limitations": [
            (
                "Each phone x pH cell contains one physical strip. "
                "pH x phone variation cannot be separated from "
                "strip-specific cell variation."
            ),
            (
                "The three-way interaction and pure error are not "
                "separately estimable."
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
        "Balanced factorial effect analysis",
        "=" * 80,
        "",
        "This implementation does not use statsmodels.",
        (
            "Sums of squares are calculated directly from balanced "
            "factorial marginal means."
        ),
        "",
        "Effect ratios",
        "-" * 80,
        "pH signal ratio = SS_pH / corrected total SS",
        (
            "nuisance-main ratio = "
            "(SS_phone + SS_light + SS_phone:light) / total SS"
        ),
        (
            "pH-dependent interaction ratio = "
            "(SS_pH:phone + SS_pH:light) / total SS"
        ),
        "",
        "Region-level summary",
        "-" * 80,
        region_summary.to_string(index=False),
    ]

    (output_dir / "analysis_notes.txt").write_text(
        "\n".join(notes),
        encoding="utf-8",
    )

    print("=" * 92)
    print("14. Balanced factorial effect analysis")
    print("=" * 92)
    print("Implementation: direct sums of squares; no statsmodels")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Rows: {len(frame)}")
    print(f"Features analyzed: {len(args.features)}")
    print("\nRegion effect summary")
    print(region_summary.to_string(index=False))
    print("\nMain files")
    print(f"  {output_dir / 'anova_long.csv'}")
    print(
        f"  {output_dir / 'feature_effect_summary.csv'}"
    )
    print(f"  {output_dir / 'effect_composition.csv'}")
    print(f"  {output_dir / 'region_effect_summary.csv'}")
    print(
        f"  {output_dir / 'paper_effect_size_table.csv'}"
    )


if __name__ == "__main__":
    main()
