#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
25_asca_permutation_inference.py

Permutation-based inference for the ASCA decomposition of script 15.

Script 15 reports effect shares and loadings only. This script adds:

1. Permutation p-values for every ASCA term
2. Holm-adjusted p-values within each region x scaling
3. Residual-bootstrap intervals for PC1 loadings of pH and pH x phone
4. The angle between the pH PC1 and pH x phone PC1 directions

This script does not use statsmodels or scipy.

Design and experimental units
-----------------------------
- pH: 13 levels, phone: 3 levels, light: 5 levels
- one image per pH x phone x light cell (195 rows)
- one physical strip per phone x pH cell (39 strips)
- every strip is photographed under all five lights

Strip is the whole-plot unit and light is the within-strip unit.
pH, phone and pH x phone are between-strip terms.
light, pH x light and phone x light are within-strip terms.

Test statistic
--------------
Multivariate pseudo-F on the SSCP trace:

    F_term = (tr(SSCP_term) / df_term) / (tr(SSCP_residual) / df_residual)

df: pH 12, phone 2, light 4, pH x phone 24, pH x light 48,
    phone x light 8, residual 96.

Permutation schemes
-------------------
restricted (primary, respects the strip structure):
    pH     : permute whole strips across pH levels within each phone
    phone  : permute whole strips across phones within each pH level
    light  : permute light labels within each strip
    two-way interactions:
             Freedman-Lane permutation of reduced-model residuals
             across all 195 rows

unrestricted (secondary, the common ASCA convention):
    main effects: permute row labels across all 195 rows
    two-way interactions: identical to the restricted scheme

Interpretation limit
--------------------
With one strip per phone x pH cell, the pH x phone test compares
strip-level deviations from additivity with within-strip (across-light)
residual variation. A significant result therefore shows that the
pH x phone pattern exceeds image-level noise. It cannot separate
smartphone-specific response from strip-to-strip variation.

Bootstrap intervals
-------------------
Residual bootstrap conditional on the observed strips:
    Y* = fitted two-way model + resampled residual rows
PC1 loadings are sign-aligned to the observed PC1 before summarizing.
These intervals reflect image-level noise only, not strip sampling.

Outputs
-------
outputs/asca_permutation/
├─ permutation_tests.csv
├─ loading_bootstrap.csv
├─ direction_angle_bootstrap.csv
├─ null_distribution_summary.csv
├─ analysis_audit.json
├─ analysis_notes.txt
└─ figures/
   ├─ restricted_vs_unrestricted_main_effects.png
   └─ pc1_loading_intervals.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGION_COLUMNS = {
    "pad": ["pad_L", "pad_a", "pad_b"],
    "bg": ["bg_L", "bg_a", "bg_b"],
    "contrast": ["contrast_L", "contrast_a", "contrast_b"],
}

CHANNEL_NAMES = ["L", "a", "b"]

TERM_ORDER = [
    "pH",
    "phone",
    "light",
    "pH_x_phone",
    "pH_x_light",
    "phone_x_light",
]

MAIN_TERMS = ["pH", "phone", "light"]
INTERACTION_TERMS = ["pH_x_phone", "pH_x_light", "phone_x_light"]

TERM_UNIT = {
    "pH": "between-strip",
    "phone": "between-strip",
    "pH_x_phone": "between-strip",
    "light": "within-strip",
    "pH_x_light": "within-strip",
    "phone_x_light": "within-strip",
}

LOADING_TERMS = ["pH", "pH_x_phone"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Permutation inference for the balanced ASCA design."
    )
    parser.add_argument(
        "--input",
        default="outputs/factorial_analysis/analysis_factorial_table.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/asca_permutation",
    )
    parser.add_argument(
        "--scalings",
        nargs="*",
        choices=["raw", "standardized"],
        default=["raw", "standardized"],
    )
    parser.add_argument("--n-permutations", type=int, default=9999)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument(
        "--reference-shares",
        default="outputs/multivariate_lab/multivariate_effect_shares.csv",
        help="Script 15 output used to verify the decomposition.",
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

    raise FileNotFoundError(f"Input not found: {path_text}")


def resolve_output(path_text: str) -> Path:
    path = Path(path_text)

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def prepare_data(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["phone", "light", "ph_nominal"] + [
        column
        for columns in REGION_COLUMNS.values()
        for column in columns
    ]
    missing = sorted(set(required) - set(frame.columns))

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    result = frame.copy()
    result["phone"] = result["phone"].astype(str)
    result["light"] = result["light"].astype(str)
    result["ph_nominal"] = pd.to_numeric(
        result["ph_nominal"], errors="raise"
    ).astype(float)

    if result.duplicated(["ph_nominal", "phone", "light"]).any():
        raise ValueError("Duplicate pH x phone x light cells were found.")

    expected = (
        result["ph_nominal"].nunique()
        * result["phone"].nunique()
        * result["light"].nunique()
    )
    if len(result) != expected:
        raise ValueError(
            f"Incomplete factorial design: rows={len(result)}, "
            f"expected={expected}."
        )

    return result


def response_array(
    frame: pd.DataFrame,
    region: str,
) -> tuple[np.ndarray, list[float], list[str], list[str]]:
    ph_levels = sorted(frame["ph_nominal"].unique().tolist())
    phone_levels = sorted(frame["phone"].unique().tolist())
    light_levels = sorted(frame["light"].unique().tolist())

    indexed = frame.set_index(["ph_nominal", "phone", "light"])
    columns = REGION_COLUMNS[region]

    values = np.empty(
        (len(ph_levels), len(phone_levels), len(light_levels), len(columns)),
        dtype=float,
    )

    for i, ph_value in enumerate(ph_levels):
        for j, phone_value in enumerate(phone_levels):
            for k, light_value in enumerate(light_levels):
                values[i, j, k, :] = indexed.loc[
                    (ph_value, phone_value, light_value), columns
                ].to_numpy(dtype=float)

    return values, ph_levels, phone_levels, light_levels


def scale_response(values: np.ndarray, scaling: str) -> np.ndarray:
    flat = values.reshape(-1, values.shape[-1])

    if scaling == "raw":
        return values.copy()

    if scaling == "standardized":
        center = flat.mean(axis=0)
        scale = flat.std(axis=0, ddof=1)

        if np.any(scale <= 0.0):
            raise ValueError("Cannot standardize a zero-variance channel.")

        return (values - center) / scale

    raise ValueError(f"Unknown scaling: {scaling}")


def effect_arrays(values: np.ndarray) -> dict[str, np.ndarray]:
    """
    Balanced three-factor decomposition, identical to script 15.
    Returns effects broadcast to the full (pH, phone, light, p) shape.
    """
    grand = values.mean(axis=(0, 1, 2))

    mean_a = values.mean(axis=(1, 2))
    mean_b = values.mean(axis=(0, 2))
    mean_c = values.mean(axis=(0, 1))

    mean_ab = values.mean(axis=2)
    mean_ac = values.mean(axis=1)
    mean_bc = values.mean(axis=0)

    effect_a = mean_a - grand
    effect_b = mean_b - grand
    effect_c = mean_c - grand

    effect_ab = mean_ab - mean_a[:, None, :] - mean_b[None, :, :] + grand
    effect_ac = mean_ac - mean_a[:, None, :] - mean_c[None, :, :] + grand
    effect_bc = mean_bc - mean_b[:, None, :] - mean_c[None, :, :] + grand

    shape = values.shape
    effects = {
        "grand": np.broadcast_to(grand, shape),
        "pH": np.broadcast_to(effect_a[:, None, None, :], shape),
        "phone": np.broadcast_to(effect_b[None, :, None, :], shape),
        "light": np.broadcast_to(effect_c[None, None, :, :], shape),
        "pH_x_phone": np.broadcast_to(effect_ab[:, :, None, :], shape),
        "pH_x_light": np.broadcast_to(effect_ac[:, None, :, :], shape),
        "phone_x_light": np.broadcast_to(effect_bc[None, :, :, :], shape),
    }

    fitted = sum(effects.values())
    effects["residual"] = values - fitted

    return effects


def term_degrees_of_freedom(a: int, b: int, c: int) -> dict[str, int]:
    return {
        "pH": a - 1,
        "phone": b - 1,
        "light": c - 1,
        "pH_x_phone": (a - 1) * (b - 1),
        "pH_x_light": (a - 1) * (c - 1),
        "phone_x_light": (b - 1) * (c - 1),
        "residual": (a - 1) * (b - 1) * (c - 1),
    }


def trace_ss(matrix: np.ndarray) -> float:
    return float(np.sum(np.square(matrix)))


def pseudo_f(
    values: np.ndarray,
    term: str,
    df: dict[str, int],
) -> float:
    effects = effect_arrays(values)
    numerator = trace_ss(effects[term]) / df[term]
    denominator = trace_ss(effects["residual"]) / df["residual"]

    return numerator / denominator


def permute_restricted(
    values: np.ndarray,
    term: str,
    rng: np.random.Generator,
) -> np.ndarray:
    a, b, c, _ = values.shape
    result = np.empty_like(values)

    if term == "pH":
        # Whole strips move across pH levels within each phone.
        for j in range(b):
            result[:, j] = values[rng.permutation(a), j]
    elif term == "phone":
        # Whole strips move across phones within each pH level.
        for i in range(a):
            result[i] = values[i, rng.permutation(b)]
    elif term == "light":
        # Light labels are permuted within each strip.
        for i in range(a):
            for j in range(b):
                result[i, j] = values[i, j, rng.permutation(c)]
    else:
        raise ValueError(f"No restricted main-effect scheme for {term}.")

    return result


def permute_unrestricted(
    values: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    flat = values.reshape(-1, values.shape[-1])

    return flat[rng.permutation(flat.shape[0])].reshape(values.shape)


def freedman_lane_sampler(
    values: np.ndarray,
    term: str,
) -> Callable[[np.random.Generator], np.ndarray]:
    """
    In a balanced orthogonal design the reduced model (all terms except
    `term`) has fitted values = full fitted - effect(term), so the reduced
    residual is effect(term) + full residual.
    """
    effects = effect_arrays(values)
    reduced_residual = effects[term] + effects["residual"]
    reduced_fitted = values - reduced_residual
    flat_residual = reduced_residual.reshape(-1, values.shape[-1])

    def sample(rng: np.random.Generator) -> np.ndarray:
        permuted = flat_residual[rng.permutation(flat_residual.shape[0])]

        return reduced_fitted + permuted.reshape(values.shape)

    return sample


def permutation_test(
    values: np.ndarray,
    term: str,
    scheme: str,
    df: dict[str, int],
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    observed = pseudo_f(values, term, df)

    if term in INTERACTION_TERMS:
        sampler = freedman_lane_sampler(values, term)
    elif scheme == "restricted":
        def sampler(generator: np.random.Generator) -> np.ndarray:
            return permute_restricted(values, term, generator)
    else:
        def sampler(generator: np.random.Generator) -> np.ndarray:
            return permute_unrestricted(values, generator)

    null = np.empty(n_permutations, dtype=float)

    for index in range(n_permutations):
        null[index] = pseudo_f(sampler(rng), term, df)

    return observed, null


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0

    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running

    return adjusted.tolist()


def pc1_loading(effect: np.ndarray) -> tuple[np.ndarray, float]:
    flat = effect.reshape(-1, effect.shape[-1])
    _, singular_values, vt = np.linalg.svd(flat, full_matrices=False)
    variance = np.square(singular_values)
    explained = float(variance[0] / variance.sum()) if variance.sum() > 0 else np.nan

    return vt[0], explained


def angle_degrees(u: np.ndarray, v: np.ndarray) -> float:
    """Angle between two axes; sign-invariant, so 0-90 degrees."""
    cosine = abs(float(np.dot(u, v))) / (
        float(np.linalg.norm(u)) * float(np.linalg.norm(v))
    )

    return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


def align_to_reference(loading: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return loading if float(np.dot(loading, reference)) >= 0.0 else -loading


def residual_bootstrap(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    effects = effect_arrays(values)
    residual = effects["residual"]
    fitted = values - residual
    flat_residual = residual.reshape(-1, values.shape[-1])

    observed = {term: pc1_loading(effects[term]) for term in LOADING_TERMS}
    observed_angle = angle_degrees(
        observed["pH"][0], observed["pH_x_phone"][0]
    )

    loadings = {
        term: np.empty((n_bootstrap, values.shape[-1]))
        for term in LOADING_TERMS
    }
    explained = {term: np.empty(n_bootstrap) for term in LOADING_TERMS}
    angles = np.empty(n_bootstrap)

    for index in range(n_bootstrap):
        draw = rng.integers(0, flat_residual.shape[0], flat_residual.shape[0])
        boot_values = fitted + flat_residual[draw].reshape(values.shape)
        boot_effects = effect_arrays(boot_values)

        for term in LOADING_TERMS:
            loading, ratio = pc1_loading(boot_effects[term])
            loadings[term][index] = align_to_reference(
                loading, observed[term][0]
            )
            explained[term][index] = ratio

        angles[index] = angle_degrees(
            loadings["pH"][index], loadings["pH_x_phone"][index]
        )

    return {
        "observed": observed,
        "observed_angle": observed_angle,
        "loadings": loadings,
        "explained": explained,
        "angles": angles,
    }


def verify_against_reference(
    observed_shares: pd.DataFrame,
    reference_path: Path | None,
) -> dict[str, Any]:
    if reference_path is None:
        return {"checked": False, "reason": "reference file not found"}

    reference = pd.read_csv(reference_path, encoding="utf-8-sig")
    merged = observed_shares.merge(
        reference[["region", "scaling", "term", "trace_share"]],
        on=["region", "scaling", "term"],
        suffixes=("", "_reference"),
    )
    max_difference = float(
        (merged["trace_share"] - merged["trace_share_reference"]).abs().max()
    )

    if max_difference > 1e-8:
        raise ValueError(
            "Decomposition does not match script 15 output: "
            f"max share difference = {max_difference}."
        )

    return {
        "checked": True,
        "rows_compared": int(len(merged)),
        "max_share_difference": max_difference,
    }


def main() -> None:
    args = parse_args()

    if args.n_permutations < 99:
        raise ValueError("--n-permutations must be at least 99.")
    if args.n_bootstrap < 99:
        raise ValueError("--n-bootstrap must be at least 99.")

    input_path = resolve_existing(args.input)
    output_dir = resolve_output(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        reference_path: Path | None = resolve_existing(args.reference_shares)
    except FileNotFoundError:
        reference_path = None

    frame = prepare_data(pd.read_csv(input_path, encoding="utf-8-sig"))
    rng = np.random.default_rng(args.seed)

    test_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    share_rows: list[dict[str, Any]] = []
    loading_rows: list[dict[str, Any]] = []
    angle_rows: list[dict[str, Any]] = []

    for region in REGION_COLUMNS:
        raw_values, ph_levels, phone_levels, light_levels = response_array(
            frame, region
        )
        df = term_degrees_of_freedom(*raw_values.shape[:3])

        for scaling in args.scalings:
            values = scale_response(raw_values, scaling)
            effects = effect_arrays(values)
            centered = values - effects["grand"]
            total = trace_ss(centered)

            for term in TERM_ORDER + ["residual"]:
                share_rows.append({
                    "region": region,
                    "scaling": scaling,
                    "term": term,
                    "trace_share": trace_ss(effects[term]) / total,
                })

            for scheme in ["restricted", "unrestricted"]:
                scheme_rows: list[dict[str, Any]] = []

                for term in TERM_ORDER:
                    if scheme == "unrestricted" and term in INTERACTION_TERMS:
                        # Identical to restricted; reported once.
                        continue

                    observed, null = permutation_test(
                        values,
                        term,
                        scheme,
                        df,
                        args.n_permutations,
                        rng,
                    )
                    exceed = int(np.sum(null >= observed - 1e-12))
                    p_value = (exceed + 1) / (args.n_permutations + 1)

                    scheme_rows.append({
                        "region": region,
                        "scaling": scaling,
                        "scheme": (
                            "freedman_lane"
                            if term in INTERACTION_TERMS
                            else scheme
                        ),
                        "term": term,
                        "unit": TERM_UNIT[term],
                        "df_term": df[term],
                        "df_residual": df["residual"],
                        "trace_share": trace_ss(effects[term]) / total,
                        "pseudo_f": observed,
                        "null_f_mean": float(null.mean()),
                        "null_f_q95": float(np.quantile(null, 0.95)),
                        "null_f_q99": float(np.quantile(null, 0.99)),
                        "null_f_max": float(null.max()),
                        "n_exceed": exceed,
                        "n_permutations": args.n_permutations,
                        "p_value": p_value,
                    })

                    null_rows.append({
                        "region": region,
                        "scaling": scaling,
                        "scheme": scheme_rows[-1]["scheme"],
                        "term": term,
                        "q01": float(np.quantile(null, 0.01)),
                        "q25": float(np.quantile(null, 0.25)),
                        "median": float(np.median(null)),
                        "q75": float(np.quantile(null, 0.75)),
                        "q95": float(np.quantile(null, 0.95)),
                        "q99": float(np.quantile(null, 0.99)),
                    })

                if scheme == "restricted":
                    adjusted = holm_adjust(
                        [row["p_value"] for row in scheme_rows]
                    )
                    for row, value in zip(scheme_rows, adjusted):
                        row["p_holm_within_region"] = value
                else:
                    for row in scheme_rows:
                        row["p_holm_within_region"] = np.nan

                test_rows.extend(scheme_rows)

            boot = residual_bootstrap(values, args.n_bootstrap, rng)

            for term in LOADING_TERMS:
                observed_loading, observed_ratio = boot["observed"][term]

                for channel_index, channel in enumerate(CHANNEL_NAMES):
                    samples = boot["loadings"][term][:, channel_index]
                    loading_rows.append({
                        "region": region,
                        "scaling": scaling,
                        "term": term,
                        "component": 1,
                        "channel": channel,
                        "loading": float(observed_loading[channel_index]),
                        "boot_mean": float(samples.mean()),
                        "boot_sd": float(samples.std(ddof=1)),
                        "ci_lower_95": float(np.quantile(samples, 0.025)),
                        "ci_upper_95": float(np.quantile(samples, 0.975)),
                        "pc1_explained_ratio": observed_ratio,
                        "pc1_explained_ci_lower_95": float(
                            np.quantile(boot["explained"][term], 0.025)
                        ),
                        "pc1_explained_ci_upper_95": float(
                            np.quantile(boot["explained"][term], 0.975)
                        ),
                    })

            angle_rows.append({
                "region": region,
                "scaling": scaling,
                "comparison": "pH PC1 vs pH_x_phone PC1",
                "angle_degrees": boot["observed_angle"],
                "boot_median": float(np.median(boot["angles"])),
                "ci_lower_95": float(np.quantile(boot["angles"], 0.025)),
                "ci_upper_95": float(np.quantile(boot["angles"], 0.975)),
                "n_bootstrap": args.n_bootstrap,
            })

    shares = pd.DataFrame(share_rows)
    verification = verify_against_reference(shares, reference_path)

    tests = pd.DataFrame(test_rows)
    tests.to_csv(
        output_dir / "permutation_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(null_rows).to_csv(
        output_dir / "null_distribution_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    loadings = pd.DataFrame(loading_rows)
    loadings.to_csv(
        output_dir / "loading_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    angles = pd.DataFrame(angle_rows)
    angles.to_csv(
        output_dir / "direction_angle_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit = {
        "input": str(input_path),
        "output": str(output_dir),
        "rows": int(len(frame)),
        "scalings": args.scalings,
        "n_permutations": args.n_permutations,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "statistic": "trace pseudo-F with full three-factor residual (df=96)",
        "schemes": {
            "restricted_pH": "whole strips permuted across pH within phone",
            "restricted_phone": "whole strips permuted across phone within pH",
            "restricted_light": "light labels permuted within strip",
            "unrestricted_main": "row labels permuted across all rows",
            "interactions": "Freedman-Lane, reduced-model residuals permuted across all rows",
        },
        "p_value": "(n_exceed + 1) / (n_permutations + 1)",
        "multiplicity": "Holm within region x scaling over the six terms (restricted scheme)",
        "bootstrap": "residual bootstrap conditional on observed strips; PC1 sign-aligned to observed",
        "verification_against_script15": verification,
        "limitations": [
            "One strip per phone x pH cell: pH x phone is tested against "
            "within-strip residual variation, not against strip-to-strip variation.",
            "The residual combines the three-way interaction with measurement error.",
            "Bootstrap intervals ignore strip sampling variability.",
        ],
    }
    (output_dir / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with pd.option_context(
        "display.width", 200,
        "display.max_columns", 20,
        "display.float_format", "{:.4g}".format,
    ):
        notes = [
            "ASCA permutation inference",
            "=" * 80,
            "",
            f"Permutations: {args.n_permutations}, bootstrap: {args.n_bootstrap}, "
            f"seed: {args.seed}",
            f"Script 15 verification: {verification}",
            "",
            "Permutation tests",
            tests[[
                "region", "scaling", "scheme", "term", "trace_share",
                "pseudo_f", "null_f_q95", "p_value", "p_holm_within_region",
            ]].to_string(index=False),
            "",
            "PC1 loading bootstrap",
            loadings[[
                "region", "scaling", "term", "channel", "loading",
                "ci_lower_95", "ci_upper_95",
            ]].to_string(index=False),
            "",
            "pH vs pH_x_phone PC1 angle",
            angles.to_string(index=False),
            "",
            "Limitations",
            *[f"- {item}" for item in audit["limitations"]],
        ]

    (output_dir / "analysis_notes.txt").write_text(
        "\n".join(notes) + "\n",
        encoding="utf-8",
    )

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_scheme_comparison(tests, figure_dir / "restricted_vs_unrestricted_main_effects.png")
    plot_loading_intervals(loadings, figure_dir / "pc1_loading_intervals.png")

    print("\n".join(notes))


REGION_LABELS = {
    "pad": "Pad",
    "bg": "Background",
    "contrast": "Contrast",
}


def plot_scheme_comparison(tests: pd.DataFrame, path: Path) -> None:
    """Observed pseudo-F vs the 95% null quantile under both schemes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = tests[
        (tests["scaling"] == "standardized") & tests["term"].isin(MAIN_TERMS)
    ]
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=False)

    for axis, region in zip(axes, REGION_COLUMNS):
        rows = subset[subset["region"] == region]
        x = np.arange(len(MAIN_TERMS))

        for offset, scheme, color in [
            (-0.18, "restricted", "#0072B2"),
            (0.18, "unrestricted", "#E69F00"),
        ]:
            part = rows[rows["scheme"] == scheme].set_index("term").loc[MAIN_TERMS]
            axis.bar(
                x + offset, part["null_f_q95"], width=0.34,
                color=color, alpha=0.45, label=f"{scheme} null 95% quantile",
            )
            for xi, (_, row) in zip(x + offset, part.iterrows()):
                axis.text(
                    xi, row["null_f_q95"], f"p={row['p_value']:.4f}",
                    ha="center", va="bottom", fontsize=7, rotation=90,
                )

        observed = rows[rows["scheme"] == "restricted"].set_index("term").loc[MAIN_TERMS]
        axis.scatter(x, observed["pseudo_f"], color="#000000", zorder=3,
                     label="observed pseudo-F", s=22)
        axis.set_yscale("log")
        axis.set_xticks(x)
        axis.set_xticklabels(MAIN_TERMS)
        axis.set_title(REGION_LABELS[region], fontsize=10)
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("pseudo-F (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, frameon=False, fontsize=8,
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04),
    )
    figure.suptitle(
        "Main-effect permutation tests: strip-restricted vs image-level "
        "(standardized Lab)",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_loading_intervals(loadings: pd.DataFrame, path: Path) -> None:
    """PC1 loadings with residual-bootstrap 95% intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scalings = list(dict.fromkeys(loadings["scaling"]))
    figure, axes = plt.subplots(
        len(scalings), 3, figsize=(12.0, 3.3 * len(scalings)), squeeze=False,
    )
    colors = {"pH": "#0072B2", "pH_x_phone": "#D55E00"}

    for row, scaling in enumerate(scalings):
        for column, region in enumerate(REGION_COLUMNS):
            axis = axes[row, column]
            x = np.arange(len(CHANNEL_NAMES))

            for offset, term in [(-0.17, "pH"), (0.17, "pH_x_phone")]:
                part = loadings[
                    (loadings["scaling"] == scaling)
                    & (loadings["region"] == region)
                    & (loadings["term"] == term)
                ].set_index("channel").loc[CHANNEL_NAMES]
                axis.bar(x + offset, part["loading"], width=0.32,
                         color=colors[term], alpha=0.8, label=term)
                axis.errorbar(
                    x + offset, part["loading"],
                    yerr=[
                        part["loading"] - part["ci_lower_95"],
                        part["ci_upper_95"] - part["loading"],
                    ],
                    fmt="none", ecolor="#000000", capsize=3, linewidth=1,
                )

            axis.axhline(0.0, color="#888888", linewidth=0.8)
            axis.set_ylim(-1.05, 1.05)
            axis.set_xticks(x)
            axis.set_xticklabels([f"{c}*" for c in CHANNEL_NAMES])
            axis.set_title(f"{REGION_LABELS[region]} ({scaling})", fontsize=9)
            axis.grid(axis="y", alpha=0.25)

    axes[0, 0].set_ylabel("PC1 loading")
    axes[0, 0].legend(frameon=False, fontsize=7)
    figure.suptitle(
        "ASCA PC1 loadings with residual-bootstrap 95% intervals "
        "(conditional on observed strips)",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
