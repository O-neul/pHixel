#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
15_multivariate_lab_asca.py

Multivariate Lab effect decomposition and ASCA diagnostics for the complete
3-factor smartphone pH colorimetry design.

This script does not use statsmodels.

Design
------
- pH: 13 levels
- phone: 3 levels
- light: 5 levels
- one observation per pH x phone x light cell

Multivariate response
---------------------
For each region:
- Pad Lab        = (pad_L, pad_a, pad_b)
- Background Lab = (bg_L, bg_a, bg_b)
- Contrast Lab   = (contrast_L, contrast_a, contrast_b)

The analysis is run twice:
1. raw Lab scale
2. channel-standardized Lab scale

The standardized analysis is primary when comparing the relative contribution
of L*, a*, and b*, because otherwise channels with larger numeric variance can
dominate the trace of the SSCP matrix.

Methods
-------
1. Direct balanced-factorial effect matrices
2. SSCP matrices for every effect
3. Trace-based multivariate effect shares
4. Descriptive MANOVA statistics from H and residual E:
   - Pillai trace
   - Wilks lambda
   - Hotelling-Lawley trace
   - Roy largest root
   No p-values are reported for these descriptive statistics.
5. ASCA-style SVD of each effect matrix

Residual interpretation
-----------------------
With one observation per three-way cell, the residual contains:
- pH x phone x light interaction
- pure measurement/extraction error

These components cannot be separated.

Outputs
-------
outputs/multivariate_lab/
├─ multivariate_effect_shares.csv
├─ effect_composition.csv
├─ sscp_matrices_long.csv
├─ manova_descriptive_statistics.csv
├─ asca_variance.csv
├─ asca_loadings.csv
├─ asca_scores.csv
├─ analysis_audit.json
├─ analysis_notes.txt
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

TERM_ORDER = [
    "pH",
    "phone",
    "light",
    "pH_x_phone",
    "pH_x_light",
    "phone_x_light",
    "residual",
]

REGION_LABELS = {
    "pad": "Pad",
    "bg": "Background",
    "contrast": "Pad - Background Contrast",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multivariate Lab decomposition and ASCA for the balanced design."
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
        default="outputs/multivariate_lab",
    )
    parser.add_argument(
        "--scalings",
        nargs="*",
        choices=["raw", "standardized"],
        default=["raw", "standardized"],
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--save-pdf", action="store_true")
    parser.add_argument(
        "--negative-tolerance",
        type=float,
        default=1e-8,
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

    return result.sort_values(
        ["ph_nominal", "phone", "light"],
        kind="stable",
    ).reset_index(drop=True)


def response_array(
    frame: pd.DataFrame,
    region: str,
) -> tuple[
    np.ndarray,
    list[float],
    list[str],
    list[str],
    pd.DataFrame,
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

    columns = REGION_COLUMNS[region]
    indexed = frame.set_index(
        ["ph_nominal", "phone", "light"]
    )

    values = np.empty(
        (
            len(ph_levels),
            len(phone_levels),
            len(light_levels),
            len(columns),
        ),
        dtype=float,
    )
    observation_rows: list[dict[str, Any]] = []

    for ph_index, ph_value in enumerate(ph_levels):
        for phone_index, phone_value in enumerate(
            phone_levels
        ):
            for light_index, light_value in enumerate(
                light_levels
            ):
                row = indexed.loc[
                    (ph_value, phone_value, light_value)
                ]

                if isinstance(row, pd.DataFrame):
                    raise ValueError(
                        "Duplicate cell encountered while creating "
                        f"{region} response array."
                    )

                values[
                    ph_index,
                    phone_index,
                    light_index,
                    :,
                ] = row[columns].to_numpy(dtype=float)

                observation_rows.append({
                    "flat_index": len(observation_rows),
                    "image_id": str(row["image_id"]),
                    "strip_id": str(row["strip_id"]),
                    "ph_nominal": float(ph_value),
                    "phone": str(phone_value),
                    "light": str(light_value),
                })

    return (
        values,
        ph_levels,
        phone_levels,
        light_levels,
        pd.DataFrame(observation_rows),
    )


def scale_response(
    values: np.ndarray,
    scaling: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = values.reshape(-1, values.shape[-1])
    center = flat.mean(axis=0)

    if scaling == "raw":
        scale = np.ones(flat.shape[1], dtype=float)
        transformed = values.copy()
    elif scaling == "standardized":
        scale = flat.std(axis=0, ddof=1)

        if np.any(scale <= 0.0):
            raise ValueError(
                "Cannot standardize a Lab channel with zero variance."
            )

        transformed = (
            values - center.reshape(1, 1, 1, -1)
        ) / scale.reshape(1, 1, 1, -1)
    else:
        raise ValueError(f"Unknown scaling: {scaling}")

    return transformed, center, scale


def balanced_effect_matrices(
    values: np.ndarray,
    negative_tolerance: float,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    """
    Return expanded observation-level effect matrices for a balanced
    pH x phone x light x response array.
    """
    if values.ndim != 4:
        raise ValueError(
            "Expected a four-dimensional response array."
        )

    a, b, c, p = values.shape
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

    effect_ab = (
        mean_ab
        - mean_a[:, None, :]
        - mean_b[None, :, :]
        + grand
    )
    effect_ac = (
        mean_ac
        - mean_a[:, None, :]
        - mean_c[None, :, :]
        + grand
    )
    effect_bc = (
        mean_bc
        - mean_b[:, None, :]
        - mean_c[None, :, :]
        + grand
    )

    expanded = {
        "pH": np.broadcast_to(
            effect_a[:, None, None, :],
            values.shape,
        ),
        "phone": np.broadcast_to(
            effect_b[None, :, None, :],
            values.shape,
        ),
        "light": np.broadcast_to(
            effect_c[None, None, :, :],
            values.shape,
        ),
        "pH_x_phone": np.broadcast_to(
            effect_ab[:, :, None, :],
            values.shape,
        ),
        "pH_x_light": np.broadcast_to(
            effect_ac[:, None, :, :],
            values.shape,
        ),
        "phone_x_light": np.broadcast_to(
            effect_bc[None, :, :, :],
            values.shape,
        ),
    }

    fitted = np.broadcast_to(
        grand.reshape(1, 1, 1, p),
        values.shape,
    ).copy()

    for matrix in expanded.values():
        fitted += matrix

    residual = values - fitted
    expanded["residual"] = residual

    centered = (
        values
        - grand.reshape(1, 1, 1, p)
    )

    total_ss = float(
        np.sum(np.square(centered))
    )
    component_ss = float(
        sum(
            np.sum(np.square(matrix))
            for matrix in expanded.values()
        )
    )

    tolerance = (
        negative_tolerance * max(1.0, abs(total_ss))
    )

    if abs(total_ss - component_ss) > tolerance:
        raise ValueError(
            "Multivariate effect decomposition did not reconstruct "
            f"the total SS. total={total_ss}, components={component_ss}."
        )

    flattened = {
        term: matrix.reshape(-1, p)
        for term, matrix in expanded.items()
    }

    return (
        flattened,
        centered.reshape(-1, p),
        grand,
    )


def sscp(matrix: np.ndarray) -> np.ndarray:
    return matrix.T @ matrix


def descriptive_manova_statistics(
    h_matrix: np.ndarray,
    e_matrix: np.ndarray,
) -> dict[str, float]:
    p = h_matrix.shape[0]
    ridge = np.finfo(float).eps * max(
        1.0,
        float(np.trace(e_matrix)),
    )

    e_regularized = (
        e_matrix + ridge * np.eye(p, dtype=float)
    )

    eigenvalues = np.linalg.eigvals(
        np.linalg.solve(e_regularized, h_matrix)
    )
    eigenvalues = np.real_if_close(eigenvalues).astype(float)
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    pillai = float(
        np.sum(eigenvalues / (1.0 + eigenvalues))
    )
    wilks = float(
        np.prod(1.0 / (1.0 + eigenvalues))
    )
    hotelling = float(np.sum(eigenvalues))
    roy = float(np.max(eigenvalues))

    return {
        "pillai_trace": pillai,
        "wilks_lambda": wilks,
        "hotelling_lawley_trace": hotelling,
        "roy_largest_root": roy,
        "eigenvalue_1": float(
            eigenvalues[0] if len(eigenvalues) > 0 else np.nan
        ),
        "eigenvalue_2": float(
            eigenvalues[1] if len(eigenvalues) > 1 else np.nan
        ),
        "eigenvalue_3": float(
            eigenvalues[2] if len(eigenvalues) > 2 else np.nan
        ),
    }


def asca_svd(
    effect_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SVD of an observation-level effect matrix.

    Returns
    -------
    scores : U * singular values
    loadings : V
    explained_ratio : squared singular values / term trace
    """
    u, singular_values, vt = np.linalg.svd(
        effect_matrix,
        full_matrices=False,
    )
    scores = u * singular_values.reshape(1, -1)
    loadings = vt.T
    variance = np.square(singular_values)
    total = float(variance.sum())

    if total > 0.0:
        explained_ratio = variance / total
    else:
        explained_ratio = np.zeros_like(variance)

    return scores, loadings, explained_ratio


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


def plot_effect_composition(
    composition: pd.DataFrame,
    scaling: str,
    stem: Path,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    subset = (
        composition.loc[
            composition["scaling"] == scaling
        ]
        .sort_values("region")
        .reset_index(drop=True)
    )

    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    x = np.arange(len(subset), dtype=float)
    bottom = np.zeros(len(subset), dtype=float)

    columns = [
        ("pH_signal_share", "pH signal"),
        ("nuisance_main_share", "Nuisance main"),
        (
            "pH_dependent_interaction_share",
            "pH-dependent interaction",
        ),
        ("residual_share", "Residual"),
    ]

    for column, label in columns:
        values = subset[column].to_numpy(dtype=float)
        axis.bar(
            x,
            values,
            bottom=bottom,
            label=label,
        )
        bottom += values

    axis.set_xticks(x)
    axis.set_xticklabels(
        [
            REGION_LABELS[region]
            for region in subset["region"]
        ]
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Trace-based multivariate variance share")
    axis.set_title(
        f"Multivariate Lab Effect Composition ({scaling})"
    )
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()

    return save_figure(
        figure,
        stem,
        dpi,
        save_pdf,
    )


def plot_term_shares(
    shares: pd.DataFrame,
    scaling: str,
    stem: Path,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    subset = shares.loc[
        shares["scaling"] == scaling
    ].copy()

    terms = TERM_ORDER
    regions = sorted(subset["region"].unique())
    x = np.arange(len(terms), dtype=float)
    width = 0.8 / max(1, len(regions))

    figure, axis = plt.subplots(figsize=(10.0, 5.6))

    for index, region in enumerate(regions):
        region_data = (
            subset.loc[subset["region"] == region]
            .set_index("term")
            .reindex(terms)
        )
        axis.bar(
            x + (index - (len(regions) - 1) / 2) * width,
            region_data["trace_share"].to_numpy(dtype=float),
            width=width,
            label=REGION_LABELS[region],
        )

    axis.set_xticks(x)
    axis.set_xticklabels(terms, rotation=30, ha="right")
    axis.set_ylabel("Trace share")
    axis.set_title(
        f"Multivariate Effect Shares by Term ({scaling})"
    )
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()

    return save_figure(
        figure,
        stem,
        dpi,
        save_pdf,
    )


def plot_ph_asca_score(
    scores: pd.DataFrame,
    region: str,
    scaling: str,
    stem: Path,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    subset = scores.loc[
        (scores["region"] == region)
        & (scores["scaling"] == scaling)
        & (scores["term"] == "pH")
        & (scores["component"] == 1)
    ].copy()

    summary = (
        subset.groupby("ph_nominal", as_index=False)
        ["score"]
        .mean()
        .sort_values("ph_nominal")
    )

    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    axis.plot(
        summary["ph_nominal"].to_numpy(dtype=float),
        summary["score"].to_numpy(dtype=float),
        marker="o",
        linewidth=1.8,
    )
    axis.set_xlabel("Nominal pH")
    axis.set_ylabel("ASCA pH score, component 1")
    axis.set_xticks(
        summary["ph_nominal"].to_numpy(dtype=float)
    )
    axis.set_title(
        f"{REGION_LABELS[region]} pH Effect ASCA Score "
        f"({scaling})"
    )
    axis.grid(True, alpha=0.25)
    figure.tight_layout()

    return save_figure(
        figure,
        stem,
        dpi,
        save_pdf,
    )


def plot_ph_phone_asca_score(
    scores: pd.DataFrame,
    region: str,
    scaling: str,
    stem: Path,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    subset = scores.loc[
        (scores["region"] == region)
        & (scores["scaling"] == scaling)
        & (scores["term"] == "pH_x_phone")
        & (scores["component"] == 1)
    ].copy()

    summary = (
        subset.groupby(
            ["phone", "ph_nominal"],
            as_index=False,
        )["score"]
        .mean()
        .sort_values(["phone", "ph_nominal"])
    )

    figure, axis = plt.subplots(figsize=(8.4, 5.2))

    for phone, group in summary.groupby(
        "phone",
        sort=True,
    ):
        axis.plot(
            group["ph_nominal"].to_numpy(dtype=float),
            group["score"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.7,
            label=str(phone),
        )

    axis.axhline(0.0, linewidth=1.0, alpha=0.5)
    axis.set_xlabel("Nominal pH")
    axis.set_ylabel(
        "ASCA pH x phone score, component 1"
    )
    axis.set_xticks(
        sorted(summary["ph_nominal"].unique())
    )
    axis.set_title(
        f"{REGION_LABELS[region]} pH x Phone ASCA Score "
        f"({scaling})"
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


def plot_loading_bars(
    loadings: pd.DataFrame,
    region: str,
    scaling: str,
    term: str,
    stem: Path,
    dpi: int,
    save_pdf: bool,
) -> list[Path]:
    subset = loadings.loc[
        (loadings["region"] == region)
        & (loadings["scaling"] == scaling)
        & (loadings["term"] == term)
        & (loadings["component"] == 1)
    ].copy()

    subset = subset.set_index("channel").reindex(
        CHANNEL_NAMES
    ).reset_index()

    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    axis.bar(
        subset["channel"],
        subset["loading"].to_numpy(dtype=float),
    )
    axis.axhline(0.0, linewidth=1.0, alpha=0.5)
    axis.set_ylabel("ASCA loading, component 1")
    axis.set_title(
        f"{REGION_LABELS[region]} {term} Loadings "
        f"({scaling})"
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

    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0.")
    if args.negative_tolerance < 0:
        raise ValueError(
            "--negative-tolerance must be nonnegative."
        )

    input_path = resolve_existing(args.input)
    output_dir = resolve_output(args.output_dir)
    figure_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    frame = prepare_data(pd.read_csv(input_path))

    effect_share_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    sscp_rows: list[dict[str, Any]] = []
    manova_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    loading_rows: list[dict[str, Any]] = []
    score_frames: list[pd.DataFrame] = []

    for region in REGION_COLUMNS:
        (
            raw_values,
            ph_levels,
            phone_levels,
            light_levels,
            observation_index,
        ) = response_array(frame, region)

        for scaling in args.scalings:
            (
                values,
                center,
                scale,
            ) = scale_response(raw_values, scaling)

            (
                effect_matrices,
                centered_matrix,
                grand,
            ) = balanced_effect_matrices(
                values,
                args.negative_tolerance,
            )

            total_sscp = sscp(centered_matrix)
            total_trace = float(np.trace(total_sscp))
            residual_sscp = sscp(
                effect_matrices["residual"]
            )

            term_traces: dict[str, float] = {}

            for term in TERM_ORDER:
                effect_matrix = effect_matrices[term]
                term_sscp = sscp(effect_matrix)
                term_trace = float(np.trace(term_sscp))
                term_traces[term] = term_trace

                effect_share_rows.append({
                    "region": region,
                    "scaling": scaling,
                    "term": term,
                    "trace_ss": term_trace,
                    "total_trace_ss": total_trace,
                    "trace_share": (
                        term_trace / total_trace
                        if total_trace > 0.0
                        else np.nan
                    ),
                })

                for row_index, row_channel in enumerate(
                    CHANNEL_NAMES
                ):
                    for column_index, column_channel in enumerate(
                        CHANNEL_NAMES
                    ):
                        sscp_rows.append({
                            "region": region,
                            "scaling": scaling,
                            "term": term,
                            "row_channel": row_channel,
                            "column_channel": column_channel,
                            "value": float(
                                term_sscp[
                                    row_index,
                                    column_index,
                                ]
                            ),
                        })

                if term != "residual":
                    statistics = descriptive_manova_statistics(
                        term_sscp,
                        residual_sscp,
                    )
                    manova_rows.append({
                        "region": region,
                        "scaling": scaling,
                        "term": term,
                        **statistics,
                    })

                (
                    scores,
                    loadings,
                    explained_ratio,
                ) = asca_svd(effect_matrix)

                component_count = min(
                    len(CHANNEL_NAMES),
                    scores.shape[1],
                )

                for component_index in range(component_count):
                    variance_rows.append({
                        "region": region,
                        "scaling": scaling,
                        "term": term,
                        "component": component_index + 1,
                        "explained_ratio_within_term": float(
                            explained_ratio[component_index]
                        ),
                        "singular_value": float(
                            np.linalg.norm(
                                scores[:, component_index]
                            )
                        ),
                    })

                    for channel_index, channel in enumerate(
                        CHANNEL_NAMES
                    ):
                        loading_rows.append({
                            "region": region,
                            "scaling": scaling,
                            "term": term,
                            "component": component_index + 1,
                            "channel": channel,
                            "loading": float(
                                loadings[
                                    channel_index,
                                    component_index,
                                ]
                            ),
                        })

                    score_frame = observation_index.copy()
                    score_frame["region"] = region
                    score_frame["scaling"] = scaling
                    score_frame["term"] = term
                    score_frame["component"] = (
                        component_index + 1
                    )
                    score_frame["score"] = (
                        scores[:, component_index]
                    )
                    score_frames.append(score_frame)

            pH_share = (
                term_traces["pH"] / total_trace
            )
            nuisance_share = (
                term_traces["phone"]
                + term_traces["light"]
                + term_traces["phone_x_light"]
            ) / total_trace
            interaction_share = (
                term_traces["pH_x_phone"]
                + term_traces["pH_x_light"]
            ) / total_trace
            residual_share = (
                term_traces["residual"] / total_trace
            )

            composition_rows.append({
                "region": region,
                "scaling": scaling,
                "pH_signal_share": pH_share,
                "nuisance_main_share": nuisance_share,
                "pH_dependent_interaction_share": (
                    interaction_share
                ),
                "residual_share": residual_share,
                "all_share_sum": (
                    pH_share
                    + nuisance_share
                    + interaction_share
                    + residual_share
                ),
                "channel_center_L": float(center[0]),
                "channel_center_a": float(center[1]),
                "channel_center_b": float(center[2]),
                "channel_scale_L": float(scale[0]),
                "channel_scale_a": float(scale[1]),
                "channel_scale_b": float(scale[2]),
            })

    effect_shares = pd.DataFrame(effect_share_rows)
    composition = pd.DataFrame(composition_rows)
    sscp_long = pd.DataFrame(sscp_rows)
    manova = pd.DataFrame(manova_rows)
    asca_variance = pd.DataFrame(variance_rows)
    asca_loadings = pd.DataFrame(loading_rows)
    asca_scores = pd.concat(
        score_frames,
        ignore_index=True,
    )

    effect_shares.to_csv(
        output_dir / "multivariate_effect_shares.csv",
        index=False,
        encoding="utf-8-sig",
    )
    composition.to_csv(
        output_dir / "effect_composition.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sscp_long.to_csv(
        output_dir / "sscp_matrices_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manova.to_csv(
        output_dir / "manova_descriptive_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    asca_variance.to_csv(
        output_dir / "asca_variance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    asca_loadings.to_csv(
        output_dir / "asca_loadings.csv",
        index=False,
        encoding="utf-8-sig",
    )
    asca_scores.to_csv(
        output_dir / "asca_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for scaling in args.scalings:
        plot_effect_composition(
            composition,
            scaling,
            figure_dir
            / f"effect_composition_{scaling}",
            args.dpi,
            args.save_pdf,
        )
        plot_term_shares(
            effect_shares,
            scaling,
            figure_dir
            / f"term_shares_{scaling}",
            args.dpi,
            args.save_pdf,
        )

        for region in REGION_COLUMNS:
            plot_ph_asca_score(
                asca_scores,
                region,
                scaling,
                figure_dir
                / f"{region}_pH_asca_pc1_{scaling}",
                args.dpi,
                args.save_pdf,
            )
            plot_ph_phone_asca_score(
                asca_scores,
                region,
                scaling,
                figure_dir
                / (
                    f"{region}_pH_x_phone_"
                    f"asca_pc1_{scaling}"
                ),
                args.dpi,
                args.save_pdf,
            )

            for term in ["pH", "pH_x_phone"]:
                plot_loading_bars(
                    asca_loadings,
                    region,
                    scaling,
                    term,
                    figure_dir
                    / (
                        f"{region}_{term}_"
                        f"loadings_pc1_{scaling}"
                    ),
                    args.dpi,
                    args.save_pdf,
                )

    audit = {
        "input": str(input_path),
        "output": str(output_dir),
        "rows": int(len(frame)),
        "regions": list(REGION_COLUMNS),
        "scalings": args.scalings,
        "phones": sorted(
            frame["phone"].unique().tolist()
        ),
        "lights": sorted(
            frame["light"].unique().tolist()
        ),
        "ph_levels": sorted(
            frame["ph_nominal"].unique().tolist()
        ),
        "method": (
            "direct balanced multivariate effect decomposition "
            "with ASCA-style SVD"
        ),
        "primary_comparison": (
            "standardized Lab trace shares across regions"
        ),
        "manova_note": (
            "Pillai, Wilks, Hotelling-Lawley, and Roy statistics "
            "are descriptive only; no formal p-values are reported."
        ),
        "limitations": [
            (
                "Each phone x pH cell contains one physical strip."
            ),
            (
                "The residual combines the three-way interaction "
                "with pure measurement/extraction error."
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

    standardized_summary = composition.loc[
        composition["scaling"] == "standardized"
    ].copy()

    notes = [
        "Multivariate Lab effect decomposition and ASCA",
        "=" * 80,
        "",
        (
            "The standardized analysis is primary for comparing Pad, "
            "Background, and Contrast because L*, a*, and b* are placed "
            "on equal variance scales."
        ),
        "",
        "Standardized multivariate composition",
        "-" * 80,
        standardized_summary.to_string(index=False),
        "",
        (
            "ASCA loadings describe which Lab channels dominate each "
            "effect component. Their signs are arbitrary because singular "
            "vectors can be multiplied by -1 without changing the result."
        ),
    ]

    (output_dir / "analysis_notes.txt").write_text(
        "\n".join(notes),
        encoding="utf-8",
    )

    print("=" * 92)
    print("15. Multivariate Lab decomposition and ASCA")
    print("=" * 92)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Rows: {len(frame)}")
    print("\nStandardized multivariate effect composition")
    print(standardized_summary.to_string(index=False))
    print("\nMain files")
    print(
        f"  {output_dir / 'multivariate_effect_shares.csv'}"
    )
    print(f"  {output_dir / 'effect_composition.csv'}")
    print(f"  {output_dir / 'asca_variance.csv'}")
    print(f"  {output_dir / 'asca_loadings.csv'}")
    print(f"  {output_dir / 'asca_scores.csv'}")
    print(f"  {figure_dir}")


if __name__ == "__main__":
    main()
