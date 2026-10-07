#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
19_boundary_stabilized_strip_holdout.py

Boundary-stabilized leave-one-strip-out validation using restricted cubic
splines (natural cubic spline basis) and ridge shrinkage.

Primary question:
Can linear-tail spline behavior + shrinkage reduce the pH 1/13 failures found
in script 18 while retaining any internal-pH benefit?

Models
------
rcs_additive:
    RCS(pH) + phone + light + phone x light

rcs_phone:
    rcs_additive + phone x RCS(pH)

rcs_phone_ridge:
    same design as rcs_phone, but phone x RCS(pH) terms are ridge-shrunk.
    Lambda is selected inside each outer training fold by inner strip-level CV.

rcs_full_ridge:
    rcs_phone + light x RCS(pH), with both interaction blocks ridge-shrunk.

Primary comparison:
    rcs_phone_ridge - rcs_additive
Negative values favor the phone-specific ridge RCS.

No statsmodels is used.
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

MODEL_ORDER = [
    "rcs_additive",
    "rcs_phone",
    "rcs_phone_ridge",
    "rcs_full_ridge",
]

COMPARISONS = [
    ("rcs_phone", "rcs_additive"),
    ("rcs_phone_ridge", "rcs_additive"),
    ("rcs_full_ridge", "rcs_additive"),
    ("rcs_phone_ridge", "rcs_phone"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="outputs/factorial_analysis/analysis_factorial_table.csv",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/boundary_stabilized_models",
    )
    p.add_argument(
        "--knot-counts",
        nargs="*",
        type=int,
        default=[3, 4, 5],
    )
    p.add_argument(
        "--primary-knot-count",
        type=int,
        default=4,
    )
    p.add_argument(
        "--lambda-grid",
        nargs="*",
        type=float,
        default=[0.01, 0.1, 1, 10, 100, 1000],
    )
    p.add_argument("--inner-folds", type=int, default=5)
    p.add_argument("--bootstrap-reps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--save-pdf", action="store_true")
    return p.parse_args()


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
    description: str,
) -> str:
    lower = {str(c).lower(): str(c) for c in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise ValueError(f"Could not find {description}.")


def prepare_data(frame: pd.DataFrame) -> pd.DataFrame:
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

    required = [
        x
        for cols in REGION_COLUMNS.values()
        for x in cols
    ]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing Lab columns: {missing}")

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

    if "image_id" not in result.columns:
        result["image_id"] = [
            f"row_{i:03d}" for i in range(len(result))
        ]
    else:
        result["image_id"] = result["image_id"].astype(str)

    if "strip_id" not in result.columns:
        result["strip_id"] = (
            result["phone"]
            + "_ph"
            + result["ph_nominal"].map(lambda x: f"{x:g}")
        )
    else:
        result["strip_id"] = result["strip_id"].astype(str)

    if result.duplicated(
        ["ph_nominal", "phone", "light"],
        keep=False,
    ).any():
        raise ValueError("Duplicate pH x phone x light cells found.")

    expected = (
        result["ph_nominal"].nunique()
        * result["phone"].nunique()
        * result["light"].nunique()
    )
    if len(result) != expected:
        raise ValueError(
            f"Incomplete factorial design: {len(result)} vs {expected}."
        )

    light_count = result["light"].nunique()
    audit = result.groupby("strip_id").agg(
        phone_count=("phone", "nunique"),
        ph_count=("ph_nominal", "nunique"),
        light_count=("light", "nunique"),
        row_count=("image_id", "size"),
    )
    valid = (
        audit["phone_count"].eq(1)
        & audit["ph_count"].eq(1)
        & audit["light_count"].eq(light_count)
        & audit["row_count"].eq(light_count)
    )
    if not valid.all():
        raise ValueError(
            "strip_id must identify one phone x pH unit under all lights."
        )

    return result.sort_values(
        ["phone", "ph_nominal", "light"],
        kind="stable",
    ).reset_index(drop=True)


def rcs_basis(
    x: np.ndarray,
    x_min: float,
    x_max: float,
    knot_count: int,
):
    """
    Restricted cubic spline basis with linear tails.
    Returns one linear column plus knot_count-2 nonlinear columns.
    """
    if knot_count < 3:
        raise ValueError("knot_count must be >= 3.")

    x = np.asarray(x, dtype=float)
    span = x_max - x_min
    if span <= 0:
        raise ValueError("Invalid pH range.")
    if np.any(x < x_min) or np.any(x > x_max):
        raise ValueError("pH outside fixed range.")

    z = (x - x_min) / span
    knots_original = np.linspace(
        x_min,
        x_max,
        knot_count,
    )
    knots = (knots_original - x_min) / span

    def cube_plus(v):
        return np.maximum(v, 0.0) ** 3

    last = knots[-1]
    penultimate = knots[-2]

    columns = [z]
    names = ["rcs_linear"]

    for j in range(knot_count - 2):
        kj = knots[j]
        term = (
            (cube_plus(z - kj) - cube_plus(z - last))
            / (last - kj)
            - (
                cube_plus(z - penultimate)
                - cube_plus(z - last)
            )
            / (last - penultimate)
        )
        columns.append(term)
        names.append(f"rcs_nonlinear_{j + 1}")

    matrix = np.column_stack(columns)
    return matrix, names, knots_original


def dummies(values, levels, prefix):
    cat = pd.Categorical(
        values,
        categories=list(levels),
        ordered=True,
    )
    codes = cat.codes
    if np.any(codes < 0):
        raise ValueError(f"Unknown {prefix} level.")

    cols = []
    names = []
    for j in range(1, len(levels)):
        cols.append((codes == j).astype(float))
        names.append(f"{prefix}[{levels[j]}]")

    if not cols:
        return np.empty((len(values), 0)), names
    return np.column_stack(cols), names


def interaction(a, a_names, b, b_names, prefix):
    cols = []
    names = []

    for i, an in enumerate(a_names):
        for j, bn in enumerate(b_names):
            cols.append(a[:, i] * b[:, j])
            names.append(f"{prefix}:{an}:{bn}")

    if not cols:
        return np.empty((a.shape[0], 0)), names
    return np.column_stack(cols), names


def build_design(
    frame,
    model,
    ph_levels,
    phone_levels,
    light_levels,
    knot_count,
):
    spline, spline_names, knots = rcs_basis(
        frame["ph_nominal"].to_numpy(dtype=float),
        float(min(ph_levels)),
        float(max(ph_levels)),
        knot_count,
    )
    phone, phone_names = dummies(
        frame["phone"],
        phone_levels,
        "phone",
    )
    light, light_names = dummies(
        frame["light"],
        light_levels,
        "light",
    )

    phone_light = interaction(
        phone,
        phone_names,
        light,
        light_names,
        "phone_x_light",
    )
    phone_spline = interaction(
        phone,
        phone_names,
        spline,
        spline_names,
        "phone_x_rcs",
    )
    light_spline = interaction(
        light,
        light_names,
        spline,
        spline_names,
        "light_x_rcs",
    )

    blocks = [
        ("base", np.ones((len(frame), 1)), ["Intercept"]),
        ("base", spline, spline_names),
        ("base", phone, phone_names),
        ("base", light, light_names),
        ("base", phone_light[0], phone_light[1]),
    ]

    if model in {
        "rcs_phone",
        "rcs_phone_ridge",
        "rcs_full_ridge",
    }:
        blocks.append(
            ("penalty", phone_spline[0], phone_spline[1])
        )

    if model == "rcs_full_ridge":
        blocks.append(
            ("penalty", light_spline[0], light_spline[1])
        )

    x = np.column_stack([b[1] for b in blocks])
    names = [n for _, _, ns in blocks for n in ns]
    penalized = np.array(
        [g == "penalty" for g, _, ns in blocks for _ in ns],
        dtype=bool,
    )
    return x, names, penalized, knots


def fit_ols(x, y):
    rank = int(np.linalg.matrix_rank(x))
    if rank != x.shape[1]:
        raise ValueError(
            f"Rank deficient: rank={rank}, cols={x.shape[1]}"
        )
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return beta, rank


def fit_ridge(x, y, penalized, ridge_lambda):
    rank = int(np.linalg.matrix_rank(x))
    xtx = x.T @ x
    xty = x.T @ y

    scale = x.std(axis=0, ddof=1)
    scale[~np.isfinite(scale)] = 0.0

    penalty = np.zeros(x.shape[1])
    penalty[penalized] = np.square(
        np.maximum(scale[penalized], 1e-8)
    )

    beta = np.linalg.solve(
        xtx + ridge_lambda * np.diag(penalty),
        xty,
    )
    return beta, rank


def outer_splits(frame):
    all_idx = np.arange(len(frame))
    strips = (
        frame[["strip_id", "phone", "ph_nominal"]]
        .drop_duplicates()
        .sort_values(["phone", "ph_nominal", "strip_id"])
    )

    result = []
    for row in strips.itertuples(index=False):
        mask = (
            frame["strip_id"].astype(str)
            == str(row.strip_id)
        ).to_numpy()
        result.append(
            (
                str(row.strip_id),
                all_idx[~mask],
                all_idx[mask],
            )
        )

    tested = np.concatenate([x[2] for x in result])
    if sorted(tested.tolist()) != list(range(len(frame))):
        raise ValueError("Outer CV coverage failed.")
    return result


def inner_splits(frame, n_folds):
    strips = (
        frame[["strip_id", "phone", "ph_nominal"]]
        .drop_duplicates()
        .sort_values(["phone", "ph_nominal", "strip_id"])
    )

    assignment = {}
    for phone, group in strips.groupby("phone", sort=True):
        group = group.sort_values(["ph_nominal", "strip_id"])
        for pos, row in enumerate(group.itertuples(index=False)):
            assignment[str(row.strip_id)] = pos % n_folds

    all_idx = np.arange(len(frame))
    result = []

    for fold in range(n_folds):
        test_mask = (
            frame["strip_id"].map(assignment).to_numpy() == fold
        )
        if not np.any(test_mask):
            continue
        train_idx = all_idx[~test_mask]
        test_idx = all_idx[test_mask]

        if frame.iloc[train_idx]["phone"].nunique() != frame["phone"].nunique():
            raise ValueError("Inner fold lost a phone level.")
        if frame.iloc[train_idx]["light"].nunique() != frame["light"].nunique():
            raise ValueError("Inner fold lost a light level.")

        result.append((train_idx, test_idx))

    return result


def choose_lambda(
    train_frame,
    response_columns,
    model,
    ph_levels,
    phone_levels,
    light_levels,
    knot_count,
    lambda_grid,
    n_inner_folds,
):
    rows = []

    for ridge_lambda in lambda_grid:
        losses = []

        for fold_index, (tr_idx, te_idx) in enumerate(
            inner_splits(train_frame, n_inner_folds),
            start=1,
        ):
            tr = train_frame.iloc[tr_idx]
            te = train_frame.iloc[te_idx]

            x_tr, _, penalized, _ = build_design(
                tr,
                model,
                ph_levels,
                phone_levels,
                light_levels,
                knot_count,
            )
            x_te, _, _, _ = build_design(
                te,
                model,
                ph_levels,
                phone_levels,
                light_levels,
                knot_count,
            )

            y_tr = tr[list(response_columns)].to_numpy(float)
            y_te = te[list(response_columns)].to_numpy(float)

            beta, _ = fit_ridge(
                x_tr,
                y_tr,
                penalized,
                ridge_lambda,
            )
            pred = x_te @ beta
            delta_e = np.linalg.norm(pred - y_te, axis=1)

            temp = te[["strip_id"]].copy()
            temp["deltaE76"] = delta_e
            loss = float(
                temp.groupby("strip_id")["deltaE76"]
                .mean()
                .mean()
            )
            losses.append(loss)

            rows.append({
                "lambda": float(ridge_lambda),
                "inner_fold": fold_index,
                "mean_strip_deltaE76": loss,
            })

        rows.append({
            "lambda": float(ridge_lambda),
            "inner_fold": 0,
            "mean_strip_deltaE76": float(np.mean(losses)),
        })

    table = pd.DataFrame(rows)
    aggregate = (
        table.loc[table["inner_fold"] == 0]
        .sort_values(["mean_strip_deltaE76", "lambda"])
    )
    selected = float(aggregate.iloc[0]["lambda"])
    return selected, table


def predict_fold(
    train,
    test,
    response_columns,
    model,
    ph_levels,
    phone_levels,
    light_levels,
    knot_count,
    ridge_lambda,
):
    x_tr, names, penalized, knots = build_design(
        train,
        model,
        ph_levels,
        phone_levels,
        light_levels,
        knot_count,
    )
    x_te, _, _, _ = build_design(
        test,
        model,
        ph_levels,
        phone_levels,
        light_levels,
        knot_count,
    )
    y_tr = train[list(response_columns)].to_numpy(float)

    if ridge_lambda is None:
        beta, rank = fit_ols(x_tr, y_tr)
        used_lambda = 0.0
    else:
        beta, rank = fit_ridge(
            x_tr,
            y_tr,
            penalized,
            ridge_lambda,
        )
        used_lambda = float(ridge_lambda)

    return (
        x_te @ beta,
        {
            "n_columns": x_tr.shape[1],
            "rank": rank,
            "residual_df_ols_reference": len(train) - rank,
            "penalized_columns": int(np.sum(penalized)),
            "lambda": used_lambda,
            "knots": ",".join(f"{x:g}" for x in knots),
            "column_names": " | ".join(names),
        },
    )


def make_prediction_rows(
    test,
    pred,
    actual,
    region,
    model,
    knot_count,
    ridge_lambda,
    fold_id,
):
    error = pred - actual
    rows = []

    for i, (_, meta) in enumerate(test.iterrows()):
        rows.append({
            "fold_id": fold_id,
            "knot_count": knot_count,
            "region": region,
            "model": model,
            "selected_lambda": float(ridge_lambda),
            "image_id": str(meta["image_id"]),
            "strip_id": str(meta["strip_id"]),
            "phone": str(meta["phone"]),
            "light": str(meta["light"]),
            "ph_nominal": float(meta["ph_nominal"]),
            "actual_L": float(actual[i, 0]),
            "actual_a": float(actual[i, 1]),
            "actual_b": float(actual[i, 2]),
            "predicted_L": float(pred[i, 0]),
            "predicted_a": float(pred[i, 1]),
            "predicted_b": float(pred[i, 2]),
            "squared_error_L": float(error[i, 0] ** 2),
            "squared_error_a": float(error[i, 1] ** 2),
            "squared_error_b": float(error[i, 2] ** 2),
            "deltaE76": float(np.linalg.norm(error[i])),
        })

    return rows


def summarize(predictions, group_columns):
    rows = []

    for keys, group in predictions.groupby(
        list(group_columns),
        sort=True,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {
            c: v for c, v in zip(group_columns, keys)
        }
        row.update({
            "n_predictions": len(group),
            "n_strips": group["strip_id"].nunique(),
            "rmse_L": float(
                np.sqrt(group["squared_error_L"].mean())
            ),
            "rmse_a": float(
                np.sqrt(group["squared_error_a"].mean())
            ),
            "rmse_b": float(
                np.sqrt(group["squared_error_b"].mean())
            ),
            "mean_deltaE76": float(group["deltaE76"].mean()),
            "median_deltaE76": float(group["deltaE76"].median()),
            "q90_deltaE76": float(group["deltaE76"].quantile(0.9)),
        })
        rows.append(row)

    return pd.DataFrame(rows)


def bootstrap_difference(frame, value_column, reps, seed):
    observed = float(frame[value_column].mean())
    if reps <= 0:
        return observed, np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    groups = {
        p: g.reset_index(drop=True)
        for p, g in frame.groupby("phone", sort=True)
    }
    values = np.empty(reps)

    for r in range(reps):
        sample = []
        for group in groups.values():
            idx = rng.integers(0, len(group), size=len(group))
            sample.append(
                group.iloc[idx][value_column].to_numpy(float)
            )
        values[r] = np.mean(np.concatenate(sample))

    return (
        observed,
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        float(np.mean(values < 0)),
    )


def paired_tables(strip_metrics, ph_levels, reps, seed):
    paired_rows = []
    subset_rows = []

    low = float(min(ph_levels))
    high = float(max(ph_levels))
    subsets = {
        "all_ph": set(ph_levels),
        "boundary_ph_1_13": {low, high},
        "interior_ph_2_12": {
            x for x in ph_levels if low < x < high
        },
        "strict_interior_ph_3_11": {
            x for x in ph_levels if low + 1 < x < high - 1
        },
    }

    join = [
        "knot_count",
        "region",
        "strip_id",
        "phone",
        "ph_nominal",
    ]

    for numerator, denominator in COMPARISONS:
        num = strip_metrics.loc[
            strip_metrics["model"] == numerator
        ]
        den = strip_metrics.loc[
            strip_metrics["model"] == denominator
        ]

        merged = num.merge(
            den,
            on=join,
            how="inner",
            suffixes=("_num", "_den"),
            validate="one_to_one",
        )
        merged["difference"] = (
            merged["mean_deltaE76_num"]
            - merged["mean_deltaE76_den"]
        )

        for (knot_count, region), group in merged.groupby(
            ["knot_count", "region"],
            sort=True,
        ):
            mean, lower, upper, prob = bootstrap_difference(
                group,
                "difference",
                reps,
                seed,
            )
            paired_rows.append({
                "knot_count": int(knot_count),
                "region": region,
                "numerator_model": numerator,
                "denominator_model": denominator,
                "comparison": f"{numerator} minus {denominator}",
                "n_strips": len(group),
                "mean_difference": mean,
                "median_difference": float(
                    group["difference"].median()
                ),
                "proportion_numerator_better": float(
                    np.mean(group["difference"] < 0)
                ),
                "bootstrap_ci_lower": lower,
                "bootstrap_ci_upper": upper,
                "bootstrap_probability_below_zero": prob,
            })

            for subset_name, ph_set in subsets.items():
                sub = group.loc[
                    group["ph_nominal"].isin(ph_set)
                ]
                mean, lower, upper, prob = bootstrap_difference(
                    sub,
                    "difference",
                    reps,
                    seed,
                )
                subset_rows.append({
                    "knot_count": int(knot_count),
                    "region": region,
                    "subset": subset_name,
                    "included_ph": ",".join(
                        f"{x:g}" for x in sorted(ph_set)
                    ),
                    "numerator_model": numerator,
                    "denominator_model": denominator,
                    "comparison": f"{numerator} minus {denominator}",
                    "n_strips": len(sub),
                    "mean_difference": mean,
                    "median_difference": float(
                        sub["difference"].median()
                    ),
                    "proportion_numerator_better": float(
                        np.mean(sub["difference"] < 0)
                    ),
                    "bootstrap_ci_lower": lower,
                    "bootstrap_ci_upper": upper,
                    "bootstrap_probability_below_zero": prob,
                })

    return pd.DataFrame(paired_rows), pd.DataFrame(subset_rows)


def save_figure(fig, stem, dpi, save_pdf):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
    )
    if save_pdf:
        fig.savefig(
            stem.with_suffix(".pdf"),
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_summary(summary, knot_count, region, stem, dpi, save_pdf):
    subset = summary.loc[
        (summary["knot_count"] == knot_count)
        & (summary["region"] == region)
    ].copy()

    subset["model"] = pd.Categorical(
        subset["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    subset = subset.sort_values("model")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(
        subset["model"].astype(str),
        subset["mean_deltaE76"].to_numpy(float),
    )
    ax.set_ylabel("Mean leave-one-strip-out Delta E")
    ax.set_title(
        f"{region}: boundary-stabilized models (knots={knot_count})"
    )
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, stem, dpi, save_pdf)


def plot_by_ph(by_ph, knot_count, region, stem, dpi, save_pdf):
    subset = by_ph.loc[
        (by_ph["knot_count"] == knot_count)
        & (by_ph["region"] == region)
    ]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))

    for model, group in subset.groupby("model", sort=True):
        group = group.sort_values("ph_nominal")
        ax.plot(
            group["ph_nominal"].to_numpy(float),
            group["mean_deltaE76"].to_numpy(float),
            marker="o",
            linewidth=1.5,
            label=model,
        )

    ax.set_xlabel("Nominal pH")
    ax.set_ylabel("Mean held-out-strip Delta E")
    ax.set_xticks(sorted(subset["ph_nominal"].unique()))
    ax.set_title(f"{region}: error by pH (knots={knot_count})")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, stem, dpi, save_pdf)


def main():
    args = parse_args()
    knot_counts = sorted(set(args.knot_counts))
    lambda_grid = sorted(set(args.lambda_grid))

    if args.primary_knot_count not in knot_counts:
        raise ValueError(
            "primary-knot-count must be included in knot-counts."
        )

    input_path = resolve_existing(args.input)
    output_dir = resolve_output(args.output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    frame = prepare_data(pd.read_csv(input_path))
    ph_levels = sorted(frame["ph_nominal"].unique())
    phone_levels = sorted(frame["phone"].unique())
    light_levels = sorted(frame["light"].unique())
    folds = outer_splits(frame)

    predictions_all = []
    lambda_rows = []
    rank_rows = []
    ridge_models = {"rcs_phone_ridge", "rcs_full_ridge"}

    for knot_count in knot_counts:
        for fold_id, train_idx, test_idx in folds:
            train = frame.iloc[train_idx].copy()
            test = frame.iloc[test_idx].copy()

            heldout_phone = str(test["phone"].iloc[0])
            heldout_ph = float(test["ph_nominal"].iloc[0])

            for region, response_columns in REGION_COLUMNS.items():
                actual = test[response_columns].to_numpy(float)
                selected = {}

                for model in ridge_models:
                    best_lambda, table = choose_lambda(
                        train,
                        response_columns,
                        model,
                        ph_levels,
                        phone_levels,
                        light_levels,
                        knot_count,
                        lambda_grid,
                        args.inner_folds,
                    )
                    selected[model] = best_lambda

                    aggregate = table.loc[
                        table["inner_fold"] == 0
                    ]
                    for _, row in aggregate.iterrows():
                        lambda_value = float(row["lambda"])
                        lambda_rows.append({
                            "outer_fold": fold_id,
                            "heldout_phone": heldout_phone,
                            "heldout_ph": heldout_ph,
                            "knot_count": knot_count,
                            "region": region,
                            "model": model,
                            "lambda": lambda_value,
                            "inner_mean_strip_deltaE76": float(
                                row["mean_strip_deltaE76"]
                            ),
                            "selected": bool(
                                np.isclose(lambda_value, best_lambda)
                            ),
                        })

                for model in MODEL_ORDER:
                    ridge_lambda = (
                        selected[model]
                        if model in ridge_models
                        else None
                    )
                    pred, audit = predict_fold(
                        train,
                        test,
                        response_columns,
                        model,
                        ph_levels,
                        phone_levels,
                        light_levels,
                        knot_count,
                        ridge_lambda,
                    )
                    rank_rows.append({
                        "outer_fold": fold_id,
                        "heldout_phone": heldout_phone,
                        "heldout_ph": heldout_ph,
                        "knot_count": knot_count,
                        "region": region,
                        "model": model,
                        **audit,
                    })
                    predictions_all.extend(
                        make_prediction_rows(
                            test,
                            pred,
                            actual,
                            region,
                            model,
                            knot_count,
                            (
                                ridge_lambda
                                if ridge_lambda is not None
                                else 0.0
                            ),
                            fold_id,
                        )
                    )

    predictions = pd.DataFrame(predictions_all)
    lambdas = pd.DataFrame(lambda_rows)
    rank_audit = pd.DataFrame(rank_rows)

    strip_table = summarize(
        predictions,
        [
            "knot_count",
            "region",
            "model",
            "strip_id",
            "phone",
            "ph_nominal",
        ],
    )
    cv_summary = summarize(
        predictions,
        ["knot_count", "region", "model"],
    )
    by_ph = summarize(
        predictions,
        ["knot_count", "region", "model", "ph_nominal"],
    )
    paired, subsets = paired_tables(
        strip_table,
        ph_levels,
        args.bootstrap_reps,
        args.seed,
    )

    lambda_frequency = (
        lambdas.loc[lambdas["selected"]]
        .groupby(
            ["knot_count", "region", "model", "lambda"],
            as_index=False,
        )
        .agg(selection_count=("selected", "size"))
    )
    lambda_frequency["selection_fraction"] = (
        lambda_frequency["selection_count"] / len(folds)
    )

    predictions.to_csv(
        output_dir / "cv_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    strip_table.to_csv(
        output_dir / "strip_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv_summary.to_csv(
        output_dir / "cv_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_ph.to_csv(
        output_dir / "metrics_by_ph.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        output_dir / "paired_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )
    subsets.to_csv(
        output_dir / "subset_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lambdas.to_csv(
        output_dir / "lambda_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lambda_frequency.to_csv(
        output_dir / "lambda_selection_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rank_audit.to_csv(
        output_dir / "design_rank_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for region in REGION_COLUMNS:
        plot_summary(
            cv_summary,
            args.primary_knot_count,
            region,
            figure_dir
            / f"{region}_model_comparison_knots{args.primary_knot_count}",
            args.dpi,
            args.save_pdf,
        )
        plot_by_ph(
            by_ph,
            args.primary_knot_count,
            region,
            figure_dir
            / f"{region}_error_by_ph_knots{args.primary_knot_count}",
            args.dpi,
            args.save_pdf,
        )

    primary = paired.loc[
        paired["knot_count"] == args.primary_knot_count
    ]
    primary_subsets = subsets.loc[
        (subsets["knot_count"] == args.primary_knot_count)
        & (subsets["numerator_model"] == "rcs_phone_ridge")
        & (subsets["denominator_model"] == "rcs_additive")
    ]

    audit = {
        "input": str(input_path),
        "output": str(output_dir),
        "rows": len(frame),
        "outer_folds": len(folds),
        "knot_counts": knot_counts,
        "primary_knot_count": args.primary_knot_count,
        "lambda_grid": lambda_grid,
        "inner_folds": args.inner_folds,
        "primary_comparison": (
            "rcs_phone_ridge minus rcs_additive"
        ),
        "lambda_selection": (
            "within each outer training fold using strip-level inner CV"
        ),
        "limitations": [
            "One physical strip per phone x pH cell.",
            "Known-phone new-cell validation, not unseen-phone validation.",
        ],
    }
    (output_dir / "analysis_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    notes = [
        "Boundary-stabilized leave-one-strip-out analysis",
        "=" * 80,
        "",
        "Primary paired comparisons",
        primary.to_string(index=False),
        "",
        "Primary ridge RCS subset comparisons",
        primary_subsets.to_string(index=False),
    ]
    (output_dir / "analysis_notes.txt").write_text(
        "\n".join(notes),
        encoding="utf-8",
    )

    print("=" * 100)
    print("19. Boundary-stabilized leave-one-strip-out models")
    print("=" * 100)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Outer folds: {len(folds)}")
    print(f"Primary knot count: {args.primary_knot_count}")
    print("\nPrimary paired comparisons")
    print(primary.to_string(index=False))
    print("\nPrimary ridge RCS boundary/interior comparisons")
    print(primary_subsets.to_string(index=False))
    print("\nSelected lambda frequencies")
    print(
        lambda_frequency.loc[
            lambda_frequency["knot_count"]
            == args.primary_knot_count
        ].to_string(index=False)
    )
    print("\nMain result:")
    print(output_dir / "subset_comparisons.csv")


if __name__ == "__main__":
    main()
