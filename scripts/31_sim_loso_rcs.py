#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
31_sim_loso_rcs.py

Leave-one-strip-out RCS predictions on the simulated data sets of script 30
(loso_datasets.csv), using the model code of script 19 unchanged.

Used by S3: when the true phone-specific curves are smooth and have no
boundary anomaly (A = 1), do RCS phone-specific models still fail at
pH 1/13 as they did on the real data?

Models (script 19): rcs_additive, rcs_phone, rcs_phone_ridge
(inner 5-fold strip CV over the script 19 lambda grid). rcs_full_ridge is
omitted to save time. Knot counts 3, 4, 5. Contrast region only.

Output
------
outputs/simulation/s3/rcs_strip_metrics.csv
    one row per data set x knot count x model x held-out strip
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESPONSE = ["contrast_L", "contrast_a", "contrast_b"]
MODELS = ["rcs_additive", "rcs_phone", "rcs_phone_ridge"]
KNOTS = [3, 4, 5]
LAMBDA_GRID = [0.01, 0.1, 1, 10, 100, 1000]
INNER_FOLDS = 5


def load_script19():
    path = PROJECT_ROOT / "scripts" / "19_boundary_stabilized_strip_holdout.py"
    spec = importlib.util.spec_from_file_location("script19", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S19 = load_script19()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulation LOSO with RCS models.")
    parser.add_argument("--input", default="outputs/simulation/s2/loso_datasets.csv")
    parser.add_argument("--output-dir", default="outputs/simulation/s3")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-datasets", type=int, default=0)
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    ph_levels = sorted(frame["ph_nominal"].unique())
    phone_levels = sorted(frame["phone"].unique())
    light_levels = sorted(frame["light"].unique())
    rows = []

    for knot_count in KNOTS:
        for fold_id, train_idx, test_idx in S19.outer_splits(frame):
            train = frame.iloc[train_idx]
            test = frame.iloc[test_idx]
            actual = test[RESPONSE].to_numpy(float)

            for model in MODELS:
                ridge_lambda = None
                if model == "rcs_phone_ridge":
                    ridge_lambda, _ = S19.choose_lambda(
                        train, RESPONSE, model, ph_levels, phone_levels, light_levels,
                        knot_count, LAMBDA_GRID, INNER_FOLDS,
                    )
                predicted, _ = S19.predict_fold(
                    train, test, RESPONSE, model, ph_levels, phone_levels, light_levels,
                    knot_count, ridge_lambda,
                )
                delta_e = np.linalg.norm(predicted - actual, axis=1)
                rows.append({
                    "A": frame["A"].iloc[0],
                    "dataset": int(frame["dataset"].iloc[0]),
                    "method": "RCS",
                    "variant": f"knots={knot_count}",
                    "model": model,
                    "strip_id": fold_id,
                    "phone": str(test["phone"].iloc[0]),
                    "ph_nominal": float(test["ph_nominal"].iloc[0]),
                    "mean_deltaE76": float(delta_e.mean()),
                })

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(resolve(args.input))
    groups = [group for _, group in data.groupby(["A", "dataset"], sort=True)]
    if args.max_datasets > 0:
        groups = [g for g in groups if g["dataset"].iloc[0] <= args.max_datasets]

    started = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(run_dataset, groups, chunksize=1)

    metrics = pd.concat(results, ignore_index=True)
    metrics.to_csv(output_dir / "rcs_strip_metrics.csv", index=False, encoding="utf-8-sig")

    notes = [
        "Simulation LOSO, RCS (script 31, script 19 model code)",
        "=" * 80,
        f"data sets: {len(groups)} | workers: {args.workers}",
        f"models: {MODELS} | knots: {KNOTS} | lambda grid: {LAMBDA_GRID}",
        f"elapsed: {(time.time() - started) / 60:.1f} min",
    ]
    (output_dir / "rcs_loso_audit.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
