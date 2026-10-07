#!/usr/bin/env python
"""Summarize all train-only nuisance decompositions across LOPO folds.

Expected input layout
---------------------
outputs/decomposition/
  capture_lopo_holdout_.../
    decomposition_config.json
    fit_summary.csv
    component_effects.csv
    nuisance_vector_library.csv

Outputs
-------
outputs/decomposition_summary/
  fit_summary_all_folds.csv
  component_level_vectors.csv
  component_magnitude_all_folds.csv
  nuisance_distribution_all_folds.csv
  fold_summary.csv
  audit_summary.json
  plots/*.png
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHANNELS_BY_REGION = {
    "pad": ["pad_L", "pad_a", "pad_b"],
    "bg": ["bg_L", "bg_a", "bg_b"],
    "contrast": ["contrast_L", "contrast_a", "contrast_b"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine and audit nuisance-decomposition results from every LOPO split."
    )
    parser.add_argument(
        "--decomposition-root",
        default="outputs/decomposition",
        help="Directory containing one subdirectory per decomposition run.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/decomposition_summary",
        help="Directory in which combined tables and plots are saved.",
    )
    parser.add_argument(
        "--run-glob",
        default="capture_lopo_holdout_*",
        help="Glob used to select decomposition run directories.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Create CSV/JSON summaries only.",
    )
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def safe_read_csv(path: Path, required_columns: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    return frame


def parse_fold_metadata(run_dir: Path, config: dict) -> dict:
    split_file = Path(str(config.get("split_file", ""))).stem
    source = split_file or run_dir.name

    holdout_match = re.search(r"holdout_(.+?)_vallight_", source)
    val_match = re.search(r"_vallight_(.+)$", source)

    return {
        "run_name": run_dir.name,
        "holdout_phone": holdout_match.group(1) if holdout_match else "",
        "validation_light": val_match.group(1) if val_match else "",
        "training_rows": int(config.get("training_rows", 0)),
        "training_phone_count": len(config.get("training_phones", [])),
        "training_light_count": len(config.get("training_lights", [])),
        "training_ph_count": len(config.get("training_ph_levels", [])),
        "training_phones": ",".join(map(str, config.get("training_phones", []))),
        "training_lights": ",".join(map(str, config.get("training_lights", []))),
    }


def build_component_level_vectors(components: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Turn separate L/a/b scalar effects into one vector per component level."""
    records: list[dict] = []
    for region, channels in CHANNELS_BY_REGION.items():
        region_part = components[components["value_column"].isin(channels)].copy()
        region_part["level_2"] = region_part["level_2"].fillna("").astype(str)
        if region_part.empty:
            continue

        for effect_type in ["phone", "light", "phone_light"]:
            part = region_part[region_part["effect_type"].eq(effect_type)].copy()
            if part.empty:
                continue

            index_columns = ["effect_type", "level_1", "level_2"]
            pivot = part.pivot_table(
                index=index_columns,
                columns="value_column",
                values="effect",
                aggfunc="first",
            ).reset_index()

            if not all(column in pivot.columns for column in channels):
                continue

            for _, row in pivot.iterrows():
                vector = [float(row[channel]) for channel in channels]
                records.append(
                    {
                        **metadata,
                        "region": region,
                        "effect_type": effect_type,
                        "level_1": str(row["level_1"]),
                        "level_2": "" if pd.isna(row["level_2"]) else str(row["level_2"]),
                        "effect_L": vector[0],
                        "effect_a": vector[1],
                        "effect_b": vector[2],
                        "effect_deltaE": float(np.linalg.norm(vector)),
                    }
                )
    return pd.DataFrame(records)


def summarize_component_vectors(vectors: pd.DataFrame) -> pd.DataFrame:
    if vectors.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    group_columns = [
        "run_name",
        "holdout_phone",
        "validation_light",
        "region",
        "effect_type",
    ]
    for keys, group in vectors.groupby(group_columns, sort=True, dropna=False):
        values = pd.to_numeric(group["effect_deltaE"], errors="coerce").dropna().to_numpy(float)
        if len(values) == 0:
            continue
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_levels": int(len(values)),
                "mean_deltaE": float(np.mean(values)),
                "median_deltaE": float(np.median(values)),
                "std_deltaE": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "q95_deltaE": float(np.quantile(values, 0.95)),
                "max_deltaE": float(np.max(values)),
                "rms_deltaE": float(np.sqrt(np.mean(values**2))),
            }
        )
    return pd.DataFrame(rows)


def summarize_vector_library(library: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    rows: list[dict] = []
    effects = ["phone_effect", "light_effect", "interaction_effect", "residual"]
    for region, region_frame in library.groupby("region", sort=True):
        for effect in effects:
            column = f"{effect}_deltaE"
            if column not in region_frame.columns:
                continue
            values = pd.to_numeric(region_frame[column], errors="coerce").dropna().to_numpy(float)
            if len(values) == 0:
                continue
            rows.append(
                {
                    **metadata,
                    "region": region,
                    "effect_type": effect,
                    "n_rows": int(len(values)),
                    "mean_deltaE": float(np.mean(values)),
                    "median_deltaE": float(np.median(values)),
                    "std_deltaE": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "q95_deltaE": float(np.quantile(values, 0.95)),
                    "max_deltaE": float(np.max(values)),
                    "rms_deltaE": float(np.sqrt(np.mean(values**2))),
                }
            )
    return pd.DataFrame(rows)


def build_fold_summary(
    fit_all: pd.DataFrame,
    component_summary: pd.DataFrame,
    nuisance_summary: pd.DataFrame,
) -> pd.DataFrame:
    metadata_columns = [
        "run_name",
        "holdout_phone",
        "validation_light",
        "training_rows",
        "training_phone_count",
        "training_light_count",
        "training_ph_count",
    ]
    records: list[dict] = []

    for run_name, fit in fit_all.groupby("run_name", sort=True):
        first = fit.iloc[0]
        record = {column: first.get(column, "") for column in metadata_columns}
        record.update(
            {
                "fit_mean_r2": float(fit["r2"].mean()),
                "fit_min_r2": float(fit["r2"].min()),
                "fit_mean_rmse": float(fit["rmse"].mean()),
                "fit_max_rmse": float(fit["rmse"].max()),
                "fit_mean_residual_std": float(fit["residual_std"].mean()),
            }
        )

        for region in CHANNELS_BY_REGION:
            region_fit = fit[fit["region"].eq(region)]
            if not region_fit.empty:
                record[f"{region}_mean_r2"] = float(region_fit["r2"].mean())
                record[f"{region}_min_r2"] = float(region_fit["r2"].min())

        if not component_summary.empty:
            component_run = component_summary[component_summary["run_name"].eq(run_name)]
            for _, row in component_run.iterrows():
                key = f"{row['region']}_{row['effect_type']}_mean_deltaE"
                record[key] = float(row["mean_deltaE"])

        if not nuisance_summary.empty:
            residual_run = nuisance_summary[
                nuisance_summary["run_name"].eq(run_name)
                & nuisance_summary["effect_type"].eq("residual")
            ]
            for _, row in residual_run.iterrows():
                record[f"{row['region']}_residual_q95_deltaE"] = float(row["q95_deltaE"])

        records.append(record)

    return pd.DataFrame(records)


def create_plots(
    fit_all: pd.DataFrame,
    component_summary: pd.DataFrame,
    nuisance_summary: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: matplotlib unavailable; plots skipped: {exc}")
        return []

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    # Plot 1: R2 by value column across folds.
    labels = sorted(fit_all["value_column"].astype(str).unique())
    data = [fit_all.loc[fit_all["value_column"].eq(label), "r2"].to_numpy(float) for label in labels]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("R² across folds")
    ax.set_xlabel("Decomposed color variable")
    ax.set_title("Nuisance decomposition fit across LOPO folds")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = plot_dir / "r2_by_value_column.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(str(path))

    # Plot 2: component magnitude by region/effect.
    if not component_summary.empty:
        agg = (
            component_summary.groupby(["region", "effect_type"], sort=True)["mean_deltaE"]
            .mean()
            .unstack("effect_type")
        )
        fig, ax = plt.subplots(figsize=(9, 6))
        agg.plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean component magnitude (ΔE76)")
        ax.set_xlabel("Region")
        ax.set_title("Average structured nuisance magnitude")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = plot_dir / "component_magnitude_by_region.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(str(path))

    # Plot 3: residual q95 by region across folds.
    if not nuisance_summary.empty:
        residual = nuisance_summary[nuisance_summary["effect_type"].eq("residual")]
        if not residual.empty:
            fig, ax = plt.subplots(figsize=(9, 6))
            for region, group in residual.groupby("region", sort=True):
                ordered = group.sort_values(["holdout_phone", "validation_light"])
                ax.plot(range(len(ordered)), ordered["q95_deltaE"], marker="o", label=region)
            ax.set_ylabel("Residual ΔE76, 95th percentile")
            ax.set_xlabel("Fold index")
            ax.set_title("Residual magnitude across folds")
            ax.grid(axis="y", alpha=0.25)
            ax.legend()
            fig.tight_layout()
            path = plot_dir / "residual_q95_across_folds.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            saved.append(str(path))

    return saved


def main() -> None:
    args = parse_args()
    decomposition_root = resolve(args.decomposition_root)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not decomposition_root.exists():
        raise FileNotFoundError(decomposition_root)

    run_dirs = sorted(
        path for path in decomposition_root.glob(args.run_glob) if path.is_dir()
    )
    if not run_dirs:
        raise ValueError(
            f"No run directories matched {args.run_glob!r} under {decomposition_root}"
        )

    fit_parts: list[pd.DataFrame] = []
    component_vector_parts: list[pd.DataFrame] = []
    component_summary_parts: list[pd.DataFrame] = []
    nuisance_summary_parts: list[pd.DataFrame] = []
    errors: list[dict] = []

    for run_dir in run_dirs:
        try:
            config_path = run_dir / "decomposition_config.json"
            if not config_path.exists():
                raise FileNotFoundError(config_path)
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            metadata = parse_fold_metadata(run_dir, config)

            fit = safe_read_csv(
                run_dir / "fit_summary.csv",
                ["value_column", "rmse", "mae", "r2", "residual_std", "region"],
            )
            for key, value in metadata.items():
                fit[key] = value
            fit_parts.append(fit)

            components = safe_read_csv(
                run_dir / "component_effects.csv",
                ["value_column", "effect_type", "level_1", "effect"],
            )
            if "level_2" not in components.columns:
                components["level_2"] = ""
            vectors = build_component_level_vectors(components, metadata)
            if not vectors.empty:
                component_vector_parts.append(vectors)
                component_summary_parts.append(summarize_component_vectors(vectors))

            library = safe_read_csv(
                run_dir / "nuisance_vector_library.csv",
                ["image_id", "phone_id", "light_id", "ph_nominal", "region"],
            )
            nuisance_summary_parts.append(summarize_vector_library(library, metadata))

            print(f"Loaded: {run_dir.name}")
        except Exception as exc:
            errors.append({"run_name": run_dir.name, "error": str(exc)})
            print(f"ERROR: {run_dir.name}: {exc}")

    if not fit_parts:
        raise RuntimeError("No valid decomposition run could be summarized")

    fit_all = pd.concat(fit_parts, ignore_index=True)
    component_vectors = (
        pd.concat(component_vector_parts, ignore_index=True)
        if component_vector_parts
        else pd.DataFrame()
    )
    component_summary = (
        pd.concat(component_summary_parts, ignore_index=True)
        if component_summary_parts
        else pd.DataFrame()
    )
    nuisance_summary = (
        pd.concat(nuisance_summary_parts, ignore_index=True)
        if nuisance_summary_parts
        else pd.DataFrame()
    )
    fold_summary = build_fold_summary(fit_all, component_summary, nuisance_summary)

    fit_all.to_csv(output_dir / "fit_summary_all_folds.csv", index=False, encoding="utf-8-sig")
    component_vectors.to_csv(output_dir / "component_level_vectors.csv", index=False, encoding="utf-8-sig")
    component_summary.to_csv(output_dir / "component_magnitude_all_folds.csv", index=False, encoding="utf-8-sig")
    nuisance_summary.to_csv(output_dir / "nuisance_distribution_all_folds.csv", index=False, encoding="utf-8-sig")
    fold_summary.to_csv(output_dir / "fold_summary.csv", index=False, encoding="utf-8-sig")

    plots = [] if args.skip_plots else create_plots(
        fit_all,
        component_summary,
        nuisance_summary,
        output_dir,
    )

    expected_run_count = 15
    audit = {
        "decomposition_root": str(decomposition_root),
        "run_glob": args.run_glob,
        "matched_run_count": len(run_dirs),
        "successful_run_count": int(fit_all["run_name"].nunique()),
        "expected_run_count": expected_run_count,
        "all_expected_runs_present": len(run_dirs) == expected_run_count and not errors,
        "fit_rows": int(len(fit_all)),
        "fold_rows": int(len(fold_summary)),
        "errors": errors,
        "plots": plots,
    }
    with (output_dir / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)

    print("\nSaved decomposition summary:", output_dir)
    print("Matched runs:", len(run_dirs))
    print("Successful runs:", fit_all["run_name"].nunique())
    print("Fit rows:", len(fit_all))
    print("Fold rows:", len(fold_summary))
    print("Mean R2:", round(float(fit_all["r2"].mean()), 4))
    print("Minimum R2:", round(float(fit_all["r2"].min()), 4))
    if errors:
        print("WARNING: Some runs failed. See audit_summary.json.")
    elif len(run_dirs) != expected_run_count:
        print(f"WARNING: expected {expected_run_count} runs, found {len(run_dirs)}.")
    else:
        print("All 15 decomposition runs were summarized successfully.")


if __name__ == "__main__":
    main()


"""
python .\scripts\04_summarize_decomposition.py
"""