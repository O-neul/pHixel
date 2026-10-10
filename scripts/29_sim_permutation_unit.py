#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
29_sim_permutation_unit.py

Simulation S1 (docs/SIMULATION_DESIGN.md): does the choice of exchangeable
unit change the validity of ASCA permutation tests in this design?

Design (same as the real data)
------------------------------
13 pH x 3 phone x 5 light, one image per cell, one strip per phone x pH.

Data-generating model (3 Lab channels, calibrated to the real Contrast data)
---------------------------------------------------------------------------
    y = phone + light + phone:light            (real balanced effects)
      + tau * sigma_e * f(pH)                  (pH effect, f scaled to unit SD)
      + u_strip ~ N(0, (r * sigma_e)^2)        (non-smooth strip deviation)
      + e ~ N(0, sigma_e^2)                    (image noise)

sigma_e per channel = GAM M2 residual SD of the real Contrast data
(script 26). f(pH) = real Contrast pH main effect, scaled per channel to
unit SD across pH levels.

S1a: pH main-effect test
    r   in {0, 0.5, 1, 2, 4, 6}   (strip SD / image SD)
    tau in {0, 0.5, 1, 2}         (pH effect SD / image SD; 0 = null)
    tests: strip-restricted permutation (script 25 primary)
           image-level unrestricted permutation (common ASCA convention)

S1b: pH x phone test without any phone-specific pH curve
    y additionally contains the full real pH effect; no pH x phone term.
    r in {0, 0.5, 1, 2, 4, 6}
    test: Freedman-Lane (script 25)
    A rejection here shows that a significant ASCA pH x phone term can arise
    from non-smooth strip deviations alone.

Statistic: trace pseudo-F on channel-standardized data, as in script 25.
Each scenario: n_datasets data sets x n_permutations permutations.
Rejection rate at alpha = 0.05 with Monte Carlo SE sqrt(p(1-p)/n).

Outputs
-------
outputs/simulation/s1/
    s1_rejection_rates.csv
    s1_parameters.json
    s1_notes.txt
    figures/s1_rejection_rates.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ["L", "a", "b"]
DF = {"pH": 12, "pH_x_phone": 24, "residual": 96}
R_LEVELS = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0]
TAU_LEVELS = [0.0, 0.5, 1.0, 2.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulation S1.")
    parser.add_argument(
        "--input",
        default="outputs/factorial_analysis/analysis_factorial_table.csv",
    )
    parser.add_argument("--gam-dir", default="outputs/gam_curves")
    parser.add_argument("--output-dir", default="outputs/simulation/s1")
    parser.add_argument("--n-datasets", type=int, default=1000)
    parser.add_argument("--n-permutations", type=int, default=999)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def real_array(frame: pd.DataFrame, region: str) -> np.ndarray:
    ph_levels = sorted(frame["ph_nominal"].unique())
    phones = sorted(frame["phone"].unique())
    lights = sorted(frame["light"].unique())
    indexed = frame.set_index(["ph_nominal", "phone", "light"])
    columns = [f"{region}_{c}" for c in CHANNELS]
    values = np.empty((len(ph_levels), len(phones), len(lights), 3))

    for i, ph in enumerate(ph_levels):
        for j, phone in enumerate(phones):
            for k, light in enumerate(lights):
                values[i, j, k] = indexed.loc[(ph, phone, light), columns].to_numpy(float)

    return values


def balanced_effects(values: np.ndarray) -> dict[str, np.ndarray]:
    """Effects of a single (a, b, c, p) array, broadcast to full shape."""
    grand = values.mean(axis=(0, 1, 2))
    m_a = values.mean(axis=(1, 2))
    m_b = values.mean(axis=(0, 2))
    m_c = values.mean(axis=(0, 1))
    m_bc = values.mean(axis=0)
    shape = values.shape

    return {
        "pH": np.broadcast_to((m_a - grand)[:, None, None, :], shape),
        "phone": np.broadcast_to((m_b - grand)[None, :, None, :], shape),
        "light": np.broadcast_to((m_c - grand)[None, None, :, :], shape),
        "phone_x_light": np.broadcast_to(
            (m_bc - m_b[:, None, :] - m_c[None, :, :] + grand)[None, :, :, :], shape
        ),
    }


def batch_ss(values: np.ndarray) -> dict[str, np.ndarray]:
    """
    Sums of squares for a batch of balanced arrays, shape (B, a, b, c, p).
    Returns arrays of length B.
    """
    _, a, b, c, _ = values.shape
    grand = values.mean(axis=(1, 2, 3), keepdims=True)
    m_a = values.mean(axis=(2, 3), keepdims=True)
    m_b = values.mean(axis=(1, 3), keepdims=True)
    m_c = values.mean(axis=(1, 2), keepdims=True)
    m_ab = values.mean(axis=3, keepdims=True)
    m_ac = values.mean(axis=2, keepdims=True)
    m_bc = values.mean(axis=1, keepdims=True)

    residual = values - m_ab - m_ac - m_bc + m_a + m_b + m_c - grand
    effect_ab = m_ab - m_a - m_b + grand

    return {
        "pH": b * c * np.sum(np.square(m_a - grand), axis=(1, 2, 3, 4)),
        "pH_x_phone": c * np.sum(np.square(effect_ab), axis=(1, 2, 3, 4)),
        "residual": np.sum(np.square(residual), axis=(1, 2, 3, 4)),
    }


def pseudo_f(ss: dict[str, np.ndarray], term: str) -> np.ndarray:
    return (ss[term] / DF[term]) / (ss["residual"] / DF["residual"])


def standardize(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1, values.shape[-1])
    return (values - flat.mean(axis=0)) / flat.std(axis=0, ddof=1)


def restricted_ph_permutations(
    values: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Whole strips permuted across pH within each phone; shape (n, a, b, c, p)."""
    a, b, c, p = values.shape
    order = np.argsort(rng.random((n, b, a)), axis=-1)       # (n, b, a)
    index = np.transpose(order, (0, 2, 1))                   # (n, a, b)
    phone_index = np.broadcast_to(np.arange(b)[None, None, :], index.shape)
    return values[index, phone_index]                        # (n, a, b, c, p)


def unrestricted_permutations(
    values: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    flat = values.reshape(-1, values.shape[-1])
    order = np.argsort(rng.random((n, flat.shape[0])), axis=-1)
    return flat[order].reshape((n,) + values.shape)


def freedman_lane_permutations(
    values: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Reduced model = all terms except pH x phone (balanced, orthogonal)."""
    grand = values.mean(axis=(0, 1, 2), keepdims=True)
    m_a = values.mean(axis=(1, 2), keepdims=True)
    m_b = values.mean(axis=(0, 2), keepdims=True)
    m_c = values.mean(axis=(0, 1), keepdims=True)
    m_ab = values.mean(axis=2, keepdims=True)
    m_ac = values.mean(axis=1, keepdims=True)
    m_bc = values.mean(axis=0, keepdims=True)

    residual = values - m_ab - m_ac - m_bc + m_a + m_b + m_c - grand
    effect_ab = m_ab - m_a - m_b + grand
    reduced_residual = effect_ab + residual
    reduced_fitted = values - reduced_residual

    permuted = unrestricted_permutations(reduced_residual, n, rng)
    return reduced_fitted[None] + permuted


def p_value(observed: float, null: np.ndarray) -> float:
    return (1 + int(np.sum(null >= observed - 1e-12))) / (len(null) + 1)


def simulate_dataset(
    base: np.ndarray,
    ph_shape: np.ndarray,
    sigma_e: np.ndarray,
    tau: float,
    r: float,
    rng: np.random.Generator,
) -> np.ndarray:
    a, b, c, p = base.shape
    strip = rng.normal(size=(a, b, 1, p)) * (r * sigma_e)
    noise = rng.normal(size=(a, b, c, p)) * sigma_e
    ph_effect = tau * sigma_e * ph_shape[:, None, None, :]
    return base + ph_effect + strip + noise


def run_scenario(
    base: np.ndarray,
    ph_shape: np.ndarray,
    sigma_e: np.ndarray,
    tau: float,
    r: float,
    tests: list[str],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p_values = {test: np.empty(args.n_datasets) for test in tests}

    for index in range(args.n_datasets):
        values = standardize(simulate_dataset(base, ph_shape, sigma_e, tau, r, rng))
        observed = batch_ss(values[None])

        if "restricted_pH" in tests:
            null = pseudo_f(
                batch_ss(restricted_ph_permutations(values, args.n_permutations, rng)), "pH"
            )
            p_values["restricted_pH"][index] = p_value(pseudo_f(observed, "pH")[0], null)

        if "unrestricted_pH" in tests:
            null = pseudo_f(
                batch_ss(unrestricted_permutations(values, args.n_permutations, rng)), "pH"
            )
            p_values["unrestricted_pH"][index] = p_value(pseudo_f(observed, "pH")[0], null)

        if "freedman_lane_pH_x_phone" in tests:
            null = pseudo_f(
                batch_ss(freedman_lane_permutations(values, args.n_permutations, rng)),
                "pH_x_phone",
            )
            p_values["freedman_lane_pH_x_phone"][index] = p_value(
                pseudo_f(observed, "pH_x_phone")[0], null
            )

    return p_values


def reference_ratios(frame: pd.DataFrame, gam_dir: Path) -> dict[str, dict[str, float]]:
    """Where the real Background and Contrast data sit on the (r, tau) grid."""
    comparison = pd.read_csv(gam_dir / "model_comparison.csv")
    comparison = comparison[comparison["primary_spec"].astype(str).str.upper() == "TRUE"]
    result = {}

    for region in ["bg", "contrast"]:
        values = real_array(frame, region)
        ph_sd = balanced_effects(values)["pH"][:, 0, 0, :].std(axis=0, ddof=1)

        for channel_index, channel in enumerate(CHANNELS):
            rows = comparison[(comparison["region"] == region) & (comparison["channel"] == channel)]
            sigma_e = float(rows.loc[rows["model"] == "M1", "residual_sd"].iloc[0])
            strip_sd = float(rows.loc[rows["model"] == "M1", "strip_sd"].iloc[0])
            result[f"{region}_{channel}"] = {
                "r_strip_over_image_M1": strip_sd / sigma_e,
                "tau_pH_over_image": float(ph_sd[channel_index]) / sigma_e,
            }

    return result


def plot_results(rates: pd.DataFrame, reference: dict, path: Path, dpi: int, alpha: float) -> None:
    figure, axes = plt.subplots(1, len(TAU_LEVELS) + 1, figsize=(17.0, 4.0), sharey=True)
    styles = {
        "restricted_pH": ("#0072B2", "o", "strip-restricted"),
        "unrestricted_pH": ("#D55E00", "s", "image-level (unrestricted)"),
    }
    bg_r = np.mean([reference[f"bg_{c}"]["r_strip_over_image_M1"] for c in CHANNELS])

    for axis, tau in zip(axes, TAU_LEVELS):
        part = rates[(rates["study"] == "S1a") & np.isclose(rates["tau"], tau)]
        for test, (color, marker, label) in styles.items():
            rows = part[part["test"] == test].sort_values("r")
            axis.errorbar(rows["r"], rows["rejection_rate"], yerr=1.96 * rows["mc_se"],
                          color=color, marker=marker, capsize=2, label=label)
        axis.axhline(alpha, color="#888888", linestyle="--", linewidth=0.8)
        axis.axvline(bg_r, color="#009E73", linestyle=":", linewidth=1.0)
        axis.set_title("H0: no pH effect" if tau == 0 else f"pH effect: tau = {tau:g}", fontsize=9)
        axis.set_xlabel("strip SD / image SD (r)")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)

    part = rates[rates["study"] == "S1b"].sort_values("r")
    axes[-1].errorbar(part["r"], part["rejection_rate"], yerr=1.96 * part["mc_se"],
                      color="#000000", marker="^", capsize=2)
    axes[-1].axhline(alpha, color="#888888", linestyle="--", linewidth=0.8)
    axes[-1].set_title("pH x phone test, no phone-specific curve", fontsize=9)
    axes[-1].set_xlabel("strip SD / image SD (r)")
    axes[-1].grid(alpha=0.25)

    axes[0].set_ylabel("rejection rate (alpha = 0.05)")
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "S1: permutation tests in the 13 x 3 x 5 design with one strip per phone x pH "
        "(green dotted = real Background r; bars = 95% Monte Carlo interval)",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(resolve(args.input), encoding="utf-8-sig")
    contrast = real_array(frame, "contrast")
    effects = balanced_effects(contrast)

    comparison = pd.read_csv(resolve(args.gam_dir) / "model_comparison.csv")
    comparison = comparison[
        (comparison["primary_spec"].astype(str).str.upper() == "TRUE")
        & (comparison["region"] == "contrast")
        & (comparison["model"] == "M2")
    ].set_index("channel")
    sigma_e = comparison.loc[CHANNELS, "residual_sd"].to_numpy(float)

    grand = contrast.mean(axis=(0, 1, 2))
    base = grand + effects["phone"] + effects["light"] + effects["phone_x_light"]
    ph_effect = effects["pH"][:, 0, 0, :]
    ph_shape = ph_effect / ph_effect.std(axis=0, ddof=1)

    reference = reference_ratios(frame, resolve(args.gam_dir))
    rows = []
    started = time.time()
    scenario = 0

    for r in R_LEVELS:
        for tau in TAU_LEVELS:
            scenario += 1
            p_values = run_scenario(base, ph_shape, sigma_e, tau, r,
                                    ["restricted_pH", "unrestricted_pH"], args,
                                    args.seed + scenario)
            for test, values in p_values.items():
                rate = float(np.mean(values < args.alpha))
                rows.append({"study": "S1a", "r": r, "tau": tau, "test": test,
                             "rejection_rate": rate,
                             "mc_se": float(np.sqrt(rate * (1 - rate) / len(values))),
                             "n_datasets": len(values)})
            print(f"S1a r={r:g} tau={tau:g} done ({time.time() - started:.0f}s)", flush=True)

    # S1b: real pH effect (full size) present, no pH x phone term.
    full_ph = ph_effect / sigma_e  # tau * sigma_e * shape == real effect when tau=1
    for r in R_LEVELS:
        scenario += 1
        p_values = run_scenario(base, full_ph, sigma_e, 1.0, r,
                                ["freedman_lane_pH_x_phone"], args, args.seed + scenario)
        values = p_values["freedman_lane_pH_x_phone"]
        rate = float(np.mean(values < args.alpha))
        rows.append({"study": "S1b", "r": r, "tau": np.nan, "test": "freedman_lane_pH_x_phone",
                     "rejection_rate": rate,
                     "mc_se": float(np.sqrt(rate * (1 - rate) / len(values))),
                     "n_datasets": len(values)})
        print(f"S1b r={r:g} done ({time.time() - started:.0f}s)", flush=True)

    rates = pd.DataFrame(rows)
    rates.to_csv(output_dir / "s1_rejection_rates.csv", index=False, encoding="utf-8-sig")

    parameters = {
        "n_datasets": args.n_datasets,
        "n_permutations": args.n_permutations,
        "alpha": args.alpha,
        "seed": args.seed,
        "sigma_e_contrast_M2": dict(zip(CHANNELS, sigma_e.tolist())),
        "r_levels": R_LEVELS,
        "tau_levels": TAU_LEVELS,
        "real_data_reference": reference,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (output_dir / "s1_parameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )
    plot_results(rates, reference, output_dir / "figures" / "s1_rejection_rates.png",
                 args.dpi, args.alpha)

    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        wide = rates[rates["study"] == "S1a"].pivot_table(
            index=["tau", "r"], columns="test", values="rejection_rate")
        notes = [
            "Simulation S1: exchangeable unit and permutation validity",
            "=" * 80,
            f"datasets per scenario: {args.n_datasets}, permutations: {args.n_permutations}",
            "",
            "S1a: pH main-effect rejection rates (tau = 0 is the null)",
            wide.to_string(),
            "",
            "S1b: Freedman-Lane pH x phone rejection without phone-specific curves",
            rates[rates["study"] == "S1b"][["r", "rejection_rate", "mc_se"]].to_string(index=False),
            "",
            "Real-data reference (r = strip SD / image SD from GAM M1; tau = pH SD / image SD):",
            json.dumps(reference, indent=2),
        ]
    (output_dir / "s1_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
