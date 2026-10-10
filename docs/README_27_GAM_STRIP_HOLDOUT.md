# GAM leave-one-strip-out validation (script 27)

## Files

- `scripts/27_gam_leave_one_strip_out.R`: 39-fold LOSO fitting (R, mgcv)
- `scripts/27_summarize_gam_strip_holdout.py`: strip metrics, paired bootstrap,
  GAM-RCS consistency table, figures
- `run_gam_leave_one_strip_out.ps1`: runs both (about 6 minutes)

## Design

- Fold = one physical strip (5 light images); 39 folds.
- Models M0/M1/M2 identical to script 26; held-out strip predicted with the
  strip random effect set to zero.
- Specs: tp k = 10 (primary), tp k = 6, 8, 12, cr k = 10. 5265 fits, all
  converged.
- Metric and bootstrap identical to script 19: strip mean Delta E76 over
  5 lights; strips resampled within phone, 3000 reps, seed 42.
- Correspondence: M1 <-> rcs_additive, M2 <-> rcs_phone / rcs_phone_ridge.
  Mean Delta E76 for M1 (Contrast 6.51) is close to rcs_additive (6.37).

## Key results: M2 minus M1 (negative favours phone-specific curves)

Contrast, mean difference [95% CI], % strips improved:

| method | pH 2-12 | pH 3-11 | pH 1, 13 |
|---|---|---|---|
| GAM tp k=10 (primary) | -1.16 [-1.92, -0.33] 67% | -1.01 [-1.86, -0.08] 63% | +0.64 [-2.93, +4.21] 67% |
| GAM, all 5 specs | -1.04 to -1.16, every CI < 0 | -0.92 to -1.01, every CI < 0 | +0.44 to +0.99, every CI spans 0 |
| RCS ridge knots=4 (draft primary) | -0.79 [-1.42, -0.17] | -0.72 [-1.35, -0.06] | +3.29 [+1.15, +5.43] |
| RCS ridge knots=3 / 5 | CI spans 0 | CI spans 0 | +1.92 / +3.16, CI > 0 |
| RCS knots=3 / 4 / 5 | only knots=4 CI < 0 | only knots=4 CI < 0 | +7.89 / +4.36 / +6.35 |

Pad: GAM pH 2-12 -0.90 to -1.01 with every CI < 0; pH 3-11 CIs span 0.
Background: no gain (pH 2-12: +0.02 [-0.04, +0.08]).

## Interpretation

1. Interior pH: the phone-specific gain is reproduced under every GAM
   specification. Under RCS it appeared only at knots = 4. The GAM result
   removes the knot-choice dependence noted for the draft section 3.8.
2. Boundary pH: GAM does not show the significant boundary failure of RCS.
   4 of 6 boundary strips improve; the mean is driven by one strip
   (iPhone 16 Pro, pH 13: +17 Delta E76). Boundary strips are extrapolations
   for the held-out phone curve and should be reported separately.
3. iPhone 16 Pro at high pH (pH 11, 13) accounts for the largest failures.
4. Background shows no predictive gain, consistent with Background being
   mainly global nuisance.

## Figures (`outputs/gam_strip_holdout/figures/`)

- `gam_rcs_consistency_<region>.png`: forest plot, GAM specs vs RCS knots
- `difference_by_ph_<region>.png`: per-strip M2 - M1 by pH and phone
- `sorted_strip_differences_<region>.png`
