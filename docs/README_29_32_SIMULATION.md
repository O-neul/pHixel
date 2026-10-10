# Simulation study (scripts 29-32)

Design: `docs/SIMULATION_DESIGN.md`. Run: `run_simulation.ps1` (about 1.5 hours).
All data are generated in the real 13 pH x 3 phone x 5 light layout with one
strip per phone x pH, calibrated to the real Contrast fit (script 26).

| script | content |
|---|---|
| `29_sim_permutation_unit.py` | S1: restricted vs image-level permutation tests |
| `30_sim_gam_identification.R` | S2: GAM identification of phone-specific curves; writes LOSO data sets |
| `31_sim_loso_gam.R`, `31_sim_loso_rcs.py` | S2-LOSO / S3: GAM and RCS (script 19 code) strip holdout on simulated data |
| `32_summarize_simulation.py` | S2 and S3 summaries and figures |

## S1: exchangeable unit (1000 data sets x 999 permutations per scenario)

r = strip SD / image SD. Real data: Background r = 0-0.95, Contrast r = 4.5-6.3.

pH main-effect test, no pH effect (type I error, nominal 0.05):

| r | 0 | 0.5 | 1 | 2 | 4 | 6 |
|---|---|---|---|---|---|---|
| strip-restricted | 0.047 | 0.051 | 0.058 | 0.053 | 0.058 | 0.046 |
| image-level | 0.066 | **0.749** | **0.999** | 1.000 | 1.000 | 1.000 |

- The image-level test is invalid as soon as strips vary at all; at the real
  Background r it rejects a true null 75-100% of the time.
- The strip-restricted test keeps the nominal level for every r. Its power
  falls as r grows (tau = 1: 0.93 at r = 1, 0.37 at r = 2), which is the
  honest cost of having only 13 strips per phone. The non-significant real
  Background pH effect (p = 0.57) therefore means "no detectable pH effect
  beyond strip variation", not proof of zero.

Freedman-Lane pH x phone test with no phone-specific curves:
rejection 0.060 at r = 0, 0.988 at r = 0.5, 1.000 for r >= 1.
At the real Contrast r (about 5), a significant ASCA pH x phone term is
guaranteed by non-smooth strip variation alone and says nothing about
phone-specific response. This justifies the GAM step.

## S2: GAM identification (a*, 500 data sets per cell)

A = phone-specific curve size (1 = real); m = strip SD multiplier.

| A | m | reject (M2, Holm over 2 phones) | reject (no strip RE) | strip var. absorbed | simult. coverage 13 Pro / 16 Pro |
|---|---|---|---|---|---|
| 0 | 0.5 / 1 / 2 | 0.10 / 0.11 / 0.11 | 0.98-1.00 | 6-8% | 0.97 / 0.97-0.98 |
| 0.25 | 1 | 0.35 | 1.00 | 17% | 0.97 / 0.36 |
| 0.5 | 1 | 0.85 | 1.00 | 36% | 0.96 / 0.34 |
| 1 | 1 | 1.00 | 1.00 | 68% | 0.97 / 0.41 |

L* and b* (A = 0 / 1, m = 1): Holm rejection 0.08 / 0.97 and 0.10 / 0.75.

Findings:
1. Omitting the strip random effect makes the phone-specific curve test
   reject almost always, even with no curves. The random effect is essential.
2. With the random effect, the approximate mgcv p-values are anti-conservative:
   about 0.10 per term (and 0.10 Holm-adjusted) instead of 0.05. The real-data
   by-smooth p-values (0.001-0.004) remain small, but should be reported as
   approximate, with this simulation as calibration.
3. Under no curves the strip-variance reduction is 6-8%; the real values
   (39-66%) are far above this null level and close to the A = 1 value (68%).
4. Simultaneous bands cover the truth (about 0.97) when the true curve is zero
   or simple (iPhone 13 Pro a*, nearly linear). For a small, wiggly true curve
   (iPhone 16 Pro a*, range 2 Lab units) coverage drops to 0.34-0.61 because
   penalization shrinks the curve (smoothing bias). Pointwise coverage
   0.82-0.96. "pH intervals where the band excludes zero" should be described
   as indicative for wiggly, small differences.

## S2-LOSO and S3: strip holdout on simulated data (200 data sets each)

Per-data-set mean strip Delta E76 difference, phone-specific minus additive
(5-95% range across data sets). Real-data values from scripts 27 and 19.

pH 2-12:

| model | A = 0 | A = 1 | real data |
|---|---|---|---|
| GAM M2 - M1 | +0.19 (-0.08, +0.40); CI < 0 in 0% | -1.03 (-1.73, -0.43); CI < 0 in 81% | -1.16 |
| RCS knots 3 / 4 / 5 | +0.55 / +0.41 / +0.62 | -0.23 / -0.72 / -0.56 | -0.18 / -1.13 / -0.85 |
| RCS ridge knots 3 / 4 / 5 | +0.02 / +0.04 / +0.03 | -0.26 / -0.69 / -0.68 | -0.13 / -0.79 / -0.35 |

pH 1, 13:

| model | A = 0 | A = 1 | real data |
|---|---|---|---|
| GAM M2 - M1 | +1.14 | -1.97 (-4.03, +0.93) | +0.64 |
| RCS knots 3 / 4 / 5 | +7.74 / +3.32 / +5.31 | +6.37 / +0.41 / +2.01 | +7.89 / +4.36 / +6.35 |
| RCS ridge knots 3 / 4 / 5 | +0.75 / +0.26 / +0.46 | +2.46 / -0.41 / -0.27 | +1.92 / +3.29 / +3.16 |

Findings:
1. No spurious gain: with no phone-specific curves the GAM never shows a
   significant interior gain (0% of data sets); the real interior gain (-1.16)
   lies far outside the A = 0 distribution and matches the A = 1 expectation
   (-1.03).
2. The RCS knot dependence seen in the real data is a property of the method:
   even under a smooth truth, unpenalized RCS with 3 knots fails at pH 1/13
   in every data set (+6.4), and the interior gain varies with the knot count.
3. The real boundary failures are larger than a smooth truth predicts for
   GAM (+0.64 vs -1.97, above the 95th percentile +0.93) and for RCS with
   4-5 knots and ridge. So the real boundary difficulty is partly
   methodological (RCS 3 knots) and partly a genuine non-smooth feature of the
   real data at pH 1/13 (largely the iPhone 16 Pro pH 13 strip).
