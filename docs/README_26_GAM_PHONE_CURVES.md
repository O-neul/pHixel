# GAM phone-specific pH curves (script 26)

## Files

- `scripts/26_gam_phone_specific_curves.R`: model fitting (R, mgcv)
- `scripts/26_plot_gam_curves.py`: figures (matplotlib)
- `run_gam_phone_specific_curves.ps1`: runs both

Requirements: R with `mgcv` and `MASS` (both recommended packages);
Python with numpy, pandas, matplotlib.

## Why GAM after ASCA

pH x phone has zero strip-level df in ASCA (one strip per phone x pH), so the
phone-specific response cannot be separated from strip-to-strip variation
(script 25). Assuming the phone-specific response is smooth in pH, the GAM
splits the pH x phone deviation into a smooth phone curve and non-smooth strip
deviations (strip random effect).

## Pre-specified models

Common terms: `light`, sum-coded `phone:light` (orthogonal to the main
effects), `s(strip_id, bs = "re")`; `method = "REML"`.

| model | added term |
|---|---|
| M0 | `s(ph, k)` |
| M1 | M0 + `phone` |
| M2 | M1 + `s(ph, by = ordered phone, k)` |

Primary: `tp`, k = 10, Contrast. Sensitivity: `tp` k = 6, 8, 12; `cr` k = 10;
Pad and Background.

Population curves exclude the strip effect and average over the five lights.
Pointwise and simultaneous (posterior simulation, 10000 draws) 95% bands.

Difference curves are reported in two components:

- `total`: phone offset + pH-dependent part
- `shape`: pH-dependent part only. This is identified up to a constant
  (centred smooth), so its band pinches at the centring point. Read it as
  differences in slope/curvature, not as level differences at a given pH.

## Key results (Contrast, primary spec)

By-smooth tests in M2 (reference phone: Galaxy S24+):

| channel | iPhone 13 Pro curve (edf, p) | iPhone 16 Pro curve (edf, p) |
|---|---|---|
| L* | 1.39, 0.0013 | 3.25, 0.0037 |
| a* | 1.00, < 0.001 | 1.66, 0.56 |
| b* | 1.00, 0.32 | 3.61, 0.0028 |

Strip random-effect variance absorbed by phone-specific curves (M1 -> M2):
L* 39%, a* 66%, b* 39% (per channel across k/basis: L* 36-39%, a* 58-69%, b* 34-41%).

pH intervals where the simultaneous band of the shape difference excludes 0:

- iPhone 13 Pro - Galaxy, a*: slope difference over the whole range
- iPhone 13 Pro - Galaxy, L*: pH 2.7-4.1 (+), 10.3-12.2 (-)
- iPhone 16 Pro - iPhone 13 Pro, a*: pH 1.0-5.2 (+), 9.2-13.0 (-)
- iPhone 16 Pro - iPhone 13 Pro, b*: pH 4.8-6.4 (-)

Shape differences are nearly identical across k = 6-12 and tp/cr bases
(`figures/k_sensitivity_shape_contrast.png`), unlike the knot sensitivity of
the RCS strip-holdout result (script 19).

## Caveats

- AIC differences between M0-M2 are near zero by construction: the strip
  random effect can absorb any strip mean. Use the by-smooth tests, variance
  reduction and strip-holdout prediction (script 27) instead.
- Background strip variances are near zero (b*: ~1e-7), so the percentage
  reductions for Background are numerically unstable.
- Background also shows significant pH-dependent iPhone 16 Pro differences in
  a* and b*. The background ROI does not overlap the strip, so pad-colour
  spill is unlikely; scene-dependent camera processing is the more plausible
  source (see `BACKGROUND_ROI_CHECK.md`).
- Curves describe observed phone-specific responses; with one strip per
  phone x pH they are not hardware-only effects.
