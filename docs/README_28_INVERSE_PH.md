# Inverse pH prediction (script 28)

## Question

Script 27 showed that phone-specific GAM curves (M2) reduce the forward
colour prediction error of unseen strips by about 1.1 Delta E76 at pH 2-12.
Does that gain translate into better pH readings?

## Files

- `scripts/28_gam_inverse_ph_prediction.R`: LOSO fits and inverse prediction
- `scripts/28_summarize_inverse_ph.py`: metrics, paired bootstrap, figures
- `run_gam_inverse_ph_prediction.ps1`: runs both (about 6 minutes)

## Procedure

Same 39 strip folds and models as script 27. For each held-out image, the
estimated pH is the value on a grid (1-13, step 0.05) whose predicted Lab
for the same phone and light is closest in Delta E76 (strip effect = 0).

| model | calibration scenario |
|---|---|
| M0 | one curve for all phones (phone unknown) |
| M1 | shared curve shape + phone offset (phone known) |
| M2 | phone-specific curves (phone known) |

Metrics per strip (mean over 5 lights): |pH error|, exact accuracy
(rounded estimate = true pH), within-1 accuracy, acidic/alkaline accuracy
(pH 7 excluded). Bootstrap as in scripts 19 and 27.
Background is a negative control.

## Results (Contrast, tp k = 10)

| model | pH 2-12 mean abs error | exact | within 1 | acidic/alkaline |
|---|---|---|---|---|
| M0 | 0.58 | 52% | 86% | 100% |
| M1 | 0.39 | 76% | 97% | 100% |
| M2 | 0.36 | 81% | 94% | 100% |

Paired strip differences, pH 2-12, mean |pH error| [95% CI]:

| comparison | Contrast | Pad |
|---|---|---|
| M1 - M0 (phone offset) | -0.20 [-0.31, -0.07] | -0.27 [-0.40, -0.14] |
| M2 - M1 (phone-specific curves) | -0.03 [-0.10, +0.05] | -0.03 [-0.11, +0.04] |

M2 - M1 is not significant under any of the five GAM specifications
(range -0.015 to -0.038 pH). At pH 1/13 M2 is worse for Contrast
(exact accuracy 100% -> 73%).

Background (negative control): mean errors of 3-8 pH units, as expected.

## Interpretation

1. The phone offset (M1) carries most of the practical benefit of device
   calibration: about 0.2 pH less error and +24 percentage points exact
   accuracy over a shared curve.
2. Phone-specific curve shape (M2) improves colour prediction (script 27) but
   not pH reading. The colour gradient along the pH trajectory is large
   (median 9.7 Delta E per pH unit at pH 2-12), so a 1.1 Delta E colour gain
   could shift pH by at most about 0.1, and only its component along the
   trajectory matters. This is consistent with ASCA: the pH x phone direction
   differs from the pH direction by about 58 degrees.
3. Per-strip colour gain and pH gain are positively related (r = 0.66), but
   the pH gains are small.
4. Errors peak at pH 11-12 for every model, where the colour gradient is
   smallest (about 6 Delta E per pH unit).
5. Acidic/alkaline classification is perfect for every phone-aware model, so
   the binary task used in the augmentation experiments could not show any
   benefit of phone-specific structure.

Statistical vs practical: the phone-specific structure is real and
reproducible as colour structure, but for pH reading with this strip its
practical impact is small compared with the phone offset.
