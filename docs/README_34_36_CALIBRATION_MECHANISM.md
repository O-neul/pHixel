# Which nuisance affects pH prediction, and how much calibration is needed (scripts 28, 34-36)

Run: `run_calibration_mechanism.ps1` (about 3 minutes, single core).
Main figure: `outputs/calibration_mechanism/figures/Fig_calibration_mechanism.png`.

## 1. Light unknown (script 28, `--light-mode unknown`)

The inverse pH prediction of script 28 was repeated with one calibration curve
averaged over the five lights. Results barely change (Contrast, pH 2-12):

| model | light known | light unknown |
|---|---|---|
| M0 shared curve | 0.583 | 0.585 |
| M1 + phone offset | 0.388 | 0.388 |
| M2 phone-specific curves | 0.361 | 0.367 |

The five lights of this experiment move the pH reading very little
(largest light component: 0.03-0.16 pH, script 34). Lighting is not a major
pH-relevant nuisance in this data set; this should be stated as a property of
the lighting range tested, not as a general claim.

## 2. pH-relevant nuisance (script 34)

A colour shift v at pH x moves the reading by (v . t_hat(x)) / s(x), where
t_hat is the unit tangent of the pH colour trajectory and s its speed
(Delta E per pH unit). Only the component along the trajectory matters.

Mean |pH shift| over pH 2-12 (Contrast; posterior median):

| component | Galaxy S24+ | iPhone 13 Pro | iPhone 16 Pro |
|---|---|---|---|
| phone offset | **0.63** | **0.52** | 0.12 |
| phone curve shape | 0.13 | 0.22 | 0.18 |
| light (largest) | 0.16 | 0.03 | 0.07 |
| strip noise (SD, all phones) | 0.30 | | |
| image noise (SD, all phones) | 0.08 | | |

- The phone offset is the largest pH-relevant nuisance for two of three
  phones: it is large and roughly aligned with the pH direction.
- Curve-shape differences are smaller and partly orthogonal to the pH
  direction.
- After the offset is removed, the dominant remaining error is non-smooth
  strip-to-strip variation (SD 0.30 pH), which device calibration cannot remove.
- Background subtraction reduces the offset effect (Pad offset: 0.84 / 0.45 /
  0.39 vs Contrast 0.63 / 0.52 / 0.12).

## 3. New-phone offset calibration (script 35)

Leave-one-phone-out: a GAM fitted on two phones gives the base curve; the
new phone's offset is estimated from k reference strips of known pH; the
remaining strips are read. Light unknown, pH 2-12:

| reference strips | Contrast mean abs pH error (10-90% over reference sets) | exact | share of full gain |
|---|---|---|---|
| 0 (no calibration) | 0.73 (0.55-0.88) | 43% | 0% |
| 1 | 0.56 (0.38-0.74) | 59% | 54% |
| 2 | 0.48 (0.34-0.62) | 65% | 77% |
| 3 | 0.45 (0.31-0.60) | 69% | 86% |
| 12 (all other strips) | 0.41 (0.33-0.48) | 74% | 100% |

- Two to three reference strips recover about 80-90% of the benefit of a
  full offset calibration.
- With one reference strip the choice matters: pH 12 is a poor reference
  (error 0.98), pH 10, 11, 13 are good (0.44-0.49).
- Without calibration Pad is worse than Contrast (0.85 vs 0.73); with
  calibration they converge. Background subtraction mainly helps when the
  new phone is uncalibrated.

## 4. Mechanism check

The pH shift predicted from colour geometry alone (script 34, full-data fit)
matches the bias observed when an uncalibrated new phone is read
(script 35, leave-one-phone-out):

| phone | predicted mean abs shift [95%] | observed mean bias, new phone |
|---|---|---|
| Galaxy S24+ | 0.63 [0.50, 0.77] | -0.57 |
| iPhone 13 Pro | 0.52 [0.39, 0.66] | +0.64 |
| iPhone 16 Pro | 0.12 [0.04, 0.24] | -0.12 |

## Caveats

- Offsets in script 35 are estimated from all five light images of each
  reference strip.
- Each reference strip carries its own non-smooth strip deviation; with one
  strip per phone x pH this cannot be separated from measurement of the
  offset. Results with independent replicate strips may differ.
- Three phones only: the leave-one-phone-out base curve is an average of two
  phones.
