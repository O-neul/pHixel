# ASCA permutation inference (script 25)

## Why this analysis exists

Script 15 reported ASCA effect shares and PC1 loadings only. For a statistics
journal submission, formal inference is needed for each ASCA term.
Script 25 adds:

1. permutation p-values for all six terms;
2. Holm adjustment within each region x scaling;
3. residual-bootstrap intervals for the pH and pH x phone PC1 loadings;
4. the angle between the pH PC1 and pH x phone PC1 directions.

The decomposition is re-implemented and checked against script 15
(`multivariate_effect_shares.csv`); the maximum share difference is ~1e-16.

## Experimental units

```text
39 strips  = 3 phones x 13 pH          (whole-plot unit)
195 images = 39 strips x 5 lights      (within-strip unit)
```

- between-strip terms: pH, phone, pH x phone
- within-strip terms: light, pH x light, phone x light

## Statistic and permutation schemes

Pseudo-F on the SSCP trace, residual df = 96.

| term | restricted (primary) | unrestricted (secondary) |
|---|---|---|
| pH | whole strips permuted across pH within phone | rows permuted across all 195 |
| phone | whole strips permuted across phone within pH | rows permuted across all 195 |
| light | light labels permuted within strip | rows permuted across all 195 |
| two-way interactions | Freedman-Lane, reduced-model residuals | same |

The restricted scheme treats the strip, not the image, as the exchangeable
unit for between-strip main effects.

## Run

```powershell
.\run_asca_permutation_inference.ps1
```

Runtime is about one minute with 9999 permutations.

## Key results (seed 20261009)

| region (standardized) | term | share | restricted p | unrestricted p |
|---|---|---|---|---|
| Background | pH | 1.03% | **0.568** | 0.0001 |
| Background | phone | 60.9% | 0.0001 | 0.0001 |
| Background | pH x phone | 2.17% | 0.0001 | (same) |
| Contrast | pH | 86.7% | 0.0001 | 0.0001 |
| Contrast | pH x phone | 3.89% | 0.0001 | (same) |

- Background pH is not significant when strips are the exchangeable unit,
  but is "highly significant" when the 195 images are treated as independent.
  This is a direct demonstration of pseudoreplication in this design.
- Contrast pH PC1 vs pH x phone PC1 angle: 57.6 deg (95% CI 56.3-59.0,
  standardized) and 65.5 deg (64.0-67.0, raw). The two effects point in
  clearly different Lab directions under both scalings.
- The dominant channel of the pH x phone PC1 depends on scaling
  (standardized: L* 0.80; raw: b* 0.77). The scaling-robust feature is the
  negative a* loading (-0.50 / -0.58), opposite in sign to the pH PC1 a*
  loading (+0.38 / +0.49).

## Interpretation limits

- With one strip per phone x pH cell, pH x phone has zero df at the strip
  level. Its test compares strip-level deviations from additivity with
  within-strip (across-light) residual variation. Significance shows the
  pattern exceeds image-level noise; it cannot separate smartphone response
  from strip-to-strip variation.
- The residual combines the three-way interaction with measurement error.
- Bootstrap intervals are conditional on the observed strips and therefore
  understate uncertainty about new strips.
