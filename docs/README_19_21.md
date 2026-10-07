# Remaining analyses: 19-21

These scripts complete the current statistical analysis before deciding whether
to rerun empirical augmentation.

## 19. Boundary-stabilized strip-holdout models

Purpose:

```text
Does restricted/natural cubic spline behavior + ridge shrinkage reduce the
pH 1 and 13 failures while retaining any interior-pH advantage?
```

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_boundary_stabilized_models.ps1
```

Primary comparison:

```text
rcs_phone_ridge - rcs_additive
```

Negative values favor the phone-specific ridge model.

Most important file:

```text
outputs/boundary_stabilized_models/subset_comparisons.csv
```

Check these subsets:

```text
all_ph
boundary_ph_1_13
interior_ph_2_12
strict_interior_ph_3_11
```

Lambda is selected strictly inside each outer training fold using inner
strip-level cross-validation. Test-fold outcomes are not used for tuning.

## 20. ASCA interaction interpretation

Purpose:

```text
Which Lab directions dominate the observed pH x phone interaction, and at
which pH values is that interaction largest for each phone?
```

Run:

```powershell
.\run_asca_interaction_summary.ps1
```

Important files:

```text
outputs/asca_interaction_summary/loading_summary.csv
outputs/asca_interaction_summary/phone_peak_interactions.csv
outputs/asca_interaction_summary/ph_interaction_magnitude.csv
```

ASCA loading signs are arbitrary. Interpret relative magnitudes, not the sign.

## 21. Final evidence synthesis

Run only after 19 and 20 finish:

```powershell
.\run_final_evidence_synthesis.ps1
```

It combines analyses 14, 15, 16, 18, 19, and 20.

Important files:

```text
outputs/final_evidence_synthesis/region_evidence_matrix.csv
outputs/final_evidence_synthesis/evidence_summary.csv
outputs/final_evidence_synthesis/decision_questions.csv
outputs/final_evidence_synthesis/final_evidence_notes.txt
```

The script intentionally does not make the scientific decision automatically.

A new pH-conditioned augmentation experiment is most justified if:

```text
1. ridge-stabilized phone RCS improves Pad/Contrast at interior pH;
2. the strip-level bootstrap evidence is reasonably supportive;
3. pH 1 and 13 are substantially stabilized;
4. the redesigned experiment separates strip variation from phone variation;
5. augmentation parameters are estimated from training folds only.
```

If the categorical pH x phone model improves only when another image from the
same strip remains in training, but the ridge RCS does not help new strips,
then a new pH-conditioned augmentation run is not well supported by the
current data.

A background/global-nuisance augmentation pilot may still be considered
separately because Background showed a much more additive structure.
