# 13. Color profiles and interaction diagnostics

This step is descriptive. Formal ANOVA and effect-size calculations are
reserved for script 14.

## Run

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_color_profiles.ps1
```

## Input

```text
outputs/factorial_analysis/analysis_factorial_table.csv
```

## Main figures

```text
outputs/color_profiles/figures/
├─ phone_profiles/
├─ light_profiles/
├─ phone_interaction_residuals/
├─ light_interaction_residuals/
├─ interaction_strength/
└─ lab_separation/
```

Every figure is saved separately. No multi-panel subplots are used.

## 1. Phone profiles

For every Pad, Background, and Contrast Lab component:

```text
x-axis: nominal pH
line: phone
point: mean across five lights
error bar: observed minimum-to-maximum across lights
```

The default error display is descriptive range rather than a confidence
interval. Each phone × pH cell uses one physical strip observed under five
lights, so the five observations are repeated lighting conditions rather than
five independent strips.

## 2. Light profiles

```text
x-axis: nominal pH
line: light
point: mean across three phones
error bar: observed minimum-to-maximum across phones
```

These plots are used to inspect whether lighting effects are approximately
parallel across pH.

## 3. Interaction residual profiles

For phone profiles, the script computes:

```text
cell mean(phone, pH)
- phone mean
- pH mean
+ grand mean
```

The same calculation is made for light profiles.

A curve near zero indicates a profile compatible with an additive main-effect
description. Large pH-specific deviations and crossing curves indicate a
possible pH × phone or pH × light interaction.

These are descriptive residuals, not p-values.

## 4. Lab separation

At each pH, the script calculates Pad, Background, and Contrast Lab centroids
for each phone and each light.

Outputs include:

- pairwise CIE76 ΔE
- RMS distance to the pH-specific grand centroid
- mean pairwise ΔE
- maximum pairwise ΔE

These tables help determine whether pH 8 or pH 12 has unusually large
phone-to-phone separation.

## Main tables

```text
outputs/color_profiles/tables/
├─ phone_profile_summary.csv
├─ light_profile_summary.csv
├─ phone_interaction_residuals.csv
├─ light_interaction_residuals.csv
├─ interaction_strength_by_ph.csv
├─ lab_centroids_by_ph.csv
├─ pairwise_deltaE_by_ph.csv
└─ domain_dispersion_by_ph.csv
```

## Suggested inspection order

1. `phone_profiles`
2. `phone_interaction_residuals`
3. `lab_separation`
4. `light_profiles`
5. `light_interaction_residuals`

Pay special attention to:

- whether phone curves are parallel
- whether Contrast reduces phone separation relative to Pad
- whether pH 8 and pH 12 are unusually separated
- whether the largest interaction residuals align with known LOPO errors

## Optional settings

Use standard deviation instead of observed range:

```powershell
--uncertainty sd
```

Use standard error:

```powershell
--uncertainty sem
```

Also save PDF figures:

```powershell
--save-pdf
```

Change highlighted pH values:

```powershell
--highlight-ph "7,8,12"
```
