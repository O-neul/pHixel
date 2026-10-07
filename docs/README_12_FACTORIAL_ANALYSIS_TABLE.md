# 12. Final factorial analysis table and design audit

This script creates the fixed analysis data set for the statistical part of
the study.

## Planned design

```text
3 phones × 13 pH levels × 5 lights = 195 images
```

Expected repeated structure:

```text
39 phone × pH strips
each strip photographed under 5 lights
```

## Run

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_factorial_analysis_table.ps1
```

## Input

The default input is:

```text
data/processed/capture_manifest.csv
```

The script automatically recognizes common aliases such as:

```text
image_id or capture_id
phone or phone_id
light or light_id
ph_nominal or ph_value
```

For Pad and Background colors, saved Lab columns are preferred. When complete
Lab columns are unavailable, mean RGB columns are converted to CIE Lab under
D65. The fallback is recorded in `design_audit.json`.

## Main outputs

```text
outputs/factorial_analysis/
├─ analysis_factorial_table.csv
├─ analysis_factorial_table_long.csv
├─ design_cell_counts.csv
├─ phone_ph_strip_audit.csv
├─ strip_repeat_audit.csv
├─ factor_level_counts.csv
├─ feature_quality_summary.csv
├─ design_term_df.csv
├─ design_audit_items.csv
├─ design_audit.json
└─ data_dictionary.csv
```

### `analysis_factorial_table.csv`

One row per image, including:

```text
image_id
strip_id
phone
light
ph_nominal
pad_L, pad_a, pad_b
bg_L, bg_a, bg_b
contrast_L, contrast_a, contrast_b
pad_deltaE_from_bg
```

### `analysis_factorial_table_long.csv`

Three rows per image:

```text
Pad
Background
Contrast
```

This is convenient for profile plots and multivariate analysis.

## Audits

The script checks:

- exactly 195 rows
- unique image identifiers
- 3 phone levels
- 5 light levels
- pH 1 through 13
- exactly one observation in each phone × pH × light cell
- one strip in each phone × pH cell
- five light repeats for every strip
- plausible Lab ranges
- full-rank design matrix for:

```text
pH + Phone + Light
+ pH:Phone
+ pH:Light
+ Phone:Light
```

For the expected balanced design, the model matrix has:

```text
99 estimable parameters including the intercept
96 residual degrees of freedom
```

## Interpretation limitations recorded by the script

- Each phone × pH cell has only one physical strip. The observed pH × phone
  profile therefore cannot be separated from strip-specific variation in that
  cell.
- There is one image in each phone × pH × light cell. Three-way interaction
  and pure measurement error cannot be separately estimated.
- Lab computed from mean RGB is an approximation to mean pixelwise Lab.

Do not proceed to profile plots or ANOVA until the overall audit status is
`PASS` or any warning has been understood.
