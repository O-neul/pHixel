# Fixed runners for the hybrid augmentation pilot

These runners explicitly use the complete 195-image local-context manifest:

```text
outputs/local_context_all_whitepad/local_context_manifest.csv
```

and the model-ready image path column:

```text
model_input_path
```

This prevents script 23 from auto-selecting the 75-image audit/subset manifest.

## 1. Dry run first

Copy:

```text
run_hybrid_pilot_dry_FULL195.ps1
```

to the project root and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_hybrid_pilot_dry_FULL195.ps1
```

Expected binary fold sizes after pH 7 is excluded:

```text
train = 96
val   = 24
test  = 60
```

for all nine folds.

## 2. Full pilot

Only after the dry run shows 96 / 24 / 60 for every fold:

```powershell
.\run_hybrid_pilot_FULL195.ps1
```

The augmentation settings remain fixed:

```text
probability              0.70
background strength      0.50
global contrast strength 0.50
trajectory strength      0.50
trajectory pH range      2-12
```

Do not tune these after viewing the test results.
