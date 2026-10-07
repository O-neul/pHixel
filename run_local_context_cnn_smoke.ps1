$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Step 1. Generate all 195 local-context inputs with fixed white padding.
# The previous audit_grid output contains only pH 2, 6, 8, 9, and 12, so it
# must not be used as the training manifest.
# ---------------------------------------------------------------------------

python .\scripts\09_make_local_context_inputs.py `
  --manifest ".\data\processed\capture_manifest.csv" `
  --annotation-dir ".\data\raw\captures\annotations" `
  --output-dir ".\outputs\local_context_all_whitepad" `
  --roi-label "roi" `
  --bg-side "left" `
  --bg-width-ratio 1.0 `
  --bg-height-ratio 1.0 `
  --bg-gap-ratio 0.25 `
  --context-margin-ratio 0.15 `
  --image-size 224 `
  --resize-mode "letterbox" `
  --letterbox-fill "white" `
  --selection "all" `
  --seed 42

if ($LASTEXITCODE -ne 0) {
    throw "Full local-context generation failed."
}

$manifestRows = (
    Import-Csv `
      ".\outputs\local_context_all_whitepad\local_context_manifest.csv"
).Count

if ($manifestRows -ne 195) {
    throw "Expected 195 local-context rows, found $manifestRows."
}

Write-Host ""
Write-Host "Full local-context rows: $manifestRows"

# ---------------------------------------------------------------------------
# Step 2. Run one development/smoke split.
# ---------------------------------------------------------------------------

$runName = "capture_lopo_holdout_galaxy_s24plus_vallight_L01"

python .\scripts\10_train_local_context_binary.py `
  --local-context-manifest `
    ".\outputs\local_context_all_whitepad\local_context_manifest.csv" `
  --split-file ".\data\splits\$runName.csv" `
  --decomposition-dir ".\outputs\decomposition_binary\$runName" `
  --output-dir `
    ".\outputs\local_context_cnn_smoke\${runName}_seed42" `
  --methods "baseline,color_jitter,nuisance_structured" `
  --epochs 12 `
  --batch-size 8 `
  --learning-rate 0.001 `
  --weight-decay 0.0001 `
  --patience 4 `
  --num-workers 0 `
  --device auto `
  --image-size 224 `
  --jitter-brightness 0.10 `
  --jitter-contrast 0.10 `
  --jitter-saturation 0.10 `
  --jitter-hue 0.02 `
  --nuisance-probability 0.80 `
  --phone-scale 1.0 `
  --light-scale 1.0 `
  --interaction-scale 0.5 `
  --residual-scale 0.0 `
  --overall-scale 1.0 `
  --feather-radius 24 `
  --seed 42

if ($LASTEXITCODE -ne 0) {
    throw "Local-context CNN smoke test failed."
}

Write-Host ""
Write-Host "Test comparison"
Import-Csv `
  ".\outputs\local_context_cnn_smoke\${runName}_seed42\test_comparison.csv" |
Select-Object `
  method,
  best_epoch,
  best_val_balanced_accuracy,
  test_accuracy,
  test_balanced_accuracy,
  test_macro_f1,
  ph6_accuracy,
  ph8_accuracy,
  ph9_accuracy,
  test_single_class_prediction |
Format-Table -AutoSize
