$ErrorActionPreference = "Stop"

$localManifest = `
  ".\outputs\local_context_all_whitepad\local_context_manifest.csv"

if (-not (Test-Path $localManifest)) {
    throw "Missing full local-context manifest: $localManifest"
}

$manifestRows = (Import-Csv $localManifest).Count
if ($manifestRows -ne 195) {
    throw "Expected 195 local-context rows, found $manifestRows."
}

$runName = "capture_lopo_holdout_galaxy_s24plus_vallight_L01"

python .\scripts\11_train_local_context_binary_layer4.py `
  --local-context-manifest $localManifest `
  --split-file ".\data\splits\$runName.csv" `
  --decomposition-dir ".\outputs\decomposition_binary\$runName" `
  --output-dir `
    ".\outputs\local_context_cnn_layer4_smoke\${runName}_seed42" `
  --methods "baseline,color_jitter,nuisance_structured" `
  --warmup-epochs 5 `
  --finetune-epochs 15 `
  --head-learning-rate 0.001 `
  --layer4-learning-rate 0.0001 `
  --batch-size 8 `
  --weight-decay 0.0001 `
  --patience 5 `
  --min-finetune-epochs 5 `
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
    throw "Layer4 local-context CNN smoke test failed."
}

Write-Host ""
Write-Host "Test comparison"

Import-Csv `
  ".\outputs\local_context_cnn_layer4_smoke\${runName}_seed42\test_comparison.csv" |
Select-Object `
  method,
  best_stage,
  best_stage_epoch,
  best_global_epoch,
  best_val_balanced_accuracy,
  test_accuracy,
  test_balanced_accuracy,
  test_macro_f1,
  ph6_accuracy,
  ph8_accuracy,
  ph9_accuracy,
  ph12_accuracy,
  test_single_class_prediction |
Format-Table -AutoSize
