# 11 Local-context ResNet18 layer4 fine-tuning

This keeps the accepted local-context input and compares the same three methods:

- baseline
- mild ColorJitter
- structured empirical nuisance augmentation

## Two-stage training

### Stage 1: head warm-up

- 5 epochs
- only the final classifier is trainable
- classifier learning rate: 1e-3

### Stage 2: layer4 fine-tuning

- maximum 15 epochs
- ResNet18 `layer4` and the final classifier are trainable
- layer4 learning rate: 1e-4
- classifier learning rate: 1e-3
- early stopping patience: 5
- minimum fine-tuning epochs: 5

Fine-tuning starts from the best warm-up validation checkpoint. The final test
model is the best checkpoint across both stages.

All BatchNorm running statistics remain frozen because the train set has only
96 images and batch size is 8.

## Run

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_local_context_cnn_layer4_smoke.ps1
```

## Main result

```text
outputs/local_context_cnn_layer4_smoke/
└─ capture_lopo_holdout_galaxy_s24plus_vallight_L01_seed42/
   ├─ test_comparison.csv
   ├─ audit.json
   ├─ baseline/
   ├─ color_jitter/
   └─ nuisance_structured/
```

`training_history.csv` records both `warmup` and `finetune` stages.

The final table now includes pH 12 because the frozen-backbone model
systematically misclassified four of the five Galaxy pH 12 images.

Galaxy/L01 is a development fold. Use it to confirm that staged training works;
do not repeatedly tune these settings against its test result.
