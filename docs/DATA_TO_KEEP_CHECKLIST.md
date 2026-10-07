# Data migration checklist

## Required

- [ ] 195 untouched direct-capture images
- [ ] `captures_final_features.csv`
- [ ] Phone identifier/model for every image
- [ ] Light identifier/label for every image
- [ ] Nominal pH for every image
- [ ] Reaction-pad color features
- [ ] Background color features

## Required for the later ROI-aware augmentation and XAI stages

- [ ] LabelMe JSON annotations, ROI masks, or reproducible ROI coordinates
- [ ] Clear identification of reaction-pad ROI and background ROI

## Strongly recommended

- [ ] Feature-extraction script that generated the CSV
- [ ] Capture protocol notes
- [ ] Original file-naming/manifest script
- [ ] Environment file or package list

## Optional external transfer test

- [ ] QUB `three_smartphones` original images
- [ ] QUB canonical manifest
- [ ] QUB LOPO split files

## Safe to regenerate rather than migrate

- [ ] Pairwise shift tables
- [ ] Augmentation previews/contact sheets
- [ ] Old model checkpoints
- [ ] Old training logs
- [ ] Temporary crops and cache files
