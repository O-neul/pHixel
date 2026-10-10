# Background ROI check (2026-10-09)

## Question

Script 26 found pH-dependent iPhone 16 Pro differences in Background a*/b*.
Does the Background ROI contain pad colour (spill), which would undermine
Contrast = Pad - Background?

## How Background was measured

Recorded in `data/raw/captures/captures_final_features.csv` (`left25_*` columns;
the extraction code itself was written by a collaborator and is not in this
repository):

- Pad ROI: the LabelMe `roi` rectangle.
- Background ROI: a rectangle of the same size, shifted left by
  2.5 x pad width (`left25_bg_shift_multiplier = 2.5`).
- Images were ICC-converted to sRGB before extraction (`icc_converted_to_srgb`).
- All 195 extractions succeeded.

Gap between the Background ROI and the pad (pixels, 4000 x 3000 images):

| phone | pad width (mean) | gap (mean, min-max) |
|---|---|---|
| Galaxy S24+ | 88 | 132 (103-165) |
| iPhone 13 Pro | 95 | 142 (108-187) |
| iPhone 16 Pro | 102 | 154 (115-193) |

Note: `/Users/soyeon/Downloads/20260707/20260707.ipynb` is a QUB
three-smartphone feature EDA (separate background images), not the
extraction code for this dataset.

## Visual check

`outputs/background_roi_check/roi_montage_L01_ph2_6_12.jpg`
(rows: Galaxy S24+, iPhone 13 Pro, iPhone 16 Pro; columns: pH 2, 6, 12 under L01;
red = pad ROI, blue = Background ROI).

The Background ROI lies on the white backing paper, about 1.5 pad widths from
the strip edge, and never overlaps the strip.

## Pad-Background correlation (strip means, n = 13 per phone)

| phone | corr(pad a*, bg a*) | bg a* range |
|---|---|---|
| Galaxy S24+ | +0.67 | 1.0 |
| iPhone 13 Pro | +0.49 | 0.4 |
| iPhone 16 Pro | -0.29 | 1.9 |

## Conclusion

- Direct spill is unlikely: the ROI is on plain paper, away from the strip.
- The pH-dependent Background differences are small (at most about 2 Lab units,
  vs 6-16 for Contrast) and, for iPhone 16 Pro, move opposite to the pad
  colour. This is more consistent with scene-dependent camera processing
  (auto white balance / tone mapping reacting to the strip colour) than with
  spill. Optical flare cannot be fully excluded.
- For the paper: describe the ROI geometry in Methods 2.2 and mention
  scene-dependent camera processing as a possible source of pH-dependent
  phone structure in the Discussion.
