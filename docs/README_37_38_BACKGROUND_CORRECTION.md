# Can the background replace reference strips? (scripts 37-38)

Run:
```powershell
python .\scripts\37_background_correction_features.py
foreach ($r in "pad","subtract","wb_linear","bradford","ratio_gamma") {
  Rscript .\scripts\35_new_phone_offset_calibration.R --input .\outputs\background_correction\corrected_features.csv `
    --region $r --output-dir .\outputs\background_correction\calibration
}
python .\scripts\38_summarize_background_correction.py
```
About one minute in total, single core.

## Corrections compared (no reference strips; same photo only)

| name | correction |
|---|---|
| pad | none |
| subtract | Lab(pad) - Lab(background) (current Contrast) |
| wb_linear | linear-RGB white balance: background -> neutral grey |
| bradford | Bradford chromatic adaptation: background -> D65 white |
| ratio_gamma | gamma-RGB ratio pad / background (close to Nelis et al. 2019) |

All computed from the mean sRGB of the ROIs; subtract reproduces the saved
Contrast exactly.

## Results (new phone, light unknown, pH 2-12)

| correction | phone share of colour variation | error, no reference | error, 3 references | error, all 12 |
|---|---|---|---|---|
| pad | 18.0% | 0.85 | 0.46 | 0.42 |
| subtract | **9.1%** | 0.73 | 0.45 | 0.41 |
| wb_linear | 11.5% | **0.71** | 0.47 | 0.41 |
| bradford | 11.9% | 0.71 | 0.46 | 0.41 |
| ratio_gamma | 11.7% | 0.72 | 0.47 | 0.42 |

Mean pH bias of an uncalibrated new phone:

| correction | Galaxy S24+ | iPhone 13 Pro | iPhone 16 Pro |
|---|---|---|---|
| pad | -0.79 | +0.63 | +0.12 |
| subtract | -0.57 | +0.65 | -0.12 |
| wb_linear / bradford / ratio | -0.53 to -0.55 | +0.64 to +0.65 | -0.18 to -0.23 |

## Interpretation

1. Any background-based correction helps an uncalibrated new phone
   (0.85 -> about 0.71-0.73), and the four variants perform almost the same.
2. None removes the phone offset. iPhone 13 Pro keeps a +0.64 pH bias under
   every correction. The offset is therefore not a white-balance problem that
   the white paper can reveal: the phones render the saturated strip colours
   differently from how they render white (colour rendering / tone mapping).
3. Consequently the reference strips cannot be replaced by the background.
   A reference with coloured patches (e.g. a colour card in the photo) might
   capture this offset, but that cannot be tested with the current data.
4. iPhone 16 Pro images are darker (background about 157 vs 176-191) and the
   pad blue channel is clipped to 0 in 39 of its 65 images (pH 2-9). Clipping
   loses information and is a further phone-specific capture effect.
