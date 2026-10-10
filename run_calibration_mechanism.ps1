$ErrorActionPreference = "Stop"

# Light-unknown inverse pH prediction (primary spec, about 1-2 minutes)
Rscript .\scripts\28_gam_inverse_ph_prediction.R --specs primary --light-mode unknown `
  --output-dir ".\outputs\gam_inverse_ph_light_unknown"
if ($LASTEXITCODE -ne 0) { throw "28 (light unknown) failed." }
python .\scripts\28_summarize_inverse_ph.py --inverse-dir ".\outputs\gam_inverse_ph_light_unknown"
if ($LASTEXITCODE -ne 0) { throw "28 summary failed." }

# pH-relevant nuisance decomposition (about 15 seconds)
Rscript .\scripts\34_ph_relevant_nuisance.R
if ($LASTEXITCODE -ne 0) { throw "34 failed." }

# New-phone offset calibration (seconds per region)
Rscript .\scripts\35_new_phone_offset_calibration.R --region contrast
if ($LASTEXITCODE -ne 0) { throw "35 (contrast) failed." }
Rscript .\scripts\35_new_phone_offset_calibration.R --region pad
if ($LASTEXITCODE -ne 0) { throw "35 (pad) failed." }

python .\scripts\36_plot_calibration_mechanism.py
if ($LASTEXITCODE -ne 0) { throw "36 failed." }

Write-Host ""
Write-Host "Main outputs:"
Write-Host ".\outputs\calibration_mechanism\notes.txt"
Write-Host ".\outputs\calibration_mechanism\figures\Fig_calibration_mechanism.png"
