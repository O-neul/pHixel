$ErrorActionPreference = "Stop"

# S1: exchangeable unit and permutation validity (about 30-40 minutes)
python .\scripts\29_sim_permutation_unit.py --n-datasets 1000 --n-permutations 999
if ($LASTEXITCODE -ne 0) { throw "29_sim_permutation_unit.py failed." }

# S2: GAM identification + LOSO data sets (about 10 minutes, 6 workers)
Rscript .\scripts\30_sim_gam_identification.R --n-reps 500 --n-loso-datasets 200 --cores 6
if ($LASTEXITCODE -ne 0) { throw "30_sim_gam_identification.R failed." }

# S2-LOSO / S3: GAM and RCS leave-one-strip-out on simulated data (about 50 minutes)
Rscript .\scripts\31_sim_loso_gam.R --cores 6
if ($LASTEXITCODE -ne 0) { throw "31_sim_loso_gam.R failed." }
python .\scripts\31_sim_loso_rcs.py --workers 6
if ($LASTEXITCODE -ne 0) { throw "31_sim_loso_rcs.py failed." }

python .\scripts\32_summarize_simulation.py --bootstrap-reps 1000
if ($LASTEXITCODE -ne 0) { throw "32_summarize_simulation.py failed." }

Write-Host ""
Write-Host "Main outputs:"
Write-Host ".\outputs\simulation\s1\s1_notes.txt"
Write-Host ".\outputs\simulation\simulation_notes.txt"
Write-Host ".\outputs\simulation\figures"
Write-Host ".\outputs\simulation\s1\figures"
