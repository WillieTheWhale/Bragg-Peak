# Water-phantom Bragg-peak benchmark

> Research software only. No clinical claims.

- Package version: `0.1.0`  ·  model: `sde`  ·  numpy `2.5.0`, python `3.14.4`
- Reference: NIST-anchored Bortfeld analytic Bragg curve
- Fitted stopping-power scale: **1.00466**
- Calibration baseline range RMS: 0.806 mm -> calibrated 0.148 mm
- Wall time: 8.1 s

## Result: FAIL

> Shape metrics (RMSE, gamma) here are against the **Bortfeld analytic** reference, whose plateau-to-peak ratio is conservative; the `gate-benchmark` (vs Geant4) is authoritative for dose shape. Range metrics below are reliable against either reference.

Threshold violations:
- 80 MeV: rmse_pct=4.496 violates <=1.5
- 80 MeV: gamma_2pct_2mm=0.337 violates >=0.95
- 100 MeV: rmse_pct=4.381 violates <=1.5
- 100 MeV: gamma_2pct_2mm=0.299 violates >=0.95
- 150 MeV: rmse_pct=3.998 violates <=1.5
- 150 MeV: gamma_2pct_2mm=0.440 violates >=0.95
- 200 MeV: rmse_pct=3.324 violates <=1.5
- 200 MeV: gamma_2pct_2mm=0.703 violates >=0.95
- 230 MeV: r80_err_mm=-0.847 violates <=0.7
- 230 MeV: r90_err_mm=-0.821 violates <=0.7
- 230 MeV: rmse_pct=3.320 violates <=1.5
- 230 MeV: gamma_2pct_2mm=0.784 violates >=0.95

## Calibrated ladder vs NIST-anchored reference

| E (MeV) | peak (mm) | ref peak | Δpeak (mm) | ΔR80 (mm) | ΔR90 (mm) | RMSE % | γ 2/2 (%) | t (ms) |
|---|---|---|---|---|---|---|---|---|
| 80 | 51.10 | 51.10 | +0.00 | +0.20 | +0.18 | 4.50 | 33.7 | 251 |
| 100 | 76.10 | 75.90 | +0.20 | +0.20 | +0.18 | 4.38 | 29.9 | 373 |
| 150 | 155.30 | 155.10 | +0.20 | +0.24 | +0.19 | 4.00 | 44.0 | 732 |
| 200 | 255.50 | 255.50 | +0.00 | -0.04 | +0.01 | 3.32 | 70.3 | 1165 |
| 230 | 324.90 | 324.70 | +0.20 | -0.85 | -0.82 | 3.32 | 78.4 | 1446 |

## Baseline (uncalibrated) ladder

| E (MeV) | peak (mm) | ref peak | Δpeak (mm) | ΔR80 (mm) | ΔR90 (mm) | RMSE % | γ 2/2 (%) | t (ms) |
|---|---|---|---|---|---|---|---|---|
| 80 | 51.50 | 51.10 | +0.40 | +0.44 | +0.43 | 5.83 | 32.6 | 255 |
| 100 | 76.50 | 75.90 | +0.60 | +0.56 | +0.55 | 5.57 | 29.1 | 370 |
| 150 | 156.30 | 155.10 | +1.20 | +0.96 | +0.88 | 5.11 | 41.5 | 733 |
| 200 | 256.90 | 255.50 | +1.40 | +1.15 | +1.17 | 4.17 | 70.0 | 1170 |
| 230 | 326.10 | 324.70 | +1.40 | +0.73 | +0.58 | 3.44 | 78.9 | 1447 |

## Success thresholds (water)

| metric | limit |
|---|---|
| peak_depth_err_mm | 0.5 |
| r80_err_mm | 0.7 |
| r90_err_mm | 0.7 |
| rmse_pct | 1.5 |
| gamma_2pct_2mm | 0.95 |
