# Water-phantom Bragg-peak benchmark

> Research software only. No clinical claims.

- Package version: `0.1.0`  ·  model: `sde`  ·  numpy `2.5.0`, python `3.14.4`
- Reference: NIST-anchored Bortfeld analytic Bragg curve
- Fitted stopping-power scale: **1.00466**
- Calibration baseline range RMS: 0.806 mm -> calibrated 0.148 mm
- Wall time: 8.4 s

## Result: FAIL

Threshold violations:
- 80 MeV: rmse_pct=4.496 violates <=1.5
- 80 MeV: gamma_2pct_2mm=0.300 violates >=0.95
- 100 MeV: rmse_pct=4.176 violates <=1.5
- 100 MeV: gamma_2pct_2mm=0.314 violates >=0.95
- 150 MeV: peak_depth_err_mm=0.800 violates <=0.5
- 150 MeV: rmse_pct=3.713 violates <=1.5
- 150 MeV: gamma_2pct_2mm=0.813 violates >=0.95
- 200 MeV: rmse_pct=3.254 violates <=1.5
- 230 MeV: peak_depth_err_mm=-1.000 violates <=0.5
- 230 MeV: r80_err_mm=-1.030 violates <=0.7
- 230 MeV: r90_err_mm=-1.034 violates <=0.7
- 230 MeV: rmse_pct=3.696 violates <=1.5

## Calibrated ladder vs NIST-anchored reference

| E (MeV) | peak (mm) | ref peak | Δpeak (mm) | ΔR80 (mm) | ΔR90 (mm) | RMSE % | γ 2/2 (%) | t (ms) |
|---|---|---|---|---|---|---|---|---|
| 80 | 51.10 | 51.10 | +0.00 | +0.18 | +0.16 | 4.50 | 30.0 | 257 |
| 100 | 75.90 | 75.90 | +0.00 | +0.21 | +0.19 | 4.18 | 31.4 | 380 |
| 150 | 155.90 | 155.10 | +0.80 | +0.24 | +0.24 | 3.71 | 81.3 | 755 |
| 200 | 255.70 | 255.50 | +0.20 | -0.02 | -0.14 | 3.25 | 99.5 | 1206 |
| 230 | 323.70 | 324.70 | -1.00 | -1.03 | -1.03 | 3.70 | 99.8 | 1507 |

## Baseline (uncalibrated) ladder

| E (MeV) | peak (mm) | ref peak | Δpeak (mm) | ΔR80 (mm) | ΔR90 (mm) | RMSE % | γ 2/2 (%) | t (ms) |
|---|---|---|---|---|---|---|---|---|
| 80 | 51.30 | 51.10 | +0.20 | +0.43 | +0.40 | 5.81 | 28.5 | 261 |
| 100 | 76.30 | 75.90 | +0.40 | +0.55 | +0.55 | 5.45 | 29.1 | 381 |
| 150 | 156.50 | 155.10 | +1.40 | +0.98 | +0.94 | 4.89 | 76.4 | 761 |
| 200 | 256.10 | 255.50 | +0.60 | +1.10 | +0.88 | 4.14 | 96.9 | 1215 |
| 230 | 324.90 | 324.70 | +0.20 | +0.49 | +0.47 | 3.45 | 99.8 | 1513 |

## Success thresholds (water)

| metric | limit |
|---|---|
| peak_depth_err_mm | 0.5 |
| r80_err_mm | 0.7 |
| r90_err_mm | 0.7 |
| rmse_pct | 1.5 |
| gamma_2pct_2mm | 0.95 |
