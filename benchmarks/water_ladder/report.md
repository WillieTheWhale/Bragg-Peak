# Water-phantom Bragg-peak benchmark

> Research software only. No clinical claims.

- Package version: `0.1.0`  ·  model: `sde`  ·  numpy `2.5.0`, python `3.14.4`
- Reference: NIST-anchored Bortfeld analytic Bragg curve
- Fitted stopping-power scale: **1.00466**
- Calibration baseline range RMS: 0.806 mm -> calibrated 0.148 mm
- Wall time: 8.1 s

## Result: FAIL

Threshold violations:
- 80 MeV: rmse_pct=5.140 violates <=1.5
- 80 MeV: gamma_2pct_2mm=0.251 violates >=0.95
- 100 MeV: rmse_pct=4.973 violates <=1.5
- 100 MeV: gamma_2pct_2mm=0.221 violates >=0.95
- 150 MeV: peak_depth_err_mm=0.800 violates <=0.5
- 150 MeV: rmse_pct=4.680 violates <=1.5
- 150 MeV: gamma_2pct_2mm=0.190 violates >=0.95
- 200 MeV: rmse_pct=4.186 violates <=1.5
- 200 MeV: gamma_2pct_2mm=0.659 violates >=0.95
- 230 MeV: peak_depth_err_mm=-1.000 violates <=0.5
- 230 MeV: r80_err_mm=-0.995 violates <=0.7
- 230 MeV: r90_err_mm=-0.977 violates <=0.7
- 230 MeV: rmse_pct=4.139 violates <=1.5
- 230 MeV: gamma_2pct_2mm=0.933 violates >=0.95

## Calibrated ladder vs NIST-anchored reference

| E (MeV) | peak (mm) | ref peak | Δpeak (mm) | ΔR80 (mm) | ΔR90 (mm) | RMSE % | γ 2/2 (%) | t (ms) |
|---|---|---|---|---|---|---|---|---|
| 80 | 51.10 | 51.10 | +0.00 | +0.18 | +0.16 | 5.14 | 25.1 | 251 |
| 100 | 75.90 | 75.90 | +0.00 | +0.21 | +0.20 | 4.97 | 22.1 | 368 |
| 150 | 155.90 | 155.10 | +0.80 | +0.25 | +0.24 | 4.68 | 19.0 | 736 |
| 200 | 255.30 | 255.50 | -0.20 | -0.02 | -0.12 | 4.19 | 65.9 | 1183 |
| 230 | 323.70 | 324.70 | -1.00 | -0.99 | -0.98 | 4.14 | 93.3 | 1455 |

## Baseline (uncalibrated) ladder

| E (MeV) | peak (mm) | ref peak | Δpeak (mm) | ΔR80 (mm) | ΔR90 (mm) | RMSE % | γ 2/2 (%) | t (ms) |
|---|---|---|---|---|---|---|---|---|
| 80 | 51.30 | 51.10 | +0.20 | +0.43 | +0.41 | 6.40 | 21.0 | 255 |
| 100 | 76.30 | 75.90 | +0.40 | +0.55 | +0.56 | 6.12 | 19.1 | 373 |
| 150 | 156.50 | 155.10 | +1.40 | +0.98 | +0.95 | 5.74 | 12.9 | 737 |
| 200 | 256.10 | 255.50 | +0.60 | +1.13 | +0.92 | 4.94 | 55.9 | 1188 |
| 230 | 324.90 | 324.70 | +0.20 | +0.50 | +0.47 | 4.24 | 89.1 | 1488 |

## Success thresholds (water)

| metric | limit |
|---|---|
| peak_depth_err_mm | 0.5 |
| r80_err_mm | 0.7 |
| r90_err_mm | 0.7 |
| rmse_pct | 1.5 |
| gamma_2pct_2mm | 0.95 |
