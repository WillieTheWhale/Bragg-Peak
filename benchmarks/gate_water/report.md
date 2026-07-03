# Water-phantom benchmark vs Geant4/OpenGATE

> Research software only. No clinical claims.

- Reference: OpenGATE/Geant4 QGSP_BIC_EMZ  ·  250000 primaries
- Candidate: calibrated SDE (scale 1.00466), 120000 histories

| E (MeV) | Δpeak (mm) | ΔR80 (mm) | ΔR90 (mm) | RMSE % | γ 2/2 (%) | γ 3/3 (%) | SDE t (s) | speedup |
|---|---|---|---|---|---|---|---|---|
| 100 | +0.00 | +0.42 | +0.42 | 3.01 | 98 | 100 | 0.38 | cached |
| 150 | +0.00 | +0.03 | +0.04 | 2.00 | 80 | 99 | 0.77 | 20.0x |
| 200 | +0.50 | +0.01 | -0.03 | 2.99 | 58 | 83 | 1.19 | 20.1x |
