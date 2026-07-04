# Water-phantom benchmark vs Geant4/OpenGATE

> Research software only. No clinical claims.

- Reference: OpenGATE/Geant4 QGSP_BIC_EMZ  ·  200000 primaries
- Candidate: calibrated SDE (scale 1.00466), 150000 histories

| E (MeV) | Δpeak (mm) | ΔR80 (mm) | ΔR90 (mm) | RMSE % | γ 2/2 (%) | γ 3/3 (%) | SDE t (s) | speedup |
|---|---|---|---|---|---|---|---|---|
| 100 | +0.00 | +0.43 | +0.43 | 2.93 | 99 | 100 | 0.49 | 20.9x |
| 150 | +0.00 | +0.01 | +0.02 | 1.10 | 100 | 100 | 0.96 | 14.1x |
| 200 | +0.00 | -0.01 | +0.02 | 1.32 | 98 | 100 | 1.49 | 13.6x |
