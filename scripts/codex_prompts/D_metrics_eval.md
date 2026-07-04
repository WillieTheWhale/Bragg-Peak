You are implementing Module D (metrics, calibration, evaluation) of BraggTransporter v3.1.

READ FIRST: `braggtransporter/INTERFACES.md`, `braggtransporter/schema.py`,
`braggtransporter/config.py` (both FROZEN — do NOT edit), and reuse the sibling
`braggpeak/scoring.py` (it already has R80/R90/R50, peak depth, FWHM, RMSE/MAE, and
a 1-D gamma index — call it, do not reimplement).

ONLY create/edit these files (touch nothing else):
- `braggtransporter/metrics.py`
- `braggtransporter/calibration.py`
- `braggtransporter/evaluate.py`
- `tests/test_bt_metrics.py`

`metrics.py`: thin, well-tested wrappers returning plain floats/dicts, operating on
numpy arrays `(Nz,)`:
- `gamma_index_1d(pred, ref, z, dose_pct=2.0, dta_mm=2.0) -> float` (pass rate %)
- `r80_r90_r50(z, dose) -> dict`
- `peak_depth_cm(z, dose) -> float`
- `distal_80_20_mm(z, dose) -> float`
- `rmse_pct(pred, ref) -> float`
- `distal_edge_error_mm(pred, ref, z) -> float`  # |R80_pred - R80_ref| in mm — THE
  headline metric for the "attention beats FNO at the sharp edge" claim.
Prefer delegating to braggpeak.scoring; only add what's missing.

`calibration.py`: `CalibrationWrapper` (temperature/quantile scaling of a predicted
sigma) and `coverage(pred, ref, sigma, level) -> float` (empirical fraction inside
the band). Keep it uncertainty-agnostic (works once a sigma exists in later phases).

`evaluate.py`: `python -m braggtransporter.evaluate --ckpt <pt> --data <h5>` loads a
trained model + the held-out loaders, computes per-energy and aggregate metrics
(incl. distal_edge_error_mm), prints a readable table, and writes CSV+JSON to
`experiments/bt/eval/`. Support `--models a.pt b.pt ...` to compare several models
side by side on the same held-out set (this is what the Phase-1 GATE needs:
v0 vs mlp vs fno1d vs dota on distal-edge error).

`tests/test_bt_metrics.py`: validate metrics against a known analytic Bragg curve
(reuse braggpeak.analytic_bragg) — e.g. R80 recovered within one voxel, gamma of a
curve vs itself == 100%, distal_edge_error_mm(self,self)==0. Run
`.venv/bin/python -m pytest tests/test_bt_metrics.py -q` until green. Do NOT run git.
Print a summary + test result.
