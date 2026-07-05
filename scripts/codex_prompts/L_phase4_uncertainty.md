Implement Phase 4 (decomposed uncertainty + calibration) for BraggTransporter v3.1.

READ FIRST: brag_deep_learning/six-month-model-plan.md (Phase 4 + the §5 uncertainty
IDENTIFIABILITY table — σ_MC, σ_input, σ_epistemic, σ_meas each need a matched data
intervention), braggtransporter/INTERFACES.md, schema.py & config.py (FROZEN),
braggtransporter/models/braggtransporter_v0.py (read-only), braggtransporter/
calibration.py, braggtransporter/data/dataset.py, braggtransporter/data/generate.py.

DESIGN CONSTRAINT: do NOT edit braggtransporter_v0.py or train.py (other work/runs
depend on them). Implement Phase 4 as a SEPARATE residual-uncertainty head trained on
top of a FROZEN, already-trained v0 checkpoint.

ONLY create:
- braggtransporter/models/uncertainty_head.py — a residual head that consumes the
  per-depth input x plus a frozen v0's mean dose prediction and outputs a per-voxel
  predictive sigma. Provide BOTH: (a) a lightweight heteroscedastic head (predict
  mean+logvar, Gaussian NLL), and (b) a conditional flow-matching residual head
  (learns the residual distribution; sample N times -> empirical sigma) with a
  few-step sampler. MPS-safe (no torch.use_deterministic_algorithms; no grid_sample).
- braggtransporter/uncertainty.py — orchestration: load frozen v0, train the chosen
  head on (target - v0_mean) residuals; a decomposition API returning separate
  channels where identifiable: sigma_aleatoric (flow/heteroscedastic) and
  sigma_epistemic (variance across an ENSEMBLE of v0 checkpoints if provided). Add
  explicit HOOKS + docstrings for sigma_MC (needs low/high-stat MC pairs) and
  sigma_meas (needs measurements) noting they are Phase-5+ data-gated.
- scripts/phase4_uncertainty.py — CLI: train the head on a frozen v0, then report
  CALIBRATION on held-out: empirical coverage of 68%/95% bands (target within +-5%),
  a reliability table, and a sharpness metric; write docs/results/phase4_calibration.csv.
- scripts/gen_mc_pairs.py — OPTIONAL small helper: extend the data engine to emit
  low-stat vs high-stat SDE dose pairs (reuse braggtransporter.data + braggpeak SDE)
  so sigma_MC becomes identifiable; write to a separate HDF5. Keep it importable.
- tests/test_bt_uncertainty.py — head forward/shape/nonneg-sigma; NLL/flow loss
  decreases 2 steps; coverage() monotonic in sigma; CPU-fast; a real MPS backward
  check (must not raise / must be finite).

VERIFY: `.venv/bin/python -m pytest tests/test_bt_uncertainty.py -q` and a 2-epoch
CPU smoke of scripts/phase4_uncertainty.py against the existing frozen checkpoint
experiments/bt/phase3/braggtransporter_v0/best.pt (or a --fast synthetic path if
absent). Run scripts with the repo root importable (the orchestrator uses PYTHONPATH=.
— make the scripts insert the repo root into sys.path so they work standalone too).
Do NOT run git. Print a summary + calibration numbers + test result.
