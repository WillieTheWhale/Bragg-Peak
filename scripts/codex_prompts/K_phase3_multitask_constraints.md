Implement Phase 3 multi-task heads + physics constraints for BraggTransporter v3.1.

READ FIRST: brag_deep_learning/six-month-model-plan.md (Phase 3), PROGRESS.md,
braggtransporter/INTERFACES.md, braggtransporter/schema.py (FROZEN — note QUANTITIES
= dose, letd, lett, fluence), braggtransporter/config.py, braggtransporter/models/
braggtransporter_v0.py, braggtransporter/train.py, braggtransporter/metrics.py,
and the braggpeak physics package (for a LETd reference).

ONLY edit/create:
- braggtransporter/models/braggtransporter_v0.py — extend the coordinate-query
  decoder to ALSO predict `lett` and `fluence` (in addition to dose/letd/r80). Keep
  the forward return a dict; existing keys unchanged; new keys added. Softplus
  (nonnegative) on lett/fluence. Backward-compatible defaults (Phase 1/2 results
  must still reproduce): gate the new heads behind ModelConfig.quantities so a model
  configured with quantities=["dose","letd"] behaves exactly as before.
- braggtransporter/config.py — you MAY ADD optional fields to TrainConfig with
  defaults that preserve current behavior (e.g. `constraint_weight: float = 0.0`,
  `monotonic_range_weight: float = 0.0`). Do NOT rename/remove/retype existing fields
  and do NOT touch schema.py.
- braggtransporter/train.py — add OPT-IN soft physics-constraint terms to
  compute_loss, active only when their weights > 0: (a) monotone range vs energy
  (within a batch, higher energy => larger predicted R80, hinge penalty on
  violations), (b) a soft energy-budget term. Default weights 0 => loss identical to
  now. Keep the existing NaN-safe masking.
- scripts/phase3_letd_validation.py — evaluate predicted LETd against a braggpeak
  LETd reference on the held-out set; write docs/results/phase3_letd.csv + print
  correlation / mean abs error.
- tests/test_bt_phase3.py — new heads present & nonnegative when enabled; absent
  when quantities excludes them (back-compat); constraint terms are 0 at default
  weights and positive on a synthetic violation; CPU-fast.

VERIFY: `.venv/bin/python -m pytest tests/test_bt_phase3.py tests/test_bt_v0.py -q`
(v0 back-compat must still pass) and a 2-epoch `--fast` CPU train smoke. Do NOT run
git. Do NOT enable torch.use_deterministic_algorithms (NaNs on MPS). Orchestrator
runs MPS training. Print a summary + test result.
