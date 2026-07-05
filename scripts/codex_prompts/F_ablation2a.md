Implement Phase 2A (representation ablation) for BraggTransporter v3.1.

READ FIRST: `brag_deep_learning/six-month-model-plan.md` (Phase 2A), `PROGRESS.md`,
`braggtransporter/INTERFACES.md`, `braggtransporter/models/braggtransporter_v0.py`,
`braggtransporter/train.py`, `braggtransporter/config.py` (config.py FROZEN).

GOAL: make the v0 model support three ablation toggles, then provide a runner that
trains the matrix and reports which representation choices reduce held-out
distal-edge error. The three factors (plan Phase 2A):
1. decoder: coordinate-query (current) vs fixed-grid per-depth head.
2. physics prior: on (current) vs off (zero the 5 prior channels, indices 4..8 of x).
3. loss: distal-edge-weighted (current, weight 5) vs uniform (weight 1).
Factor 3 is ALREADY controlled by TrainConfig.distal_edge_weight — do not change
train.py for it; the runner just sets it.

ONLY edit/create:
- `braggtransporter/models/braggtransporter_v0.py`  (add toggles via ModelConfig.extra:
   `decoder_mode` in {"coord_query","fixed_grid"} default "coord_query";
   `use_physics_prior` bool default True. When False, zero x[..., 4:9] at forward
   input. Keep DEFAULT behavior identical to today so Phase-1 results reproduce.)
- `scripts/ablation_2a.py`  (CLI: trains the 8-config matrix by writing temp YAMLs
   that set model.extra + train.distal_edge_weight + train.epochs, invoking
   `braggtransporter.train`, then `braggtransporter.evaluate`; aggregates held-out
   distal_edge_error_mm + gamma + rmse into `docs/results/phase2a.csv` and prints a
   ranked table with a one-line conclusion per factor. Accept --epochs and --data.)
- `tests/test_bt_ablation2a.py`

CONSTRAINTS: do NOT run the full MPS ablation yourself — the orchestrator runs it.
Verify ONLY with a fast CPU smoke: `.venv/bin/python -m pytest
tests/test_bt_ablation2a.py -q` (forward works with each decoder_mode and with
use_physics_prior False/True; fixed_grid and coord_query both give (B,Nz) dose) and
a 1-config 2-epoch CPU run of scripts/ablation_2a.py on a tiny synthetic/--fast path.
Do NOT run git. Print a summary + test result + the exact command the orchestrator
should run for the full ablation.
