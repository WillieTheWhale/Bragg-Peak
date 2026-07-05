Implement Phase 2B (backbone ablation) for BraggTransporter v3.1.

READ FIRST: `brag_deep_learning/six-month-model-plan.md` (Phase 2B and the
transformer-vs-operator investigation), `braggtransporter/INTERFACES.md`,
`braggtransporter/models/braggtransporter_v0.py`, `braggtransporter/models/fno1d.py`,
`braggtransporter/train.py` (its `build_model` registry), `braggtransporter/config.py`
(FROZEN).

GOAL: add a selective state-space (Mamba-style) 1-D backbone and a runner that
compares transformer (v0) vs FNO vs SSM at MATCHED parameter budget on held-out
distal-edge error — the empirical Phase-2B experiment.

ONLY edit/create:
- `braggtransporter/models/mamba1d.py`  — `Mamba1d(nn.Module)` with the SAME forward
   contract as the baselines: `forward(x:(B,Nz,9), scalars:(B,4)) -> {"dose","letd","r80"}`,
   softplus dose, `.param_count()`. Implement a REAL selective SSM (S6-style) over the
   depth sequence — input-dependent A/B/C/Δ, a sequential or cumulative-sum scan —
   in pure torch that RUNS AND BACKPROPS ON APPLE MPS (avoid ops lacking MPS backward;
   a Python-loop scan over depth is acceptable for Nz≈680). Target ~0.9M params to
   match v0. Condition on scalars via an injected token/bias.
- `braggtransporter/train.py`  — add `"mamba1d"` to the `build_model` registry ONLY
   (one line; do not change anything else).
- `braggtransporter/evaluate.py` — add `"mamba1d"` to the canonical-name set used in
   load_model ONLY (one line).
- `scripts/ablation_2b.py` — trains {braggtransporter_v0, fno1d, mamba1d} at matched
   param budget, evaluates held-out, writes `docs/results/phase2b.csv`, prints a ranked
   distal-edge table + which backbone owns the sharp edge. Accept --epochs/--data.
- `tests/test_bt_mamba.py` — forward shapes, nonnegative dose, param_count in a sane
   range, and a REAL MPS backward check (like tests/test_bt_v0_interp.py) that must NOT
   raise NotImplementedError and must produce finite grads.

CONSTRAINTS: do NOT run the full MPS ablation (orchestrator runs it). Verify with
`.venv/bin/python -m pytest tests/test_bt_mamba.py -q` and the MPS backward snippet.
Note: `torch.use_deterministic_algorithms` is disabled on MPS in train.py by design —
do not re-enable it. Do NOT run git. Print a summary + param counts + test result.
