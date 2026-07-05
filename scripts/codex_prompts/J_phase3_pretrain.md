Implement Phase 3 Stage-0 masked transport pretraining for BraggTransporter v3.1.

READ FIRST: brag_deep_learning/six-month-model-plan.md (Phase 3 / Stage 0),
braggtransporter/INTERFACES.md, braggtransporter/schema.py & config.py (FROZEN),
braggtransporter/data/dataset.py, braggtransporter/models/braggtransporter_v0.py
(read-only — do NOT edit it).

GOAL: a self-supervised pretraining task that masks spans of the per-depth input
channels and/or the dose profile and reconstructs them (masked autoencoding), plus
a next-slab prediction objective. Pretrained encoder weights can initialize the
Stage-1 supervised v0 model. Demonstrate improved data efficiency.

ONLY create:
- braggtransporter/pretrain.py  — `MaskedTransportPretrainer` + a CLI
  `python -m braggtransporter.pretrain --config configs/bt_v0_1d.yaml --epochs N`.
  Reuse the v0 encoder (import BraggTransporterV0's encoder submodule or wrap the
  model) WITHOUT modifying v0. Mask random depth spans of x (and optionally dose),
  reconstruct with an MSE loss over masked positions; add next-slab prediction.
  Save pretrained weights to experiments/bt/pretrain/encoder.pt. MPS-safe; do NOT
  enable torch.use_deterministic_algorithms (it NaNs on MPS).
- scripts/phase3_data_efficiency.py — trains v0 with vs without Stage-0 init on
  {25%,50%,100%} of the training data, evaluates held-out gamma/distal-edge, writes
  docs/results/phase3_data_efficiency.csv and prints whether Stage-0 helps.
- tests/test_bt_pretrain.py — masking shapes, loss decreases 2 steps, save/load
  round-trip, CPU-fast.

Verify: `.venv/bin/python -m pytest tests/test_bt_pretrain.py -q` and a 2-epoch
`--fast` CPU smoke of pretrain.py. Do NOT run git. Orchestrator runs the MPS
data-efficiency study. Print a summary + test result.
