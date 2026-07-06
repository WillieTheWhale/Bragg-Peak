Implement the "sharp-data architecture test" for BraggTransporter (overnight goal #1).

MOTIVATION (read PROGRESS.md Phase 2B + docs/FINAL_REPORT.md finding #2): on CLEAN
analytic 1-D data the transformer's distal-edge advantage over FNO was within noise.
The claim (transformer-vs-operator-investigation.md) is that attention beats FNO's
spectral smoothing on SHARP/heterogeneous/noisy edges. This test proves or disproves it.

READ FIRST: braggtransporter/INTERFACES.md, schema.py & config.py (FROZEN),
braggtransporter/data/generate.py + physics_engine.py (the existing generator),
braggtransporter/models/{braggtransporter_v0,fno1d,mlp}.py, braggtransporter/metrics.py,
braggtransporter/train.py, scripts/ablation_2b.py (the matched-budget comparison pattern).

ONLY create:
- braggtransporter/data/sharp.py — a generator of 1-D targets with DELIBERATELY SHARP
  distal edges: (a) heterogeneous multi-slab geometries (bone/lung/air interfaces that
  sharpen and split the distal falloff via WEPL mixing), and (b) finer depth resolution
  (dz down to ~0.02 cm) so the 80-20 falloff spans few voxels. Reuse braggpeak physics.
  Emit the SAME HDF5 schema as data/generate.py (train/val/heldout_energy groups, norm
  group) so the existing dataset/train/eval code works unchanged. CLI:
  `python -m braggtransporter.data.sharp --out data/generated/sharp_1d.h5 ...`.
- scripts/sharp_comparison.py — train matched-budget v0-transformer vs fno1d vs a
  conv baseline (use mlp as the third if a conv model doesn't exist; else add a tiny
  1-D conv model INSIDE this script) on sharp_1d.h5, evaluate held-out with distal-edge
  error AND TIGHT gamma (1%/1mm and 2%/2mm), write docs/results/sharp_comparison.csv,
  and print a ranked table + a one-line honest conclusion (does the transformer beat FNO
  on the sharp edge, and by how much, with significance across the held-out set?).
- tests/test_bt_sharp.py — CPU-fast: sharp generator emits valid schema with sharper
  80-20 falloff than the clean generator; comparison runner works on a tiny synthetic set.

Use `.venv/bin/python`; run training with `-u`. Verify with
`.venv/bin/python -m pytest tests/test_bt_sharp.py -q` and a 2-epoch CPU smoke of
scripts/sharp_comparison.py. Do NOT run git. Do NOT enable torch.use_deterministic_
algorithms on MPS. Print a summary + smoke result. Note: the orchestrator runs the full
MPS comparison (40+ epochs) — keep your smoke tiny.
