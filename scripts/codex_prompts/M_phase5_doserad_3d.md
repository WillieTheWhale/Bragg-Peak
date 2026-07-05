Implement Phase 5 (3-D lift + DoseRAD2026 real data) for BraggTransporter v3.1.

CONTEXT: Real DoseRAD2026 proton data is downloaded under
`data/doserad2026/1ABB006/`: `ct.mha` (164x493x498, spacing 1x1x3 mm, HU),
`plan.json` (iso_center; 36 beams each with gantry_angle and `rays` of
ray_source/ray_target 3-D points), and `dose/Dose_B{b}_R{r}_L{l}.mha` beamlet dose
volumes on the CT grid. Some dose files are tiny/corrupt (~15 bytes) — SKIP those.
Read .mha with SimpleITK (installed). This machine is a 48 GB M4 Max (no CUDA) — you
MUST heavily downsample; the full volume (40M voxels) will not train.

READ FIRST: brag_deep_learning/six-month-model-plan.md (Phase 5), PROGRESS.md,
braggtransporter/INTERFACES.md, schema.py & config.py (FROZEN),
braggtransporter/models/braggtransporter_v0.py (coordinate-query decoder — reuse the
idea in 3-D), braggtransporter/metrics.py, braggtransporter/config.py (get_device).

ONLY create:
- braggtransporter/data/doserad.py — SimpleITK readers; parse plan.json; for each
  VALID beamlet (b,r,l): use the ray_source->ray_target direction to extract a
  BEAM'S-EYE-VIEW sub-volume from the CT (resample so axis 0 = depth along the ray),
  crop laterally around the ray, DOWNSAMPLE to a small grid (e.g. depth<=64,
  lateral<=24x24). Pair (BEV CT patch in HU->density/RSP, energy-layer index l,
  gantry angle) with the matching BEV crop of the beamlet dose. A torch Dataset +
  `make_doserad_loaders(root, patients, max_beamlets, val_frac, ...)`. Cache extracted
  tensors to disk (npz) so re-runs are fast. Handle corrupt/missing files gracefully.
- braggtransporter/models/bragg3d.py — `Bragg3D(nn.Module)`: reuse the depth-sequence
  transformer idea (BEV depth = token sequence; lateral patch flattened/embedded per
  depth) with a coordinate-query decoder over (depth,y,x); softplus dose head;
  `.param_count()`. Keep it SMALL (<5M params). MPS-safe (no grid_sample; no
  torch.use_deterministic_algorithms — it NaNs on MPS).
- scripts/phase5_doserad.py — CLI: build/cache the dataset from
  data/doserad2026, train Bragg3D on the beamlet subset (train/val split over
  beamlets), evaluate held-out beamlets with a 3-D gamma (3%/3mm) + range/RMSE, write
  docs/results/phase5_doserad.csv. Insert repo root into sys.path so it runs
  standalone. Print gamma + a note that laptop-scale training is a proof-of-concept.
- tests/test_bt_doserad.py — SYNTHETIC-only unit tests (do NOT require the real .mha
  or network): a fake tiny CT+dose+plan in a tmp dir exercises the BEV extraction,
  dataset shapes, model forward/backward finite on CPU, and a 3-D gamma sanity
  (curve vs itself = 100%). CPU-fast.

VERIFY: `.venv/bin/python -m pytest tests/test_bt_doserad.py -q` and a real-data smoke
`.venv/bin/python scripts/phase5_doserad.py --patients 1ABB006 --max-beamlets 16
--epochs 2 --device cpu` (must load real .mha, train 2 epochs, print a gamma). Do NOT
run git. Print a summary + the real-data smoke gamma + test result. Note honestly if
laptop-scale gamma is far from clinical — full-scale training is cloud-gated per the plan.
