Fix and scale the 3-D DoseRAD pipeline for GPU training (overnight goal #2).

PROBLEM (read PROGRESS.md Phase 5 + docs/FINAL_REPORT.md finding #7): the Bragg3D 3-D
pipeline works on real DoseRAD2026 .mha data but COLLAPSED on 49 beamlets / 1 patient —
it overfit to a blob (predicted peak ~10 cm off, gamma 0% after training). Root causes:
(a) beamlet doses are tiny (~1e-3) and NOT per-beamlet normalized, and (b) far too few
beamlets/patients. Fix both so it can train to real gamma at scale on ONE GPU.

READ FIRST: braggtransporter/data/doserad.py, braggtransporter/models/bragg3d.py,
scripts/phase5_doserad.py, braggtransporter/metrics.py, PROGRESS.md Phase 5.

ONLY edit/create:
- braggtransporter/data/doserad.py — add PER-BEAMLET dose normalization: normalize each
  beamlet dose to unit max (store the scale), so targets are O(1); the model predicts
  normalized dose and gamma is computed on the normalized field (scale-invariant, standard
  for beamlet dose). Support MULTI-PATIENT loading (a list of patient dirs), and a
  `download_patients(patients, dest, max_beamlets_per_patient)` helper that pulls valid
  (>50KB) proton beamlet .mha + ct.mha + plan.json from HuggingFace
  (LMUK-RADONC-PHYS-RES/DoseRAD2026, proton/training/<patient>/...) via direct HTTPS
  (no auth needed). Skip corrupt/tiny files. Cache extracted BEV tensors to npz.
- scripts/train_doserad_gpu.py — a robust GPU training entrypoint: build the multi-patient
  DoseRAD dataset, train Bragg3D with the per-beamlet-normalized relative loss + a
  distal-weighted term, AdamW+cosine, grad clip, deterministic seeds (NO
  torch.use_deterministic_algorithms), AMP on CUDA, checkpoint EVERY N steps to a local
  dir AND (if `--gcs gs://...` given) upload checkpoints+metrics to GCS via gsutil,
  resumable from the latest checkpoint (spot VMs get preempted!). Print per-epoch loss +
  held-out gamma3d(3%/3mm) with `-u`. CLI: patients list, max-beamlets, epochs, device
  (cuda/mps/cpu), batch-size, --gcs, --resume.
- tests/test_bt_doserad_scale.py — CPU-fast, SYNTHETIC only (no network/real .mha):
  per-beamlet normalization yields unit-max targets; multi-patient dataset concatenation
  works; training step reduces loss; resume-from-checkpoint restores state.

Verify: `.venv/bin/python -m pytest tests/test_bt_doserad_scale.py tests/test_bt_doserad.py -q`
and a `--device cpu --fast`-style 2-epoch smoke IF real data is present under
data/doserad2026 (else synthetic). Do NOT run git. Do NOT actually download at scale
(the orchestrator/GPU VM does that). Print a summary + test result.
