Fix the SCALABILITY bottleneck in scripts/train_doserad_gpu.py: per-epoch 3-D gamma
evaluation over the full held-out set is Python-CPU-bound and stalls training (GPU idle
at 0% while a large held-out gamma eval runs). Make training GPU-bound and fast.

READ FIRST: scripts/train_doserad_gpu.py (the epoch loop + eval + gamma_index_3d call),
braggtransporter/data/doserad.py (gamma_index_3d).

ONLY edit:
- scripts/train_doserad_gpu.py:
  1. Add `--eval-subsample N` (default 96): each epoch, compute the PROGRESS gamma3d on a
     FIXED random subsample of N held-out beamlets (seeded, same subset every epoch for a
     comparable curve) instead of the full set. Do a FULL held-out gamma eval only (a) at
     the very end and (b) every `--full-eval-every` epochs (default 0 = only at end).
     Print both "gamma(sub)" each epoch and "gamma(full)" when computed.
  2. Wire the training DataLoader with num_workers=min(8, os.cpu_count()) and
     pin_memory=True + persistent_workers so data loading is parallel (was likely
     num_workers=0 -> CPU-serial bottleneck).
  3. Keep best-gamma checkpointing on the SUBSAMPLE metric during training, and re-evaluate
     the best checkpoint on the FULL held-out set at the end for the reported number.
  4. Make gamma_index_3d calls skip beamlets whose ref peak is ~0 (already returns nan) and
     cap the local search neighborhood so a degenerate beamlet can't hang the eval.

VERIFY: `.venv/bin/python -m pytest tests/test_bt_doserad_scale.py tests/test_bt_lr_schedule.py -q`
and a `--fast --eval-subsample 8` 3-epoch CPU smoke that prints per-epoch gamma(sub) quickly
(no multi-minute stall). Confirm the epoch loop no longer calls full-held-out gamma every
epoch. Do NOT run git. Print the per-epoch timing before/after if measurable, and the smoke result.
