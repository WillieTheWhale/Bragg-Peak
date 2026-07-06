Add a DoTA-faithful warm-restart LR schedule + long-training support to the 3-D trainer.

MOTIVATION: DoTA (arXiv 2202.02653) trains with LR halved every 4 epochs and a WARM
RESTART after 28 epochs, up to 56-118 epochs. Our trainer uses a plain cosine schedule
(anneals monotonically to ~0), which cannot produce the plateau-escaping post-restart
jumps DoTA relies on. This may be why our 3-D gamma plateaus ~57%. Add warm restarts.

READ FIRST: scripts/train_doserad_gpu.py (its optimizer/scheduler + arg parsing),
braggtransporter/train.py (compute_loss reuse). Only touch the DoseRAD trainer.

ONLY edit/create:
- scripts/train_doserad_gpu.py:
  * Add `--lr-schedule` arg with choices {cosine (current default), warmrestart, dota}.
    - `warmrestart`: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts (T_0 configurable
      via `--restart-epochs` default 28, T_mult=1), stepped per epoch.
    - `dota`: replicate DoTA exactly — LR halved every `--lr-halve-epochs` (default 4)
      epochs, hard RESTART to the base LR after `--restart-epochs` (default 28), repeating.
    - keep `cosine` as-is for back-compat (default unchanged).
  * Add `--weight-decay` arg (default 0.05 — grokking/generalization needs meaningful weight
    decay; current may be ~0). Wire into AdamW.
  * Add `--save-best-by gamma|val_loss` (default gamma): track and checkpoint the BEST
    held-out gamma3d over ALL epochs (not just final), and print a running "best gamma so
    far @ epoch N" each epoch, so a late spike is captured even if the run is cut off.
  * Ensure the per-epoch print includes the current LR so restarts are visible in logs.
- tests/test_bt_lr_schedule.py: CPU-fast — each schedule produces a valid per-epoch LR
  sequence (warmrestart/dota show the LR jumping back up at restart boundaries); best-gamma
  checkpointing keeps the max, not the last.

VERIFY: `.venv/bin/python -m pytest tests/test_bt_lr_schedule.py -q` and a 3-epoch
`--fast --lr-schedule dota` smoke that prints LR per epoch. Do NOT change the model or
data pipeline. Do NOT run git. Print a summary + the LR sequence from a short warmrestart run.
