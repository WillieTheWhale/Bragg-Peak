DRAFT — Phase 3 (physics prior + multi-task heads + Stage-0 masked pretraining).
Finalize after the Phase-2B gate. Reference: brag_deep_learning/six-month-model-plan.md
(Phase 3 / Stage 0), PROGRESS.md, braggtransporter/INTERFACES.md.

The physics prior is already wired (PRIOR_FIELDS) and validated in Phase 2A
(prior-off collapses the model). Phase 3 adds:

1. Multi-task query heads on BraggTransporter-v0: extend the coordinate-query
   decoder to also predict LETt and fluence (in addition to dose/LETd/R80), each a
   queryable quantity in schema.QUANTITIES. Add PHYSICS CONSTRAINTS as soft penalty
   terms in train.compute_loss (opt-in via TrainConfig flags): dose>=0 (already via
   softplus), dRange/dEnergy>0 (monotonicity across a batch sorted by energy), and a
   soft energy-budget term. Keep defaults backward-compatible with Phase 1/2.

2. Stage-0 masked transport pretraining: a self-supervised task that masks spans of
   the per-depth input channels and/or the dose profile and reconstructs them, plus
   next-slab prediction. New module braggtransporter/pretrain.py + a CLI. Pretrained
   weights initialize the Stage-1 supervised model; show it improves data efficiency
   (train on 25%/50%/100% of data, compare held-out gamma with vs without Stage-0).

3. LETd validation: compare predicted LETd against the braggpeak LET reference (and,
   where available, note the Geant4/TOPAS path) on the held-out set.

Disjoint file ownership (assign across agents at launch):
- braggtransporter/models/braggtransporter_v0.py (heads) — one agent only
- braggtransporter/train.py (constraint loss terms) — same agent as heads
- braggtransporter/pretrain.py + tests — separate agent
- scripts/phase3_*.py runners + docs — separate agent
Verify CPU-fast; orchestrator runs MPS. Do NOT run git.
