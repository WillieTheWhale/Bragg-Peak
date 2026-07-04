# BraggTransporter v3.1 — Build Progress

Orchestration log. Reference: `brag_deep_learning/six-month-model-plan.md`.
Bulk code authored by Codex (`gpt-5.5`, high reasoning) under senior-MLE review;
each phase is gated (tests + metrics pass) before the next begins.

## Environment
- Python 3.11 venv at `.venv` (default system python is 3.14 → no torch wheels).
- torch 2.12.1, MPS verified on Apple M4 Max, **48 GB RAM** (plan text says 128 GB;
  real machine is 48 GB — batch sizes sized conservatively).
- Physics reuse: sibling `braggpeak/` package (validated SDE vs Geant4, sub-0.1 mm).

## Phase gates
- [ ] **Phase 0 — Scaffold + env + remote overwrite.** contracts frozen (`schema.py`,
      `config.py`, `INTERFACES.md`), repo pushed to `WillieTheWhale/Bragg-Peak`.
- [ ] **Phase 1 (Month 1) — Data engine + baselines + v0.**
      - [ ] tier-1 1-D data generator (SDE/analytic) → HDF5, train/val/heldout-energy
      - [ ] baselines: MLP, FNO1d, DoTA-style transformer
      - [ ] BraggTransporter-v0 (physics prior + transformer + coord-query + det heads)
      - [ ] metrics + eval + calibration harness
      - [ ] **GATE:** v0 beats MLP/FNO/DoTA on distal-edge error on held-out energies;
            all tests green.
- [ ] **Phase 2A — Representation ablation** (fixed-grid vs coord-query; prior vs none).
- [ ] **Phase 2B — Backbone ablation** (transformer vs FNO vs Mamba, matched budget).
- [ ] **Phase 3 — Physics prior + multi-task heads + Stage-0 masked pretraining.**
- [ ] **Phase 4 — Decomposed uncertainty + calibration.**
- [ ] **Phase 5 — 3-D lift + DoseRAD2026 train/val.**
- [ ] **Phase 6 — Benchmark, ablate, write up.**

## Log
- Phase 0 started.
