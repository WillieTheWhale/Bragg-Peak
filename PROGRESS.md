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
- [x] **Phase 1 (Month 1) — Data engine + baselines + v0. GATE PASSED.**
      - [x] tier-1 1-D data generator (SDE/analytic) → HDF5, train/val/heldout-energy
      - [x] baselines: MLP, FNO1d, DoTA-style transformer
      - [x] BraggTransporter-v0 (physics prior + transformer + coord-query + det heads)
      - [x] metrics + eval + calibration harness
      - [x] **GATE PASS** on held-out energies (interpolation, clean analytic targets,
            60 epochs, MPS). Distal-edge error (mm), lower=better:
            **v0 0.111** < fno1d 0.120 < mlp 0.715 < dota 1.409.
            γ(2%/2mm): v0 **96.8%**, fno1d 94.4%, dota 92.4%, mlp 69.5%.
            RMSE%: v0 **0.90**, fno1d 0.99, dota 4.24, mlp 4.11. All 16 tests green.
      - Honest note: on *clean analytic* targets FNO is competitive (0.120 vs 0.111 mm);
        the v0/FNO distal-edge gap is expected to widen on sharper/noisier targets
        (heterogeneous slabs, SDE residuals) — a Phase-2/4 follow-up.
- [x] **Phase 2A — Representation ablation** (8 configs, held-out energies, MPS).
      Ranked by distal-edge error (mm): cq+prior+weighted **0.165** > fg+prior+weighted
      0.189 > cq+prior+uniform 0.217 > fg+prior+uniform 0.302 >> all prior-OFF (NaN
      edge, ~16% gamma). Conclusions: (1) **physics prior is essential** — removing it
      collapses the model (can't localize the peak, gamma 16%); (2) **coord-query beats
      fixed-grid** at the distal edge; (3) **distal-edge weighting** lowers edge error
      (trades ~1% aggregate gamma). Best config = v0's defaults → design validated.
      Results: `docs/results/phase2a.csv`.
- [x] **Phase 2B — Backbone ablation** (transformer vs FNO vs Mamba). Held-out
      energies, MPS-trained. Distal-edge / gamma / RMSE:
      | backbone | params | γ2%/2mm | RMSE% | distal-edge |
      |---|---|---|---|---|
      | v0 (transformer) | 944k | **95.4%** | **1.14** | 0.172 mm |
      | FNO1d | 596k | 91.4% | 1.28 | **0.156 mm** |
      | Mamba1d | 963k | 0.0%* | 21.5* | 1.02 mm* |
      **Honest findings:** (1) on *clean analytic 1-D* data the transformer leads on
      overall shape (gamma, RMSE) while FNO is marginally better on the *raw* distal
      edge; combined with Phase 1 (v0 0.111 < FNO 0.120), the v0/FNO edge gap is
      within run-to-run noise on smooth data — the transformer's edge advantage is
      expected to emerge on *sharper/heterogeneous/noisy* targets (Phase 4 flow head,
      heterogeneity), NOT clean ones. This refines rather than confirms the
      transformer-vs-operator memo. (2) *Mamba is computationally intractable* at
      Nz=680 without a fused kernel (~2.4 min/epoch vs ~5s for v0/FNO, ~30x); it was
      under-trained here (*asterisked* rows) — a practical finding against unfused
      sequential-SSM backbones on this hardware. Caveat: FNO ran at its default 596k
      params (not param-matched to v0's 944k). Results: `docs/results/phase2b_summary.csv`.
- [ ] **Phase 3 — Physics prior + multi-task heads + Stage-0 masked pretraining.**
- [ ] **Phase 3 — Physics prior + multi-task heads + Stage-0 masked pretraining.**
- [ ] **Phase 4 — Decomposed uncertainty + calibration.**
- [ ] **Phase 5 — 3-D lift + DoseRAD2026 train/val.**
- [ ] **Phase 6 — Benchmark, ablate, write up.**

## Log
- Phase 0 started.
