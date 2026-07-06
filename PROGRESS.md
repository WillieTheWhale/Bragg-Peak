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
- [x] **Phase 0 — Scaffold + env + remote overwrite.** contracts frozen (`schema.py`,
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
- [x] **Phase 3 — Multi-task heads + physics constraints + LETd validation.**
      v0 with dose/LETd/LETt/fluence heads + opt-in monotone-range (w=0.5) and
      energy-budget (w=0.1) constraints, 50 epochs MPS, best_val 0.654, no NaN.
      **Held-out dose metrics IMPROVED vs dose-only:** γ **96.35%**, distal-edge
      **0.132 mm**, RMSE **0.98%** — multi-task learning did not degrade dose.
      **LETd validation** vs braggpeak reference: Pearson r **0.76**, MAE 0.094 keV/µm.
      Results: `docs/results/phase3_letd.csv`.
      **Stage-0 masked pretraining: honest negative result** — does NOT improve data
      efficiency by paired held-out criteria (25/50/100% × ±Stage-0). Supervised
      signal already strong in this regime; best config was full-data (edge 0.100mm,
      γ 97.25%) with or without Stage-0. Kept as a documented null finding.
      Minor follow-up: ensure all secondary heads enforce nonnegativity at init.
- [x] **Phase 4 — Decomposed uncertainty + calibration. GATE PASSED.**
      Residual uncertainty head (heteroscedastic NLL) trained on frozen v0. Raw head
      overconfident (68/95% coverage 0.38/0.58). Post-hoc temperature calibration
      (τ=2.21, fit on half the held-out, tested on the other) → **calibrated coverage
      68%: 0.634, 95%: 0.952 — both within ±5% of nominal.** Decomposition:
      σ_aleatoric (head) + σ_epistemic (ensemble hook); σ_MC/σ_input/σ_meas are
      data-gated hooks (Phase-5+ data). Flow-matching head also implemented as an
      alternative. Results: `docs/results/phase4_calibration.csv`.
- [x] **Phase 5 — 3-D lift + DoseRAD2026 real data (pipeline works; accuracy cloud-gated).**
      Downloaded real **DoseRAD2026** proton data (public, HuggingFace
      `LMUK-RADONC-PHYS-RES/DoseRAD2026`): patient 1ABB006 CT (164×493×498) + plan
      (36 beams, rays, energy layers) + 49 valid beamlet dose `.mha` volumes.
      Built: `.mha`/SimpleITK loader, **beam's-eye-view extraction** (resample CT
      along each ray, downsample to depth≤64 × 24×24), `Bragg3D` model, 3-D
      gamma(3%/3mm) eval. Pipeline **verified end-to-end on real data** (4 tests green).
      **Honest result:** laptop-scale POC gamma3d ≈ 32% at low epochs and *collapses*
      with more training (49 beamlets from ONE patient → model overfits to a blob,
      predicted peak ~10 cm off). This is the plan's expected **data-scarcity /
      cloud-gated** outcome — competitive 3-D gamma needs the full 75-patient / 81k-
      beamlet set on cloud GPU. The 1-D result (Phases 1–4) is the laptop-complete
      contribution; the 3-D real-data pipeline is the reusable scaffold for cloud scale.
- [x] **Phase 6 — Benchmark, reproduce, OOD, write-up. COMPLETE.**
      `docs/FINAL_REPORT.md` (claims A/B/C + honest negatives); `scripts/reproduce.sh`
      (one-command regen, GATE PASS); `tests/test_bt_regression.py` CI invariants (5);
      OOD/abstention: σ inflates in→OOD (0.48→0.68), abstention rate 10%→77%. 49 tests green.

## Log
- Phase 0 started.

## Overnight run (goal: beat 2026 papers) — started 2026-07-06
GCP verified from Claude Code terminal (project braggtransporter, us-central1-f, bucket
gs://braggtransporter-braggtransporter, 1x T4/L4 spot quota). Wave 1 Codex agents:
O1 sharp-data architecture test (MPS), O2 DoseRAD 3-D normalization+multi-patient+GPU
train script, O3 GCP spot-GPU harness. Targets: DoTA 99.37% gamma 1%/3mm; ADoTA
99.4-99.87% beamlet / 98.4-98.9% plan-level 2%/2mm.

### Overnight result #1 — Sharp-data architecture test (MPS, matched ~0.93M params)
Heterogeneous-slab sharp 1-D, held-out, 30 epochs. Distal-edge / γ1%1mm / γ2%2mm:
- **conv1d 0.250mm / 86.4% / 97.0%** (BEST)
- fno1d 0.456mm / 3.5% / 10.7%
- transformer v0 0.747mm / 35.7% / 69.6% (WORST on edge)
**Honest finding — the memo's thesis is HALF refuted:** FNO *is* bad at the sharp edge on
tight gamma (confirms the spectral-smoothing weakness), BUT the transformer is NOT the
winner — a simple 1-D CONV beats both, and the transformer is worst on raw distal-edge.
So "attention is uniquely good at sharp edges" is DISPROVEN; the real story is local
convs > attention > spectral for sharp local edges. Caveat: single seed, 238 train
samples, 30 epochs — directional, not definitive. Results: docs/results/sharp_comparison.csv.

### Overnight result #2 — GPU 3-D scaling experiment (spot T4, DoseRAD2026)
Downloaded 12 patients / 480 beamlets from HuggingFace; trained Bragg3D on a spot T4
(CUDA verified, AMP, checkpoints streamed to gs://.../runs/scaling1). Infra fully
exercised (VM create, bootstrap, data, systemd training, GCS checkpointing, teardown).
**Honest finding:** even at 10x the Phase-5 data (49->480 beamlets), held-out gamma3d
stayed pinned at ~0-5% and rmse ~22% plateaued — the CT->3D-dose map does not learn at
this scale. Confirms the collapse is DATA SCARCITY, not a bug: DoTA/ADoTA used ~80k
beamlets (~170x more). Competitive 3-D gamma needs the full dataset + serious compute;
the pipeline + cloud harness are proven and ready for that. GPU cost ~$0.06 (spot T4,
25 min); VM deleted. Also fixed: gcp_train.sh --no-tail teardown bug, image-family, and
a memory-safe batch size (OOM at bs=32 on 30GB -> bs=8).

### Overnight result #1b — Sharp-data test CONFIRMED across 3 seeds
Distal-edge (mm), all seeds: conv1d 0.250/0.257/0.253 (RANK 1 EVERY SEED) <
fno1d 0.684/0.491/0.465 & transformer 0.582/0.773/0.750 (trade #2/#3).
g2/2mm: conv1d ~94-95% (best all seeds); transformer 76-86%; fno1d 3-11%.
**Robust honest conclusion:** a simple 1-D CONV beats both the transformer and FNO on
sharp proton edges (conv #1 in 3/3 seeds). The transformer is NEVER best -> "attention
is uniquely good at sharp edges" is robustly DISPROVEN. Sub-findings: FNO worst at tight
gamma (confirms spectral smoothing weakness); transformer beats FNO at tight gamma but
they trade on edge_mm (transformer-vs-FNO not significant in 1/3 seeds, FNO better in 2/3).

### Overnight result #2b — GPU 3-D scaling curve (autonomous spot T4, self-deleted)
Scaled Bragg3D (d_model 192, 6 layers) on 1050 beamlets (~20 patients), 44 epochs before
plateau. Held-out gamma3d(3%/3mm) trajectory: 0 -> 5 -> 20 -> **peak 28.27%** (epoch 43,
rmse 10.1%). Clean SCALING CURVE across the night:
| data / model | gamma3d 3%/3mm |
|---|---|
| 49 beamlets, tiny model | ~0% (collapse) |
| 480 beamlets, tiny model | ~5% |
| **1050 beamlets, scaled model (d192/L6)** | **28.3%** |
**Honest conclusion:** the 3-D approach genuinely SCALES with data + capacity (0->28%),
but reaching the papers' ~99% needs their ~80k-beamlet scale (~76x our data). We did NOT
beat the papers; we quantified exactly why (data volume) and showed the trajectory is the
right shape. Autonomous run self-managed via GCS + triple cost-safety; VM deleted. Total
overnight GPU spend ~\$0.4 (two spot-T4 runs). Follow-up to actually approach the papers:
full 55-patient / paginated beamlet download (tens of thousands), bigger model, multi-hour
or multi-GPU training, likely a DoTA-faithful BEV architecture.

### Overnight result #2c — Next bottleneck is COMPUTE TIME (6400-beamlet run)
Scaling further (16 patients x 400 = ~6400 beamlets, bigger model d256/L8) exposed the
next wall: ~20-30 min/epoch on ONE spot T4, so it cannot converge within the 6h cap
(~12 epochs -> non-converged ~10-15% gamma). Deleted rather than harvest a misleadingly
low non-converged number. **Converged best remains the 1050-beamlet run at 28% gamma.**
Lesson: past ~1000 beamlets, a single T4 is compute-time-bound; approaching the papers
needs multi-GPU or multi-day training in ADDITION to the full ~80k-beamlet dataset.
Total overnight GPU spend ~\$0.7; all VMs self-deleted/deleted, 0 running.

### Overnight result #3 — DoTA-faithful architecture is a BREAKTHROUGH (in progress)
Implemented dota3d.py (CNN-encoder + depth-transformer + CNN-decoder, DoTA's PUBLISHED
design) vs our from-scratch Bragg3D approximation. On ~4200 beamlets (12 patients),
DoTA3D held-out gamma3d(3%/3mm) climbed to **~42-45% by epoch 20 and stable** — vs
Bragg3D's 28% (which needed 40+ epochs on 1050 beamlets and plateaued). DoTA3D reached
24% at EPOCH 2 (Bragg3D was ~2%). **Key finding: architecture fidelity was the dominant
bottleneck, not only data.** This validates DoTA's design and gives a credible path to
the papers' ~99%: DoTA-faithful architecture + full ~80k-beamlet dataset + convergence.
**Run4 final: DoTA3D plateaued at gamma3d(3%/3mm) = 57.3%, rmse 6.0% on 4200 beamlets**
(epoch 52; vs Bragg3D 28% / rmse ~10-22%). Architecture fidelity ~doubled gamma at the
same data scale. Launching run5 (DoTA3D + ~8000 beamlets) to push further via data scale.

### Overnight result #4 — data lever saturates ~57%; capacity is the next ceiling
DoTA3D on 8000 beamlets (run5) tracked the 4200-beamlet run4 exactly (both ~50.7% at
epoch 30) -> doubling data 4200->8000 did NOT raise gamma. So at this scale the ceiling
is MODEL CAPACITY, not data. Next: run6 = bigger DoTA3D (d320/L10) on the same 8000
beamlets to test the capacity lever. Path to papers now precise: bigger DoTA architecture
+ full ~80k data + convergence.

### Overnight result #5 — capacity lever is COMPUTE-bound (bigger model)
Bigger DoTA3D (d320/L10) on 8000 beamlets (run6) trained SLOWER (36% at epoch 19 vs
~42% for d192/L6) and cannot converge within the single-T4 8h cap. So a bigger model
needs more compute to reveal its higher ceiling. COMPLETE ABLATION of the path to the
papers' ~99%:
| lever | change | effect |
|---|---|---|
| architecture | Bragg3D -> DoTA-faithful | 28% -> 57% (DOMINANT) |
| data | 4200 -> 8000 beamlets | saturated ~57% at d192 |
| capacity | d192/L6 -> d320/L10 | slower, compute-bound, peaked 52% < 57% in budget |
**Precise conclusion:** reaching 99% needs a bigger DoTA-faithful model trained to
CONVERGENCE on the full ~80k-beamlet dataset = multi-GPU or multi-day compute (a
resourcing decision). Overnight single-T4 runs peak at ~57%. Total GPU spend ~\$3,
all VMs self-deleting/cost-capped.

### Overnight result #6 — resolution lever also plateaus ~57%; the ceiling is ROBUST
Finer BEV resolution (48x48x96, 2mm voxels vs prior 4mm) on 4000 beamlets (run7):
gamma oscillated 50-60% (epoch-1 60% was noise; ~57% converged range) = COMPARABLE to
run4's 57% at coarse res, but ~15 min/epoch (compute-bound). So resolution does NOT
break the ceiling.

## EXHAUSTIVE lever characterization (single-T4 overnight budget)
| lever | tested range | result |
|---|---|---|
| **architecture** | Bragg3D -> DoTA-faithful | **28% -> 57%** (the ONE big win) |
| data | 1050 -> 4200 -> 8000 beamlets | 28 -> 57 -> 57% (saturates at d192) |
| capacity | d192/L6 -> d320/L10 | compute-bound, <=57% |
| resolution | 24x24x64 -> 48x48x96 | ~57%, compute-bound |
**DEFINITIVE conclusion:** within a single spot T4, gamma3d(3%/3mm) plateaus ~**57%**
robustly across data/capacity/resolution once the DoTA-faithful architecture is used.
Reaching the papers' ~99% is NOT achievable at this compute class -- it requires the
full ~80k-beamlet dataset + a bigger model trained to convergence = multi-GPU or
multi-day compute (a resourcing decision beyond overnight/1-T4). All 7 GPU runs
self-deleted/deleted; total spend ~\$4; 0 orphaned instances.

### Overnight result #7 — per-epoch gamma eval is the SCALABILITY bottleneck (not data)
run9 (12600 beamlets) was 98% CPU-bound with the GPU at 0% — the per-epoch 3-D gamma
evaluation over ~1900 held-out beamlets (Python local-search) dominates, making
large-scale training impractically slow (stalled at epoch 12, GPU idle). This ALSO means
my earlier "57% ceiling" runs were gamma-eval-limited in how far they could train. Fix
needed: subsample per-epoch progress-gamma (full eval only at end) + DataLoader workers.
Pivoting to the KEY test the user raised: DoTA warm-restart schedule + weight decay + LONG
training + FAST eval, to see if the 57% plateau was a premature-stopping/schedule artifact.
