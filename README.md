# Bragg-Peak — BraggTransporter (v3.1)

A **learned, calibrated proton-transport engine** for Bragg-peak dose simulation,
built on top of a validated first-principles physics platform.

> **Research software only. No clinical claims, ever.**

Two layers live in this repo:

1. **`braggpeak/`** — the physics engine and data source: analytic Bortfeld/PSTAR
   stopping power, a stochastic **SDE Monte Carlo** transport model, and an
   OpenGATE/**Geant4** reference adapter. The SDE matches Geant4 range to
   **sub-0.1 mm** at 150/200 MeV and is ~14–21× faster. This is tier-1/tier-3 of
   the data engine and the source of the physics prior.
2. **`braggtransporter/`** — the deep-learning project (this is v3.1): a
   physics-prior → latent transformer encoder → coordinate-query decoder →
   decomposed-uncertainty system, trained on a multi-fidelity data ladder and
   evaluated on public benchmarks (DoseRAD2026) plus real measurements.

The full research plan is `brag_deep_learning/six-month-model-plan.md`; the
architecture decision record is `brag_deep_learning/transformer-vs-operator-investigation.md`;
the module contract is `braggtransporter/INTERFACES.md`; build status is `PROGRESS.md`;
the **final consolidated report (honest claims A/B/C, results, negatives) is
`docs/FINAL_REPORT.md`**.

**Build status:** Phases 0–6 complete. The 1-D learned-transport system (physics
prior + transformer/coord-query backbone + multi-task LET heads + calibrated
uncertainty + OOD abstention) is laptop-complete and gated; the 3-D DoseRAD2026
pipeline runs on real data but competitive 3-D accuracy and Claim-B measurement
validation are cloud/data-gated (as the plan flagged up front). Regenerate with
`bash scripts/reproduce.sh` and `pytest tests/test_bt_*.py`.

## Three separately-proven claims (the scientific contract)
- **A** — reproduce OpenGATE/Geant4 *faster* (public Monte-Carlo benchmarks).
- **B** — *beat* OpenGATE/Geant4 against **real measurements** (gated on measurement data).
- **C** — behave *trustworthily*: decomposed calibrated uncertainty + OOD abstention.

Claim-A evidence is never reported as Claim B: OpenGATE can only be the answer key
for copying OpenGATE.

## Setup
```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch numpy scipy matplotlib pyyaml tqdm einops h5py pandas scikit-learn
```
(The default system Python here is 3.14, which has no torch wheels yet; the project
runs on the 3.11 venv. Training uses Apple MPS.)

## Quick start (Phase 1)
```bash
.venv/bin/python -m braggtransporter.data.generate            # build the 1-D dataset
.venv/bin/python -m braggtransporter.train --config configs/bt_v0_1d.yaml
.venv/bin/python -m braggtransporter.evaluate --ckpt experiments/bt/best.pt
```
