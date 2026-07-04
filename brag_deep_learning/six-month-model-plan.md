# A Six-Month Plan for a Learned, Calibrated Proton-Transport Engine

_Research software only. No clinical claims, ever. This plan synthesizes 2026
state-of-the-art deep learning onto the existing `braggpeak` physics platform._

> **Locked decisions.** Framework = **PyTorch + MPS** (best M4 Max support,
> dominant in 2026 operator/diffusion research, clean cloud portability).
> Validation = **both** paths — iterate on the DoseRAD2026 train/val split (public
> from April 2026, CC BY-NC 4.0) plus our own matched-material Geant4 references,
> and use the challenge evaluation server for the official hidden-test score *if and
> when available* (the 40-patient test set is withheld until March 2030). Do **not**
> assume a locally-scorable held-out gamma; report a frozen internal validation
> split with a reproducible train/val protocol as the primary deliverable.
>
> **Version history.** v1 proposed a Fourier Neural Operator (FNO) backbone.
> v2 corrected that to a transformer after the head-to-head in
> `transformer-vs-operator-investigation.md`. v3 reframed the project from
> *"find the best backbone"* to *"build a full-stack learned transport system."*
> **v3.1 (this document)** tightens execution: digitized curves are pilot data not
> Claim-B proof; uncertainty channels must be made *identifiable* by design; the
> first build is a humble deterministic v0; the hybrid router is physically
> supervised and logged; the Month-2 ablation is split into runnable pieces. The
> v3 systems framing — physics prior, latent-state encoder, coordinate-query
> decoding, decomposed uncertainty, a multi-fidelity data engine with active
> learning, and — decisively — a measurement-calibration track, because you
> **cannot prove "more accurate than OpenGATE" using OpenGATE as the answer key.**

---

## 1. Thesis

The taste error in v1/v2 was treating this as an architecture contest. The 2026
AI-lab analogs for this problem are not medical dose-prediction papers; they are
**learned physical simulators** — GraphCast/GenCast, MeshGraphNets, and above all
**NeuralGCM**, which wins by combining a differentiable physical core with learned
corrections rather than replacing physics wholesale. None of them are "a cool
network." They are systems: data engine, physics prior, task framing, uncertainty,
speed, and a brutal benchmark loop.

So the revised thesis:

> We are building a **calibrated learned proton-transport operator**, not a
> black-box dose surrogate. It uses known stopping-power/material physics as a
> prior, a transformer/operator latent state to model nonlocal transport through
> heterogeneous matter, coordinate-query decoding for arbitrary-resolution dose
> and LET, and probabilistic residual heads that **decompose** uncertainty
> (Monte-Carlo noise, input/material, epistemic, measurement). It is trained on a
> multi-fidelity ladder from analytic physics to OpenGATE/Geant4/TOPAS and
> **calibrated against real measurements**. It claims success only in three
> *separately measured* regimes (§6): reproduce OpenGATE faster, improve agreement
> with real experimental data beyond OpenGATE, and behave trustworthily (knows
> when it doesn't know).

The single most important logical correction, stated plainly: **if OpenGATE writes
the answer key, the model can at best prove it copied the key efficiently. To beat
OpenGATE, the key must come from the lab.** That reshapes the data plan, the
training stack, and the success criteria below.

**Working name:** `BraggTransporter` (a learned transport engine), retiring the
architecture-first name "BraggFormer-Flow."

---

## 2. The four-layer architecture

The four layers are the *target* design. **Build them in order of earned
complexity, not all at once.** The first implementation is a deliberately humble
**BraggTransporter-v0**:

```text
physics-prior features → simple latent transformer encoder
  → coordinate-query decoder → deterministic dose/range heads → calibration wrapper
```

v0's one hard proof, before any router/SSM/spectral/flow machinery exists: *given
energy + material profile + query coordinate, does it predict dose and R80/R90
better than a plain MLP, an FNO, and a DoTA-style transformer on the same 1-D
data?* Only after v0 clears that bar do we add — each gated by an ablation that
earns it — SSM depth roll-out, the spectral plateau branch, the SIREN decoder, the
mixer router, and finally the flow/DiT decomposed-uncertainty head. This keeps the
v3 taste while preventing Month 2 from becoming an architecture swamp.

The full target system has four layers, each with a clear job.

```
Inputs (explicit, unit-tagged — extends SurrogateSample in ml_surrogate.py):
  per-depth material: density, RSP, I-value, material class
  beam: energy, energy spread, spot size; analytic beam-shape projection (ADoTA-style)
  fidelity tag: {PSTAR, Bortfeld, SDE, MCsquare, GATE/Geant4, TOPAS, measurement}

 ┌─ (1) PHYSICS PRIOR ─────────────────────────────────────────────┐
 │ Precompute known structure so the net never relearns range-energy:│
 │  water-equiv depth (WEPL), Bortfeld/PSTAR range features,          │
 │  CSDA stopping power, expected peak depth. NeuralGCM lesson:       │
 │  known physics in, learned *correction* out.                       │
 └───────────────────────────────────────────────────────────────────┘
 ┌─ (2) LATENT PHYSICAL-STATE ENCODER ───────────────────────────────┐
 │ Transolver/Perceiver/UPT-style transformer → learned physical      │
 │ state tokens (NOT just "depth slice 1..150"): distal-edge state,    │
 │ material-interface state, nuclear-halo state, beam-energy state,    │
 │ uncertainty state. Physics-/linear-attention → O(N) at 3D scale.    │
 └───────────────────────────────────────────────────────────────────┘
 ┌─ (3) COORDINATE-QUERY DECODER ────────────────────────────────────┐
 │ Answer queries (x,y,z, quantity): dose, LETd, LETt, fluence,       │
 │ R80/R90, uncertainty — at ARBITRARY coordinates (Perceiver-IO      │
 │ philosophy + SIREN-style MLP for sharp distal-edge interpolation).  │
 │ Decouples the model from one voxel spacing / fixed depth grid.      │
 └───────────────────────────────────────────────────────────────────┘
 ┌─ (4) DECOMPOSED PROBABILISTIC RESIDUAL ───────────────────────────┐
 │ Flow-matching / DiT residual head, distilled to ≤4 steps, but      │
 │ reporting SEPARATE channels where identifiable:                     │
 │  σ_MC (shot noise) · σ_input (CT/RSP) · σ_epistemic (ensemble/     │
 │  disagreement) · σ_meas (residual vs real data)                     │
 └───────────────────────────────────────────────────────────────────┘

Physics constraints (soft): dose ≥ 0 ; dRange/dE > 0 ; Σdeposit+escaped ≈ incident
```

**Hybrid mixers, chosen by physical regime — not by ideology.** The encoder is a
router/mixture over blocks, each owning the regime it is best at (Jamba/Mamba-2
show hybrids beat monocultures):

| Block | Owns | Why |
|---|---|---|
| Physics-/linear-attention (Transolver) | distal edge, material interfaces, nonlocal halo | attention concentrates capacity at sharp fronts; O(N) |
| SSM / Mamba along depth | long sequential transport roll-out | linear-time, memory-cheap depth propagation |
| Small conv / spectral branch | smooth entrance plateau | Fourier view *is* efficient for low-frequency bulk |
| SIREN-style coordinate MLP | high-gradient interpolation near distal edge | represents sharp continuous fields without grid aliasing |

**The router is physically supervised and logged — not a free black box.** It gates
on *interpretable* physical features, not raw activations:

```text
distance_to_predicted_range · local_density_gradient · local_RSP_gradient
material_interface_flag · depth_normalized_by_CSDA_range · energy_remaining_estimate
```

and we **log which expert is active where**. If the spectral branch is handling the
distal edge, or the SIREN branch fires everywhere, that is a red flag, not a
curiosity — the model must not just be accurate, it must use the right tool in the
right part of the beam path. The design question is not "which architecture is
correct?" but **"which regime should each block own?"** — and the Month-2 ablation
(§7) measures it.

**On FNO — softened from v2.** The correct claim is *not* "FNO is disproven." It is:
**a pure FNO backbone is a risky default for distal-edge accuracy, so it belongs as
a baseline and an auxiliary plateau branch, not the main bet.** The literature
(spectral bias is data-irreducible on sharp gradients, arXiv 2601.11428/2602.19265)
justifies demoting it; only our own same-data ablation justifies any stronger word.

---

## 3. The multi-fidelity data engine (the part v1/v2 under-built)

Scaling-law and self-supervised lessons (Chinchilla; MAE/DINOv2) say performance
depends on data *mixture and quality*, and that curated pretraining precedes
supervised fine-tuning. We build a **five-tier Bragg data ladder**, each sample
carrying its fidelity tag so the model always knows how much to trust its target:

1. **Cheap** — analytic/PSTAR/Bortfeld/SDE water + slab phantoms. Effectively free
   on the laptop (SDE is 14–21× faster than Geant4). Unlimited volume for
   pretraining and for the smooth bulk.
2. **Medium** — fast MC (MCsquare / FRED-style) where available; DoTA's public
   MCsquare beamlets (`github.com/opaserr/dota`, GPL-3.0) live here.
3. **High** — OpenGATE/Geant4/TOPAS with matched physics lists and materials
   (our `monte_carlo_gate.py`, sub-0.1 mm range today; the DoseRAD2026 MC labels).
4. **Real** — measured Bragg/IDD curves (§4). Scarce and precious; the *only* tier
   that can support a beat-OpenGATE claim.
5. **Adversarial** — bone/lung/air interfaces, dental fillings, range shifters,
   OOD energies. Generated cheaply from tiers 1–3 to stress the distal edge and to
   drive abstention training.

**Active learning, not passive training.** Run the expensive simulator/experiment
only where (a) the model is uncertain, (b) tier baselines disagree, or (c)
distal-edge error is large. This is what makes it frontier rather than "train on a
static dump." Concretely: an acquisition loop that ranks candidate
(material, energy) configs by predicted σ_epistemic and spends the Geant4/measure
budget there.

---

## 4. Datasets — simulator *and* measurement (all public / online)

**Simulator-side (proves competitiveness, Claim A):**

- **DoseRAD2026 Grand Challenge** — the anchor **Claim-A MC benchmark** (not a
  Claim-B measurement benchmark). `doserad2026.grand-challenge.org`, dataset paper
  arXiv 2604.12778, Zenodo `zenodo.org/records/19714006`. 115 patients (75 train /
  40 test), **81,000 proton beamlets**, open-source MC ground truth, **CC BY-NC
  4.0**, training data public from **April 2026**, test set **withheld until March
  2030**. Therefore: iterate on train/val, report a frozen internal validation
  split with a reproducible protocol as the primary number, and use the challenge
  evaluation server for the hidden-test gamma only if/when it opens. Never phrase a
  DoseRAD result as evidence for Claim B — it is MC-vs-MC.
- **DoTA repo + MCsquare beamlets** — `github.com/opaserr/dota` (GPL-3.0). Reproduce
  DoTA, then run BraggTransporter on the identical split (apples-to-apples).
- **SynthRAD2025** (parent of DoseRAD2026) — CT/MRI material realism + SPR-perturbation
  input-uncertainty experiments.
- **Own Geant4/OpenGATE + SDE ladder** — controlled ablations, matched materials.
- **NIST PSTAR** (vendored) — range-energy anchoring.

**Measurement-side (the ONLY path to Claim B — and it is genuinely hard).**
There is **no turnkey open repository of measured proton IDDs**; real data is
scarce, mostly paper-embedded or institution-controlled. Honest sourcing plan:

- **Digitize published benchmark curves** — e.g. the 67.5 MeV benchmark set
  (0.15%/0.3% stated accuracy, PMID 26133619), 78 MeV multi-chamber curves
  (PMID 11190961), and 130–235 MeV PTW Bragg-peak-chamber IDDs (PMC9415752).
  **Treat these as measurement-adjacent *pilot* calibration, not Claim-B proof.** A
  digitized figure is measurement → processing → figure rendering → screenshot →
  digitization: it can start Stage 3 and exercise the software path, but it is not
  strong Claim-B evidence unless the added error from plot digitization, detector
  response, energy spread, chamber geometry, and absolute normalization is
  explicitly modeled. A *strong* Claim B needs raw commissioning IDDs **with**
  detector/beamline metadata, chamber geometry, energy spread, spot size, water
  density/temperature corrections, and absolute-dose normalization.
- **RaLPh heterogeneous phantom** (115 MeV, film + IC, 0.2–0.3% MC agreement,
  ScienceDirect S1120179721003434) — a real heterogeneous range benchmark.
- **ESTRO-EPTN dosimetry guidelines** (PMC11364130) — defines the exact measured
  data format (IDD, in-air lateral profiles, absolute output) our schema targets.
- **Open GATE PBS commissioning model** (arXiv 2107.12184) — an open-source MC
  model validated against measurements; a bridge between tiers 3 and 4.
- **Collaboration/commissioning data** — pursue a data-sharing agreement with a
  proton center for de-identified commissioning IDDs. Flagged as the project's
  **critical external dependency** (§8): if it lands, Claim B is reachable; if not,
  the project still fully delivers Claims A and C.

Ground-truth hygiene (existing AGENTS.md rule): reference outputs stay strictly
separate from candidate outputs, materials/I-values matched exactly — the
`add_material_weights` I-value bug in STATUS.md is exactly how one fakes an
accuracy gain.

---

## 5. Training stack — five stages, not two

**Stage 0 — Masked transport pretraining (new).** Mask chunks of material /
depth-dose / energy / LET profiles and reconstruct them; add next-slab prediction
(given upstream state, predict downstream dose/energy/fluence). MAE/foundation-style
self-supervision on unlimited tier-1 data, before any supervised regression.

**Stage 1 — Supervised physics imitation.** Train on tiers 1–3 with the fidelity
tag as an explicit input, so the model knows whether a target is PSTAR, SDE,
MCsquare, GATE, or TOPAS.

**Stage 2 — High-fidelity residual learning.** Learn the correction from cheap
physics → high-fidelity MC (the NeuralGCM pattern: predict the residual, not the
whole field).

**Stage 3 — Measurement calibration.** Learn the correction from high-fidelity MC →
**real measured data**, with uncertainty. The only stage that licenses a
"more accurate than OpenGATE" claim.

**Stage 4 — Abstention / OOD.** Train the model to output "I don't know" when
material, geometry, or beam conditions leave the validated domain — a deployment
behavior, not an afterthought.

**Making the four uncertainty channels *identifiable* (not just four output
numbers).** A network does not learn to separate `σ_MC / σ_input / σ_epistemic /
σ_meas` because we give it four heads; it only can if the *data* contains
situations where **only one** source varied. Each channel therefore requires a
specific training intervention:

| Channel | What makes it identifiable |
|---|---|
| `σ_MC` (shot noise) | noisy-vs-high-statistics MC pairs at *known* particle/history counts |
| `σ_input` (CT/RSP) | controlled CT/RSP/SPR perturbation ensembles, same geometry |
| `σ_epistemic` (model) | ensemble / bootstrap-training disagreement; active-learning disagreement |
| `σ_meas` (reality) | paired MC-vs-measurement residuals *with* detector/beamline metadata |

Without these interventions the model will learn one generic error bar and split it
arbitrarily. Calibration (§6, Claim C) is therefore validated **per channel** on
held-out cases where that channel's source was the one deliberately varied — not
just on aggregate coverage.

---

## 6. Success criteria — three *separately proven* claims

Do not let evidence for Claim A be reported as Claim B.

- **Claim A — Faster OpenGATE-like simulation.** Reproduce Geant4/OpenGATE within
  strict tolerances (peak-depth ≤ 0.5 mm, R80/R90 ≤ 0.7 mm, RMSE ≤ 1.5% water,
  plan-level gamma ≥ 98.5% at 2%/2mm on the DoseRAD2026 **validation** split and
  our own Geant4 references), materially faster (ms/beamlet, competitive with
  ADoTA's 1.72 ms). "Same answer, faster." (Hidden-test gamma reported later if the
  challenge server opens.)
- **Claim B — Better measured-data agreement.** Beat Geant4/OpenGATE on *real*
  measured Bragg/heterogeneous-phantom data (§4). "More accurate than OpenGATE."
  Gated on the measurement-data dependency; scoped small and honest.
- **Claim C — Trustworthy behavior.** Calibrated *decomposed* uncertainty
  (empirical 68/95% coverage within ±5%), graceful OOD degradation, abstention,
  auditable one-CLI reproducibility. "Could anyone trust it?"

---

## 7. Six-month timeline

**Month 1 — Data engine + baselines + v0.** PyTorch/MPS harness; extend
`SurrogateSample` with fidelity tags + coordinate-query targets. Build the 5-tier
data ladder and the active-learning acquisition-loop skeleton. Download DoseRAD2026;
reproduce DoTA (the number to beat). Digitize the first published measured IDDs.
Stand up **BraggTransporter-v0** (physics prior + simple transformer encoder +
coordinate decoder + deterministic heads + calibration wrapper).

**Month 2A — Representation ablation (1-D).** On identical 1-D data, distal-edge
scored: fixed-grid vs coordinate-query decoder · physics-prior vs none ·
distal-edge-weighted vs uniform loss. Settles the *representation* before touching
backbones.

**Month 2B — Backbone ablation (1-D).** Same data, **same parameter budget**, same
distal-edge metrics: small transformer vs FNO vs Mamba/SSM (and softmax vs linear
vs Physics-Attention within the transformer). This is where v2's distal-edge claim
is confirmed or refuted empirically — the experiment, not the argument, decides.

**Month 3 — Physics prior + multi-task heads.** Wire WEPL/Bortfeld/CSDA features;
add LETd/LETt/range/fluence query heads with constraints; validate LETd vs
Geant4/TOPAS. Stage-0 masked pretraining stood up.

**Month 4 — Decomposed uncertainty + calibration.** Flow/DiT residual with separate
σ channels, each trained with its identifiability intervention (§5); distill to ≤4
steps; **uncertainty ablation** (deterministic vs ensemble vs flow) scored on
per-channel calibration/coverage; MC-denoising on Geant4 noisy/clean pairs. Begin
Stage-3 on digitized (pilot) measured curves. (Most cloud-heavy.)

**Month 5 — Lift to 3-D + measurement track.** Coordinate decoder to 3-D beamlets;
train on the DoseRAD2026 train/val split; report the frozen internal validation
number (challenge-server submission only if it is open). Push Stage-3 measurement
calibration as far as the acquired real data allows.

**Month 6 — Benchmark, ablate, write up.** Report Claims A/B/C *separately*;
SPR-perturbation input-realism; OOD + abstention; failure catalog; CI
reproducibility gate; manuscript + released weights/config.

---

## 8. Compute budget & the critical dependency

- **Laptop-first.** Tiers 1–3 data generation and Stages 0–2 in 1-D/multi-task run
  entirely on the **M4 Max (128 GB)** via PyTorch MPS. At beamlet scale the depth
  sequence is short (~150 tokens → attention ≈ 23k terms, free); the model is
  DoTA-scale (~10–30M params). Unlimited cheap SDE data is a surplus a transformer
  wants.
- **Cloud only where it pays.** Flow/distillation (Month 4) + full 3-D DoseRAD2026
  (Month 5): ~200–400 GPU-hours on one A100/H100 spot (~$1–2/hr) ≈ **$250–500**.
- **Critical external dependency = measurement data.** Claim B lives or dies on
  acquiring real measured curves (digitized published sets get it started; a
  center collaboration makes it strong). This is explicitly *decoupled* from Claims
  A and C, which are fully deliverable on the laptop + public simulator data alone.
  If the measurement track stalls, the project still ships a calibrated, uncertainty-
  aware, faster-than-OpenGATE transport engine — just without the beat-OpenGATE claim.

---

## 9. Clinical-trust pathway (unchanged direction, sharpened)

Adoption path: research tool → QA/second-check → adaptive-planning accelerator →
locked, validated dose-engine component. It earns trust by behaving like a fast
*instrument that reports its calibration range*, not magic AI: locked version,
explicit input domain, decomposed calibrated uncertainty, OOD rejection, audit logs,
site commissioning, conventional-calc fallback. FDA's predetermined change control
plan (PCCP) guidance exists precisely because adaptive models must have specified,
validated, controlled updates — the abstention + decomposed-uncertainty design maps
onto that lifecycle rather than fighting it.

---

## 10. One-paragraph summary

Build **BraggTransporter**: a learned proton-transport operator that uses known
stopping-power and material physics as a prior, learns residual transport structure
from a multi-fidelity ladder (analytic → SDE → fast MC → Geant4/OpenGATE/TOPAS), and
calibrates its final residuals against real measurements **when available**. Start
humble — a deterministic v0 (physics prior + simple transformer encoder +
coordinate-query decoder + calibration wrapper) that must first beat a plain MLP,
an FNO, and a DoTA-style transformer on the same 1-D data — then add, each earned by
an ablation, a Transolver/Perceiver latent state, hybrid regime-specialized mixers
under a *physically-supervised, logged* router (attention owns the distal edge, SSM
owns depth roll-out, spectral/conv owns the plateau), and a flow-matching head whose
*decomposed* uncertainty channels are each made identifiable by a matched data
intervention. It makes **three separate claims** — OpenGATE-like accuracy *faster*
than OpenGATE (public MC), measurement-*superior* accuracy *only when real data
supports it* (the gated stretch goal, on raw commissioning data, not digitized
pilots), and *trustworthy* behavior (decomposed calibration + OOD abstention) —
never conflating the first for the second. Sized for an M4 Max plus ≤ $500 of spot
GPU over six months, with the measurement track as the one honestly-flagged
external dependency.
