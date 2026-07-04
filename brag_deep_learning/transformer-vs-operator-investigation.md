# Rigorous Investigation: Should the Backbone Be a Transformer or a Neural Operator?

_A deliberate stress-test of the `six-month-model-plan.md` architecture choice.
Conclusion up front: **my original FNO-first recommendation is disproven. The
backbone should be a transformer.** Below is the evidence, including the parts
that contradict my earlier plan._

---

## 0. Why this memo exists

The first plan proposed a **Fourier Neural Operator (FNO)** trunk with
state-space mixing and argued a transformer was the "lazy catch-up" choice. That
argument rested on two load-bearing claims:

1. **Efficiency:** transformers are O(N²) and too expensive; FNO/SSM are cheaper.
2. **Novelty:** DoTA/ADoTA already did transformers, so a transformer is not the
   frontier.

Both claims collapse under scrutiny. Neither survives contact with the 2026
literature or with DoTA's own numbers. I was wrong on the central architectural
call. Here is the proof, organized as four falsifiable questions.

---

## 1. Is a transformer actually too expensive here? — **No. The efficiency argument was a non-problem.**

The whole FNO/SSM efficiency case assumed the attention sequence is long. It is
not. DoTA's public config (`github.com/opaserr/dota/hyperparam.json`) tokenizes a
beamlet as **150 depth slices + 1 energy token = 151 tokens**, each a
low-dimensional projection (6×6×12 ≈ 432-dim), 16 heads, a handful of layers,
trained 56 epochs on a single GPU.

- Attention cost at that scale is **151² ≈ 23,000** pairwise terms — utterly
  negligible. Quadratic attention is only a problem when N is large; at
  beamlet/depth-sequence scale it is free.
- DoTA predicts a beamlet in **5 ms** with **99.37% gamma at 1%/3mm** — faster
  and more accurate than the FNO proton-transport surrogate I built the plan
  around (which needs ~23 s per beam and hits 99.95% gamma only because it
  reports a coarser metric on a laterally-homogeneous cylinder).
- The model is small enough to train on an M4 Max. My "need FNO/SSM to fit the
  laptop budget" premise was false: **DoTA-scale transformers already fit.**

**And even at full 3D resolution, where N does get large, the transformer still
wins on cost**, because 2026 operator-transformers are no longer O(N²):

- **Transolver's Physics-Attention** groups N mesh points into M ≪ N learned
  physical "slices" and attends among slices → **O(NMC + M²C), linear in N**
  (arXiv 2402.02366, 2502.02414, 2602.04940 Transolver-3 scales to
  hundred-million-point meshes).
- **Linear Attention Neural Operators** replace softmax to get **O(N)** while, per
  the authors, *"better preserving local feature information crucial for capturing
  rapid spatial variations"* than both FNO and softmax transformers
  (arXiv 2510.16816).

So: at beamlet scale, attention is free; at 3D scale, linear/Physics-Attention is
also O(N). **The efficiency justification for choosing FNO over a transformer does
not exist.**

---

## 2. Is FNO actually more accurate on the feature that matters? — **No. FNO fails at the Bragg peak's sharpest feature, and the failure is data-irreducible.**

This is the decisive finding, and it is specifically about *our* problem.

The single most important feature of a Bragg curve is the **distal falloff** — the
80%→20% dose drop over ~3–5 mm, the sharpest gradient in radiotherapy dosimetry
and the whole reason proton therapy exists. An architecture that smooths that
edge is disqualified regardless of aggregate gamma.

FNO smooths exactly that edge, by construction:

- **Spectral bias is architectural, not a tuning issue.** FNO truncates
  high-frequency Fourier modes, so it *"excel[s] at… low-frequency features while
  underrepresenting high-frequency modes… trace[d] to the truncation of
  high-frequency modes and poor resolution of sharp gradients"* (spectral-bias
  analysis, arXiv 2602.19265).
- **The error does not go away with more data.** The failure-mode study across
  Schrödinger/Poisson/Navier-Stokes/Black-Scholes/KS finds FNO error on *"sharp
  jumps, steep gradients, and localized high-frequency features"* is **"not
  substantially reducible through increased training data alone"** (arXiv
  2601.11428). Reported extremes reach **6× error** at discontinuities
  (spectral-bias literature). More cheap SDE data — my main laptop-budget
  advantage — would **not** fix the distal edge.
- Mitigations exist (SpecB-FNO residual ensembles, LOGLO-FNO local branches,
  frequency-aware losses) but they are **patches that bolt a local/high-frequency
  module onto FNO** — i.e. they concede FNO alone cannot do the sharp part.

Meanwhile the transformer-operator resolves sharp fronts as a *native* strength:

- Transolver's learned slices *"align with shock waves, material junctions, and
  fluid-structure interfaces — regions where FNO typically struggles"* (arXiv
  2402.02366). Quantitatively it beats FNO by **~42% on Navier-Stokes (0.090 vs
  0.156)** and **~47% on Darcy (0.0057 vs 0.0108)**, with **linear** complexity.
- Attention places representational capacity *where the signal is* (the peak and
  distal edge) instead of spreading it across a fixed truncated spectrum.

A Bragg peak is, physically, a **localized sharp front riding on a smooth
plateau** — the precise regime where the literature says FNO is weakest and
attention is strongest. Choosing FNO as the backbone would compromise the one
feature the entire project exists to get right.

---

## 3. Is a transformer really not the frontier here? — **No. The frontier operator *is* a transformer; DoTA just used a first-generation one.**

My "transformers are catch-up" claim conflated "a transformer" with "DoTA's 2022
transformer." The 2026 state of the art in scientific ML is a *family of
transformer operators* — Transolver/Transolver++/Transolver-3, Universal Physics
Transformers (arXiv 2402.12365), GAOT, DPOT, Unisolver, and the "Physics
Foundation Model" line — all attention-based, all beating FNO. The frontier move
is not "avoid the transformer." It is **"bring the 2026 operator-transformer
advances into proton dose, which still runs a 2022-era transformer (DoTA) or no
transformer at all."** Concretely, none of the published proton dose transformers
use: Physics-Attention slice tokenization, linear attention, flow-matching
generative heads, or multi-task LET/uncertainty. That gap *is* the contribution.

So the honest novelty statement flips: a next-gen **transformer** dose engine is
*more* frontier than an FNO one, not less.

---

## 4. Where would a transformer genuinely lose? — Intellectual-honesty check.

Being rigorous cuts both ways. The real, non-fatal risks of the transformer path:

- **Data hunger in general.** Transformers can overfit with little data. *Mitigant,
  and it is strong here:* the beam's-eye-view depth-slice tokenization is a heavy
  physical inductive bias (DoTA needed only 56 epochs), and our SDE generates
  effectively unlimited cheap, physically-consistent training data — the one thing
  transformers want most. Data is our surplus, not our scarcity.
- **The smooth plateau/bulk.** FNO's spectral view *is* efficient for the smooth,
  low-frequency bulk of the curve. This is the one place the operator idea keeps
  value — hence the corrected design keeps an **optional spectral/conv branch as a
  supporting term**, not the backbone (mirroring LOGLO-FNO/SpecB-FNO, but with the
  roles inverted: attention leads, spectral assists).
- **3D memory at full resolution.** Real, but solved by Physics-Attention/linear
  attention (§1) and by keeping the depth-autoregressive factorization so the
  model never holds the whole 3D volume in one quadratic attention.

None of these overturn the verdict; they shape the design.

---

## 5. Verdict

| Claim in original plan | Status after investigation | Evidence |
|---|---|---|
| Transformer is too expensive → use FNO/SSM | **Disproven** | DoTA = 151 tokens, 5 ms, single GPU; Physics-/linear attention is O(N) (2402.02366, 2510.16816) |
| FNO backbone is accurate enough | **Disproven for the distal edge** | Spectral bias is data-irreducible on sharp gradients; 6× error at discontinuities (2601.11428, 2602.19265) |
| Transformer is "catch-up," not frontier | **Disproven** | 2026 SOTA operators are transformers and beat FNO 42–47% (Transolver 2402.02366, ++/‑3) |
| Flow head, LET/uncertainty, physics constraints | **Retained** | Architecture-agnostic; still the biology+uncertainty frontier |
| Depth-autoregressive problem structure | **Retained** | Good factorization; keep it, swap the per-step operator to attention |

**Corrected architecture: a transformer is the backbone.** Specifically a
**linear / Physics-Attention operator-transformer** over the beam's-eye-view
depth-slice sequence (DoTA's proven formulation, modernized with 2026 attention),
carrying the flow-matching residual head, multi-task dose/LETd/range/uncertainty
outputs, and physics constraints. An optional spectral/conv branch assists only on
the smooth bulk. This is *more* efficient than DoTA (linear attention at 3D),
*more* accurate at the distal edge than any FNO backbone (attention resolves sharp
fronts), and *more* powerful than both (generative uncertainty + LET). The
user's instinct was correct; the plan is updated accordingly.

---

## Addendum (v3): softening "disproven"

A later review rightly noted that "FNO is disproven" is rhetorically stronger than
the evidence warrants. The precise, defensible claim is:

> **A pure FNO backbone is a risky default for distal-edge accuracy; it should be a
> baseline and an auxiliary plateau branch, not the main bet.**

Two reasons for the softer wording. First, the literature evidence is about a
*general* spectral bias; the *magnitude* of the distal-edge penalty for our
specific 1-D/beamlet Bragg problem is only established by our own same-data
ablation (plan §7, Month 2), not by the papers alone. Second, good research taste
wins on a brutal ablation table, not on declaring an architecture dead — so FNO
stays in the comparison as a first-class baseline, and a spectral/conv branch keeps
a real job on the smooth entrance plateau (where a Fourier view genuinely is
efficient). The transformer remains the *lead* choice; FNO is *demoted*, not
*deleted*. The v3 plan (`six-month-model-plan.md`) reflects this and reframes the
whole effort from "which backbone?" to "a full-stack learned, calibrated transport
system."
