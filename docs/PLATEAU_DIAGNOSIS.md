# Why the 3-D model plateaus at ~63% gamma — root-cause diagnosis (2026-07-07)

Question: are we compute/data-limited vs DoTA/ADoTA, or stuck on a pipeline bug?

## Answer: it is an ARCHITECTURE bug, not compute, not the reward pipeline.

### 1. NOT a compute deficit
DoTA trained 56 epochs (up to 118 on a new dataset) on an **Nvidia T4**. We trained run11
for 150–360 epochs on the **same T4 class** with warm restarts + weight decay. So we have
**matched or exceeded the papers' training compute.** Compute is not the limiter.

### 2. Reward (gamma) pipeline is CORRECT
`gamma_index_3d` is a standard local-search global gamma (3%/3mm, 10% low-dose mask),
if anything **more lenient** than the papers' 1%/3mm. Train target and eval both use the
same per-beamlet unit-max normalization; gamma is scale-invariant. No bug here.

### 3. ROOT CAUSE — the "DoTA3D" model is blind to lateral position
`braggtransporter/models/dota3d.py` encodes each transverse slice with
`AdaptiveAvgPool2d(1)` — it **averages the entire (H,W) plane to one vector per channel
BEFORE the transformer.** Empirical proof: feeding two inputs that differ ONLY in lateral
position (a dose blob at (5,5) vs (18,18)) yields encoder tokens that differ by **6e-8
(numerically identical)**. The decoder then reconstructs each (H,W) slice from a **4×4
learned seed**. Consequences:
- The model can learn the DEPTH/Bragg-peak profile (transformer runs along depth) — which
  is why we get ~63%, not 0% — but it **cannot place dose laterally**; it emits a generic
  blurry transverse blob. That caps gamma exactly where lateral precision matters.
- This is **NOT DoTA-faithful.** DoTA preserves transverse structure via patch tokens /
  spatial feature maps. Our model was built by Codex from a prompt referencing paper files
  that **do not exist in the repo**, so the transverse-preserving design was never encoded.

### 4. Secondary factors (real but not the main cap)
- **Data scale:** ~6k beamlets (run11) vs the full ~55-patient / ~60–80k set (~10%).
- **Input resolution:** BEV is 24×24 lateral over 96 mm = **4 mm voxels**, coarser than the
  3 mm gamma DTA; and only 3 channels [HU, density, RSP] with **no beam-fluence channel**.

## The fix (real path past 63%)
Replace the average-pool encoder + 4×4-seed decoder with a **spatially-preserving**
design: patch-embed or conv encoder that keeps the (H,W) feature map, transformer that
mixes along depth while preserving lateral features, U-Net-style conv decoder from
full-resolution features. Add a beam-fluence input channel and finer lateral voxels.
THEN scale data. This — not more epochs or grokking — is the lever to approach the papers.

## Consensus audit (2026-07-07): me + Codex (blind, adversarial) independently agree

Codex audited BLIND (not told my suspicions) and ADVERSARIALLY (told to default to "code is
correct", write the strongest defense first, only flag a bug it could disprove). It still
confirmed TWO bugs — genuine consensus, not sycophancy. Both verified empirically.

### CONSENSUS BUG 1 (CRITICAL) — model is blind to tissue stopping power
- Me: `hu_to_density_rsp` sets `rsp = density.copy()` and a no-op `np.where` -> density==RSP
  (duplicate channels), crude linear conversion, no real stopping-power contrast.
- Codex (the mechanism I MISSED): the 3 channels are linearly dependent ([a, 1+a, 1+a]), and
  `slice_norm = GroupNorm(1, c_in)` then REMOVES the absolute level -> a homogeneous water
  slab and a dense slab become IDENTICAL tensors before the transformer. VERIFIED: after
  GroupNorm, water [0,1,1] == dense [1,2,2]. So the model cannot see tissue density, the
  variable that sets Bragg-peak DEPTH. Concrete failure: same geometry/energy, water vs
  dense -> model predicts same peak depth; real peak differs by >3mm -> gamma fails.
- Fix: real piecewise HU->density and HU->RSP (density != RSP); do NOT normalize away the
  absolute CT/RSP level (fixed dataset-constant standardization, not per-slice GroupNorm);
  add a cumulative-WEPL physics-prior channel for peak depth.

### CONSENSUS BUG 2 (HIGH) — physical depth scale hidden from the model
- Me: depth spacing is a fixed ~7.8mm (coarser than the 3mm gamma DTA) and NOT range-adapted;
  a 45MeV beamlet's peak occupies ~4 of 64 depth bins.
- Codex: the variable physical depth spacing (7.1-8.0mm, patient/gantry-dependent) is computed
  but NEVER fed to the model; it gets only index positions + 4 scalars. Two BEVs spanning
  450 vs 500mm with identical normalized input must output the same depth INDEX -> ~20-25mm
  peak error. Energy->physical-range is unlearnable without the mm scale.
- Fix: feed physical depth spacing/coords to the model AND/OR resample every BEV to a fixed
  physical-mm, range-adapted depth grid (fine ~1mm spacing so the peak is well resolved).

### Independently judged CORRECT by BOTH (consensus on what's NOT broken)
BEV geometry/axis ordering, ray direction, dose-file<->metadata matching, train/eval
normalization consistency (the gamma "reward" is fine), the scalar/energy path exists.

### Where we DIVERGED (honest)
I flagged lateral 4mm voxels as a bug; Codex did NOT independently flag lateral resolution.
Downgrade: lateral coarseness is a limitation, not a confirmed bug. The confirmed, consensus
root causes are the MATERIAL-BLINDNESS (critical) and DEPTH-SCALE (high) bugs above.
