Optimize the scan in `braggtransporter/models/mamba1d.py`. It is CORRECT but far
too slow: it runs a sequential Python `for` loop over ~680 depth steps, launching
hundreds of tiny MPS kernels per forward, so one training epoch takes minutes
(v0/FNO take seconds). Training is intractable. Fix the SPEED without changing the
math or the public API.

The SSM is DIAGONAL: per channel (inner_dim × d_state),
  h_t = dA_t * h_{t-1} + bu_t,   where dA_t = exp(delta_t * a),  bu_t = delta_t * B_t * u_t,
  y_t = C_t · h_t + D * u_t.
A first-order linear recurrence with per-step diagonal multiplier → parallelizable.

REQUIRED:
- Replace the per-timestep Python loop with a VECTORIZED / CHUNKED PARALLEL SCAN
  (e.g. a chunked associative scan: sequential across chunks of ~32–64, fully
  vectorized within a chunk; or a numerically-stable cumulative-product formulation).
  It must run and BACKPROP on Apple MPS with finite grads, and be dramatically
  faster (target: a 60-epoch run of ~544 samples finishes in a few minutes, i.e.
  seconds per epoch — same order as v0/FNO).
- Keep a private `_scan_reference` implementing the original naive loop, used ONLY
  by a test to prove equivalence.
- Do NOT change: forward signature, outputs {"dose","letd","r80"}, softplus dose,
  .param_count(), param initialization, or the module's public attributes.

ONLY edit:
- `braggtransporter/models/mamba1d.py`
- `tests/test_bt_mamba.py`  (add: (1) fast scan == reference scan within 1e-4 on a
  random (B=2,Nz=128) input, forward AND grads; keep the existing MPS backward test.)

VERIFY (this shell may lack MPS — that's fine, the orchestrator re-checks on MPS):
- `.venv/bin/python -m pytest tests/test_bt_mamba.py -q`
- print a quick timing: forward+backward for (B=8, Nz=680) before/after should show a
  large speedup (report the numbers).
Do NOT run git. Do NOT re-enable torch.use_deterministic_algorithms. Print a summary
with the equivalence max-abs-diff and the timing improvement.
