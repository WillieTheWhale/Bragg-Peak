Build a cost-safe GCP spot-GPU training harness for BraggTransporter (overnight goal #2 infra).

CONTEXT: GCP is set up and CLI-verified. See docs/gcp_env.md for PROJECT_ID
(braggtransporter), REGION (us-central1), ZONE (us-central1-f), BUCKET
(gs://braggtransporter-braggtransporter). Quota = 1 GLOBAL GPU: 1x NVIDIA T4 or L4
SPOT/preemptible in us-central1. The training entrypoint is
scripts/train_doserad_gpu.py (written by another agent; assume its CLI:
`--patients ... --max-beamlets N --epochs N --device cuda --gcs gs://... --resume`).

ONLY create (shell + small python, no edits to existing modules):
- scripts/gcp_train.sh — one command that: (1) creates a SPOT VM in us-central1-f with 1x
  T4 (fallback L4) using a Deep Learning PyTorch CUDA image
  (--image-family from deeplearning-platform-release, e.g. pytorch-latest-gpu),
  g2-standard-8 for L4 or n1-standard-8 + nvidia-tesla-t4 for T4, 100GB boot disk,
  --provisioning-model=SPOT, --maintenance-policy=TERMINATE, and a STARTUP SCRIPT that
  runs scripts/gcp_bootstrap.sh; (2) waits for the VM, streams its serial/console or
  SSH-tails the training log; (3) on completion (or a --teardown flag) DELETES the VM.
  Must be idempotent and print the exact SSH + delete commands. NEVER leave a VM running:
  trap EXIT to offer teardown; print a loud reminder of the running instance name.
- scripts/gcp_bootstrap.sh — runs ON the VM: clone github.com/WillieTheWhale/Bragg-Peak
  (or copy if provided), create a python venv, pip install torch (CUDA build present in
  the DL image) + repo deps + SimpleITK + huggingface deps, download the DoseRAD subset
  via the loader's download helper, then run scripts/train_doserad_gpu.py with
  checkpointing to the GCS bucket. Make it RESUMABLE (spot preemption): on restart, pull
  the latest checkpoint from GCS and pass --resume.
- scripts/gcp_watch.py — poll: instance status, current GCS checkpoint/metrics, and
  today's spend if available (`gcloud billing` best-effort); print a compact status line;
  exit when training is done or the VM is gone. Cost guard: warn if a GPU VM has run
  > --max-hours (default 10) so the orchestrator can kill it.
- docs/gcp_runbook.md — the exact commands to launch, watch, SSH, fetch results, and
  DELETE, plus the cost note (spot T4 ~$0.11/hr) and the "always delete idle GPU" rule.

Do NOT actually create a VM or spend money — the orchestrator launches it after review.
Validate scripts with `bash -n` (syntax) and `.venv/bin/python -c "import ast; ast.parse(open('scripts/gcp_watch.py').read())"`.
Do NOT run git. Print a summary + the launch/watch/teardown commands.
