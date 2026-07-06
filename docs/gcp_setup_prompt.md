# Prompt for the Codex app (computer use) — Google Cloud setup for BraggTransporter

Paste everything in the fenced block below into the Codex app. It uses computer use
for the browser/console steps and the real terminal for the CLI steps.

```
You are setting up Google Cloud on my MacBook (Apple M4 Max, macOS) so that BOTH you
AND a separate Claude Code agent running in my terminal can deploy and train PyTorch
models on GCP GPUs. This is for my research project "BraggTransporter" at
/Users/williamkeffer/NerdsInc/Personal_Projects/bragg_peak (Python 3.11 venv at
`.venv`, repo github.com/WillieTheWhale/Bragg-Peak). The near-term goal is full-scale
3-D proton-dose training on the DoseRAD2026 dataset, which is too big for my laptop.

CRITICAL REQUIREMENT — do not offload everything to yourself. The end state must be a
LOCALLY installed, authenticated gcloud CLI + SDK on THIS machine that the Claude Code
agent can drive from a normal terminal in the repo directory. So install into my real
system (Homebrew / my shell profile, not a throwaway sandbox), authenticate globally,
and verify it works from a freshly opened terminal. Where a step is browser-only
(billing, quota), do it in the Console with computer use, then confirm the result on
the CLI.

Use my Google account's free credit: I have a $300 free trial / student credit —
activate it and make sure billing is backed by that credit. Be cost-careful: I only
have $300.

Do the following, verifying and reporting after each numbered step:

1. ACCOUNT + BILLING (browser/computer-use):
   - Sign in to https://console.cloud.google.com with my Google account.
   - Activate the $300 free credit (Free Trial or the student/education offer,
     whichever my account is eligible for). Create/confirm a Billing Account funded by
     that credit. Report the billing account ID and remaining credit + expiry date.
   - IMPORTANT GPU GOTCHA: free-trial accounts get 0 GPU quota and often must be
     "upgraded" to a full (paid) account to be granted GPU quota — the $300 credit
     still covers usage after upgrading, you just remove the trial spending cap. If
     needed, upgrade the account (this does NOT charge me while credit remains) so we
     can request GPU quota. Tell me exactly what you did here.

2. PROJECT:
   - Create a project named `braggtransporter` (let GCP assign the ID, or use
     `braggtransporter-<random>`). Link it to the billing account. Report the PROJECT_ID.

3. COST CONTROLS (do this BEFORE creating any GPU):
   - Create a Budget of $250 with alert thresholds at 50/90/100% emailed to me.
   - Note the plan: we will use SPOT/preemptible GPUs and shut instances down when
     idle. Confirm the budget is active.

4. ENABLE APIs (CLI preferred once gcloud is installed in step 6, else Console):
   compute.googleapis.com, storage.googleapis.com, iam.googleapis.com,
   aiplatform.googleapis.com, artifactregistry.googleapis.com. Report which succeeded.

5. GPU QUOTA:
   - In IAM & Admin → Quotas, request quota for a cost-effective GPU we can actually
     get: prefer `NVIDIA L4 GPUs` or `NVIDIA T4 GPUs` (and the Preemptible/Spot variant
     if listed) = 1, in a region with availability (try us-central1, then us-east1,
     us-west1). Also ensure the region's Compute CPU quota is >= 12.
   - Submit the increase request. Quota approval can be instant or take hours/days —
     report the request status and which GPU/region you targeted. If instant-approved,
     say so.

6. LOCAL gcloud CLI (real terminal on this machine — this is the part Claude Code needs):
   - Install the Google Cloud SDK via Homebrew: `brew install --cask google-cloud-sdk`.
     Ensure `gcloud`, `gsutil`, `bq` are on PATH for a NEW terminal (add the SDK
     path.zsh.inc to ~/.zshrc if the cask doesn't). Verify: open a fresh terminal and
     run `gcloud --version`.
   - Authenticate globally so the config lives in ~/.config/gcloud (shared with Claude
     Code): `gcloud auth login` (browser) AND `gcloud auth application-default login`
     (sets Application Default Credentials for the Python SDKs).
   - Set defaults: `gcloud config set project <PROJECT_ID>` and
     `gcloud config set compute/region <REGION>` and `compute/zone <ZONE>`.
   - Verify: `gcloud auth list`, `gcloud config list`, and
     `gcloud projects describe <PROJECT_ID>` all succeed.

7. PYTHON SDK (into the project venv so Claude Code can use it):
   `/Users/williamkeffer/NerdsInc/Personal_Projects/bragg_peak/.venv/bin/pip install \
     google-cloud-storage google-cloud-aiplatform` and verify
   `.venv/bin/python -c "import google.cloud.storage, google.cloud.aiplatform; print('ok')"`.

8. STORAGE for data + checkpoints:
   - Create a bucket `gs://braggtransporter-<PROJECT_ID>` in the chosen region.
   - Verify read/write: `gsutil cp` a small test file up and down, then remove it.

9. SANITY GPU VM (only if GPU quota was granted; otherwise skip and tell me):
   - Create a SPOT Deep Learning VM with 1 L4 or T4, e.g.:
     `gcloud compute instances create bt-gpu-test --zone=<ZONE>
      --machine-type=g2-standard-8 (L4) or n1-standard-8 + --accelerator=type=nvidia-tesla-t4,count=1
      --provisioning-model=SPOT --image-family=<pytorch-latest-gpu family> --image-project=deeplearning-platform-release
      --boot-disk-size=100GB --maintenance-policy=TERMINATE`.
   - SSH in (`gcloud compute ssh bt-gpu-test --zone=<ZONE>`), run `nvidia-smi` to confirm
     the GPU, then DELETE the instance (`gcloud compute instances delete bt-gpu-test`)
     so it stops costing money. Report nvidia-smi output.

10. HANDOFF FILE (so Claude Code can pick up without guessing) — write a NON-SECRET
    file at `/Users/williamkeffer/NerdsInc/Personal_Projects/bragg_peak/docs/gcp_env.md`
    containing: PROJECT_ID, REGION, ZONE, BUCKET name, billing account ID, remaining
    credit + expiry, chosen GPU type, GPU quota status (granted/pending), and the exact
    gcloud commands to (a) create a spot GPU VM, (b) SSH in, (c) delete it. Do NOT put
    any private key or token in this file, and do NOT commit service-account key files
    to git.

SECURITY: prefer Application Default Credentials over downloadable service-account
keys. If a service-account key is unavoidable, save it outside the repo (e.g.
~/.config/gcloud/) and add it to .gitignore — never commit it.

When done, give me a concise summary: PROJECT_ID, REGION/ZONE, BUCKET, billing/credit
status, GPU quota status, and confirmation that `gcloud --version` + `gcloud auth list`
work in a fresh terminal (so the Claude Code agent can use them). List anything still
pending (e.g. GPU quota approval) and what I need to click to finish it.
```
