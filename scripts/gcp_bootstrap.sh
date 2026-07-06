#!/usr/bin/env bash
set -Eeuo pipefail

BOOT_LOG=/var/log/braggtransport-bootstrap.log
TRAIN_LOG=/var/log/braggtransport-training.log
DONE_MARKER=/var/log/braggtransport-training.done
FAILED_MARKER=/var/log/braggtransport-training.failed

mkdir -p /var/log
touch "$BOOT_LOG" "$TRAIN_LOG"
exec > >(tee -a "$BOOT_LOG") 2>&1

metadata() {
  local key="$1"
  local default_value="${2:-}"
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/${key}" \
    2>/dev/null || printf '%s' "$default_value"
}

PROJECT_ID="$(metadata project_id braggtransporter)"
REGION="$(metadata region us-central1)"
ZONE="$(metadata zone us-central1-f)"
BUCKET="$(metadata bucket gs://braggtransporter-braggtransporter)"
RUN_GCS="$(metadata run_gcs "${BUCKET%/}/runs/$(hostname)")"
REPO_URL="$(metadata repo_url https://github.com/WillieTheWhale/Bragg-Peak.git)"
SOURCE_GCS="$(metadata source_gcs '')"
PATIENTS_RAW="$(metadata patients 1ABB006)"
MAX_BEAMLETS="$(metadata max_beamlets 256)"
EPOCHS="$(metadata epochs 50)"
TRAIN_ARGS="$(metadata train_args '')"
DATA_ROOT="${DATA_ROOT:-data/doserad2026}"
WORKDIR="${WORKDIR:-/opt/braggtransport}"

finish() {
  local status=$?
  mkdir -p /tmp/braggtransport-markers
  if [[ "$status" -eq 0 ]]; then
    date -Is > "$DONE_MARKER"
    cp "$DONE_MARKER" /tmp/braggtransport-markers/DONE
    gcloud storage cp /tmp/braggtransport-markers/DONE "${RUN_GCS%/}/DONE" >/dev/null 2>&1 || true
  else
    date -Is > "$FAILED_MARKER"
    cp "$FAILED_MARKER" /tmp/braggtransport-markers/FAILED
    gcloud storage cp /tmp/braggtransport-markers/FAILED "${RUN_GCS%/}/FAILED" >/dev/null 2>&1 || true
  fi
  gcloud storage cp "$BOOT_LOG" "${RUN_GCS%/}/logs/bootstrap.log" >/dev/null 2>&1 || true
  gcloud storage cp "$TRAIN_LOG" "${RUN_GCS%/}/logs/training.log" >/dev/null 2>&1 || true
  exit "$status"
}
trap finish EXIT

echo "BraggTransporter GCP bootstrap started at $(date -Is)"
echo "Project=$PROJECT_ID Region=$REGION Zone=$ZONE"
echo "Run GCS prefix=$RUN_GCS"
echo "Patients=$PATIENTS_RAW max_beamlets=$MAX_BEAMLETS epochs=$EPOCHS"
echo "Research software only; no clinical claims."

export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HUB_ENABLE_HF_TRANSFER=1

apt-get update
apt-get install -y --no-install-recommends git python3-venv python3-pip ca-certificates jq

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

if [[ -n "$SOURCE_GCS" ]]; then
  echo "Using source tarball from $SOURCE_GCS"
  gcloud storage cp "$SOURCE_GCS" /tmp/braggtransport-source.tgz
  tar -xzf /tmp/braggtransport-source.tgz -C "$WORKDIR" --strip-components=1
else
  echo "Cloning $REPO_URL"
  git clone --depth=1 "$REPO_URL" "$WORKDIR"
fi

cd "$WORKDIR"

python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip wheel setuptools

if ! .venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print("cuda_available=", torch.cuda.is_available())
PY
then
  .venv/bin/python -m pip install torch
fi

.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install SimpleITK huggingface_hub hf_transfer tqdm google-cloud-storage

.venv/bin/python - <<PY
from pathlib import Path
import os
import shutil

patients = "${PATIENTS_RAW}".replace(",", " ").split()
root = Path("${DATA_ROOT}")
max_beamlets = int("${MAX_BEAMLETS}")
root.mkdir(parents=True, exist_ok=True)

try:
    from braggtransporter.data import doserad
except Exception as exc:
    raise RuntimeError(f"Could not import DoseRAD loader before download: {exc}") from exc

helper = getattr(doserad, "download_patients", None)
if helper is not None:
    print(f"Downloading DoseRAD subset with braggtransporter.data.doserad.download_patients: {patients}")
    try:
        helper(patients=patients, dest=root, max_beamlets_per_patient=max_beamlets)
    except TypeError:
        helper(patients, root, max_beamlets)
else:
    print("download_patients helper not found; using huggingface_hub snapshot_download fallback.")
    from huggingface_hub import snapshot_download

    allow_patterns = []
    for patient in patients:
        allow_patterns.extend([
            f"proton/training/{patient}/**",
            f"training/{patient}/**",
            f"{patient}/**",
        ])
    snapshot_download(
        repo_id="LMUK-RADONC-PHYS-RES/DoseRAD2026",
        repo_type="dataset",
        local_dir=str(root),
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
    )
    for patient in patients:
        target = root / patient
        if target.exists():
            continue
        for candidate in (root / "proton" / "training" / patient, root / "training" / patient):
            if candidate.exists():
                try:
                    target.symlink_to(candidate, target_is_directory=True)
                except OSError:
                    shutil.copytree(candidate, target)
                break

print("DoseRAD subset prepared under", root)
PY

mkdir -p experiments/gcp_resume
latest_ckpt="$(
  { gcloud storage ls --recursive --long "${RUN_GCS%/}" 2>/dev/null || true; } \
    | awk 'NF >= 3 && ($NF ~ /(checkpoint|ckpt)/ || $NF ~ /\.(pt|pth)$/) {print $(NF-1), $NF}' \
    | sort \
    | tail -n 1 \
    | awk '{print $2}'
)"

resume_args=()
if [[ -n "$latest_ckpt" ]]; then
  echo "Pulling latest checkpoint from GCS: $latest_ckpt"
  gcloud storage cp "$latest_ckpt" experiments/gcp_resume/ || true
  resume_args+=(--resume)
else
  echo "No existing checkpoint found under $RUN_GCS; starting fresh."
fi

if [[ ! -f scripts/train_doserad_gpu.py ]]; then
  echo "Expected training entrypoint scripts/train_doserad_gpu.py is missing after source checkout." | tee -a "$TRAIN_LOG"
  exit 2
fi

read -r -a patient_args <<< "$(printf '%s' "$PATIENTS_RAW" | tr ',' ' ')"
extra_args=()
if [[ -n "$TRAIN_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args=($TRAIN_ARGS)
fi

nvidia-smi | tee -a "$TRAIN_LOG"

train_cmd=(
  .venv/bin/python scripts/train_doserad_gpu.py
  --patients "${patient_args[@]}"
  --max-beamlets "$MAX_BEAMLETS"
  --epochs "$EPOCHS"
  --device cuda
  --gcs "$RUN_GCS"
  "${resume_args[@]}"
  "${extra_args[@]}"
)

printf 'Training command:' | tee -a "$TRAIN_LOG"
printf ' %q' "${train_cmd[@]}" | tee -a "$TRAIN_LOG"
printf '\n' | tee -a "$TRAIN_LOG"

"${train_cmd[@]}" 2>&1 | tee -a "$TRAIN_LOG"

gcloud storage cp --recursive experiments "${RUN_GCS%/}/artifacts/" >/dev/null 2>&1 || true
echo "BraggTransporter training finished at $(date -Is)"
