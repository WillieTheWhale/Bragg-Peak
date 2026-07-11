#!/bin/bash
# Launch one self-deleting GPU training VM for an audit-iteration branch.
# Usage: scripts/gcp_iter_launch.sh --run-name run18 --branch audit-iter1 \
#          [--gpu t4|l4] [--zone ZONE] [--on-demand] [--resume] [--train-args "..."]
# The startup script is generated here (no hand-edited constants), clones the
# given branch, trains, streams logs/checkpoints to gs://.../runs/<run-name>,
# writes a DONE marker, and self-deletes the VM.
set -euo pipefail

PROJECT=braggtransporter
BUCKET="gs://braggtransporter-braggtransporter"
GPU=t4
ZONE=""
RUN_NAME=""
BRANCH=main
TRAIN_ARGS=""
EXPECT_SHA=""
RESUME_ARG=""
PROVISIONING_MODEL=SPOT
PATIENTS="1ABB006,1ABB011,1ABB020,1ABB021,1ABB030,1ABB031,1ABB035,1ABB036,1ABB039,1ABB041,1ABB042,1ABB045"
PER_PATIENT=500
BOOT_DISK_SIZE_GB=100

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --train-args) TRAIN_ARGS="$2"; shift 2 ;;
    --expect-sha) EXPECT_SHA="$2"; shift 2 ;;
    --resume) RESUME_ARG="--resume latest"; shift ;;
    --resume-checkpoint) RESUME_ARG="--resume $2"; shift 2 ;;
    --on-demand) PROVISIONING_MODEL=STANDARD; shift ;;
    --provisioning-model) PROVISIONING_MODEL="$2"; shift 2 ;;
    --patients) PATIENTS="$2"; shift 2 ;;
    --per-patient) PER_PATIENT="$2"; shift 2 ;;
    --boot-disk-size-gb) BOOT_DISK_SIZE_GB="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
[[ -n "$RUN_NAME" ]] || { echo "--run-name is required" >&2; exit 2; }
[[ -n "$TRAIN_ARGS" ]] || { echo "--train-args is required" >&2; exit 2; }

if [[ "$GPU" == "l4" ]]; then
  MACHINE=g2-standard-12
  ACCEL="type=nvidia-l4,count=1"
  ZONES_TO_TRY=${ZONE:-"us-central1-a us-central1-b us-central1-c"}
else
  MACHINE=n1-standard-8
  ACCEL="type=nvidia-tesla-t4,count=1"
  ZONES_TO_TRY=${ZONE:-"us-central1-f us-central1-a us-central1-b"}
fi

RUN="$BUCKET/runs/$RUN_NAME"
INSTANCE="bt-$RUN_NAME"
STARTUP=$(mktemp "${TMPDIR:-/tmp}/bt-startup.XXXXXX")
trap 'rm -f "$STARTUP"' EXIT
if [[ "$PROVISIONING_MODEL" == "SPOT" ]]; then
  COST_LABEL=spot
else
  COST_LABEL=on-demand
fi

cat > "$STARTUP" <<EOF
#!/bin/bash
set -x
LOG=/var/log/bt_scaling.log
exec > >(tee -a "\$LOG") 2>&1
BUCKET="$BUCKET"
RUN="$RUN"
BRANCH="$BRANCH"
PATIENTS="$PATIENTS"
PER_PATIENT=$PER_PATIENT

TRAIN_STATUS=1
push() { gsutil -q cp "\$LOG" "\$RUN/startup.log" 2>/dev/null || true; }
finish() {
  push
  gsutil -q cp /opt/bt/train.log "\$RUN/train.log" 2>/dev/null || true
  for ARTIFACT in metrics_best_full.json metrics_test.json; do
    gsutil -q cp "/opt/bt/runs/$RUN_NAME/\$ARTIFACT" "\$RUN/\$ARTIFACT" 2>/dev/null || true
  done
  if [[ "\$TRAIN_STATUS" -eq 0 ]]; then
    echo "DONE \$(date -u +%FT%TZ)" | gsutil -q cp - "\$RUN/DONE" 2>/dev/null || true
  else
    echo "FAILED status=\$TRAIN_STATUS \$(date -u +%FT%TZ)" | gsutil -q cp - "\$RUN/FAILED" 2>/dev/null || true
  fi
  NAME=\$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name || hostname -s)
  ZONE=\$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print \$NF}')
  gcloud compute instances delete "\$NAME" --zone="\$ZONE" --quiet || true
}
trap finish EXIT

apt-get update -y && apt-get install -y --no-install-recommends git python3-venv python3-pip ca-certificates jq
rm -rf /opt/bt && git clone --depth 1 --branch "\$BRANCH" https://github.com/WillieTheWhale/Bragg-Peak.git /opt/bt
cd /opt/bt
echo "GIT_COMMIT=\$(git rev-parse HEAD) BRANCH=\$BRANCH RUN=\$RUN"
EXPECT_SHA="$EXPECT_SHA"
if [[ -n "\$EXPECT_SHA" ]] && [[ "\$(git rev-parse HEAD)" != "\$EXPECT_SHA"* ]]; then
  echo "FATAL: cloned commit \$(git rev-parse HEAD) does not match expected \$EXPECT_SHA"
  exit 1
fi
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -q --upgrade pip wheel setuptools
.venv/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || .venv/bin/python -m pip install -q torch
.venv/bin/python -m pip install -q SimpleITK huggingface_hub numpy scipy h5py pyyaml einops google-cloud-storage pymedphys numba
push

( while true; do sleep 180; push; done ) &

echo "=== DOWNLOAD \$(date -u +%FT%TZ) ==="
.venv/bin/python -u scripts/vm_download_doserad.py "\$PATIENTS" "\$PER_PATIENT" || true
NBEAM=\$(find data/doserad2026 -path '*/dose/*.mha' -size +50000c | wc -l)
echo "VALID_BEAMLETS=\$NBEAM"
push

echo "=== TRAIN \$(date -u +%FT%TZ) ==="
PATS=\$(ls data/doserad2026 | grep 1ABB | tr '\n' ' ')
if [[ -n "$RESUME_ARG" ]]; then
  mkdir -p /opt/bt/runs/$RUN_NAME
  for ARTIFACT in latest.pt best.pt metrics.jsonl metrics_latest.json; do
    gsutil -q cp "\$RUN/\$ARTIFACT" "/opt/bt/runs/$RUN_NAME/\$ARTIFACT" 2>/dev/null || true
  done
fi
set -o pipefail
.venv/bin/python -u scripts/train_doserad_gpu.py --patients \$PATS \
  --max-beamlets "\$PER_PATIENT" --device cuda \
  $RESUME_ARG \
  $TRAIN_ARGS \
  --gcs "\$RUN" --out-dir /opt/bt/runs/$RUN_NAME 2>&1 | tee /opt/bt/train.log
TRAIN_STATUS=\$?
set +o pipefail
echo "=== TRAIN EXIT \$TRAIN_STATUS \$(date -u +%FT%TZ) ==="
EOF

for TRY_ZONE in $ZONES_TO_TRY; do
  echo "launching $INSTANCE ($MACHINE, $ACCEL, zone $TRY_ZONE, provisioning=$PROVISIONING_MODEL) branch=$BRANCH run=$RUN resume=${RESUME_ARG:-none}"
  CREATE_ARGS=(
    gcloud compute instances create "$INSTANCE"
    --project="$PROJECT" \
    --zone="$TRY_ZONE" \
    --machine-type="$MACHINE" \
    --provisioning-model="$PROVISIONING_MODEL" \
    --maintenance-policy=TERMINATE \
    --accelerator="$ACCEL" \
    --image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size="${BOOT_DISK_SIZE_GB}GB" \
    --boot-disk-type=pd-balanced \
    --metadata-from-file=startup-script="$STARTUP" \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --labels=project=braggtransporter,cost="$COST_LABEL",run="$RUN_NAME"
  )
  if [[ "$PROVISIONING_MODEL" == "SPOT" ]]; then
    CREATE_ARGS+=(--instance-termination-action=DELETE)
  fi
  if "${CREATE_ARGS[@]}"; then
    echo "launched in $TRY_ZONE. logs: $RUN/startup.log ; success marker: $RUN/DONE ; failure marker: $RUN/FAILED"
    exit 0
  fi
  echo "zone $TRY_ZONE failed (likely capacity); trying next" >&2
done
echo "all zones failed for $GPU" >&2
exit 1
