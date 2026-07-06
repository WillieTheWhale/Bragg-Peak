# BraggTransporter GCP Environment

Last verified: 2026-07-06 00:06 America/Los_Angeles

## Core settings

- Google account: `williamkeffer2005@gmail.com`
- Project name: `BraggTransporter`
- Project ID: `braggtransporter`
- Project number: `359754878565`
- Region: `us-central1`
- Default zone: `us-central1-f`
- Storage bucket: `gs://braggtransporter-braggtransporter`
- Billing account: `01B3E3-02E639-8AAE96` (`My Billing Account`)
- Billing account status: open, linked, account type shown in Console as `Direct`
- Credit balance/expiry: pending manual Console confirmation. The billing account is open and usable, but the Console Credits page did not expose readable credit balance text through automation.

## Cost controls

- Budget: 250 USD monthly budget for project `braggtransporter`
- Budget scope: `projects/359754878565`
- Credit treatment: `EXCLUDE_ALL_CREDITS` so alerts track gross usage before promotional credits mask spend
- Alert thresholds: 50%, 90%, 100%
- GPU plan: use Spot VMs, delete when idle, avoid service-account keys

## APIs enabled

- `compute.googleapis.com`
- `storage.googleapis.com`
- `iam.googleapis.com`
- `aiplatform.googleapis.com`
- `artifactregistry.googleapis.com`
- `billingbudgets.googleapis.com`
- `cloudquotas.googleapis.com`
- `cloudresourcemanager.googleapis.com`
- `serviceusage.googleapis.com`

## Local auth and SDK

The local Homebrew Cloud SDK is installed and authenticated for both gcloud user credentials and Application Default Credentials.

```bash
gcloud auth list
gcloud config list
gcloud projects describe braggtransporter
gcloud auth application-default set-quota-project braggtransporter
```

The project venv has Google Python SDK packages installed:

```bash
.venv/bin/python -m pip install google-cloud-storage google-cloud-aiplatform
.venv/bin/python - <<'PY'
import google.cloud.storage
import google.cloud.aiplatform
print("ok")
PY
```

## Storage check

The bucket was created in `US-CENTRAL1` and a test upload/download/delete round trip succeeded.

```bash
printf 'hello\n' > /tmp/braggtransport-gcs-test.txt
gcloud storage cp /tmp/braggtransport-gcs-test.txt gs://braggtransporter-braggtransporter/sanity/test.txt
gcloud storage cp gs://braggtransporter-braggtransporter/sanity/test.txt /tmp/braggtransport-gcs-test.down.txt
cmp /tmp/braggtransport-gcs-test.txt /tmp/braggtransport-gcs-test.down.txt
gcloud storage rm gs://braggtransporter-braggtransporter/sanity/test.txt
```

## GPU quota status

- Global GPU quota (`GPUS_ALL_REGIONS`): approved to 1
- Quota preference: `projects/braggtransporter/locations/global/quotaPreferences/braggtransporter-gpus-all-regions-1`
- Trace ID: `3f0c6589-43a3-4d1a-b662-4b440b238ae4`
- `us-central1` CPU quota: 200 CPUs
- `us-central1` L4 quota: 1 regular, 1 Spot/preemptible
- `us-central1` T4 quota: 1 regular, 1 Spot/preemptible
- L4 Spot in `us-central1-a/b/c`: quota granted, but capacity was stocked out during setup
- T4 Spot in `us-central1-f`: provisioned successfully

Sanity `nvidia-smi` result from a deleted Spot T4 VM:

```text
NVIDIA-SMI 580.159.03
Driver Version: 580.159.03
CUDA Version: 13.0
GPU 0: Tesla T4, 15360 MiB
```

## Working Spot T4 VM commands

Create:

```bash
gcloud compute instances create braggtransport-gpu \
  --project=braggtransporter \
  --zone=us-central1-f \
  --machine-type=n1-standard-8 \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --maintenance-policy=TERMINATE \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --metadata=install-nvidia-driver=True \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --labels=project=braggtransporter,cost=spot
```

SSH:

```bash
gcloud compute ssh braggtransport-gpu \
  --project=braggtransporter \
  --zone=us-central1-f
```

Check GPU:

```bash
gcloud compute ssh braggtransport-gpu \
  --project=braggtransporter \
  --zone=us-central1-f \
  --command='nvidia-smi'
```

Delete:

```bash
gcloud compute instances delete braggtransport-gpu \
  --project=braggtransporter \
  --zone=us-central1-f \
  --quiet
```

## L4 alternative

Use this when L4 Spot capacity is available. During setup, `us-central1-a/b/c` were stocked out.

```bash
gcloud compute instances create braggtransport-l4 \
  --project=braggtransporter \
  --zone=us-central1-a \
  --machine-type=g2-standard-12 \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --maintenance-policy=TERMINATE \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --metadata=install-nvidia-driver=True \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --labels=project=braggtransporter,cost=spot
```

## Security notes

- No service-account key was created.
- Use Application Default Credentials for Python clients.
- If a service-account key ever becomes unavoidable, store it outside this repo and add the path to `.gitignore`.
