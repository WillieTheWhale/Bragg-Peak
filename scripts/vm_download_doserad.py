#!/usr/bin/env python
"""Robust DoseRAD2026 proton downloader (direct HF API+resolve). Populates
data/doserad2026/<patient>/{ct.mha,plan.json,dose/*.mha} to match the loader's
expected layout. Skips corrupt/tiny (<50KB) beamlet files."""
from __future__ import annotations
import json, os, sys, pathlib, urllib.request

REPO = "LMUK-RADONC-PHYS-RES/DoseRAD2026"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/proton/training/{{pat}}/dose?recursive=true&expand=true"
RES = f"https://huggingface.co/datasets/{REPO}/resolve/main/{{path}}"


def _get(url: str) -> bytes:
    for _ in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


def download(patients: list[str], per_patient: int, root: str = "data/doserad2026") -> None:
    total = 0
    for pat in patients:
        d = pathlib.Path(root) / pat
        (d / "dose").mkdir(parents=True, exist_ok=True)
        try:
            (d / "plan.json").write_bytes(_get(RES.format(path=f"proton/training/{pat}/{pat}.json")))
            (d / "ct.mha").write_bytes(_get(RES.format(path=f"proton/training/{pat}/image/ct.mha")))
        except Exception as e:  # noqa: BLE001
            print(f"{pat}: CT/plan failed ({e}); skipping", flush=True); continue
        try:
            files = json.loads(_get(API.format(pat=pat)).decode())
        except Exception as e:  # noqa: BLE001
            print(f"{pat}: dose listing failed ({e})", flush=True); continue
        valid = [f["path"] for f in files if f.get("type") == "file" and f.get("size", 0) > 50000][:per_patient]
        n = 0
        for p in valid:
            out = d / "dose" / os.path.basename(p)
            if out.exists() and out.stat().st_size > 50000:
                n += 1; continue
            try:
                out.write_bytes(_get(RES.format(path=p))); n += 1
            except Exception as e:  # noqa: BLE001
                print(f"  {os.path.basename(p)} failed: {e}", flush=True)
        total += n
        print(f"{pat}: {n} beamlets + ct + plan", flush=True)
    print(f"TOTAL valid beamlets downloaded: {total}", flush=True)


if __name__ == "__main__":
    pats = sys.argv[1].split(",") if len(sys.argv) > 1 else ["1ABB006"]
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    download(pats, per)
