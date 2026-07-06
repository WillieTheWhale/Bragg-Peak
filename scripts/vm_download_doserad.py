#!/usr/bin/env python
"""Robust DoseRAD2026 proton downloader (direct HF API+resolve). Populates
data/doserad2026/<patient>/{ct.mha,plan.json,dose/*.mha} to match the loader's
expected layout. Skips corrupt/tiny (<50KB) beamlet files."""
from __future__ import annotations
import json, os, sys, pathlib, urllib.request

REPO = "LMUK-RADONC-PHYS-RES/DoseRAD2026"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/proton/training/{{pat}}/dose?recursive=true&expand=true"
RES = f"https://huggingface.co/datasets/{REPO}/resolve/main/{{path}}"


_SIBLINGS: list[str] | None = None


def _all_siblings() -> list[str]:
    """Complete file list from the dataset info API (all ~99k files, no pagination cap)."""
    global _SIBLINGS
    if _SIBLINGS is None:
        d = json.loads(_get(f"https://huggingface.co/api/datasets/{REPO}").decode())
        _SIBLINGS = [s["rfilename"] for s in d.get("siblings", [])]
    return _SIBLINGS


def _list_dose_files(pat: str) -> list[tuple[str, int]]:
    """ALL proton dose .mha files for a patient (from the complete siblings list)."""
    pref = f"proton/training/{pat}/dose/"
    paths = [p for p in _all_siblings() if p.startswith(pref) and p.endswith(".mha")]
    if paths:
        return [(p, 10**6) for p in paths]  # size unknown; validate after download
    # fallback: tree API first page
    files = json.loads(_get(API.format(pat=pat)).decode())
    return [(f["path"], f.get("size", 0)) for f in files if f.get("type") == "file"]


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
            listed = _list_dose_files(pat)
        except Exception as e:  # noqa: BLE001
            print(f"{pat}: dose listing failed ({e})", flush=True); continue
        paths = [p for p, _ in listed][:per_patient]
        n = 0
        def _one(p: str) -> int:
            out = d / "dose" / os.path.basename(p)
            if out.exists() and out.stat().st_size > 50000:
                return 1
            try:
                data = _get(RES.format(path=p))
                if len(data) > 50000:  # skip corrupt/empty stubs
                    out.write_bytes(data); return 1
            except Exception as e:  # noqa: BLE001
                print(f"  {os.path.basename(p)} failed: {e}", flush=True)
            return 0

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=16) as ex:  # concurrent -> ~10x faster
            n = sum(ex.map(_one, paths))
        total += n
        print(f"{pat}: {n} beamlets + ct + plan", flush=True)
    print(f"TOTAL valid beamlets downloaded: {total}", flush=True)


if __name__ == "__main__":
    pats = sys.argv[1].split(",") if len(sys.argv) > 1 else ["1ABB006"]
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    download(pats, per)
