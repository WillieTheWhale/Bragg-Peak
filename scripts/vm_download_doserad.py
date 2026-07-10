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
    import time
    last: Exception = RuntimeError("no attempt")
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        except urllib.error.HTTPError as e:  # noqa: PERF203
            last = e
            if e.code == 429:  # HF rate limit -> exponential backoff + jitter
                time.sleep(min(60.0, 2.0 ** attempt) + (os.getpid() % 5) * 0.1)
                continue
            if e.code in (500, 502, 503, 504):
                time.sleep(2.0 ** attempt); continue
            raise
        except Exception as e:  # noqa: BLE001
            last = e; time.sleep(1.0 + attempt)
    raise last


def stratified_beamlet_paths(plan: dict, pat: str, per_patient: int) -> list[str]:
    """Round-robin across beams (all gantry angles), ray/layer order within a beam.

    A plain listing-prefix covers only the lexicographically-first beams (17/36
    gantry angles for the DoseRAD proton plans), so train AND val would omit
    half the angular/anatomical distribution. Round-robin keeps the selection
    deterministic while spanning every beam evenly.
    """

    groups: list[list[str]] = []
    for beam in plan.get("beams", []):
        b = int(beam.get("beam_idx", 0))
        items: list[str] = []
        for ray in beam.get("rays", []):
            r = int(ray.get("ray_idx", 0))
            for bl in ray.get("beamlets", []):
                l = int(bl.get("beamlet_idx", bl.get("layer_idx", 0)))
                items.append(f"proton/training/{pat}/dose/Dose_B{b}_R{r}_L{l}.mha")
        if items:
            groups.append(items)
    out: list[str] = []
    rank = 0
    while len(out) < per_patient and any(rank < len(g) for g in groups):
        for g in groups:
            if rank < len(g) and len(out) < per_patient:
                out.append(g[rank])
        rank += 1
    return out


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
        available = {p for p, _ in listed}
        try:
            plan = json.loads((d / "plan.json").read_bytes())
            stratified = [p for p in stratified_beamlet_paths(plan, pat, per_patient) if p in available]
        except Exception as e:  # noqa: BLE001
            print(f"{pat}: stratification failed ({e}); falling back to listing prefix", flush=True)
            stratified = []
        if stratified:
            paths = stratified
            if len(paths) < per_patient:  # top up from the listing if upstream files are missing
                seen = set(paths)
                paths += [p for p, _ in listed if p not in seen][: per_patient - len(paths)]
            (d / "manifest.json").write_text(json.dumps(sorted(os.path.basename(p) for p in paths)))
        else:
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
        with ThreadPoolExecutor(max_workers=8) as ex:  # concurrent -> ~10x faster
            n = sum(ex.map(_one, paths))
        total += n
        print(f"{pat}: {n} beamlets + ct + plan", flush=True)
    print(f"TOTAL valid beamlets downloaded: {total}", flush=True)


if __name__ == "__main__":
    pats = sys.argv[1].split(",") if len(sys.argv) > 1 else ["1ABB006"]
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    download(pats, per)
