"""DoseRAD2026 beamlet loading and BEV extraction.

Research software only. Coordinates and spacings are millimetres, matching
SimpleITK physical coordinates for the provided CT and dose volumes. Extracted
tensors are small beam's-eye-view (BEV) grids with axis order
``(depth, lateral_y, lateral_x)`` after SimpleITK's array conversion.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split


DOSERAD_INPUT_CHANNELS = 4
DOSERAD_HF_BASE_URL = "https://huggingface.co/datasets/LMUK-RADONC-PHYS-RES/DoseRAD2026/resolve/main"
DOSERAD_MIN_DOWNLOAD_BYTES = 50 * 1024


@dataclass(frozen=True)
class BeamletRecord:
    patient: str
    beam_idx: int
    ray_idx: int
    layer_idx: int
    energy_mev: float
    gantry_angle_deg: float
    ray_source_mm: tuple[float, float, float]
    ray_target_mm: tuple[float, float, float]
    dose_path: Path
    cache_path: Path


def read_mha(path: str | Path, pixel_type: int = sitk.sitkFloat32) -> sitk.Image:
    """Read an MHA file as a SimpleITK image, raising the underlying ITK error."""

    return sitk.ReadImage(str(path), pixel_type)


def parse_plan(path: str | Path) -> dict[str, Any]:
    """Parse a DoseRAD2026 ``plan.json`` file."""

    with Path(path).open("r", encoding="utf-8") as f:
        plan = json.load(f)
    if "beams" not in plan:
        raise ValueError(f"{path} does not contain a DoseRAD2026 beams list.")
    return plan


def hu_to_density_rsp(hu: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Approximate HU-to-density/RSP conversion for DoseRAD BEV inputs.

    The curves are deterministic piecewise-linear CT calibration priors:
    density is a Schneider-like electron-density/mass-density proxy, while RSP
    is a separate proton stopping-power proxy. Values are raw physical inputs;
    any model standardization must happen downstream.
    """

    hu32 = np.asarray(hu, dtype=np.float32)
    hu_knots = np.asarray([-1000.0, -950.0, -700.0, -300.0, -100.0, 0.0, 100.0, 300.0, 1000.0, 1600.0, 3000.0])
    density_knots = np.asarray([0.0, 0.0, 0.30, 0.70, 0.93, 1.00, 1.06, 1.16, 1.55, 1.90, 2.20])
    rsp_knots = np.asarray([0.0, 0.0, 0.25, 0.68, 0.94, 1.00, 1.03, 1.10, 1.43, 1.65, 1.90])
    density = np.interp(hu32, hu_knots, density_knots).astype(np.float32)
    rsp = np.interp(hu32, hu_knots, rsp_knots).astype(np.float32)
    density = np.clip(density, 0.0, 2.2).astype(np.float32, copy=False)
    rsp = np.clip(rsp, 0.0, 2.2).astype(np.float32, copy=False)
    return density, rsp


def extract_bev_pair(
    ct_image: sitk.Image,
    dose_image: sitk.Image,
    ray_source_mm: Iterable[float],
    ray_target_mm: Iterable[float],
    *,
    depth_size: int = 128,
    depth_extent_mm: float = 400.0,
    lateral_size: int = 24,
    lateral_extent_mm: float = 96.0,
) -> dict[str, NDArray[np.float32] | tuple[float, float, float] | float]:
    """Extract a downsampled BEV CT/dose pair for one ray.

    Output ``x`` has raw channels ``[HU/1000, density_g_cm3, RSP, WEPL/300mm]``
    and shape ``(4, depth, lateral_y, lateral_x)``. ``dose`` has shape
    ``(depth, lateral_y, lateral_x)``. The beam-depth axis follows
    ``ray_source -> ray_target`` and the first depth slice is placed at the CT
    entrance along that ray. The depth grid spans fixed ``depth_extent_mm`` so
    ``depth_spacing_mm`` is consistent across beamlets and patients.
    """

    if depth_size <= 1 or lateral_size <= 1:
        raise ValueError("depth_size and lateral_size must both be > 1.")
    fixed_depth_extent_mm = float(depth_extent_mm)
    if not np.isfinite(fixed_depth_extent_mm) or fixed_depth_extent_mm <= 0.0:
        raise ValueError("depth_extent_mm must be positive.")

    source = np.asarray(tuple(ray_source_mm), dtype=np.float64)
    target = np.asarray(tuple(ray_target_mm), dtype=np.float64)
    direction = target - source
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("ray_source and ray_target must be distinct 3-D points.")
    depth_axis = direction / norm

    bounds_min, bounds_max = _image_physical_bounds(ct_image)
    t0, t1 = _ray_box_interval(source, depth_axis, bounds_min, bounds_max)
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        # Fall back to a target-centred crop so synthetic edge cases still test
        # the extraction path instead of failing before resampling.
        start = target - 0.5 * fixed_depth_extent_mm * depth_axis
    else:
        start = source + max(t0, 0.0) * depth_axis

    lateral_u, lateral_v = _orthonormal_lateral_axes(depth_axis)
    depth_spacing_mm = fixed_depth_extent_mm / float(depth_size - 1)
    lateral_spacing_mm = float(lateral_extent_mm) / float(lateral_size - 1)

    output_size = (int(lateral_size), int(lateral_size), int(depth_size))
    output_spacing = (lateral_spacing_mm, lateral_spacing_mm, depth_spacing_mm)
    output_direction = _direction_matrix(lateral_u, lateral_v, depth_axis)
    output_origin = (
        start
        - lateral_u * lateral_spacing_mm * (lateral_size - 1) / 2.0
        - lateral_v * lateral_spacing_mm * (lateral_size - 1) / 2.0
    )

    ct_bev = _resample_image(
        ct_image,
        output_size,
        output_spacing,
        output_origin,
        output_direction,
        default_value=-1000.0,
    )
    dose_bev = _resample_image(
        dose_image,
        output_size,
        output_spacing,
        output_origin,
        output_direction,
        default_value=0.0,
    )

    hu = sitk.GetArrayFromImage(ct_bev).astype(np.float32, copy=False)
    dose = np.maximum(sitk.GetArrayFromImage(dose_bev).astype(np.float32, copy=False), 0.0)
    density, rsp = hu_to_density_rsp(hu)
    wepl_mm = np.empty_like(rsp, dtype=np.float32)
    wepl_mm[0, :, :] = 0.0
    wepl_mm[1:, :, :] = np.cumsum(rsp[:-1, :, :], axis=0, dtype=np.float32) * np.float32(depth_spacing_mm)
    wepl_norm = (wepl_mm / np.float32(300.0)).astype(np.float32, copy=False)
    x = np.stack([hu / 1000.0, density, rsp, wepl_norm], axis=0).astype(np.float32, copy=False)

    return {
        "x": x,
        "dose": dose.astype(np.float32, copy=False),
        "spacing_mm": (float(depth_spacing_mm), float(lateral_spacing_mm), float(lateral_spacing_mm)),
        "depth_spacing_mm": float(depth_spacing_mm),
    }


class DoseRADBeamletDataset(Dataset[dict[str, torch.Tensor | dict[str, Any]]]):
    """Torch dataset of cached downsampled DoseRAD2026 beamlet BEV tensors."""

    def __init__(
        self,
        root: str | Path = "data/doserad2026",
        patients: Iterable[str] | str = ("1ABB006",),
        *,
        max_beamlets: int | None = None,
        max_beamlets_per_patient: int | None = None,
        depth_size: int = 128,
        depth_extent_mm: float = 400.0,
        lateral_size: int = 24,
        lateral_extent_mm: float = 96.0,
        cache_dir: str | Path | None = None,
        rebuild_cache: bool = False,
        min_file_bytes: int = 1024,
    ) -> None:
        self.root = Path(root)
        self.patients = _patients_list(patients)
        self.depth_size = int(depth_size)
        self.depth_extent_mm = float(depth_extent_mm)
        self.lateral_size = int(lateral_size)
        self.lateral_extent_mm = float(lateral_extent_mm)
        self.rebuild_cache = bool(rebuild_cache)
        self.min_file_bytes = int(min_file_bytes)
        self.max_beamlets_per_patient = None if max_beamlets_per_patient is None else int(max_beamlets_per_patient)
        self.cache_root = Path(cache_dir) if cache_dir is not None else self.root / ".cache" / "bev_npz"
        self.records = self._prepare_records(max_beamlets)
        if not self.records:
            raise RuntimeError(f"No valid DoseRAD beamlets found under {self.root} for patients {self.patients}.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict[str, Any]]:
        rec = self.records[idx]
        with np.load(rec.cache_path, allow_pickle=False) as z:
            x = z["x"].astype(np.float32, copy=False)
            dose = z["dose"].astype(np.float32, copy=False)
            dose_scale = z["dose_scale"].astype(np.float32, copy=False) if "dose_scale" in z else np.asarray(1.0, dtype=np.float32)
            scalars = z["scalars"].astype(np.float32, copy=False)
            spacing_mm = z["spacing_mm"].astype(np.float32, copy=False)
            depth_spacing_mm = (
                z["depth_spacing_mm"].astype(np.float32, copy=False)
                if "depth_spacing_mm" in z
                else np.asarray(spacing_mm[0], dtype=np.float32)
            )
        return {
            "x": torch.from_numpy(x),
            "dose": torch.from_numpy(dose),
            "dose_scale": torch.as_tensor(dose_scale, dtype=torch.float32),
            "scalars": torch.from_numpy(scalars),
            "spacing_mm": torch.from_numpy(spacing_mm),
            "depth_spacing_mm": torch.as_tensor(depth_spacing_mm, dtype=torch.float32),
            "meta": {
                "patient": rec.patient,
                "beam_idx": rec.beam_idx,
                "ray_idx": rec.ray_idx,
                "layer_idx": rec.layer_idx,
                "energy_mev": rec.energy_mev,
                "gantry_angle_deg": rec.gantry_angle_deg,
            },
        }

    def _prepare_records(self, max_beamlets: int | None) -> list[BeamletRecord]:
        records: list[BeamletRecord] = []
        for patient in self.patients:
            patient_dir = self.root / patient
            ct_path = patient_dir / "ct.mha"
            plan_path = patient_dir / "plan.json"
            if not _valid_file(ct_path, self.min_file_bytes) or not plan_path.exists():
                continue
            plan = parse_plan(plan_path)
            ct_image = read_mha(ct_path)
            patient_count = 0
            for rec in self._iter_patient_records(patient, patient_dir, plan):
                if self.max_beamlets_per_patient is not None and patient_count >= self.max_beamlets_per_patient:
                    break
                if max_beamlets is not None and len(records) >= int(max_beamlets):
                    return records
                try:
                    if self.rebuild_cache or not rec.cache_path.exists():
                        self._write_cache(rec, ct_image)
                    records.append(rec)
                    patient_count += 1
                except Exception as exc:
                    print(
                        f"Skipping DoseRAD beamlet {patient} B{rec.beam_idx} R{rec.ray_idx} L{rec.layer_idx}: {exc}"
                    )
        return records

    def _iter_patient_records(self, patient: str, patient_dir: Path, plan: dict[str, Any]) -> Iterable[BeamletRecord]:
        for beam in plan.get("beams", []):
            beam_idx = int(beam.get("beam_idx", len(list(()))))
            gantry_angle = float(beam.get("gantry_angle", 0.0))
            for ray in beam.get("rays", []):
                ray_idx = int(ray.get("ray_idx", 0))
                source = tuple(float(v) for v in ray["ray_source"])
                target = tuple(float(v) for v in ray["ray_target"])
                for beamlet in ray.get("beamlets", []):
                    layer_idx = int(beamlet.get("beamlet_idx", beamlet.get("layer_idx", 0)))
                    dose_path = patient_dir / "dose" / f"Dose_B{beam_idx}_R{ray_idx}_L{layer_idx}.mha"
                    if not _valid_file(dose_path, self.min_file_bytes):
                        continue
                    cache_path = self._cache_path(patient, beam_idx, ray_idx, layer_idx)
                    yield BeamletRecord(
                        patient=patient,
                        beam_idx=beam_idx,
                        ray_idx=ray_idx,
                        layer_idx=layer_idx,
                        energy_mev=float(beamlet.get("energy", float("nan"))),
                        gantry_angle_deg=gantry_angle,
                        ray_source_mm=source,
                        ray_target_mm=target,
                        dose_path=dose_path,
                        cache_path=cache_path,
                    )

    def _cache_path(self, patient: str, beam_idx: int, ray_idx: int, layer_idx: int) -> Path:
        tag = (
            f"d{self.depth_size}_de{self.depth_extent_mm:g}_l{self.lateral_size}"
            f"_e{self.lateral_extent_mm:g}_c4_wepl_norm1"
        )
        return self.cache_root / patient / tag / f"B{beam_idx}_R{ray_idx}_L{layer_idx}.npz"

    def _write_cache(self, rec: BeamletRecord, ct_image: sitk.Image) -> None:
        dose_image = read_mha(rec.dose_path)
        bev = extract_bev_pair(
            ct_image,
            dose_image,
            rec.ray_source_mm,
            rec.ray_target_mm,
            depth_size=self.depth_size,
            depth_extent_mm=self.depth_extent_mm,
            lateral_size=self.lateral_size,
            lateral_extent_mm=self.lateral_extent_mm,
        )
        gantry_rad = math.radians(rec.gantry_angle_deg)
        scalars = np.asarray(
            [rec.energy_mev, float(rec.layer_idx), math.sin(gantry_rad), math.cos(gantry_rad)],
            dtype=np.float32,
        )
        dose_norm, dose_scale = normalize_beamlet_dose(np.asarray(bev["dose"], dtype=np.float32))
        rec.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            rec.cache_path,
            x=np.asarray(bev["x"], dtype=np.float32),
            dose=dose_norm,
            dose_scale=np.asarray(dose_scale, dtype=np.float32),
            scalars=scalars,
            spacing_mm=np.asarray(bev["spacing_mm"], dtype=np.float32),
            depth_spacing_mm=np.asarray(bev["depth_spacing_mm"], dtype=np.float32),
        )


def make_doserad_loaders(
    root: str | Path = "data/doserad2026",
    patients: Iterable[str] | str = ("1ABB006",),
    *,
    max_beamlets: int | None = None,
    val_frac: float = 0.25,
    batch_size: int = 2,
    seed: int = 0,
    depth_size: int = 128,
    depth_extent_mm: float = 400.0,
    lateral_size: int = 24,
    lateral_extent_mm: float = 96.0,
    cache_dir: str | Path | None = None,
    rebuild_cache: bool = False,
    num_workers: int = 0,
    max_beamlets_per_patient: int | None = None,
    split_by: str = "patient",
) -> tuple[DataLoader, DataLoader]:
    """Build cached DoseRAD beamlet dataset and split train/val deterministically."""

    dataset = DoseRADBeamletDataset(
        root,
        patients,
        max_beamlets=max_beamlets,
        depth_size=depth_size,
        depth_extent_mm=depth_extent_mm,
        lateral_size=lateral_size,
        lateral_extent_mm=lateral_extent_mm,
        cache_dir=cache_dir,
        rebuild_cache=rebuild_cache,
        max_beamlets_per_patient=max_beamlets_per_patient,
    )
    train_ds, val_ds, actual_split, val_patient_ids, train_patient_ids, fallback = _split_doserad_dataset(
        dataset,
        val_frac=float(val_frac),
        seed=int(seed),
        split_by=str(split_by),
    )
    if fallback:
        print(
            "warning: split_by=patient requested but only one patient is available; "
            "falling back to beamlet split, validation metrics are optimistic.",
            flush=True,
        )
    print(f"DoseRAD split_by={actual_split} val_patients={json.dumps(val_patient_ids)}", flush=True)
    gen = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    _attach_split_metadata(
        train_loader,
        val_loader,
        split_by=actual_split,
        val_patient_ids=val_patient_ids,
        train_patient_ids=train_patient_ids,
    )
    return train_loader, val_loader


def _split_doserad_dataset(
    dataset: DoseRADBeamletDataset,
    *,
    val_frac: float,
    seed: int,
    split_by: str,
) -> tuple[Subset, Subset, str, list[str], list[str], bool]:
    mode = str(split_by).lower()
    if mode not in {"patient", "beamlet"}:
        raise ValueError("split_by must be 'patient' or 'beamlet'.")

    patient_ids = _unique_record_patients(dataset)
    if mode == "patient" and len(patient_ids) > 1:
        shuffled = list(patient_ids)
        rng = np.random.default_rng(int(seed))
        rng.shuffle(shuffled)
        n_val = max(1, int(math.ceil(float(val_frac) * len(shuffled))))
        if n_val >= len(shuffled):
            n_val = len(shuffled) - 1
        val_patient_ids = list(shuffled[:n_val])
        val_patient_set = set(val_patient_ids)
        train_indices = [i for i, rec in enumerate(dataset.records) if rec.patient not in val_patient_set]
        val_indices = [i for i, rec in enumerate(dataset.records) if rec.patient in val_patient_set]
        train_patient_ids = [p for p in patient_ids if p not in val_patient_set]
        return (
            Subset(dataset, train_indices),
            Subset(dataset, val_indices),
            "patient",
            val_patient_ids,
            train_patient_ids,
            False,
        )

    train_ds, val_ds = _beamlet_split(dataset, val_frac=val_frac, seed=seed)
    train_indices = [int(i) for i in train_ds.indices]
    val_indices = [int(i) for i in val_ds.indices]
    return (
        train_ds,
        val_ds,
        "beamlet",
        _record_patients_for_indices(dataset, val_indices),
        _record_patients_for_indices(dataset, train_indices),
        mode == "patient" and len(patient_ids) <= 1,
    )


def _beamlet_split(dataset: DoseRADBeamletDataset, *, val_frac: float, seed: int) -> tuple[Subset, Subset]:
    val_len = max(1, int(round(len(dataset) * float(val_frac)))) if len(dataset) > 1 else 1
    train_len = max(1, len(dataset) - val_len)
    if train_len + val_len > len(dataset):
        train_len = len(dataset) - val_len
    gen = torch.Generator().manual_seed(int(seed))
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=gen)
    return train_ds, val_ds


def _unique_record_patients(dataset: DoseRADBeamletDataset) -> list[str]:
    return list(dict.fromkeys(rec.patient for rec in dataset.records))


def _record_patients_for_indices(dataset: DoseRADBeamletDataset, indices: Iterable[int]) -> list[str]:
    return list(dict.fromkeys(dataset.records[int(i)].patient for i in indices))


def _attach_split_metadata(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    split_by: str,
    val_patient_ids: list[str],
    train_patient_ids: list[str],
) -> None:
    for loader in (train_loader, val_loader):
        loader.split_by = str(split_by)
        loader.val_patient_ids = list(val_patient_ids)
        loader.train_patient_ids = list(train_patient_ids)


def normalize_beamlet_dose(dose: NDArray[np.float32]) -> tuple[NDArray[np.float32], float]:
    """Return unit-maximum beamlet dose plus the original positive scale."""

    arr = np.maximum(np.asarray(dose, dtype=np.float32), 0.0)
    scale = float(np.max(arr))
    if not np.isfinite(scale) or scale <= 0.0:
        return arr.astype(np.float32, copy=False), 1.0
    return (arr / np.float32(scale)).astype(np.float32, copy=False), scale


def download_patients(
    patients: Iterable[str] | str,
    dest: str | Path,
    max_beamlets_per_patient: int | None = None,
    *,
    min_file_bytes: int = DOSERAD_MIN_DOWNLOAD_BYTES,
    base_url: str = DOSERAD_HF_BASE_URL,
) -> dict[str, dict[str, int]]:
    """Download selected DoseRAD2026 proton training patients via direct HTTPS.

    The downloaded layout matches ``DoseRADBeamletDataset``:
    ``<dest>/<patient>/ct.mha``, ``plan.json``, and ``dose/Dose_B*_R*_L*.mha``.
    MHA files smaller than ``min_file_bytes`` or unreadable by SimpleITK are
    removed and counted as skipped.
    """

    root = Path(dest)
    root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, int]] = {}
    for patient in _patients_list(patients):
        patient_dir = root / patient
        dose_dir = patient_dir / "dose"
        dose_dir.mkdir(parents=True, exist_ok=True)
        counts = {"plan": 0, "ct": 0, "beamlets": 0, "skipped": 0}

        plan_path = patient_dir / "plan.json"
        if not _download_url(_hf_url(base_url, patient, "plan.json"), plan_path, min_file_bytes=0):
            counts["skipped"] += 1
            summary[patient] = counts
            continue
        try:
            plan = parse_plan(plan_path)
            counts["plan"] = 1
        except Exception:
            plan_path.unlink(missing_ok=True)
            counts["skipped"] += 1
            summary[patient] = counts
            continue

        ct_path = patient_dir / "ct.mha"
        if _download_url(_hf_url(base_url, patient, "ct.mha"), ct_path, min_file_bytes=min_file_bytes) and _readable_mha(ct_path):
            counts["ct"] = 1
        else:
            ct_path.unlink(missing_ok=True)
            counts["skipped"] += 1

        for beam_idx, ray_idx, layer_idx in _plan_beamlet_ids(plan):
            if max_beamlets_per_patient is not None and counts["beamlets"] >= int(max_beamlets_per_patient):
                break
            rel_path = f"dose/Dose_B{beam_idx}_R{ray_idx}_L{layer_idx}.mha"
            dose_path = patient_dir / rel_path
            ok = _download_url(_hf_url(base_url, patient, rel_path), dose_path, min_file_bytes=min_file_bytes)
            if ok and _readable_mha(dose_path):
                counts["beamlets"] += 1
            else:
                dose_path.unlink(missing_ok=True)
                counts["skipped"] += 1
        summary[patient] = counts
    return summary


def gamma_index_3d(
    pred: NDArray[np.float64],
    ref: NDArray[np.float64],
    spacing_mm: Iterable[float],
    *,
    dose_pct: float = 3.0,
    dta_mm: float = 3.0,
    low_dose_threshold: float = 0.1,
) -> float:
    """Return a local-search 3-D global gamma pass rate in percent."""

    pred_arr = np.asarray(pred, dtype=np.float64)
    ref_arr = np.asarray(ref, dtype=np.float64)
    if pred_arr.shape != ref_arr.shape or pred_arr.ndim != 3:
        raise ValueError("pred and ref must be 3-D arrays with identical shape.")
    spacing = np.asarray(tuple(spacing_mm), dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0.0):
        raise ValueError("spacing_mm must contain positive (depth,y,x) spacings.")

    peak = float(np.max(ref_arr))
    if peak <= 0.0:
        return float("nan")
    dose_norm = (float(dose_pct) / 100.0) * peak
    mask = ref_arr >= float(low_dose_threshold) * peak
    points = np.argwhere(mask)
    if points.size == 0:
        return float("nan")

    radii = np.ceil(float(dta_mm) / spacing).astype(int)
    passed = 0
    for iz, iy, ix in points:
        z0, z1 = max(0, iz - radii[0]), min(ref_arr.shape[0], iz + radii[0] + 1)
        y0, y1 = max(0, iy - radii[1]), min(ref_arr.shape[1], iy + radii[1] + 1)
        x0, x1 = max(0, ix - radii[2]), min(ref_arr.shape[2], ix + radii[2] + 1)
        zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
        dist_term = (
            ((zz - iz) * spacing[0] / dta_mm) ** 2
            + ((yy - iy) * spacing[1] / dta_mm) ** 2
            + ((xx - ix) * spacing[2] / dta_mm) ** 2
        )
        dose_term = ((pred_arr[z0:z1, y0:y1, x0:x1] - ref_arr[iz, iy, ix]) / dose_norm) ** 2
        if float(np.sqrt(np.min(dist_term + dose_term))) <= 1.0:
            passed += 1
    return float(100.0 * passed / len(points))


def depth_profile(dose: NDArray[np.float64]) -> NDArray[np.float64]:
    """Collapse a BEV dose cube to a depth-dose curve by lateral summation."""

    arr = np.asarray(dose, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("dose must be a 3-D array.")
    return arr.sum(axis=(1, 2))


def r80_mm_from_profile(profile: NDArray[np.float64], depth_spacing_mm: float) -> float:
    """Return distal R80 in millimetres from a depth-dose profile."""

    curve = np.asarray(profile, dtype=np.float64)
    if curve.ndim != 1 or curve.size < 3 or float(curve.max()) <= 0.0:
        return float("nan")
    peak_idx = int(np.argmax(curve))
    target = 0.8 * float(curve[peak_idx])
    for i in range(peak_idx, curve.size - 1):
        if curve[i] >= target >= curve[i + 1]:
            if curve[i] == curve[i + 1]:
                return float(i * depth_spacing_mm)
            frac = (curve[i] - target) / (curve[i] - curve[i + 1])
            return float((i + frac) * depth_spacing_mm)
    return float("nan")


def rmse_pct_3d(pred: NDArray[np.float64], ref: NDArray[np.float64], low_dose_threshold: float = 0.0) -> float:
    """RMSE normalised to the reference peak, in percent."""

    pred_arr = np.asarray(pred, dtype=np.float64)
    ref_arr = np.asarray(ref, dtype=np.float64)
    if pred_arr.shape != ref_arr.shape:
        raise ValueError("pred and ref must have identical shapes.")
    peak = float(ref_arr.max())
    if peak <= 0.0:
        return float("nan")
    mask = ref_arr >= float(low_dose_threshold) * peak
    return float(np.sqrt(np.mean(((pred_arr[mask] - ref_arr[mask]) / peak) ** 2)) * 100.0)


def _patients_list(patients: Iterable[str] | str) -> list[str]:
    if isinstance(patients, str):
        return [p for p in patients.replace(",", " ").split() if p]
    return [str(p) for p in patients]


def _valid_file(path: Path, min_file_bytes: int) -> bool:
    try:
        return path.exists() and path.stat().st_size >= int(min_file_bytes)
    except OSError:
        return False


def _hf_url(base_url: str, patient: str, rel_path: str) -> str:
    rel = "/".join(part.strip("/") for part in ("proton", "training", patient, rel_path))
    return f"{base_url.rstrip('/')}/{rel}"


def _download_url(url: str, path: Path, *, min_file_bytes: int) -> bool:
    if _valid_file(path, min_file_bytes):
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "braggpeak-doserad/0.1"})
        with urllib.request.urlopen(req, timeout=120) as response, tmp.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        if not _valid_file(tmp, min_file_bytes):
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(path)
        return True
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        tmp.unlink(missing_ok=True)
        return False


def _readable_mha(path: Path) -> bool:
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        return True
    except Exception:
        return False


def _plan_beamlet_ids(plan: dict[str, Any]) -> Iterable[tuple[int, int, int]]:
    for beam in plan.get("beams", []):
        beam_idx = int(beam.get("beam_idx", 0))
        for ray in beam.get("rays", []):
            ray_idx = int(ray.get("ray_idx", 0))
            for beamlet in ray.get("beamlets", []):
                layer_idx = int(beamlet.get("beamlet_idx", beamlet.get("layer_idx", 0)))
                yield beam_idx, ray_idx, layer_idx


def _image_physical_bounds(image: sitk.Image) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    size = np.asarray(image.GetSize(), dtype=np.int64)
    corners = []
    for ix in (0, int(size[0] - 1)):
        for iy in (0, int(size[1] - 1)):
            for iz in (0, int(size[2] - 1)):
                corners.append(image.TransformIndexToPhysicalPoint((ix, iy, iz)))
    pts = np.asarray(corners, dtype=np.float64)
    return pts.min(axis=0), pts.max(axis=0)


def _ray_box_interval(
    source: NDArray[np.float64],
    direction: NDArray[np.float64],
    bounds_min: NDArray[np.float64],
    bounds_max: NDArray[np.float64],
) -> tuple[float, float]:
    tmin = -np.inf
    tmax = np.inf
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if source[axis] < bounds_min[axis] or source[axis] > bounds_max[axis]:
                return float("nan"), float("nan")
            continue
        t1 = (bounds_min[axis] - source[axis]) / direction[axis]
        t2 = (bounds_max[axis] - source[axis]) / direction[axis]
        low, high = min(t1, t2), max(t1, t2)
        tmin = max(tmin, low)
        tmax = min(tmax, high)
    return float(tmin), float(tmax)


def _orthonormal_lateral_axes(depth_axis: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    helper = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(depth_axis, helper))) > 0.9:
        helper = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    u = np.cross(helper, depth_axis)
    u /= np.linalg.norm(u)
    v = np.cross(depth_axis, u)
    v /= np.linalg.norm(v)
    return u, v


def _direction_matrix(
    lateral_u: NDArray[np.float64],
    lateral_v: NDArray[np.float64],
    depth_axis: NDArray[np.float64],
) -> tuple[float, ...]:
    matrix = np.column_stack([lateral_u, lateral_v, depth_axis])
    return tuple(float(x) for x in matrix.reshape(-1))


def _resample_image(
    image: sitk.Image,
    size: tuple[int, int, int],
    spacing: tuple[float, float, float],
    origin: NDArray[np.float64],
    direction: tuple[float, ...],
    *,
    default_value: float,
) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(size)
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputOrigin(tuple(float(x) for x in origin))
    resampler.SetOutputDirection(direction)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    return resampler.Execute(image)
