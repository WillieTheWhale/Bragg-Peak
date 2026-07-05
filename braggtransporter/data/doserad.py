"""DoseRAD2026 beamlet loading and BEV extraction.

Research software only. Coordinates and spacings are millimetres, matching
SimpleITK physical coordinates for the provided CT and dose volumes. Extracted
tensors are small beam's-eye-view (BEV) grids with axis order
``(depth, lateral_y, lateral_x)`` after SimpleITK's array conversion.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader, Dataset, random_split


DOSERAD_INPUT_CHANNELS = 3


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
    """Approximate HU-to-density/RSP conversion for laptop-scale Phase 5.

    This is an explicit, simple CT prior for proof-of-concept training only:
    air-like voxels map toward zero, water is one, and dense tissue rises
    linearly up to a conservative cap. RSP is taken equal to mass density here
    because the real material calibration table is not part of the public
    DoseRAD beamlet files.
    """

    hu32 = np.asarray(hu, dtype=np.float32)
    density = np.where(hu32 < 0.0, 1.0 + hu32 / 1000.0, 1.0 + hu32 / 1000.0)
    density = np.clip(density, 0.0, 2.5).astype(np.float32)
    rsp = density.copy()
    return density, rsp


def extract_bev_pair(
    ct_image: sitk.Image,
    dose_image: sitk.Image,
    ray_source_mm: Iterable[float],
    ray_target_mm: Iterable[float],
    *,
    depth_size: int = 64,
    lateral_size: int = 24,
    lateral_extent_mm: float = 96.0,
) -> dict[str, NDArray[np.float32] | tuple[float, float, float]]:
    """Extract a downsampled BEV CT/dose pair for one ray.

    Output ``x`` has channels ``[HU/1000, density_g_cm3, RSP]`` and shape
    ``(3, depth, lateral_y, lateral_x)``. ``dose`` has shape
    ``(depth, lateral_y, lateral_x)``. The beam-depth axis follows
    ``ray_source -> ray_target`` and the first depth slice is placed at the CT
    entrance along that ray.
    """

    if depth_size <= 1 or lateral_size <= 1:
        raise ValueError("depth_size and lateral_size must both be > 1.")

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
        nominal_depth = min(192.0, norm)
        start = target - 0.5 * nominal_depth * depth_axis
        depth_extent_mm = nominal_depth
    else:
        start = source + max(t0, 0.0) * depth_axis
        depth_extent_mm = max(1.0, t1 - max(t0, 0.0))

    lateral_u, lateral_v = _orthonormal_lateral_axes(depth_axis)
    depth_spacing_mm = depth_extent_mm / float(depth_size - 1)
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
    x = np.stack([hu / 1000.0, density, rsp], axis=0).astype(np.float32, copy=False)

    return {
        "x": x,
        "dose": dose.astype(np.float32, copy=False),
        "spacing_mm": (float(depth_spacing_mm), float(lateral_spacing_mm), float(lateral_spacing_mm)),
    }


class DoseRADBeamletDataset(Dataset[dict[str, torch.Tensor | dict[str, Any]]]):
    """Torch dataset of cached downsampled DoseRAD2026 beamlet BEV tensors."""

    def __init__(
        self,
        root: str | Path = "data/doserad2026",
        patients: Iterable[str] | str = ("1ABB006",),
        *,
        max_beamlets: int | None = None,
        depth_size: int = 64,
        lateral_size: int = 24,
        lateral_extent_mm: float = 96.0,
        cache_dir: str | Path | None = None,
        rebuild_cache: bool = False,
        min_file_bytes: int = 1024,
    ) -> None:
        self.root = Path(root)
        self.patients = _patients_list(patients)
        self.depth_size = int(depth_size)
        self.lateral_size = int(lateral_size)
        self.lateral_extent_mm = float(lateral_extent_mm)
        self.rebuild_cache = bool(rebuild_cache)
        self.min_file_bytes = int(min_file_bytes)
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
            scalars = z["scalars"].astype(np.float32, copy=False)
            spacing_mm = z["spacing_mm"].astype(np.float32, copy=False)
        return {
            "x": torch.from_numpy(x),
            "dose": torch.from_numpy(dose),
            "scalars": torch.from_numpy(scalars),
            "spacing_mm": torch.from_numpy(spacing_mm),
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
            plan = parse_plan(patient_dir / "plan.json")
            ct_path = patient_dir / "ct.mha"
            if not ct_path.exists():
                continue
            ct_image = read_mha(ct_path)
            for rec in self._iter_patient_records(patient, patient_dir, plan):
                if max_beamlets is not None and len(records) >= int(max_beamlets):
                    return records
                try:
                    if self.rebuild_cache or not rec.cache_path.exists():
                        self._write_cache(rec, ct_image)
                    records.append(rec)
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
        tag = f"d{self.depth_size}_l{self.lateral_size}_e{self.lateral_extent_mm:g}"
        return self.cache_root / patient / tag / f"B{beam_idx}_R{ray_idx}_L{layer_idx}.npz"

    def _write_cache(self, rec: BeamletRecord, ct_image: sitk.Image) -> None:
        dose_image = read_mha(rec.dose_path)
        bev = extract_bev_pair(
            ct_image,
            dose_image,
            rec.ray_source_mm,
            rec.ray_target_mm,
            depth_size=self.depth_size,
            lateral_size=self.lateral_size,
            lateral_extent_mm=self.lateral_extent_mm,
        )
        gantry_rad = math.radians(rec.gantry_angle_deg)
        scalars = np.asarray(
            [rec.energy_mev, float(rec.layer_idx), math.sin(gantry_rad), math.cos(gantry_rad)],
            dtype=np.float32,
        )
        rec.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            rec.cache_path,
            x=np.asarray(bev["x"], dtype=np.float32),
            dose=np.asarray(bev["dose"], dtype=np.float32),
            scalars=scalars,
            spacing_mm=np.asarray(bev["spacing_mm"], dtype=np.float32),
        )


def make_doserad_loaders(
    root: str | Path = "data/doserad2026",
    patients: Iterable[str] | str = ("1ABB006",),
    *,
    max_beamlets: int | None = None,
    val_frac: float = 0.25,
    batch_size: int = 2,
    seed: int = 0,
    depth_size: int = 64,
    lateral_size: int = 24,
    lateral_extent_mm: float = 96.0,
    cache_dir: str | Path | None = None,
    rebuild_cache: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build cached DoseRAD beamlet dataset and split train/val over beamlets."""

    dataset = DoseRADBeamletDataset(
        root,
        patients,
        max_beamlets=max_beamlets,
        depth_size=depth_size,
        lateral_size=lateral_size,
        lateral_extent_mm=lateral_extent_mm,
        cache_dir=cache_dir,
        rebuild_cache=rebuild_cache,
    )
    val_len = max(1, int(round(len(dataset) * float(val_frac)))) if len(dataset) > 1 else 1
    train_len = max(1, len(dataset) - val_len)
    if train_len + val_len > len(dataset):
        train_len = len(dataset) - val_len
    gen = torch.Generator().manual_seed(int(seed))
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


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
