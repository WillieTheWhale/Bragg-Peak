from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from braggtransporter.data.doserad import extract_bev_pair


def test_depth_extent_400_captures_distal_falloff_after_330mm_peak() -> None:
    ct, dose = _synthetic_depth_peak_pair(peak_depth_mm=330.0)

    profile_400 = _extracted_profile(ct, dose, depth_extent_mm=400.0)
    peak_idx_400 = int(np.argmax(profile_400))
    peak_400 = float(profile_400[peak_idx_400])

    assert peak_idx_400 < profile_400.size - 5
    assert float(np.min(profile_400[peak_idx_400 + 1 :])) < 0.5 * peak_400

    profile_300 = _extracted_profile(ct, dose, depth_extent_mm=300.0)
    peak_idx_300 = int(np.argmax(profile_300))

    assert peak_idx_300 >= profile_300.size - 5


def _extracted_profile(ct: sitk.Image, dose: sitk.Image, *, depth_extent_mm: float) -> np.ndarray:
    depth_size = 401
    bev = extract_bev_pair(
        ct,
        dose,
        ray_source_mm=(-50.0, 6.0, 6.0),
        ray_target_mm=(500.0, 6.0, 6.0),
        depth_size=depth_size,
        depth_extent_mm=depth_extent_mm,
        lateral_size=5,
        lateral_extent_mm=4.0,
    )
    return np.asarray(bev["dose"], dtype=np.float32).sum(axis=(1, 2))


def _synthetic_depth_peak_pair(*, peak_depth_mm: float) -> tuple[sitk.Image, sitk.Image]:
    shape_zyx = (13, 13, 450)
    z, y, x = np.mgrid[0 : shape_zyx[0], 0 : shape_zyx[1], 0 : shape_zyx[2]]
    ct_arr = np.zeros(shape_zyx, dtype=np.float32)
    dose_arr = np.exp(
        -0.5
        * (
            ((x.astype(np.float32) - float(peak_depth_mm)) / 18.0) ** 2
            + ((y.astype(np.float32) - 6.0) / 1.5) ** 2
            + ((z.astype(np.float32) - 6.0) / 1.5) ** 2
        )
    ).astype(np.float32)

    ct = sitk.GetImageFromArray(ct_arr)
    dose = sitk.GetImageFromArray(dose_arr)
    for image in (ct, dose):
        image.SetSpacing((1.0, 1.0, 1.0))
        image.SetOrigin((0.0, 0.0, 0.0))
    return ct, dose
