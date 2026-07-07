from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from braggtransporter.data.doserad import DOSERAD_INPUT_CHANNELS, extract_bev_pair, hu_to_density_rsp


def test_hu_to_density_rsp_has_separate_physical_curves() -> None:
    hu = np.asarray([-1000.0, 0.0, 1000.0], dtype=np.float32)
    density, rsp = hu_to_density_rsp(hu)

    assert density[0] == pytest.approx(0.0, abs=1e-6)
    assert rsp[0] == pytest.approx(0.0, abs=1e-6)
    assert density[1] == pytest.approx(1.0, abs=1e-6)
    assert rsp[1] == pytest.approx(1.0, abs=1e-6)
    assert density[2] > 1.4
    assert 1.3 < rsp[2] < 1.7
    assert density[2] != pytest.approx(float(rsp[2]))


def test_bev_extracts_four_channels_with_monotone_wepl() -> None:
    ct, dose = _constant_sitk_pair(hu=0.0)
    bev = extract_bev_pair(
        ct,
        dose,
        ray_source_mm=(-20.0, 10.0, 10.0),
        ray_target_mm=(360.0, 10.0, 10.0),
        depth_size=16,
        lateral_size=5,
        lateral_extent_mm=8.0,
    )

    x = np.asarray(bev["x"], dtype=np.float32)
    wepl = x[3]
    assert x.shape == (DOSERAD_INPUT_CHANNELS, 16, 5, 5)
    assert x.shape[0] == 4
    assert np.allclose(wepl[0], 0.0)
    assert np.all(np.diff(wepl, axis=0) >= -1e-6)


def test_fixed_depth_spacing_is_constant_across_rays() -> None:
    ct, dose = _constant_sitk_pair(hu=0.0)
    kwargs = {"depth_size": 16, "lateral_size": 5, "lateral_extent_mm": 8.0}
    bev_a = extract_bev_pair(
        ct,
        dose,
        ray_source_mm=(-20.0, 8.0, 8.0),
        ray_target_mm=(360.0, 8.0, 8.0),
        **kwargs,
    )
    bev_b = extract_bev_pair(
        ct,
        dose,
        ray_source_mm=(-120.0, 12.0, 12.0),
        ray_target_mm=(500.0, 12.0, 12.0),
        **kwargs,
    )

    expected_spacing = 300.0 / 15.0
    assert bev_a["depth_spacing_mm"] == pytest.approx(expected_spacing)
    assert bev_b["depth_spacing_mm"] == pytest.approx(expected_spacing)
    assert np.asarray(bev_a["spacing_mm"])[0] == pytest.approx(expected_spacing)
    assert np.asarray(bev_b["spacing_mm"])[0] == pytest.approx(expected_spacing)


def _constant_sitk_pair(*, hu: float) -> tuple[sitk.Image, sitk.Image]:
    shape_zyx = (12, 12, 220)
    ct_arr = np.full(shape_zyx, hu, dtype=np.float32)
    dose_arr = np.zeros(shape_zyx, dtype=np.float32)
    ct = sitk.GetImageFromArray(ct_arr)
    dose = sitk.GetImageFromArray(dose_arr)
    for image in (ct, dose):
        image.SetSpacing((2.0, 2.0, 2.0))
        image.SetOrigin((0.0, 0.0, 0.0))
    return ct, dose
