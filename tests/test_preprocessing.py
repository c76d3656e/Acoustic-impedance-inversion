import numpy as np

from preprocessing import (
    write_segy,
    read_segy,
    read_las,
    well_acoustic_impedance,
    write_synthetic_las,
    depth_to_twt,
    resample_to_time,
)


def test_segy_roundtrip(tmp_path):
    cube = np.random.default_rng(0).standard_normal((5, 6, 40)).astype(np.float32)
    path = str(tmp_path / "test.sgy")
    write_segy(path, cube, dt_us=2000)
    out, dt = read_segy(path)
    assert out.shape == cube.shape
    assert np.isclose(dt, 0.002)
    assert np.allclose(out, cube, atol=1e-4)


def test_las_roundtrip_and_impedance(tmp_path):
    depth = np.arange(1000.0, 1100.0, 0.5)
    vp = np.full_like(depth, 2500.0)
    rho = np.full_like(depth, 2300.0)
    path = str(tmp_path / "well.las")
    write_synthetic_las(path, depth, vp, rho, well_name="TEST-1")

    las = read_las(path)
    d, ai = well_acoustic_impedance(las)
    assert np.allclose(d, depth, atol=1e-6)
    # AI = Vp * rho = 2500 * 2300
    assert np.allclose(ai, 2500.0 * 2300.0, rtol=1e-3)


def test_depth_time_monotonic():
    depth = np.arange(0.0, 1000.0, 1.0)
    vp = np.full_like(depth, 2000.0)
    twt = depth_to_twt(depth, vp)
    assert np.all(np.diff(twt) >= 0)
    t_axis, vals = resample_to_time(depth, depth, vp, dt=0.002)
    assert t_axis[0] == 0.0
    assert np.all(np.diff(t_axis) > 0)
