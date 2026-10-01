"""The loop, the path and the sensors. Kept short enough to run in a normal test pass."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mhe_outlier_robust import EstimatorConfig, RedundantNTrailer, VehicleGeometry
from mhe_outlier_robust import path as paths
from mhe_outlier_robust import sim
from mhe_outlier_robust._nmppfc import nmppfc
from mhe_outlier_robust.sensors import SensorModel

TS = 0.05


def model(n_trailers=1):
    geo = VehicleGeometry(n_trailers=n_trailers, hitch=(0.342, 0.0)[:n_trailers],
                          link=(1.08, 0.78)[:n_trailers],
                          body_length=(0.99,) + (1.0,) * n_trailers,
                          body_width=(0.67,) + (0.8,) * n_trailers)
    return RedundantNTrailer(geo, ts=TS)


def drive(m, est_cfg, mode="clean", delta=0.25, steps=40, seed=7, use_mhe=True, n_clean=10):
    s = SensorModel(n_trailers=m.n, outlier_rows=m.outlier_rows, n_window=est_cfg.n_window,
                    mode=mode, delta=delta, seed=seed, n_clean=n_clean)
    cc = nmppfc.ControllerConfig(n_control=6, n_prediction=6, beta_max=est_cfg.beta_max,
                                 max_iter=100)
    return sim.run(m, paths.agricultural(), est_cfg=est_cfg, ctrl_cfg=cc, sensors=s,
                   steps=steps, speed=1.0, use_mhe=use_mhe)


# ------------------------------------------------------------------------- the path
def test_path_matches_the_extents_of_figure_two():
    pl = paths.polyline()
    assert pl[:, 0].min() == pytest.approx(-3.854, abs=0.02)
    assert pl[:, 0].max() == pytest.approx(23.854, abs=0.02)
    assert pl[:, 1].max() == pytest.approx(1.25, abs=0.02)
    assert pl[:, 1].min() == pytest.approx(-7.25, abs=0.02)
    assert paths.length() == pytest.approx(5 * 20.0 + 4 * 11.03, abs=0.5)


def test_bulb_turns_leave_the_rows():
    """A short arc instead of the long one collapses the turn: catch that regression."""
    pl = paths.polyline()
    assert (pl[:, 0] > paths.ROW_LENGTH + 1.0).any()
    assert (pl[:, 0] < -1.0).any()


def test_path_is_parameterised_by_arc_length():
    p = paths.agricultural()
    a, b = p.point(10.0), p.point(11.0)
    assert np.hypot(*(b - a)) == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------- the sensors
def test_sensor_draws_are_reproducible_and_ordered():
    m = model()
    r = m.measure(m.q_from_minimal(m.minimal.state_from_pose(0, 0, 0, betas=[0.0])))
    kw = dict(n_trailers=1, outlier_rows=m.outlier_rows, n_window=10, mode="scattered",
              delta=0.3, seed=11)
    a = SensorModel(**kw)(r)
    s = SensorModel(**kw)
    assert np.allclose(s(r), a), "two fresh models with the same seed must agree"
    first = [a] + [s(r) for _ in range(4)]
    s.reset()
    again = [s(r) for _ in range(5)]
    assert all(np.allclose(x, y) for x, y in zip(first, again)), "reset must rewind the draws"


def test_burst_mode_corrupts_exactly_the_outage_window():
    m = model()
    r = m.measure(m.q_from_minimal(m.minimal.state_from_pose(0, 0, 0, betas=[0.0])))
    s = SensorModel(n_trailers=1, outlier_rows=m.outlier_rows, n_window=10, mode="burst",
                    delta=1.0, n_clean=10, seed=0)
    for _ in range(25):
        s(r)
    hit = np.asarray(s.corrupted)
    assert not hit[:10].any()
    assert hit[10:20].all()
    assert not hit[20:].any()


def test_outliers_only_reach_the_gnss_rows():
    m = model()
    q = m.q_from_minimal(m.minimal.state_from_pose(0, 0, 0, betas=[0.0]))
    s = SensorModel(n_trailers=1, outlier_rows=m.outlier_rows, n_window=10, mode="burst",
                    delta=1.0, n_clean=0, seed=1, sigma_beta=0.0, sigma_theta=0.0,
                    sigma_xy=0.0)
    r = s(m.measure(q))
    clean = m.measure(q)
    assert np.allclose(r[:m.n + 1], clean[:m.n + 1])            # joints and heading untouched
    assert not np.allclose(r[m.n + 1:], clean[m.n + 1:])        # the coordinates are hit


# ------------------------------------------------------------------- the closed loop
def test_closed_loop_runs_and_reports_the_metrics():
    m = model()
    res = drive(m, EstimatorConfig(n_window=6, ts=TS), steps=30)
    for k in ("mean_deviation_m", "mean_est_error_m", "mean_rtf", "solver_failures"):
        assert k in res.metrics
    assert res.states.shape == (30, m.nq)
    assert res.metrics["mean_rtf"] < 1.0, "the estimator is not keeping up with real time"


def test_the_controller_is_driven_by_the_estimate():
    """Not by the true state -- the whole point of the paper."""
    m = model()
    est = drive(m, EstimatorConfig(n_window=6, ts=TS), mode="scattered", steps=30)
    true = drive(m, EstimatorConfig(n_window=6, ts=TS), mode="scattered", steps=30,
                 use_mhe=False)
    assert not np.allclose(est.controls, true.controls)
    assert np.allclose(true.est_error, 0.0)


def test_the_robust_loss_beats_the_quadratic_in_the_loop():
    """OUR quadratic baseline, not the paper's competitors, and not the paper's numbers."""
    m = model()
    cfg = EstimatorConfig(n_window=8, ts=TS, alpha=1.0, c=1.0)
    robust = drive(m, cfg, mode="scattered", delta=0.25, steps=60)
    quad = drive(m, sim.l2_baseline_config(cfg), mode="scattered", delta=0.25, steps=60)
    assert robust.metrics["mean_est_error_m"] < quad.metrics["mean_est_error_m"]
    assert robust.metrics["mean_est_error_m"] < 0.2
