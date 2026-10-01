"""Tests for the things a reader should not have to take on trust."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mhe_outlier_robust import loss
from mhe_outlier_robust.config import (BETA_MAX, OUTLIER_STD, SIGMA_BETA, SIGMA_THETA,
                                       SIGMA_XY, Config, EstimatorConfig)
from mhe_outlier_robust.estimator import RobustMHE, wrap
from mhe_outlier_robust.model import RedundantNTrailer, VehicleGeometry

TS = 0.2


def model(n_trailers=1):
    geo = VehicleGeometry(n_trailers=n_trailers, hitch=(0.342, 0.0)[:n_trailers],
                          link=(1.08, 0.78)[:n_trailers],
                          body_length=(0.99,) + (1.0,) * n_trailers,
                          body_width=(0.67,) + (0.8,) * n_trailers)
    return RedundantNTrailer(geo, ts=TS)


def run(m, cfg, steps=40, outlier_p=0.0, seed=0, u=(0.12, 0.7)):
    """Drive the vehicle, corrupt the measurements, and estimate. Returns (errors, infos)."""
    rng = np.random.default_rng(seed)
    q = m.q_from_minimal(m.minimal.state_from_pose(0.0, 0.0, 0.1, betas=[0.05] * m.n))
    est = RobustMHE(m, cfg)
    est.reset(q.copy())
    sd = np.array([SIGMA_BETA] * m.n + [SIGMA_THETA, SIGMA_XY, SIGMA_XY])
    errors, infos = [], []
    for k in range(steps):
        r = m.measure(q) + rng.normal(0.0, sd)
        if outlier_p and rng.random() < outlier_p:
            for i in m.outlier_rows:
                r[i] += rng.normal(0.0, OUTLIER_STD)
        q_hat, info = est.update(r, u_prev=u)
        if info.ok:
            errors.append(np.hypot(*(q_hat[m.i_xy(0):m.i_xy(0) + 2] - q[m.i_xy(0):m.i_xy(0) + 2])))
            infos.append(info)
        q = m.simulate(q, u)
    return np.asarray(errors), infos


# ------------------------------------------------------------------------- the loss
def test_psi_is_bounded_at_alpha_one_and_quadratic_near_two():
    assert loss.psi(1e6, 1.0, 1.0) == pytest.approx(loss.saturation(1.0, 1.0), rel=1e-6)
    assert loss.psi(1e6, 1.0, 2.0) == pytest.approx(loss.saturation(1.0, 2.0), rel=1e-6)
    assert loss.psi(3.0, 1.9999) == pytest.approx(loss.l2(3.0), rel=2e-3)
    assert loss.tail_exponent(1.0) == 0.0 and loss.tail_exponent(1.5) == 1.0


def test_psi_influence_vanishes_at_alpha_one():
    """The derivative of the stage cost is what drags the estimate. It must die off."""
    h = 1e-4
    slope = lambda x, a: (loss.psi(x + h, a) - loss.psi(x - h, a)) / (2 * h)
    assert abs(slope(100.0, 1.0)) < 1e-5
    assert abs(slope(100.0, 1.9)) > 1.0


def test_alpha_outside_the_design_space_is_refused():
    for bad in (0.0, 0.99, 2.0, 2.5):
        with pytest.raises(ValueError):
            loss.check_alpha(bad)
    for good in (1.0, 1.5, 1.9999):
        assert loss.check_alpha(good) == good


def test_garl_reduces_to_the_named_losses():
    assert loss.garl(1.0, 1.0) == pytest.approx(np.sqrt(2) - 1, rel=1e-6)   # Charbonnier


# ------------------------------------------------------- the redundant coordinates
def test_a_consistent_q_satisfies_the_chain_equalities():
    m = model()
    q = m.q_from_minimal(m.minimal.state_from_pose(1.0, -0.5, 0.3, betas=[0.4]))
    assert np.abs(m.chain_residual(q)).max() < 1e-12
    assert np.allclose(m.q_from_minimal(m.minimal_from_q(q)), q, atol=1e-12)


def test_free_propagation_leaves_the_chain_manifold():
    """Why the equalities have to be constraints and not an afterthought."""
    m = model()
    q = m.q_from_minimal(m.minimal.state_from_pose(0, 0, 0.1, betas=[0.15]))
    for _ in range(60):
        q = m.simulate(q, [0.3, 0.8])
    assert np.abs(m.chain_residual(q)).max() > 1e-8


def test_output_map_is_equation_one():
    m = model(2)
    q = m.q_from_minimal(m.minimal.state_from_pose(2.0, 3.0, 0.2, betas=[0.1, -0.1]))
    r = m.measure(q)
    assert r.size == m.ny == m.n + 3
    assert np.allclose(r[:m.n], [0.1, -0.1], atol=1e-12)
    assert r[m.n] == pytest.approx(0.2)
    assert np.allclose(r[m.n + 1:], [2.0, 3.0])
    assert m.outlier_rows == (m.ny - 2, m.ny - 1)


# -------------------------------------------------------------- the estimator
def test_estimate_satisfies_the_chain_constraints():
    m = model()
    errs, infos = run(m, EstimatorConfig(n_window=5, ts=TS), steps=12)
    assert infos, "no successful solve"
    assert max(i.chain_violation for i in infos) < 1e-6


def test_clean_measurements_are_filtered():
    m = model()
    errs, _ = run(m, EstimatorConfig(n_window=8, ts=TS), steps=25, outlier_p=0.0)
    assert errs.size and errs.mean() < SIGMA_XY * 1.5


def test_outliers_do_not_drag_the_estimate():
    """alpha = 1 against the quadratic this paper replaces, same everything else.

    The L2 configuration here is OUR baseline -- a quadratic stage cost with the residual box
    opened up -- not any of the five methods the paper compares against, and these numbers are
    not the paper's numbers.
    """
    m = model()
    robust, _ = run(m, EstimatorConfig(n_window=8, ts=TS, alpha=1.0, c=1.0),
                    steps=45, outlier_p=0.25, seed=3)
    quad, _ = run(m, EstimatorConfig(n_window=8, ts=TS, alpha=1.999, c=1.0,
                                     v_bound=1e3), steps=45, outlier_p=0.25, seed=3)
    assert robust.size and quad.size
    assert robust.mean() < quad.mean()
    assert robust.mean() < 0.6      # the paper reports 0.36 m at 25 %, on its own scenario


def test_alpha_and_c_are_parameters_not_rebuilds():
    """One NLP per window length while the window fills, and none after that."""
    m = model()
    est = RobustMHE(m, EstimatorConfig(n_window=4, ts=TS))
    q = m.q_from_minimal(m.minimal.state_from_pose(0, 0, 0.0, betas=[0.0]))
    est.reset(q)
    for _ in range(6):                       # fill the window
        est.update(m.measure(q), u_prev=[0.0, 0.5])
        q = m.simulate(q, [0.0, 0.5])
    assert sorted(est._problems) == [1, 2, 3, 4]
    built = len(est._problems)
    for k in range(8):                       # now sweep alpha and c
        est.update(m.measure(q), u_prev=[0.0, 0.5],
                   alpha=1.0 if k % 2 else 1.7, c=1.0 if k % 3 else 0.5)
        q = m.simulate(q, [0.0, 0.5])
    assert len(est._problems) == built, "the NLP was rebuilt for a different alpha or c"


def test_angular_residuals_are_wrapped():
    m = model()
    cfg = EstimatorConfig(n_window=4, ts=TS)
    q = m.q_from_minimal(m.minimal.state_from_pose(0.0, 0.0, np.pi - 0.02, betas=[0.0]))
    est = RobustMHE(m, cfg)
    est.reset(q.copy())
    qq, ok = q.copy(), []
    for _ in range(10):
        r = m.measure(qq)
        r[m.n] = wrap(r[m.n])            # heading delivered on the other branch
        q_hat, info = est.update(r, u_prev=[0.05, 0.4])
        if info.ok:
            ok.append(abs(wrap(q_hat[m.i_theta(0)] - qq[m.i_theta(0)])))
        qq = m.simulate(qq, [0.05, 0.4])
    assert ok and max(ok) < 0.1


def test_estimator_and_controller_share_the_joint_limit():
    cfg = Config(beta_max=1.1)
    e = cfg.build_estimator_config()
    c = cfg.build_controller_config()
    assert e.beta_max == c.beta_max == 1.1
    assert EstimatorConfig().beta_max == BETA_MAX == pytest.approx(70 * np.pi / 180)


def test_the_estimate_respects_the_shared_joint_limit():
    """The bound the controller enforces is enforced here too, so the estimator can never
    hand the controller a jackknife the controller must refuse.

    The vehicle is steered hard so the true joint angle goes past the limit, and the residual
    box is opened (``v_bound``) only so the problem stays feasible while that happens -- the
    point of the test is the bound on the state, not the paper's residual set.
    """
    m = model()
    cfg = EstimatorConfig(n_window=4, ts=TS, beta_max=0.3, v_bound=1.0)
    q = m.q_from_minimal(m.minimal.state_from_pose(0, 0, 0.0, betas=[0.0]))
    est = RobustMHE(m, cfg)
    est.reset(q.copy())
    qq, seen, truth = q.copy(), [], []
    for _ in range(8):
        q_hat, info = est.update(m.measure(qq), u_prev=[0.8, 0.5])
        if info.ok:
            seen.append(abs(q_hat[m.i_beta(0)]))
        truth.append(abs(qq[m.i_beta(0)]))
        qq = m.simulate(qq, [0.8, 0.5])
    assert max(truth) > cfg.beta_max, "the scenario never tests the bound"
    assert seen and max(seen) <= cfg.beta_max + 1e-6
