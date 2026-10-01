"""Closed loop: sensors -> robust MHE -> reference -> controller -> plant.

The controller is not written here. It is ``nmppfc.NMPPFC``, the path-following NMPC of

    Deniz and Auat Cheein, IEEE T-ASE 22:7018-7032, 2025,

driven with no obstacles, and it is handed the estimator's full state vector q -- which is how
the controllers in these papers are formulated. The only thing this module adds is the loop,
the sensor corruption and the paper's three metrics.

Reference: the guided segment is projected onto the path and the reference pose is taken a
lookahead ahead of that projection, held over the whole control window. That is simpler than
the reference generator of the T-ASE framework, which is not what this paper is about; it also
avoids the trap of a moving reference trajectory the closed loop can never catch up with.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ._nmppfc import nmppfc
from .config import EstimatorConfig
from .estimator import RobustMHE
from .model import RedundantNTrailer
from .sensors import SensorModel


@dataclass
class Result:
    t: np.ndarray
    states: np.ndarray            # true q at every step
    estimates: np.ndarray         # q_hat
    measurements: np.ndarray      # r, corrupted
    controls: np.ndarray
    corrupted: np.ndarray         # which samples carried an outlier
    deviation: np.ndarray         # distance of the tractor from the path [m]
    est_error: np.ndarray         # |tractor position error| [m]
    rtf: np.ndarray               # estimator real-time factor per step
    ctrl_cpu: np.ndarray
    solver_failures: int
    metrics: dict = field(default_factory=dict)


def _metrics(res: Result, ts: float) -> dict:
    """Delta, Psi and eta of Section IV, plus the worst cases."""
    return {"mean_deviation_m": float(res.deviation.mean()),
            "max_deviation_m": float(res.deviation.max()),
            "mean_est_error_m": float(res.est_error.mean()),
            "max_est_error_m": float(res.est_error.max()),
            "mean_rtf": float(res.rtf.mean()),
            "max_rtf": float(res.rtf.max()),
            "mean_estimator_cpu_s": float((res.rtf * ts).mean()),
            "outlier_fraction": float(res.corrupted.mean()),
            "solver_failures": int(res.solver_failures),
            "steps": int(res.t.size)}


def run(model: RedundantNTrailer, path, est_cfg: EstimatorConfig | None = None,
        ctrl_cfg=None, sensors: SensorModel | None = None, steps: int = 200,
        speed: float = 1.0, lookahead: float | None = None, use_mhe: bool = True,
        start_offset=(0.0, 0.0, 0.0), progress=None) -> Result:
    """Drive the vehicle along ``path`` with the estimator in the loop.

    ``use_mhe=False`` closes the loop on the true state instead, which is how you tell a
    controller fault from an estimator fault.
    """
    est_cfg = est_cfg or EstimatorConfig()
    ctrl_cfg = ctrl_cfg or nmppfc.ControllerConfig(n_control=8, n_prediction=8,
                                                   beta_max=est_cfg.beta_max, max_iter=120)
    sensors = sensors or SensorModel(n_trailers=model.n, outlier_rows=model.outlier_rows,
                                     n_window=est_cfg.n_window, mode="clean")
    ts = model.ts
    lookahead = speed * ctrl_cfg.n_control * ts if lookahead is None else lookahead

    controller = nmppfc.NMPPFC(model.minimal, ctrl_cfg, 0)
    est = RobustMHE(model, est_cfg)

    p0 = path.pose(0.0)
    q = model.q_from_minimal(model.minimal.state_from_pose(
        p0[0] + start_offset[0], p0[1] + start_offset[1], p0[2] + start_offset[2],
        betas=[0.0] * model.n))
    est.reset(q.copy())
    sensors.reset()

    u_prev = np.zeros(model.nu)
    T, S, E, M, U, DEV, PSI, RTF, CPU = [], [], [], [], [], [], [], [], []
    failures = 0
    for k in range(steps):
        r = sensors(model.measure(q))
        if use_mhe:
            q_hat, info = est.update(r, u_prev=u_prev)
            RTF.append(info.real_time_factor)
        else:
            q_hat, RTF = q.copy(), RTF + [0.0]

        xy = q[model.i_xy(0):model.i_xy(0) + 2]
        s = path.project(xy)
        ref = path.pose(min(s + lookahead, path.gamma_range[1]))

        t0 = time.perf_counter()
        u, flags = controller(q_hat, u_prev, ref, None, None, None)
        CPU.append(time.perf_counter() - t0)
        failures += int(not flags["ok"])

        T.append(k * ts); S.append(q.copy()); E.append(np.asarray(q_hat, float).copy())
        M.append(r.copy()); U.append(np.asarray(u, float).copy())
        DEV.append(path.distance(xy))
        PSI.append(float(np.hypot(*(np.asarray(q_hat)[model.i_xy(0):model.i_xy(0) + 2] - xy))))

        q = model.simulate(q, u)
        u_prev = np.asarray(u, float)
        if progress and (k + 1) % progress == 0:
            print(f"  step {k + 1}/{steps}  dev {DEV[-1]:.3f} m  est {PSI[-1]:.3f} m", flush=True)

    res = Result(np.asarray(T), np.asarray(S), np.asarray(E), np.asarray(M), np.asarray(U),
                 np.asarray(sensors.corrupted[:steps]), np.asarray(DEV), np.asarray(PSI),
                 np.asarray(RTF), np.asarray(CPU), failures)
    res.metrics = _metrics(res, ts)
    return res


def l2_baseline_config(cfg: EstimatorConfig) -> EstimatorConfig:
    """OUR quadratic baseline: the same estimator with the stage cost this paper replaces.

    It is not any of the five methods the paper compares against, and nothing measured with it
    should be read as a statement about those methods.
    """
    from dataclasses import replace
    return replace(cfg, alpha=1.9999, v_bound=1e3)
