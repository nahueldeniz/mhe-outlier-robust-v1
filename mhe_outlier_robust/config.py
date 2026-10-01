"""One place for the numbers, so the estimator and the controller cannot disagree.

``beta_max`` in particular is shared on purpose: an estimator allowed a wider jackknife than
the controller would hand the controller states the controller must refuse. ``build_*`` below
push the single value into both configurations, and there is a test that they match.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Section IV of the paper. The simulations add zero-mean Gaussian noise with these standard
# deviations, and draw outliers from a normal distribution of variance 100.
SIGMA_BETA = np.pi / 180          # 1 deg on each joint angle
SIGMA_THETA = 0.2 * np.pi / 180   # 0.2 deg on the tractor's attitude
SIGMA_XY = 0.025                  # 2.5 cm on the tractor's coordinates
OUTLIER_STD = 10.0                # variance 100

# Eq. (13) of the T-ASE path-following paper: |beta| <= pi/2 - delta_beta, delta_beta = 20 deg.
# The obstacle-avoidance controller uses 120 deg because evading manoeuvres need the room; a
# plain path-following problem, which is what this paper runs, uses the value below.
BETA_MAX = np.pi / 2 - 20 * np.pi / 180     # 70 deg


@dataclass
class EstimatorConfig:
    """Tuning of the robust moving horizon estimator (Eqs. 8-11)."""

    n_window: int = 10                  # N_e; Experiment 1 uses 10, the field run 15
    alpha: float = 1.0                  # shape parameter of the loss, in [1, 2)
    c: float = 1.0                      # scale parameter
    ts: float = 0.05                    # T_s [s]; the field run's 5 Hz is set explicitly
    w_gain: float = 1e6                 # W = w_gain * I, as the paper states
    arrival_gain: float = 1.0           # Gamma_chi = arrival_gain * I
    w_bound: float = 0.01               # W: |w_hat| <= 0.01
    v_bound: float = 0.1                # V: |v_hat| <= 0.1 on residuals free of outliers
    beta_max: float = BETA_MAX          # shared with the controller -- see the module docstring
    grow_window: bool = True            # estimate from the first sample instead of waiting
    wrap_angles: bool = True            # see NOTES.md; a deviation from the paper, and a fix
    max_iter: int = 500
    print_level: int = 0


@dataclass
class Config:
    """Whole-experiment settings. ``beta_max`` lives here, once."""

    beta_max: float = BETA_MAX
    n_trailers: int = 1
    ts: float = 0.05
    estimator: dict = field(default_factory=dict)
    controller: dict = field(default_factory=dict)

    def build_estimator_config(self) -> EstimatorConfig:
        kw = dict(self.estimator)
        kw.update(beta_max=self.beta_max, ts=kw.get("ts", self.ts))
        return EstimatorConfig(**kw)

    def build_controller_config(self):
        from ._nmppfc import nmppfc
        kw = dict(self.controller)
        kw.update(beta_max=self.beta_max)
        return nmppfc.ControllerConfig(**kw)
