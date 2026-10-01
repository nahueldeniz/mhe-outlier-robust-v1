"""Measurements: Gaussian noise on everything, plus GNSS outliers on the tractor's coordinates.

The paper runs two outlier scenarios (Section IV):

* **scattered** -- each sample is corrupted with probability ``delta``, the setting behind
  Fig. 3(c) and 3(d) at 1, 5, 10, 25 and 50 per cent;
* **burst** -- ``n_clean`` outlier-free samples followed by ``delta * N_e`` corrupted ones, the
  outage of Fig. 4. ``n_clean = N_e`` is Fig. 4(a); ``n_clean = 35`` is Fig. 4(b).

Draws happen in a FIXED order -- noise first, then the outlier decision, then the outlier
itself, every step, whether or not they are used -- so a cached result stays reproducible when
a configuration changes something downstream.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import OUTLIER_STD, SIGMA_BETA, SIGMA_THETA, SIGMA_XY


@dataclass
class SensorModel:
    """Noise and outliers for the output of Eq. (1)."""

    n_trailers: int = 1
    outlier_rows: tuple = ()
    sigma_beta: float = SIGMA_BETA
    sigma_theta: float = SIGMA_THETA
    sigma_xy: float = SIGMA_XY
    outlier_std: float = OUTLIER_STD
    mode: str = "scattered"          # "scattered" | "burst" | "clean"
    delta: float = 0.25              # corruption probability, or outage length / N_e
    n_clean: int = 10                # burst only: clean samples before the outage
    n_window: int = 10               # burst only: N_e, which scales the outage length
    seed: int = 0

    def __post_init__(self):
        if self.mode not in ("scattered", "burst", "clean"):
            raise ValueError(f"unknown outlier mode '{self.mode}'")
        self._rng = np.random.default_rng(self.seed)
        self._k = 0
        self.sigma = np.array([self.sigma_beta] * self.n_trailers
                              + [self.sigma_theta, self.sigma_xy, self.sigma_xy])
        self.corrupted = []          # which samples were hit, for the plots

    def _hit(self) -> bool:
        if self.mode == "clean":
            return False
        if self.mode == "scattered":
            return bool(self._rng.random() < self.delta)
        outage = int(round(self.delta * self.n_window))
        return self.n_clean <= self._k < self.n_clean + outage

    def __call__(self, truth_output) -> np.ndarray:
        r = np.asarray(truth_output, float).ravel().copy()
        noise = self._rng.normal(0.0, self.sigma)          # drawn every step, fixed order
        spike = self._rng.normal(0.0, self.outlier_std, size=len(self.outlier_rows))
        r = r + noise
        hit = self._hit()
        if hit:
            for j, i in enumerate(self.outlier_rows):
                r[i] += spike[j]
        self.corrupted.append(bool(hit))
        self._k += 1
        return r

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._k = 0
        self.corrupted = []
