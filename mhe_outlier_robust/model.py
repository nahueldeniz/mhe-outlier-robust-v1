"""The vehicle in the paper's own coordinates: the redundant vector q, with the chain
equalities carried explicitly.

The paper writes the N-trailer state as

    q = (beta_1..beta_N, theta_0, theta_1..theta_N, x_0, y_0, x_1, y_1, ..., x_N, y_N)

of dimension 4N+3, the dynamics as ``qdot = G(q) u`` with ``u = (omega_0, v_0)``, and the
output as ``r = h(q) = (beta_1..beta_N, theta_0, x_0, y_0)`` (Eq. 1): the joint angles and the
tractor's attitude and coordinates are measured, the trailers' poses are not.

Those coordinates are redundant -- 4N+3 numbers describe a 3+N dimensional configuration -- and
the redundancy is the point. It is what lets the estimator impose

    beta_i = theta_{i-1} - theta_i
    (x_i, y_i) = (x_{i-1}, y_{i-1}) - L_h,i (cos theta_{i-1}, sin theta_{i-1})
                                    - L_i   (cos theta_i,     sin theta_i)

as hard equality constraints inside the same optimisation that fights the outliers. A filter
has nowhere to put them. ``chain_constraints`` builds them symbolically.

The geometry and the minimal-coordinate helpers come from ``nmppfc.model``; nothing about the
vehicle is re-derived here.
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from ._nmppfc import NTrailer, VehicleGeometry

__all__ = ["NTrailer", "VehicleGeometry", "RedundantNTrailer"]


class RedundantNTrailer:
    """``qdot = G(q) u``, its RK4 discretisation, the output map and the chain equalities."""

    def __init__(self, geometry: VehicleGeometry | None = None, ts: float = 0.05):
        self.minimal = NTrailer(geometry or VehicleGeometry(n_trailers=1, hitch=(0.342,),
                                                            link=(1.08,),
                                                            body_length=(0.99, 1.0),
                                                            body_width=(0.67, 0.8)), ts=ts)
        self.geo = self.minimal.geo
        self.ts = float(ts)
        self.n = self.geo.n_trailers
        self.nq = 4 * self.n + 3
        self.nu = 2
        self.ny = self.n + 3
        self.layout = self.minimal.q_layout()

        q = ca.SX.sym("q", self.nq)
        u = ca.SX.sym("u", self.nu)
        self.G = ca.Function("G", [q, u], [self._ode(q, u)], ["q", "u"], ["qdot"])
        self.step = ca.Function("step", [q, u], [self._rk4(q, u)], ["q", "u"], ["q_next"])
        self.h = ca.Function("h", [q], [self._output(q)], ["q"], ["r"])
        self.chain = ca.Function("chain", [q], [self._chain(q)], ["q"], ["g"])

    # ------------------------------------------------------------------ indices
    def i_beta(self, i):  return self.layout["beta"].start + i          # i = 0 .. n-1
    def i_theta(self, i): return self.layout["theta0"] if i == 0 else self.layout["theta"].start + i - 1
    def i_xy(self, i):    return (self.layout["xy0"].start if i == 0
                                  else self.layout["xy"].start + 2 * (i - 1))

    # ----------------------------------------------------------------- dynamics
    def _velocities(self, q, u):
        """(v_i, omega_i) for every segment, from the chain recursion."""
        v, w = [u[1]], [u[0]]
        for i in range(self.n):
            b = q[self.i_beta(i)]
            lh, ll = self.geo.hitch[i], self.geo.link[i]
            v.append(v[-1] * ca.cos(b) + lh * w[-1] * ca.sin(b))
            w.append((v[-2] * ca.sin(b) - lh * w[-1] * ca.cos(b)) / ll)
        return v, w

    def _ode(self, q, u):
        v, w = self._velocities(q, u)
        qdot = [None] * self.nq
        for i in range(self.n):
            qdot[self.i_beta(i)] = w[i] - w[i + 1]
        for i in range(self.n + 1):
            qdot[self.i_theta(i)] = w[i]
            k = self.i_xy(i)
            qdot[k] = v[i] * ca.cos(q[self.i_theta(i)])
            qdot[k + 1] = v[i] * ca.sin(q[self.i_theta(i)])
        return ca.vertcat(*qdot)

    def _rk4(self, q, u):
        h = self.ts
        k1 = self._ode(q, u)
        k2 = self._ode(q + h / 2 * k1, u)
        k3 = self._ode(q + h / 2 * k2, u)
        k4 = self._ode(q + h * k3, u)
        return q + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # ------------------------------------------------------------------- output
    def _output(self, q):
        """Eq. (1): the joint angles, then the tractor's heading and coordinates."""
        rows = [q[self.i_beta(i)] for i in range(self.n)]
        rows += [q[self.i_theta(0)], q[self.i_xy(0)], q[self.i_xy(0) + 1]]
        return ca.vertcat(*rows)

    @property
    def outlier_rows(self) -> tuple:
        """Rows of the output that the GNSS outliers hit: the tractor's coordinates."""
        return (self.ny - 2, self.ny - 1)

    @property
    def angular_rows(self) -> tuple:
        """Rows of the output that are angles, so their residuals must be wrapped."""
        return tuple(range(self.n + 1))

    # ----------------------------------------------------------- the constraints
    def _chain(self, q):
        g = []
        for i in range(1, self.n + 1):
            th_p, th_i = q[self.i_theta(i - 1)], q[self.i_theta(i)]
            g.append(q[self.i_beta(i - 1)] - (th_p - th_i))
            lh, ll = self.geo.hitch[i - 1], self.geo.link[i - 1]
            kp, ki = self.i_xy(i - 1), self.i_xy(i)
            g.append(q[ki] - (q[kp] - lh * ca.cos(th_p) - ll * ca.cos(th_i)))
            g.append(q[ki + 1] - (q[kp + 1] - lh * ca.sin(th_p) - ll * ca.sin(th_i)))
        return ca.vertcat(*g)

    def chain_constraints(self, q_sym):
        """Symbolic chain equalities for a single instant, to add to an NLP as ``g = 0``."""
        return self.chain(q_sym)

    # ------------------------------------------------------------------ numeric
    def q_from_minimal(self, x) -> np.ndarray:
        return self.minimal.q_from_x(x)

    def minimal_from_q(self, q) -> np.ndarray:
        return self.minimal.x_from_q(q)

    def chain_residual(self, q) -> np.ndarray:
        return np.array(self.chain(ca.DM(np.asarray(q, float).ravel()))).ravel()

    def measure(self, q) -> np.ndarray:
        return np.array(self.h(ca.DM(np.asarray(q, float).ravel()))).ravel()

    def simulate(self, q, u, n_steps: int = 1) -> np.ndarray:
        q = ca.DM(np.asarray(q, float).ravel())
        for _ in range(n_steps):
            q = self.step(q, ca.DM(np.asarray(u, float).ravel()))
        return np.array(q).ravel()
