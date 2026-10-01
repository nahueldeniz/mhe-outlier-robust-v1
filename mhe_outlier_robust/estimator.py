"""The robust moving horizon estimator of the paper, Eqs. (8) to (11).

Over the last ``N_e`` sampling instants it solves

    min   Gamma_chi(q_0 - qbar)  +  sum_j w_j' W w_j  +  sum_j sum_i psi(v_ij, alpha, c)
    s.t.  q_{j+1} = Phi(q_j, u_j) + w_j                 (RK4, multiple shooting)
          r_j     = h(q_j) + v_j                        (Eq. 1)
          chain(q_j) = 0                                (the vehicle's own geometry)
          |beta| <= beta_max,  w_j in W,  v_j in V

Three things distinguish it from a textbook L2 moving horizon estimator, and all three are the
paper's:

1. the stage cost on the residuals is ``psi`` of Eq. (3), not a quadratic, so a wild GNSS
   reading stops dragging the estimate (``loss.py``);
2. the residual components the outliers hit -- the tractor's coordinates -- are left
   UNBOUNDED, and the loss deals with them, while the residuals we do believe are boxed;
3. the chain equalities are hard constraints in the same optimisation.

``alpha`` and ``c`` are symbolic parameters of the NLP, so sweeping them costs a
``set_value``, not a rebuild. Nothing here hard-codes a shape parameter.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import casadi as ca
import numpy as np

from .config import EstimatorConfig
from .loss import check_alpha, psi_symbolic
from .model import RedundantNTrailer


def wrap(d):
    """Angle difference mapped to (-pi, pi], symbolically or numerically."""
    if isinstance(d, (ca.SX, ca.MX, ca.DM)):
        return ca.atan2(ca.sin(d), ca.cos(d))
    return np.arctan2(np.sin(d), np.cos(d))


@dataclass
class EstimateInfo:
    ok: bool
    cost: float
    cpu: float
    iterations: int
    chain_violation: float
    status: str

    @property
    def real_time_factor(self):
        """eta of the paper: solve time divided by the sampling period. Set by the estimator."""
        return self._rtf

    _rtf: float = 0.0


class RobustMHE:
    """One NLP per window length, built once and cached; solved with IPOPT."""

    def __init__(self, model: RedundantNTrailer, cfg: EstimatorConfig | None = None):
        self.model = model
        self.cfg = cfg or EstimatorConfig()
        check_alpha(self.cfg.alpha)
        if abs(model.ts - self.cfg.ts) > 1e-12:
            raise ValueError(f"model ts {model.ts} and estimator ts {self.cfg.ts} disagree")
        self._problems: dict[int, dict] = {}
        self._history_r: list[np.ndarray] = []
        self._history_u: list[np.ndarray] = []
        self._qbar: np.ndarray | None = None
        self._guess: np.ndarray | None = None
        self.last_info: EstimateInfo | None = None

    # -------------------------------------------------------------- the problem
    def _build(self, ne: int) -> dict:
        m, cfg = self.model, self.cfg
        nq, ny = m.nq, m.ny
        opti = ca.Opti()

        Q = opti.variable(nq, ne + 1)
        W = opti.variable(nq, ne)
        V = opti.variable(ny, ne + 1)

        R = opti.parameter(ny, ne + 1)          # the measurement window
        U = opti.parameter(m.nu, ne)            # the inputs applied over it
        qbar = opti.parameter(nq)               # arrival state
        alpha = opti.parameter()
        c = opti.parameter()
        psi = psi_symbolic(alpha, c)

        dq = Q[:, 0] - qbar
        cost = cfg.arrival_gain * ca.dot(dq, dq) + cfg.w_gain * ca.sumsqr(W)

        for j in range(ne + 1):
            if j < ne:
                opti.subject_to(Q[:, j + 1] == m.step(Q[:, j], U[:, j]) + W[:, j])
            # the geometry, as a hard constraint
            opti.subject_to(m.chain(Q[:, j]) == 0)
            # Eq. (1), with angular residuals wrapped so a heading on the branch cut does not
            # pay 2 pi for being there
            pred = m.h(Q[:, j])
            for i in range(ny):
                d = R[i, j] - pred[i]
                if cfg.wrap_angles and i in m.angular_rows:
                    d = wrap(d)
                opti.subject_to(V[i, j] == d)
                cost = cost + psi(V[i, j], alpha, c)

        # bounds: the joint angles the controller also enforces, the disturbance set W, and the
        # residual set V -- unbounded exactly where the outliers land
        for j in range(ne + 1):
            for i in range(m.n):
                opti.subject_to(opti.bounded(-cfg.beta_max, Q[m.i_beta(i), j], cfg.beta_max))
            for i in range(ny):
                if i not in m.outlier_rows:
                    opti.subject_to(opti.bounded(-cfg.v_bound, V[i, j], cfg.v_bound))
        opti.subject_to(opti.bounded(-cfg.w_bound, ca.vec(W), cfg.w_bound))

        opti.minimize(cost)
        opti.solver("ipopt",
                    {"print_time": False, "expand": True},
                    {"print_level": cfg.print_level, "sb": "yes", "max_iter": cfg.max_iter,
                     "tol": 1e-6, "warm_start_init_point": "yes"})
        return dict(opti=opti, Q=Q, W=W, V=V, R=R, U=U, qbar=qbar,
                    alpha=alpha, c=c, cost=cost, ne=ne)

    def _problem(self, ne: int) -> dict:
        if ne not in self._problems:
            self._problems[ne] = self._build(ne)
        return self._problems[ne]

    # ------------------------------------------------------------------ the loop
    def reset(self, q0_guess) -> None:
        """Start a new run from an initial guess of the state (the paper's qbar_0)."""
        self._history_r.clear()
        self._history_u.clear()
        self._qbar = np.asarray(q0_guess, float).ravel().copy()
        self._guess = None

    def update(self, r, u_prev=None, alpha=None, c=None):
        """Feed one measurement and return the current state estimate.

        ``u_prev`` is the input applied between the previous measurement and this one.
        ``alpha`` and ``c`` override the configured values for this step only -- they are NLP
        parameters, so nothing is rebuilt.
        """
        m, cfg = self.model, self.cfg
        if self._qbar is None:
            raise RuntimeError("call reset(q0_guess) before update()")
        self._history_r.append(np.asarray(r, float).ravel())
        if u_prev is not None:
            self._history_u.append(np.asarray(u_prev, float).ravel())

        ne_full = cfg.n_window
        available = len(self._history_r) - 1
        ne = min(ne_full, available) if cfg.grow_window else ne_full
        if ne < 1:                      # nothing to estimate over yet
            return self._qbar.copy(), EstimateInfo(False, float("nan"), 0.0, 0,
                                                   float(np.abs(m.chain_residual(self._qbar)).max()),
                                                   "window empty")
        if available < ne:
            return self._qbar.copy(), EstimateInfo(False, float("nan"), 0.0, 0,
                                                   float(np.abs(m.chain_residual(self._qbar)).max()),
                                                   "window filling")

        p = self._problem(ne)
        opti = p["opti"]
        R = np.stack(self._history_r[-(ne + 1):], axis=1)
        U = (np.stack(self._history_u[-ne:], axis=1) if len(self._history_u) >= ne
             else np.zeros((m.nu, ne)))
        opti.set_value(p["R"], R)
        opti.set_value(p["U"], U)
        opti.set_value(p["qbar"], self._qbar)
        opti.set_value(p["alpha"], check_alpha(cfg.alpha if alpha is None else alpha))
        opti.set_value(p["c"], cfg.c if c is None else c)

        # a feasible-by-construction guess: roll the arrival state forward, so every column
        # already satisfies the chain equalities
        guess = np.stack([m.simulate(self._qbar, U[:, min(j, ne - 1)], j) for j in range(ne + 1)],
                         axis=1)
        opti.set_initial(p["Q"], guess)
        opti.set_initial(p["W"], np.zeros((m.nq, ne)))

        t0 = time.perf_counter()
        try:
            sol = opti.solve()
            Qopt = np.array(sol.value(p["Q"])).reshape(m.nq, ne + 1)
            cost, status, ok = float(sol.value(p["cost"])), "solved", True
        except RuntimeError:                       # IPOPT gave up; keep the last iterate
            Qopt = np.array(opti.debug.value(p["Q"])).reshape(m.nq, ne + 1)
            cost, status, ok = float("inf"), opti.stats().get("return_status", "failed"), False
        cpu = time.perf_counter() - t0

        q_hat = Qopt[:, -1]
        self._qbar = Qopt[:, 1] if ne > 1 else Qopt[:, -1]   # the state leaving the window
        info = EstimateInfo(ok, cost, cpu, int(opti.stats().get("iter_count", 0)),
                            float(np.abs(m.chain_residual(q_hat)).max()), status)
        info._rtf = cpu / cfg.ts
        self.last_info = info
        return q_hat, info
