# Implementation notes

This repository is a Python + CasADi implementation of

> N. N. Deniz, G. M. Sanchez, F. A. Auat Cheein and L. L. Giovanini, "Robust Moving Horizon
> Estimation for Autonomous Agricultural Vehicles With GNSS Outliers Using a Robust Loss
> Function", *IEEE Robotics and Automation Letters*, vol. 10, no. 10, pp. 10815-10820, 2025.
> [doi:10.1109/LRA.2025.3606377](https://doi.org/10.1109/LRA.2025.3606377)

The estimator is re-implemented in Python and CasADi so that it runs on free tools and anyone
can read it, change it and extend it. The notes below record what the implementation chooses,
where it deviates from the paper, and what its numbers do and do not mean.

**The numbers produced here are not the paper's numbers.** The scenario, the disturbances, the
speeds and the tuning all differ, and this repository's scenarios are easier than the paper's.
Section 8 gives what they actually are, next to what the paper reports, so the two are never
confused.

## 1. What is here and what is imported

The vehicle model, the reference path machinery and the controller are **not** re-implemented.
They come from

> https://github.com/nahueldeniz/nmppfc

the implementation of the path-following framework of Deniz and Auat Cheein, *IEEE T-ASE*
22:7018-7032, 2025. `requirements.txt` says how to obtain it; `mhe_outlier_robust/_nmppfc.py`
locates it. This repository adds four things: the loss, the vehicle in the paper's own
coordinates, the estimator, and the loop that joins them.

```
mhe_outlier_robust/loss.py       Eqs. (2) and (3); alpha and c are arguments, never baked in
mhe_outlier_robust/model.py      the redundant vector q, qdot = G(q)u, h(q), the chain equalities
mhe_outlier_robust/estimator.py  the estimator of Eqs. (8)-(11)
mhe_outlier_robust/sensors.py    Gaussian noise plus scattered or burst GNSS outliers
mhe_outlier_robust/path.py       the agricultural path of Fig. 2, rebuilt from its geometry
mhe_outlier_robust/sim.py        the closed loop and the paper's metrics
mhe_outlier_robust/config.py     the numbers, in one place -- including the shared beta_max
```

## 2. The loss

`psi(x, alpha, c) = (x^2/c^4) (x^2/(c^2|alpha-2|) + 1)^(alpha-2)` is Eq. (3), the square of the
derivative of the GARL, and it is the stage cost on the measurement residuals.

* **`alpha` and `c` are symbolic parameters of the NLP**, not constants folded into it. Sweeping
  them costs a `set_value`; there is a test asserting the problem is not rebuilt when they
  change. Nothing here hard-codes a shape parameter.
* `alpha` is validated against `[1, 2)`. Outside that interval the paper's argument does not
  apply, so the code refuses rather than silently extrapolating.
* The tail is `psi ~ |x|^(2 alpha - 2)`, so **alpha = 1 gives a bounded cost** -- it saturates
  at `1/c^2`, and its derivative, which is what drags the estimate, goes to zero. `alpha -> 2`
  returns the quadratic the paper replaces. `loss.tail_exponent` and `loss.saturation` state
  this, and both are tested, including that the influence at a 100 m residual is below 1e-5 at
  alpha = 1 and above 1 at alpha = 1.9.
* A small regulariser, `EPS_ALPHA = 1e-6`, is added to `|alpha - 2|` so the exponent base stays
  defined when alpha is numerically close to 2. It is not in the paper; it costs nothing at the
  alpha values the paper recommends and it stops the expression from blowing up at the edge of
  the interval.

## 3. The state vector, and why it is the redundant one

The paper writes the state as `q = (beta_i, theta_0, theta_i, x_0, y_0, x_i, y_i)`, of dimension
4N+3, and the output as `r = h(q) = (beta_1..beta_N, theta_0, x_0, y_0)` (Eq. 1): the joint
angles and the tractor's attitude and coordinates are measured, the trailers' poses are not.

Those coordinates are redundant, and that is the point. The chain equalities

```
beta_i     = theta_{i-1} - theta_i
(x_i, y_i) = (x_{i-1}, y_{i-1}) - L_h,i (cos theta_{i-1}, sin theta_{i-1})
                                - L_i   (cos theta_i,     sin theta_i)
```

are imposed as **hard equality constraints at every instant of the window**, in the same
optimisation that fights the outliers. A filter has nowhere to put them.

The imported nmppfc package works in the minimal state `[x_0, y_0, theta_0, beta_1..beta_N]`
and recovers the trailer poses algebraically. That is a clean choice there, but it is *not* the
choice this paper makes, and adopting it would have deleted the constraint the method relies
on. So this repository carries the paper's `q`, propagates it with the redundant `qdot = G(q)u`
and constrains it. `tests/test_estimator.py::test_free_propagation_leaves_the_chain_manifold`
shows why the constraints cannot be an afterthought: free propagation in `q` drifts off the
manifold, by 8e-7 after 25 steps at `Ts = 0.2` s and growing.

Every controller in nmppfc now accepts either vector: `NTrailer.q_from_x`, `x_from_q`,
`as_minimal` and `chain_residual` were added there, and the controllers coerce their state
argument once. `x_from_q` takes the tractor pose and the joint angles -- what Eq. (1) actually
measures -- and regenerates the trailer poses from the chain, so an inconsistent `q` cannot
reach the controller as a silent fault; `chain_residual` reports how inconsistent it was.

Note the two orderings, both deliberate and both tested: **a pose is `(x, y, theta)`, while the
controller's error vector and its weights are `(theta, x, y)`.**

## 4. The estimator

Over the last `N_e` sampling instants, Eqs. (8)-(11):

```
min   Gamma_chi |q_0 - qbar|^2 + sum_j w_j' W w_j + sum_j sum_i psi(v_ij, alpha, c)
s.t.  q_{j+1}    = Phi(q_j, u_j) + w_j       RK4, multiple shooting
      r_j        = h(q_j) + v_j              Eq. (1)
      chain(q_j) = 0
      |beta| <= beta_max,   w_j in W,   v_j in V
```

* **The residual set V is open exactly where the outliers land.** Residual components free of
  outliers are boxed to `|v| <= 0.1`; the tractor's coordinates, which the outliers hit, are
  left **unbounded** and the loss deals with them. This is the paper's specification, and it is
  what makes the robust stage cost do any work at all -- box them and the problem simply goes
  infeasible under an outlier.
* `W`: `|w| <= 0.01`, weight `1e6 * I`, as the paper states. See section 7 -- this box is tight.
* **The joint-angle limit is shared with the controller**, from a single value in `Config`. An
  estimator allowed a wider jackknife than the controller would hand the controller states the
  controller must refuse. Two tests cover it: that the two configurations carry the same
  `beta_max`, and that the estimate is clipped when the truth exceeds it. The default is the
  paper's `pi/2 - 20 deg = 70 deg`; nmppfc's own default of 120 deg belongs to the
  obstacle-avoidance controller, which needs the room for evading manoeuvres.
* **Deviation from the paper: angular residuals are wrapped** with `atan2(sin d, cos d)`, so a
  heading sitting on the branch cut costs what it should instead of `2 pi` for being there.
  `wrap_angles = False` restores the literal reading of Eq. (1). There is a test for it.
* **Simplification: the arrival cost is fixed.** `Gamma_chi = arrival_gain * I`, with `qbar` set
  to the previous window's estimate of the state that has just left the window. The paper's
  reference for the arrival-cost update is not implemented. With `N_e = 10` and well-conditioned
  sensors the difference does not show; for short windows or a poor initial guess it would.
* **The window grows.** One NLP is built and cached per window length, so there is an estimate
  from the second sample instead of a warm-up period spent on unfiltered measurements.
  `grow_window = False` waits for a full window.
* The initial guess for each solve is the arrival state rolled forward under the applied inputs,
  so every column of the guess already satisfies the chain equalities.
* IPOPT with MUMPS. The estimator reports `chain_violation` per step, and the tests require it
  below 1e-6.

## 5. The path

`path.py` rebuilds Fig. 2 from its geometry: five 20 m rows 1.5 m apart, joined at alternating
ends by bulb turns of radius 2 m tangent to both rows. Tangency fixes where a row has to end,
and at `r = 2` m with a 1.5 m spacing the neighbouring circles at the same end overlap and
cross -- which is what the published figure shows, and is the check that the reconstruction is
right rather than a coincidence. The reconstruction spans x in [-3.85, 23.85] and y in
[-7.25, 1.25] against roughly [-3.9, 23.7] and [-7.2, 1.25] in the figure, and measures 144.1 m.

**It is a reconstruction from the figure, not the paper's own path file.** Taking the short arc
instead of the long one collapses the turn onto the rows; there is a test against that.

## 6. The reference, and what is not reproduced

The guided segment is projected onto the path and the reference pose is taken a lookahead ahead
of that projection, held over the whole control window. The reference generator of the T-ASE
framework -- a target advancing at a commanded speed, slowed while occluded, with a reference
selector and a local planner -- is **not** used: there are no obstacles here and none of that
machinery is what this paper is about. Holding a single reference pose also avoids the trap of a
moving reference trajectory the closed loop can never catch up with.

## 7. Sampling rate

The paper states `Ts = 0.2` s for the field experiment and does not state it for the
simulations. **The simulations here run at `Ts = 0.05` s.** A 5 Hz rate is a property of the
field hardware, not of the problem, and a simulation has no reason to inherit it: at 1 m/s the
2 m bulb turns of Fig. 2 rotate the vehicle about 3 degrees per sample at 0.05 s and about 12
degrees at 0.2 s, and the paper's tight `|w| <= 0.01` disturbance box cannot absorb the
resulting integration mismatch at the coarse end.

What that costs, on `configs/agricultural_25.yaml` with `N_e = 10` and 40 m of driving:

| `Ts` | mean est. error | worst est. error | mean deviation | mean RTF | solve failures |
|---|---|---|---|---|---|
| 0.2 s | 0.047 m | 2.26 m | 0.168 m | 0.16 | 3 |
| 0.1 s | 0.022 m | 0.069 m | 0.048 m | 0.11 | 0 |
| **0.05 s** | **0.022 m** | **0.093 m** | **0.025 m** | **0.17** | **0** |
| 0.025 s | 0.022 m | 0.107 m | 0.054 m | 0.35 | 0 |

Estimation accuracy is flat below 0.1 s -- the loss is doing its job at every rate -- but path
tracking is twice as good at 0.05 s as at 0.1 s, and the 0.2 s spikes disappear. Going further
to 0.025 s buys nothing and tracks worse: the control horizon `N_c = 8` then covers only 0.2 s
of prediction, so the controller turns myopic, and the real-time factor doubles because RTF is
the solve time divided by a period half as long. `Ts = 0.05` s is where those two meet.

Two things worth knowing about the spikes at 0.2 s, because they were nearly misdiagnosed:
they are **not** caused by runs of consecutive outliers -- the worst one falls on a clean
sample -- and closing the loop on the true state (`--no-mhe`) removes the solve failures
entirely, so they come from the estimator, not the controller or the reference.

`configs/field_like.yaml` deliberately keeps the field run's 0.2 s and `N_e = 15`, at a gentler
speed. It is the one coarse configuration here, and it exists to show what the hardware's rate
costs: it still shows a worst estimation error of about a metre and a few controller solve
failures where the 0.05 s configurations show none.

## 8. What the numbers are

800 steps (40 m) on `configs/agricultural_*.yaml`, single passive trailer, `alpha = c = 1`,
`N_e = 10`, `Ts = 0.05` s, 1 m/s, seed 7:

| scenario | mean deviation | mean est. error | worst est. error | mean RTF | worst RTF |
|---|---|---|---|---|---|
| 25 % of samples corrupted | 0.025 m | 0.022 m | 0.093 m | 0.17 | 0.78 |
| 50 % of samples corrupted | 0.037 m | 0.062 m | 0.820 m | 0.17 | 0.75 |
| burst outage, whole window, `N_e = 20` | 0.025 m | 0.012 m | 0.430 m | 0.22 | 1.45 |
| 25 %, field sampling rate (`Ts = 0.2` s, `N_e = 15`, 0.5 m/s, 300 steps) | 0.046 m | 0.035 m | 1.010 m | 0.16 | 0.95 |
| 25 %, **our** quadratic baseline | 0.247 m | 1.698 m | 8.737 m | 0.18 | 0.82 |

The mean real-time factor sits around a fifth of the sampling period everywhere, but note the
worst case: the burst configuration, with its longer `N_e = 20` window, has one solve that
overruns a single 0.05 s period (RTF 1.45). It is one spike out of 800 steps and the loop
recovers, but a deployment at 20 Hz with a twenty-step window would want a solve-time budget
rather than an average.

The paper reports a mean deviation of 0.36 m at 25 % and 0.44 m at 50 % on its own scenario,
and a field run of 224.92 m with a mean estimation error of 0.127 m. **This repository's figures
are better, because its problem is easier**: one trailer, a reconstructed path, a projected
reference, its own speeds and sampling rate, and no model disturbance beyond the integration
mismatch. Nothing here contradicts the paper; nothing here reproduces it either.

## 9. The baseline is ours

`sim.l2_baseline_config` is the same estimator with the quadratic stage cost this paper
replaces and the residual box opened so the problem stays feasible under an outlier. **It is
not any of the five methods the paper compares against** -- GMHE, EKF+LSTM, VBKLDPF, KF-AM and
the standard nonlinear MHE were implemented and tuned by the authors from their own references,
and are not in this repository. Nothing measured here should be read as a statement about any of
them, and the paper's Fig. 3(c) and 3(d) are the place to look for that comparison.

Two things in the paper's own comparison are worth repeating because they are easy to overstate:
at 1 to 10 per cent outliers three of the six estimators sit between 0.358 and 0.375 m and are
indistinguishable -- the claim is the flat column, not a win at low outlier rates -- and the
Kalman-type filters run twenty to thirty times faster than the proposed estimator. They are
cheaper. They simply stop working when the measurements lie, and, on a chained vehicle, they
have nowhere to put the geometry either.

## 10. Reproducibility

Random draws happen in a fixed order -- noise, then the outlier decision, then the outlier
itself, every step, whether or not they are used -- so changing something downstream does not
reshuffle the sequence. Every configuration carries its seed, and `results/*.json` records the
metrics together with the estimator and sensor settings that produced them.

## 11. Still open

1. The arrival cost is fixed rather than updated as the paper's reference does (section 4).
2. Only the scattered and burst outlier scenarios are shipped as configurations. The notebook
   runs a small `alpha`/`c` sweep, and `--alpha`/`--c` make a larger one a loop over
   `examples/run_sim.py`; the paper's full sweep (its Fig. 3(a) and 3(b)) is not reproduced.
   The small sweep does land where the paper does -- see section 12 -- but note that the
   paper's own sweep is five runs per cell, so its columns are readable and its neighbouring
   cells are not. The same caution applies to anything measured here.
3. Multi-trailer chains work in the model and the estimator but only `n_trailers = 1` is
   exercised by the shipped configurations, which is what the paper's simulations use.
4. No field data. The field experiment of Section IV-B cannot be reproduced from the paper alone.

## 12. One thing this repository does corroborate

The paper recommends `alpha` near 1 and `c` near 1, and reads its sweep by columns: `c = 1`
gives the lowest error, and `c <= 0.1` is not real time at all. A small sweep here -- 200 steps
at 25 % outliers, `N_e = 10`, `Ts = 0.05` s on the reconstructed path -- lands in the same
place. Mean estimation error, with the real-time factor in brackets:

| | `c = 0.1` | `c = 1` | `c = 10` |
|---|---|---|---|
| `alpha = 1.0` | 0.254 m (0.41) | **0.026 m (0.18)** | 0.070 m (0.12) |
| `alpha = 1.3` | 0.209 m (0.33) | 0.046 m (0.22) | 0.073 m (0.12) |
| `alpha = 1.6` | 0.285 m (0.21) | 0.250 m (0.17) | 0.078 m (0.12) |
| `alpha = 1.9` | 1.515 m (0.24) | 1.445 m (0.21) | 0.098 m (0.12) |

Three of the paper's readings survive: the best cell is `alpha = 1`, `c = 1`; error grows with
`alpha` down every column; and the small-`c` column is both the least accurate and the most
expensive, because a narrow quadratic region makes almost every residual an outlier and the
solver works harder for a worse answer. The large-`c` column is cheap and mediocre, which is
also what the paper's own tables show.

This is a different scenario, a different vehicle configuration and a different sampling rate
from the paper's, so it corroborates the *shape* of the recommendation and nothing more -- and
it is one run per cell, thinner than the paper's five. Do not read the cells against each other.
