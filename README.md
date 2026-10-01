# mhe-outlier-robust-v1

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nahueldeniz/mhe-outlier-robust-v1/blob/main/notebooks/mhe_outlier_robust_demo.ipynb)

A moving horizon estimator that keeps working when the GNSS lies.

Python + CasADi implementation of

> N. N. Deniz, G. M. Sanchez, F. A. Auat Cheein and L. L. Giovanini, **"Robust Moving Horizon
> Estimation for Autonomous Agricultural Vehicles With GNSS Outliers Using a Robust Loss
> Function"**, *IEEE Robotics and Automation Letters*, 10(10):10815-10820, 2025.
> [doi:10.1109/LRA.2025.3606377](https://doi.org/10.1109/LRA.2025.3606377)

Under a canopy a receiver does not politely get noisier: it reports a position tens of metres
away and attaches no warning. A quadratic stage cost believes it — a 30 m outlier weighs as
much as a million 3 cm ones — so the estimate goes out to meet it, and on an agricultural
vehicle the estimate is what the controller steers on.

This estimator replaces that quadratic with the square of the derivative of the General
Adaptive Robust Loss (Eq. 3 of the paper). Two things follow. The singularities of the loss
family at `alpha = 0` and `alpha = 2` disappear, so the tuning lives in one well-behaved
interval `[1, 2)`. And at `alpha = 1` the stage cost is **bounded**: it saturates at `1/c^2`, so
an outlier a hundred metres out costs no more than one thirty metres out and the optimiser stops
chasing it. `c` sets where that changeover sits.

There is a second reason a moving horizon estimator belongs on this vehicle, and it has nothing
to do with outliers: a tractor and its trailer are one chain, and the joint angle and the
trailer's position are not free to disagree. The estimator carries that geometry as a hard
equality constraint, in the same optimisation. A filter has nowhere to put it.

## Install

```bash
git clone https://github.com/nahueldeniz/mhe-outlier-robust-v1
cd mhe-outlier-robust-v1
pip install -r requirements.txt

# the vehicle model, the path machinery and the controller come from:
git clone https://github.com/nahueldeniz/nmppfc ../nmppfc
export NMPPFC_PATH=$PWD/../nmppfc
```

That checkout has to be current: this package uses the redundant-coordinate helpers
(`q_from_x`, `x_from_q`, `as_minimal`, `chain_residual`) that let nmppfc's controllers take the
full state vector the estimator produces. An older copy is refused at import with a message
saying so, rather than failing obscurely later.

## Run

```bash
python examples/run_sim.py configs/agricultural_25.yaml           # a quarter of the samples corrupted
python examples/run_sim.py configs/agricultural_50.yaml           # half of them
python examples/run_sim.py configs/burst_outage.yaml              # a whole-window outage
python examples/run_sim.py configs/field_like.yaml                # the field run's 5 Hz sampling

python examples/run_sim.py configs/agricultural_25.yaml --alpha 1.5 --c 0.1   # sweep the tuning
python examples/run_sim.py configs/agricultural_25.yaml --baseline l2         # our quadratic baseline
python examples/run_sim.py configs/agricultural_25.yaml --no-mhe              # true state in the loop
```

Metrics land in `results/<name>.json`, a three-panel figure in `results/<name>.png`. The
notebook `notebooks/mhe_outlier_robust_demo.ipynb` runs the same thing in Colab.

`--no-mhe` is the diagnostic worth knowing about: it closes the loop on the true state, so a
controller fault and an estimator fault stop looking alike.

## Use it directly

```python
from mhe_outlier_robust import EstimatorConfig, RedundantNTrailer, RobustMHE

model = RedundantNTrailer(ts=0.1)                      # tractor + one passive trailer
est = RobustMHE(model, EstimatorConfig(n_window=10, alpha=1.0, c=1.0))
est.reset(q0_guess)

for r, u_prev in measurements:                         # r = (beta_i, theta_0, x_0, y_0)
    q_hat, info = est.update(r, u_prev=u_prev)         # q_hat is the paper's full state vector
    print(info.real_time_factor, info.chain_violation)
```

`alpha` and `c` are NLP parameters, so `est.update(r, u_prev, alpha=1.5, c=0.5)` changes the
tuning without rebuilding anything.

## What it does on the shipped scenarios

800 steps (40 m of driving), one trailer, `alpha = c = 1`, `N_e = 10`, `Ts = 0.05` s, 1 m/s,
seed 7:

| scenario | mean deviation from the path | mean estimation error | mean real-time factor |
|---|---|---|---|
| 25 % of samples corrupted | 0.025 m | 0.022 m | 0.17 |
| 50 % of samples corrupted | 0.037 m | 0.062 m | 0.17 |
| burst outage, the whole window | 0.025 m | 0.012 m | 0.22 |
| 25 %, **our** quadratic baseline | 0.247 m | 1.698 m | 0.18 |

The simulations run at `Ts = 0.05` s. The paper's 5 Hz is a property of the field hardware, not
of the problem, and `configs/field_like.yaml` is the one configuration that keeps it, to show
what that rate costs. [NOTES.md](NOTES.md) section 7 has the comparison across sampling rates.

**These are not the paper's numbers, and they are better than them, because this problem is
easier.** The paper reports 0.36 m mean deviation at 25 % outliers and 0.44 m at 50 % on its own
scenario, and a 224.92 m field run with a mean estimation error of 0.127 m. The quadratic
baseline above is *ours*, not any of the five methods the paper compares against.
[NOTES.md](NOTES.md) sets out every deviation, simplification and caveat — read it before
quoting anything from here.

## Tests

```bash
python -m pytest tests -q
```

With ROS sourced, its `launch_testing` package registers a pytest plugin whose hook signature
modern pytest no longer accepts, and collection dies before any test runs. Nothing to do with
this repository; these tests need no entry-point plugins at all, so turn them off:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
```

They pin the things worth doubting: that the stage cost is bounded at `alpha = 1` and quadratic
as `alpha -> 2`, that its influence really does vanish; that `alpha` and `c` do not trigger a
rebuild; that the estimate satisfies the chain equalities while free propagation does not; that
the estimator and the controller enforce the *same* joint-angle limit; that the reconstructed
path matches the extents of Fig. 2; and that the sensor draws are reproducible.

## Licence

MIT. If you use it, please cite the paper — see [CITATION.cff](CITATION.cff).
