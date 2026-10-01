#!/usr/bin/env python3
"""Drive the vehicle along the Fig. 2 path with the robust estimator in the loop.

    python examples/run_sim.py configs/agricultural_25.yaml
    python examples/run_sim.py configs/agricultural_25.yaml --baseline l2
    python examples/run_sim.py configs/agricultural_25.yaml --alpha 1.5 --c 0.1
    python examples/run_sim.py configs/agricultural_25.yaml --no-mhe   # true state in the loop

Writes ``results/<name>.json`` with the metrics and, unless ``--no-plot``, ``<name>.png``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path as _P

import numpy as np
import yaml

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from mhe_outlier_robust import EstimatorConfig, RedundantNTrailer, VehicleGeometry
from mhe_outlier_robust import path as paths
from mhe_outlier_robust import sim
from mhe_outlier_robust._nmppfc import nmppfc
from mhe_outlier_robust.sensors import SensorModel


def build(spec: dict, args):
    n = int(spec.get("n_trailers", 1))
    ts = float(spec.get("ts", 0.2))
    beta_max = float(np.deg2rad(spec.get("beta_max_deg", 70.0)))
    geo = VehicleGeometry(n_trailers=n, hitch=(0.342, 0.0)[:n], link=(1.08, 0.78)[:n],
                          body_length=(0.99,) + (1.0,) * n, body_width=(0.67,) + (0.8,) * n)
    model = RedundantNTrailer(geo, ts=ts)

    est_kw = dict(spec.get("estimator", {}))
    est_kw.update(ts=ts, beta_max=beta_max)
    if args.window:
        est_kw["n_window"] = args.window
    if args.alpha is not None:
        est_kw["alpha"] = args.alpha
    if args.c is not None:
        est_kw["c"] = args.c
    est_cfg = EstimatorConfig(**est_kw)
    if args.baseline == "l2":
        est_cfg = sim.l2_baseline_config(est_cfg)

    ctrl_kw = dict(spec.get("controller", {}))
    ctrl_kw.update(beta_max=beta_max)
    ctrl_cfg = nmppfc.ControllerConfig(**ctrl_kw)

    s = dict(spec.get("sensors", {}))
    if args.outlier_prob is not None:
        s["delta"] = args.outlier_prob
    if args.seed is not None:
        s["seed"] = args.seed
    sensors = SensorModel(n_trailers=n, outlier_rows=model.outlier_rows,
                          n_window=est_cfg.n_window, **s)
    return model, est_cfg, ctrl_cfg, sensors


def plot(res, path, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = res.states
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.9))
    pts = path.sample(1200)
    ax[0].plot(pts[:, 0], pts[:, 1], color="0.75", lw=1.5, label="reference (Fig. 2)")
    ax[0].plot(res.states[:, -4], res.states[:, -3], lw=2, label="tractor")
    ax[0].plot(res.estimates[:, -4], res.estimates[:, -3], lw=1, ls="--", label="estimate")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
    # the path fills the axes, so the legend goes above it rather than on top of a row
    ax[0].legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                 frameon=False)

    hit = res.corrupted.astype(bool)
    ax[1].plot(res.t, res.measurements[:, -2], ".", ms=3, color="0.7", label="measured x")
    ax[1].plot(res.t[hit], res.measurements[hit, -2], ".", ms=4, color="tab:red",
               label="corrupted")
    ax[1].plot(res.t, res.states[:, -4], lw=2, label="true x")
    ax[1].plot(res.t, res.estimates[:, -4], lw=1, ls="--", label="estimated x")
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("x [m]"); ax[1].legend(fontsize=8)

    ax[2].plot(res.t, res.est_error, label="estimation error [m]")
    ax[2].plot(res.t, res.deviation, label="deviation from path [m]")
    ax[2].plot(res.t, res.rtf, label="real-time factor")
    ax[2].axhline(1.0, color="0.8", lw=1)
    ax[2].set_xlabel("t [s]"); ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--alpha", type=float, help="shape parameter, in [1, 2)")
    p.add_argument("--c", type=float, help="scale parameter")
    p.add_argument("--window", type=int, help="N_e")
    p.add_argument("--outlier-prob", type=float, help="override the corruption probability")
    p.add_argument("--seed", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--baseline", choices=["l2"],
                   help="run OUR quadratic baseline instead (not the paper's competitors)")
    p.add_argument("--no-mhe", action="store_true", help="close the loop on the true state")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    spec = yaml.safe_load(_P(args.config).read_text())
    model, est_cfg, ctrl_cfg, sensors = build(spec, args)
    path = paths.agricultural()
    steps = args.steps or int(spec.get("steps", 200))

    name = _P(args.config).stem + ("_l2" if args.baseline else "") + ("_true" if args.no_mhe else "")
    print(f"{name}: N_e={est_cfg.n_window} alpha={est_cfg.alpha} c={est_cfg.c} "
          f"beta_max={np.rad2deg(est_cfg.beta_max):.0f} deg, {sensors.mode} "
          f"delta={sensors.delta}, {steps} steps at Ts={model.ts} s")
    res = sim.run(model, path, est_cfg=est_cfg, ctrl_cfg=ctrl_cfg, sensors=sensors,
                  steps=steps, speed=float(spec.get("speed", 1.0)),
                  use_mhe=not args.no_mhe, progress=max(steps // 5, 1))

    out = _P(args.out); out.mkdir(exist_ok=True)
    payload = {"config": args.config, "name": name, "metrics": res.metrics,
               "estimator": asdict(est_cfg),
               "sensors": {"mode": sensors.mode, "delta": sensors.delta, "seed": sensors.seed},
               "note": ("These are this repository's numbers on this scenario, not the paper's. "
                        "Scenario, disturbances and tuning differ.")}
    (out / f"{name}.json").write_text(json.dumps(payload, indent=2))
    for k, v in res.metrics.items():
        print(f"  {k:24s} {v}")
    if not args.no_plot:
        plot(res, path, out / f"{name}.png")
        print(f"  plot                     {out / (name + '.png')}")


if __name__ == "__main__":
    main()
