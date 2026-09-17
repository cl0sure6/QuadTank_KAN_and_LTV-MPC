# Copyright 2026 Adilkhan Salkimbayev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Stage 10 -- the state constraint made active, and a widened robustness campaign.

Part A. The scenario campaign of stage 5 keeps the plant well inside its level
limits, which is precisely the regime in which an explicit law loses nothing by
having no constraint mechanism. Here the commanded set-point is placed at the top
of the reachable set (h1 = h2 = 19.5 cm, still only ~4 V of steady input), so the
transient of a large upward step runs into the 20 cm overflow limit. The teacher
carries 0 <= x <= 20 as an inequality inside its QP and rides the limit; the
distilled law inherits only the input box through clipping and has no mechanism
for the state constraint at all. The gap is measured rather than asserted.

Part B. The Monte-Carlo campaign of stage 4 perturbs valve ratios and pump gains
only. This stage adds three axes a deployed controller would actually meet --
outlet cross-sections, actuator delay and measurement noise -- and reports each
stable fraction with a Wilson score interval, since it is a binomial proportion
over a finite sample.

The LTV-MPC is included in Part A but not in Part B: the robustness campaign is
200 trials x 2000 steps per configuration, which is cheap for a polynomial and
prohibitive for an online QP.

Outputs results/constraint_robustness.json.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
from collections import deque

import numpy as np

import controllers as C
import policyform as PF
import qtlib as Q

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

DT = 0.1
T_TRACK = 200.0

# Two set-points, chosen to separate two failure mechanisms that the first
# version of this stage confounded.
#
#   in_box      r = 15 cm is the top of the training reference box (7-15 cm), so
#               the law is interpolating. Any overflow here is a constraint
#               -handling failure and nothing else.
#   out_of_box  r = 19.5 cm is the top of the *reachable* set but well outside
#               the training box, so the law is also extrapolating. The gap
#               between the two cases is what extrapolation adds.
#
# Both start from a near-empty plant, at the corner of the training initial
# -condition box, so the fill transient is the most demanding one in
# distribution.
SCENARIOS = {
    "in_box":     dict(ref_cm=15.0, x0=[1.0, 1.0, 0.5, 0.5]),
    "out_of_box": dict(ref_cm=19.5, x0=[1.0, 1.0, 0.5, 0.5]),
}
TRAIN_REF_BOX = (7.0, 15.0)
N_MC = 200
SPREAD = 0.20
T_MC = 200.0


# --------------------------------------------------------------- helpers
@contextlib.contextmanager
def outlet_areas(a):
    """Temporarily perturb the outlet cross-sections.

    f_cont reads the module-level A_OUT at call time, so swapping it here is
    enough; it is always restored, including on exception.
    """
    old = Q.A_OUT
    Q.A_OUT = np.asarray(a, float)
    try:
        yield
    finally:
        Q.A_OUT = old


def delayed(policy, n):
    """Apply the command computed n samples ago (actuator/transport delay)."""
    if n <= 0:
        return policy
    buf = deque(maxlen=n + 1)

    def pi(x, ref):
        u = np.asarray(policy(x, ref), float).ravel()
        if not buf:                       # warm start, so t=0 is not a step
            for _ in range(n + 1):
                buf.append(u)
        buf.append(u)
        return buf[0]
    return pi


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100.0 * max(0.0, c - h), 2), round(100.0 * min(1.0, c + h), 2)]


# ------------------------------------------------- Part A: active constraint
def constraint_scenario(regime, reg, ref_cm, x0, pols=None):
    x_eq, u_eq = Q.equilibrium(ref_cm, ref_cm, reg)
    x0 = np.asarray(x0, float)
    pols = pols if pols is not None else C.build_all(reg, DT, regime)
    res = {"commanded_h12_cm": ref_cm,
           "inside_training_ref_box": bool(TRAIN_REF_BOX[0] <= ref_cm <= TRAIN_REF_BOX[1]),
           "reference_cm": [round(float(v), 3) for v in x_eq],
           "u_eq_V": [round(float(v), 3) for v in u_eq],
           "x0_cm": x0.tolist(), "limit_cm": Q.H_MAX, "controllers": {}}
    for name, pol in pols.items():
        sim = Q.run_closed_loop(pol, reg, x0, lambda t: x_eq,
                                dt=DT, T=T_TRACK, seed=5)
        x = sim["x"]
        over = np.maximum(x - Q.H_MAX, 0.0)
        res["controllers"][name] = {
            "peak_level_cm": round(float(x.max()), 3),
            "peak_tank": int(np.unravel_index(np.argmax(x), x.shape)[1] + 1),
            "overflow": bool(x.max() > Q.H_MAX),
            "violation_area_cm_s": round(float(over.sum() * DT), 3),
            "time_over_limit_s": round(float((over.max(1) > 0).sum() * DT), 2),
            "max_overshoot_above_ref_cm": round(float(x[:, :2].max() - ref_cm), 3),
            "rmse_h12_cm": round(float(np.sqrt(np.mean(
                (x[:, :2] - x_eq[:2]) ** 2))), 3),
        }
        r = res["controllers"][name]
        print("    %-22s peak %6.2f (tank %d)  overflow=%-5s  area %6.2f cm.s"
              % (name, r["peak_level_cm"], r["peak_tank"],
                 r["overflow"], r["violation_area_cm_s"]), flush=True)
    return res


# --------------------------------------------- Part B: widened robustness
CONFIGS = {
    "gamma_k":             dict(areas=False, delay=0, noise=0.0),
    "gamma_k_areas":       dict(areas=True,  delay=0, noise=0.0),
    "gamma_k_delay":       dict(areas=False, delay=5, noise=0.0),
    "gamma_k_noise":       dict(areas=False, delay=0, noise=2.37),
    "all_axes":            dict(areas=True,  delay=5, noise=2.37),
}


def robustness(policy, reg, cfg, seed=1):
    rng = np.random.default_rng(seed)
    x_eq, _ = Q.equilibrium(10.0, 10.0, reg)
    n_stable = n_over = n_dry = 0
    errs = []
    for _ in range(N_MC):
        r = reg.copy()
        r.k = reg.k * rng.uniform(1 - SPREAD, 1 + SPREAD, 2)
        r.gamma = np.clip(reg.gamma * rng.uniform(1 - SPREAD, 1 + SPREAD, 2),
                          0.05, 0.95)
        a = (Q.A_OUT * rng.uniform(1 - SPREAD, 1 + SPREAD, 4) if cfg["areas"]
             else Q.A_OUT)
        d = int(rng.integers(0, cfg["delay"] + 1)) if cfg["delay"] else 0
        pol = delayed(policy, d)
        with outlet_areas(a):
            sim = Q.run_closed_loop(
                pol, r, reg.x0, lambda t: x_eq, DT, T_MC,
                seed=int(rng.integers(1e6)),
                use_ekf=bool(cfg["noise"]),
                meas_noise_var=cfg["noise"], proc_noise_var=0.0)
        x = sim["x"]
        xf = x[-1]
        settled = np.max(np.abs(x[-100:, :2] - xf[:2])) < 0.05
        stable = bool(np.all(np.isfinite(xf)) and x.max() <= Q.H_MAX and settled)
        n_stable += stable
        n_over += bool(x.max() > Q.H_MAX)
        n_dry += bool(x.min() <= 1e-6)
        errs.append(float(np.max(np.abs(xf[:2] - x_eq[:2]))))
    errs = np.array(errs)
    return {
        "n": N_MC,
        "stable_pct": round(100.0 * n_stable / N_MC, 2),
        "stable_wilson95_pct": wilson(n_stable, N_MC),
        "overflow_pct": round(100.0 * n_over / N_MC, 2),
        "overflow_wilson95_pct": wilson(n_over, N_MC),
        "dryout_pct": round(100.0 * n_dry / N_MC, 2),
        "p95_abs_err_cm": round(float(np.percentile(errs, 95)), 3),
        "median_abs_err_cm": round(float(np.median(errs)), 3),
    }


def main():
    out = {"config": {"scenarios": SCENARIOS,
                      "training_ref_box_cm": list(TRAIN_REF_BOX),
                      "n_monte_carlo": N_MC, "spread_pct": 100 * SPREAD,
                      "delay_samples_max": CONFIGS["gamma_k_delay"]["delay"],
                      "delay_s_max": CONFIGS["gamma_k_delay"]["delay"] * DT,
                      "noise_var": CONFIGS["gamma_k_noise"]["noise"]}}

    for regime, reg in (("MP", Q.MP), ("NMP", Q.NMP)):
        block = {"constraint_scenarios": {}}
        pols = C.build_all(reg, DT, regime)   # built once, reused by both cases
        for tag, sc in SCENARIOS.items():
            inbox = TRAIN_REF_BOX[0] <= sc["ref_cm"] <= TRAIN_REF_BOX[1]
            print(f"=== {regime}: constraint-active tracking at {sc['ref_cm']} cm "
                  f"({'in' if inbox else 'outside'} the training box) ===", flush=True)
            block["constraint_scenarios"][tag] = constraint_scenario(
                regime, reg, sc["ref_cm"], sc["x0"], pols=pols)

        print(f"=== {regime}: widened robustness ===", flush=True)
        law = C.SymbolicLaw(regime)
        block["robustness"] = {}
        for tag, cfg in CONFIGS.items():
            r = robustness(law, reg, cfg)
            block["robustness"][tag] = {"axes": cfg, **r}
            print("    %-16s stable %6.2f%%  95%% CI %s  overflow %5.2f%%"
                  % (tag, r["stable_pct"], r["stable_wilson95_pct"],
                     r["overflow_pct"]), flush=True)
        out[regime] = block

    p = os.path.join(RES, "constraint_robustness.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", p, flush=True)


if __name__ == "__main__":
    main()
