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
Stage 4c -- own-channel and cross-channel error derivatives, law versus teacher.

The shape constraint of the read-out is imposed on the own-channel derivative
du_j/de_j only.  This stage asks what the *cross*-channel derivatives du_j/de_i
(i != j) do, and -- the point of the exercise -- whether the teacher's own cross
derivatives are sign-definite.  If the LTV-MPC changes the sign of its cross
response across the operating box, then constraining that sign in the distilled
law would exclude behaviour the law is supposed to imitate, and the asymmetry
between the diagonal and the off-diagonal treatment is a modelling decision
rather than an oversight.

Both derivatives are taken in the same coordinates: the state x is held fixed and
the reference component r_i is perturbed, so e_i = r_i - x_i moves while the level
features do not.  References are exact equilibria, so every sample is a physically
realisable operating point.

Outputs results/cross_channel.json.
"""

from __future__ import annotations

import json
import os

import numpy as np

import controllers as C
import policyform as PF
import qtlib as Q
import symbolic as SY

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

N_SAMPLES = 300
H_LAW = 1e-6      # analytic-quality step for the smooth polynomial law
H_MPC = 0.10      # cm; the MPC policy is piecewise affine, so use a finite facet
SEED = 20260917
DT = 0.1


# ----------------------------------------------------------------- the law
def _learned(phi, law, regime, x):
    """g(x) * w(phi) * U_max * m(clip(phi))^T c, in volts.

    The level fade is included because this stage reports the derivatives of the
    law as deployed, and the sampling here does reach levels where g < 1. It
    scales the own- and cross-channel derivatives by the same factor, so it
    cannot change which of them is sign-definite -- only the magnitudes.
    """
    phi = np.asarray(phi, float).reshape(1, 8)
    fc = PF.clip_features(phi, regime)
    y = np.array([SY.design(fc, law.sup[j]) @ law.coef[j] for j in range(2)]).ravel()
    return PF.level_fade(x)[0] * PF.gate(phi)[0] * y * Q.NORM_U


def law_du_de(x, ref, law, regime):
    """Full du/de of the deployed law, in V/cm, as a 2x2 block on (e1, e2)."""
    K = PF.lqr_gain(regime)
    J = np.zeros((2, 2))
    for i in range(2):
        out = []
        for sgn in (+1, -1):
            r = np.array(ref, float)
            r[i] += sgn * H_LAW * Q.NORM_X
            out.append(_learned(Q.features(x, r), law, regime, x))
        # feed-forward u_eq(r) is excluded: it is a function of the commanded
        # reference, not of the tracking error, and the constraint of the
        # read-out is stated on the error channel only.
        J[:, i] = K[:, i] + (out[0] - out[1]) / (2 * H_LAW * Q.NORM_X)
    return J


# -------------------------------------------------------------- the teacher
SAT_TOL = 1e-3     # V; a command this close to a bound is treated as saturated


def _interior(u):
    """True when both pump commands are strictly inside the actuator box.

    A saturated MPC does not respond to a reference perturbation at all, so such
    samples carry no derivative information and must be excluded rather than
    recorded as an exact zero.
    """
    u = np.asarray(u, float).ravel()
    return bool(np.all(u > Q.U_MIN + SAT_TOL) and np.all(u < Q.U_MAX - SAT_TOL))


def mpc_du_de(x, ref, mpc, reg, u_prev):
    """Central-difference du/de of the LTV-MPC, in V/cm, on (e1, e2).

    Returns None if the QP fails or if the command saturates at the nominal
    point or at either perturbation.
    """
    u0, ok = mpc.solve(x, ref, u_prev, DT, reg)
    if not ok or not _interior(u0):
        return None
    J = np.zeros((2, 2))
    for i in range(2):
        out = []
        for sgn in (+1, -1):
            r = np.array(ref, float)
            r[i] += sgn * H_MPC
            u, ok = mpc.solve(x, r, u_prev, DT, reg)
            if not ok or not _interior(u):
                return None
            out.append(u)
        J[:, i] = (out[0] - out[1]) / (2 * H_MPC)
    return J


def stats(v):
    v = np.asarray(v, float)
    return {"min": round(float(v.min()), 4),
            "median": round(float(np.median(v)), 4),
            "max": round(float(v.max()), 4),
            "frac_nonneg": round(float((v >= 0).mean()), 4),
            "n": int(v.size)}


def gate_region():
    """How much of the scenario campaign sits in the gate's non-smooth region.

    w(e) = min(||e||^2/s^2, 1) is continuous everywhere but its gradient jumps
    across ||e|| = s X_n. The closed loop is therefore locally Lipschitz and its
    solutions unique; the Jacobian is set-valued only on that surface. What is
    worth measuring is how often the campaign is anywhere near it.
    """
    import glob
    s = PF.GATE_SCALE
    files = sorted(glob.glob(os.path.join(RES, "traj", "S*_*.npz")))
    per, tot = {}, {"n": 0, "active": 0, "cross": 0}
    for f in files:
        d = np.load(f)
        if "ref" not in d or "x" not in d:
            continue
        e = (d["ref"][:, :4] - d["x"][:, :4]) / Q.NORM_X
        r = np.sqrt((e ** 2).sum(1)) / s        # <1 quadratic branch, >1 saturated
        active = r < 1.0
        tag = os.path.basename(f).split("_")[0]
        a = per.setdefault(tag, {"n": 0, "active": 0, "cross": 0})
        for acc in (a, tot):
            acc["n"] += int(len(r))
            acc["active"] += int(active.sum())
            acc["cross"] += int(np.sum(np.diff(active.astype(int)) != 0))
    if not tot["n"]:
        return {"note": "no trajectories on disk (results/traj is gitignored)"}
    out = {"switch_at_norm_e_cm": round(float(s * Q.NORM_X), 3),
           "total_samples": tot["n"],
           "pct_in_quadratic_branch": round(100.0 * tot["active"] / tot["n"], 2),
           "switching_surface_crossings": tot["cross"],
           "per_scenario": {k: {"samples": v["n"],
                                "pct_in_quadratic_branch": round(100.0 * v["active"] / v["n"], 2),
                                "crossings": v["cross"]} for k, v in sorted(per.items())}}
    print("=== gate region over the scenario campaign ===", flush=True)
    print("    switch at ||e|| = %.1f cm; %d samples, %.1f%% in the smooth branch, "
          "%d crossings" % (out["switch_at_norm_e_cm"], out["total_samples"],
                            out["pct_in_quadratic_branch"],
                            out["switching_surface_crossings"]), flush=True)
    for k, v in out["per_scenario"].items():
        print("      %-4s %7d samples  %6.2f%% smooth  %3d crossings"
              % (k, v["samples"], v["pct_in_quadratic_branch"], v["crossings"]), flush=True)
    return out


def main():
    rng = np.random.default_rng(SEED)
    out = {"config": {"n_samples": N_SAMPLES, "h_mpc_cm": H_MPC, "seed": SEED,
                      "coordinates": "x held fixed, r_i perturbed"},
           "gate_region": gate_region()}

    for regime, reg in (("MP", Q.MP), ("NMP", Q.NMP)):
        law = C.SymbolicLaw(regime)
        mpc = Q.LTVMPC()
        Jl, Jm = [], []
        n_fail = n_tried = 0
        print(f"=== {regime} ===", flush=True)
        while len(Jl) < N_SAMPLES and n_tried < 40 * N_SAMPLES:
            n_tried += 1
            h1r, h2r = rng.uniform(7.0, 15.0, 2)
            try:
                ref, u_eq = Q.equilibrium(h1r, h2r, reg)
            except ValueError:
                continue
            # Tracking errors of the size the controller actually sees in closed
            # loop. Sampling x independently of r puts most draws far outside the
            # actuator box, where every derivative is trivially zero.
            x = ref + np.concatenate([rng.uniform(-4.0, 4.0, 2),
                                      rng.uniform(-2.0, 2.0, 2)])
            x = np.clip(x, 0.5, 19.5)
            jm = mpc_du_de(x, ref, mpc, reg, u_eq)
            if jm is None:
                n_fail += 1
                continue
            if not _interior(law(x, ref)):
                n_fail += 1
                continue
            Jl.append(law_du_de(x, ref, law, regime))
            Jm.append(jm)
            if len(Jl) % 100 == 0:
                print(f"  {len(Jl)}/{N_SAMPLES}", flush=True)
        Jl, Jm = np.array(Jl), np.array(Jm)

        block = {}
        for name, J in (("deployed_law", Jl), ("ltv_mpc_teacher", Jm)):
            block[name] = {
                "du1_de1": stats(J[:, 0, 0]), "du1_de2": stats(J[:, 0, 1]),
                "du2_de1": stats(J[:, 1, 0]), "du2_de2": stats(J[:, 1, 1]),
            }
        # the headline question: does the teacher's cross response change sign?
        block["teacher_cross_changes_sign"] = bool(
            0.0 < block["ltv_mpc_teacher"]["du1_de2"]["frac_nonneg"] < 1.0
            or 0.0 < block["ltv_mpc_teacher"]["du2_de1"]["frac_nonneg"] < 1.0)
        block["samples_rejected_saturated_or_failed"] = n_fail
        block["samples_drawn"] = n_tried
        out[regime] = block

        for who in ("deployed_law", "ltv_mpc_teacher"):
            print(f"  {who}")
            for k in ("du1_de1", "du1_de2", "du2_de1", "du2_de2"):
                s = block[who][k]
                print("    %-8s min %8.3f  med %8.3f  max %8.3f  frac>=0 %.3f"
                      % (k, s["min"], s["median"], s["max"], s["frac_nonneg"]),
                      flush=True)
        print(f"  teacher cross response changes sign: "
              f"{block['teacher_cross_changes_sign']}", flush=True)

    p = os.path.join(RES, "cross_channel.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", p, flush=True)


if __name__ == "__main__":
    main()
