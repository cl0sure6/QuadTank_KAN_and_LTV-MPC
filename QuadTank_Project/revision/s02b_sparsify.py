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
Stage 2b -- simplification of the symbolic read-out, and the accuracy-versus-
complexity comparison against direct sparse polynomial regression.

For a term budget k the deployed law keeps the k monomials of the KAN-selected
support that orthogonal matching pursuit ranks highest, and re-estimates their
coefficients under the negative-feedback shape constraint of Eq. (12).  The
identical sweep is run with the support chosen from the *full* degree-4
dictionary, which is exactly "fitting the polynomial directly using
conventional regression techniques"; comparing the two curves isolates what the
KAN actually contributes.

Local exponential stability is not something this stage has to search for: the
gate w(e) of policyform.py makes the learned correction and its Jacobian vanish
at every set-point, so the closed-loop linearisation there is A(r) - BK for the
LQR gain K of the teacher's own cost, whatever was learned.  The certificate is
nevertheless recomputed for every candidate and recorded, so the claim is
checked rather than assumed.

Outputs results/sparsity_sweep.json and models/symbolic_law_{regime}.npz.
"""

from __future__ import annotations

import json
import os
from itertools import combinations_with_replacement

import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit

import policyform as PF
import qtlib as Q
import symbolic as SY
import verify as V

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
MODELS = os.path.join(HERE, "models")

BUDGETS = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 144]
# Deployed budget: the smallest certified-stable candidate whose imitation error
# is within DEPLOY_TOL_PP percentage points of the best certified candidate.
DEPLOY_TOL_PP = 1.0
RHO_MAX = 0.9995
DT_CERT = 0.1


def full_dictionary(degree=4):
    sup = []
    for d in range(degree + 1):
        for comb in combinations_with_replacement(range(8), d):
            m = [0] * 8
            for i in comb:
                m[i] += 1
            sup.append(tuple(m))
    return sorted(set(sup))


def load(regime):
    d = np.load(os.path.join(DATA, f"dataset_{regime}.npz"), allow_pickle=True)
    y = PF.encode_targets(d["u"], d["x"], d["ref"], regime)
    o = {"regime": regime, "reg": PF.REGIMES[regime]}
    for s in ("train", "val", "test"):
        m = d["split"] == s
        o[s] = (d["feat"][m].astype(np.float64), y[m].astype(np.float64))
        o[s + "_ref"] = d["ref"][m]
        o[s + "_x"] = d["x"][m]
        o[s + "_u"] = d["u"][m]
    return o


def nmae(pred, ds, split="test"):
    p = PF.decode(pred, ds[split + "_x"], ds[split + "_ref"], ds["regime"])
    t = np.asarray(ds[split + "_u"], float)
    return 100.0 * float(np.mean(np.abs(p - t))) / (Q.U_MAX - Q.U_MIN)


def make_law(subs, coefs, regime):
    def law(x, ref):
        f = Q.features(x, ref)
        y = np.array([SY.design(f, subs[j]) @ coefs[j] for j in range(2)]).ravel()
        return PF.decode(y, x, ref, regime).ravel()
    return law


def sweep(ds, support, budgets, Fc, log, tag, regime, certify=True):
    """Accuracy / complexity / certificate for each term budget."""
    w = PF.gate(ds["train"][0])[:, None]
    A_full = SY.design(ds["train"][0], support) * w
    out = []
    for k in budgets:
        k = int(min(k, len(support)))
        subs, coefs, viol, viol_un = [], [], [], []
        viol_x, viol_x_un = [], []
        preds = np.zeros_like(ds["test"][1])
        preds_un = np.zeros_like(ds["test"][1])
        for j in (0, 1):
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=False)
            omp.fit(A_full, ds["train"][1][:, j])
            sel = np.flatnonzero(omp.coef_)
            if len(sel) == 0:
                sel = np.array([0])
            sub = [support[i] for i in sel]
            A = SY.design(ds["train"][0], sub) * w
            G, g0 = SY.shape_rows_for_output(Fc, sub, j, regime)
            c_un = SY.fit_unconstrained(A, ds["train"][1][:, j])
            c, _ = SY.fit_shape_constrained(A, ds["train"][1][:, j], G=G, g0=g0)
            preds[:, j] = SY.design(ds["test"][0], sub) @ c
            preds_un[:, j] = SY.design(ds["test"][0], sub) @ c_un
            subs.append(sub); coefs.append(c)
            viol.append(SY.total_sign_violation(c, Fc, sub, j, regime))
            viol_un.append(SY.total_sign_violation(c_un, Fc, sub, j, regime))
            # cross-channel violation is tracked in both regimes, whether or not
            # it is constrained, so the effect of constraining it is measurable
            xi = 1 - j
            viol_x.append(SY.total_sign_violation(c, Fc, sub, j, regime, i=xi))
            viol_x_un.append(SY.total_sign_violation(c_un, Fc, sub, j, regime, i=xi))
        e = nmae(preds, ds)
        e_un = nmae(preds_un, ds)
        fl = sum(SY.flops(s) for s in subs) + PF.CORE_FLOPS
        cert = (V.summarise(V.certify(make_law(subs, coefs, regime),
                                      PF.REGIMES[regime], DT_CERT,
                                      assume_equilibrium=True))
                if certify else {})
        out.append(dict(k=k, nmae=e, nmae_unconstrained=e_un, flops=fl,
                        certificate=cert,
                        n_terms=[len(s) for s in subs],
                        sign_violation=[float(v) for v in viol],
                        sign_violation_unconstrained=[float(v) for v in viol_un],
                        cross_sign_violation=[float(v) for v in viol_x],
                        cross_sign_violation_unconstrained=[float(v) for v in viol_x_un],
                        _subs=subs, _coefs=coefs))
        cmsg = ("" if not cert else
                f"  rho={cert['max_spectral_radius']:.5f}"
                f" off={cert['worst_offset_cm']:.4f}cm stable={cert['all_locally_stable']}")
        log(f"    [{tag}] k={k:4d}  nMAE={e:6.2f}% (unconstr {e_un:6.2f}%)  flops={fl:5d}"
            f"  viol {[round(v,3) for v in viol_un]} -> {[round(v,3) for v in viol]}{cmsg}")
    return out


def main():
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    dict4 = full_dictionary(4)
    result = {}
    for regime in ("MP", "NMP"):
        log(f"=== {regime} ===")
        ds = load(regime)
        rng = np.random.default_rng(0)
        Fc = SY.constraint_points(ds["train"][0], rng, reg=PF.REGIMES[regime])

        with open(os.path.join(MODELS, f"symbolic_{regime}.json")) as fh:
            forms = json.load(fh)["pretty"]
        kan_sup = sorted(set().union(*[set(SY.formula_support(f)) for f in forms]))
        log(f"  KAN support re-derived from formulas: {len(kan_sup)} monomials")

        log("  KAN-selected support:")
        kan_curve = sweep(ds, kan_sup, BUDGETS, Fc, log, "KAN", regime)
        log("  full degree-4 dictionary (direct sparse polynomial regression):")
        poly_curve = sweep(ds, dict4, BUDGETS, Fc, log, "POLY", regime)

        def ok(c):
            ct = c.get("certificate") or {}
            return (ct.get("all_locally_stable")
                    and ct.get("max_spectral_radius", 9e9) <= RHO_MAX)

        cands = [c for c in kan_curve if ok(c)]
        if cands:
            best = min(c["nmae"] for c in cands)
            deployed = min([c for c in cands if c["nmae"] <= best + DEPLOY_TOL_PP],
                           key=lambda c: c["flops"])
            reason = (f"smallest certified-stable budget within {DEPLOY_TOL_PP} pp "
                      "of the best certified accuracy")
        else:
            deployed = min(kan_curve, key=lambda c: c["nmae"])
            reason = "NO budget met the certificate; falling back to best nMAE"
        log(f"  deployed: k={deployed['k']} nMAE={deployed['nmae']:.2f}% "
            f"flops={deployed['flops']} ({reason})")

        np.savez(os.path.join(MODELS, f"symbolic_law_{regime}.npz"),
                 coef0=deployed["_coefs"][0], coef1=deployed["_coefs"][1],
                 sup0=np.array(deployed["_subs"][0]), sup1=np.array(deployed["_subs"][1]),
                 full_sup=np.array(kan_sup))

        def strip(cur):
            return [{kk: vv for kk, vv in c.items() if not kk.startswith("_")}
                    for c in cur]

        result[regime] = {
            "kan_support_curve": strip(kan_curve),
            "direct_polynomial_curve": strip(poly_curve),
            "dictionary_size": len(dict4),
            "kan_support_size": len(kan_sup),
            "deployed": {kk: vv for kk, vv in deployed.items() if not kk.startswith("_")},
            "selection_rule": {"rho_max": RHO_MAX, "tol_pp": DEPLOY_TOL_PP,
                               "reason": reason},
            "lqr_gain": PF.lqr_gain(regime).tolist(),
            "gate_scale": PF.GATE_SCALE,
            "deployed_terms": {
                "pump1": [list(m) for m in deployed["_subs"][0]],
                "pump2": [list(m) for m in deployed["_subs"][1]],
            },
            "deployed_coefficients": {
                "pump1": [float(v) for v in deployed["_coefs"][0]],
                "pump2": [float(v) for v in deployed["_coefs"][1]],
            },
        }

    with open(os.path.join(RES, "sparsity_sweep.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(RES, "s02b_log.txt"), "w") as f:
        f.write("\n".join(lines))
    print("wrote", os.path.join(RES, "sparsity_sweep.json"))


if __name__ == "__main__":
    main()
