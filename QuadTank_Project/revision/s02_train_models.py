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
Stage 2 -- distillation of the LTV-MPC policy into every candidate surrogate,
and the two-step symbolic read-out described in symbolic.py.

Surrogates, all trained on the identical dataset and split of stage 1:
  * KAN (spline)              -- intermediate representation
  * symbolic KAN, raw         -- pykan's auto_symbolic output
  * symbolic KAN + refit      -- KAN-selected support, coefficients re-estimated
  * symbolic KAN + refit + SC -- additionally shape (negative-feedback) constrained
  * MLP (32-32) and MLP (8)   -- conventional approximate-MPC baselines
  * DeepONet                  -- operator-learning baseline
  * polynomial ridge, deg 2/3 -- direct regression on the same data
  * sparse polynomial (OMP)   -- same term budget as the symbolic KAN, but with
                                 the support chosen by orthogonal matching
                                 pursuit instead of by the KAN.  This isolates
                                 what the KAN actually contributes.

Errors are open-loop policy-imitation errors on the held-out test split, in
volts and as a percentage of the full actuator range (12 V), stated explicitly.
Predictions are clipped to the actuator box before scoring, because every
deployed controller clips.

Outputs results/openloop_metrics.json, models/*.
"""

from __future__ import annotations

import copy
import json
import os
import time

import numpy as np
import sympy
import torch

import policyform as PF
import qtlib as Q
import symbolic as SY

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
MODELS = os.path.join(HERE, "models")
os.makedirs(RES, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

DEV = torch.device("cpu")
U_RANGE = Q.U_MAX - Q.U_MIN

# --- KAN hyper-parameters (reported verbatim in the manuscript) -----------
KAN_WIDTH = [8, 5, 2]
KAN_GRID = 8
KAN_K = 3
# Default 42 reproduces the deployed law. Stage 9 overrides it through the
# environment to measure how much of the result depends on this one draw.
KAN_SEED = int(os.environ.get("KAN_SEED", "42"))
KAN_STEPS_1 = 40
KAN_STEPS_2 = 25
KAN_LAMB = 1e-4          # sparsification weight
KAN_LAMB_ENTROPY = 2.0
PRUNE_EDGE_TH = 5e-3
PRUNE_NODE_TH = 1e-3
SYMB_LIB = ["x", "x^2"]  # polynomial family -> fixed-length multiply-add chain
WEIGHT_SIMPLE = 0.0      # rank symbolic candidates by fit quality, not simplicity


def load(regime):
    """Targets are the *deviation* from the analytic feed-forward input (see
    policyform.py), so every surrogate learns the same quantity."""
    d = np.load(os.path.join(DATA, f"dataset_{regime}.npz"), allow_pickle=True)
    reg = PF.REGIMES[regime]
    y = PF.encode_targets(d["u"], d["x"], d["ref"], regime)
    out = {"regime": regime, "reg": reg}
    for s in ("train", "val", "test"):
        m = d["split"] == s
        out[s] = (d["feat"][m].astype(np.float64), y[m].astype(np.float64))
        out[s + "_ref"] = d["ref"][m]
        out[s + "_x"] = d["x"][m]
        out[s + "_u"] = d["u"][m]
    out["raw"] = d
    return out


def metrics(pred_y, ds, split="test", clip=True):
    """Errors are computed on the decoded pump commands, in volts."""
    p = PF.decode(pred_y, ds[split + "_x"], ds[split + "_ref"], ds["regime"], clip=clip)
    t = np.asarray(ds[split + "_u"], float)
    err = p - t
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((t - t.mean(axis=0)) ** 2))
    return {
        "mae_V": float(np.mean(np.abs(err))),
        "rmse_V": float(np.sqrt(np.mean(err ** 2))),
        "max_abs_V": float(np.max(np.abs(err))),
        "nmae_pct_of_range": 100.0 * float(np.mean(np.abs(err))) / U_RANGE,
        "nrmse_pct_of_range": 100.0 * float(np.sqrt(np.mean(err ** 2))) / U_RANGE,
        "r2": 1.0 - ss_res / ss_tot,
    }


# ------------------------------------------------------------------- KAN
def train_kan(regime, ds, log):
    from kan import KAN

    tr, va, te = ds["train"], ds["val"], ds["test"]
    dataset = {
        "train_input": torch.tensor(tr[0], dtype=torch.float32, device=DEV),
        "train_label": torch.tensor(tr[1], dtype=torch.float32, device=DEV),
        "test_input": torch.tensor(va[0], dtype=torch.float32, device=DEV),
        "test_label": torch.tensor(va[1], dtype=torch.float32, device=DEV),
    }
    torch.manual_seed(KAN_SEED)
    model = KAN(width=KAN_WIDTH, grid=KAN_GRID, k=KAN_K, seed=KAN_SEED, device=DEV)

    t0 = time.time()
    model.fit(dataset, opt="LBFGS", steps=KAN_STEPS_1,
              lamb=KAN_LAMB, lamb_entropy=KAN_LAMB_ENTROPY)
    model = model.prune(node_th=PRUNE_NODE_TH, edge_th=PRUNE_EDGE_TH)
    model.fit(dataset, opt="LBFGS", steps=KAN_STEPS_2,
              lamb=KAN_LAMB, lamb_entropy=KAN_LAMB_ENTROPY)
    train_time = time.time() - t0

    Xte = torch.tensor(te[0], dtype=torch.float32, device=DEV)
    with torch.no_grad():
        pred_spline = model(Xte).cpu().numpy()
    m_spline = metrics(pred_spline, ds)
    log(f"[{regime}] spline-KAN  test nMAE={m_spline['nmae_pct_of_range']:.2f}%  R2={m_spline['r2']:.4f}")

    # ---- step 1: structure selection ------------------------------------
    model(dataset["train_input"])
    edge_r2 = []
    for l in range(len(model.width_in) - 1):
        for i in range(model.width_in[l]):
            for j in range(model.width_out[l + 1]):
                sm = float(model.symbolic_fun[l].mask[j, i])
                am = float(model.act_fun[l].mask[i][j])
                if sm > 0.0 and am == 0.0:
                    continue
                if sm == 0.0 and am == 0.0:
                    model.fix_symbolic(l, i, j, "0", verbose=False, log_history=False)
                    continue
                try:
                    name, fn, r2, c = model.suggest_symbolic(
                        l, i, j, lib=SYMB_LIB, verbose=False, weight_simple=WEIGHT_SIMPLE)
                    r2v = float(np.asarray(r2.detach().cpu() if hasattr(r2, "detach") else r2).ravel()[0])
                except Exception as exc:  # pragma: no cover
                    log(f"   suggest_symbolic failed at ({l},{i},{j}): {exc}; using 'x'")
                    name, r2v = "x", float("nan")
                model.fix_symbolic(l, i, j, name, verbose=False, log_history=False)
                edge_r2.append({"layer": l, "in": i, "out": j, "fun": name, "r2": r2v})

    with torch.no_grad():
        pred_sym_raw = model(Xte).cpu().numpy()
    m_sym_raw = metrics(pred_sym_raw, ds)
    log(f"[{regime}] symbolic RAW test nMAE={m_sym_raw['nmae_pct_of_range']:.2f}%  R2={m_sym_raw['r2']:.4f}")

    formulas = model.symbolic_formula()[0]

    # Persist the network's raw outputs on the test split, together with the
    # exact features they were produced from. The KAN itself is not serialisable
    # here in a form later stages can reload, and training is not reproducible
    # even at a fixed seed, so without this the two KAN rows of Table 4 cannot be
    # re-evaluated when the policy skeleton changes -- they would have to be
    # retrained, which changes them for unrelated reasons. `s02c_reevaluate.py`
    # consumes this file and verifies it against the recorded metrics before use.
    np.savez(os.path.join(MODELS, f"kan_testpred_{regime}.npz"),
             feat_test=np.asarray(te[0], float),
             pred_spline=np.asarray(pred_spline, float),
             pred_sym_raw=np.asarray(pred_sym_raw, float),
             nmae_spline=m_spline["nmae_pct_of_range"],
             nmae_sym_raw=m_sym_raw["nmae_pct_of_range"])

    return dict(model=model, pred_spline=pred_spline, m_spline=m_spline,
                m_sym_raw=m_sym_raw, edge_r2=edge_r2, formulas=formulas,
                train_time=train_time)


# ---------------------------------------------- step 2: constrained refit
def symbolic_refit(regime, ds, formulas, log):
    """
    Refit the coefficients on the KAN-selected support, once unconstrained and
    once under the negative-feedback shape constraint of the FULL deployed law
    (gate and LQR core included -- see policyform.py and symbolic.total_shape_rows).
    """
    tr, te = ds["train"], ds["test"]
    rng = np.random.default_rng(0)
    Fc = SY.constraint_points(tr[0], rng, reg=PF.REGIMES[regime])
    w_tr = PF.gate(tr[0])[:, None]

    res, coefs, supports = {}, [], []
    pred_un = np.zeros_like(te[1])
    pred_sc = np.zeros_like(te[1])
    for j, f in enumerate(formulas):
        sup = SY.formula_support(f)
        A_tr = SY.design(tr[0], sup) * w_tr
        A_te = SY.design(te[0], sup)
        G, g0 = SY.shape_rows_for_output(Fc, sup, j, regime)
        c_un = SY.fit_unconstrained(A_tr, tr[1][:, j])
        c_sc, ok = SY.fit_shape_constrained(A_tr, tr[1][:, j], G=G, g0=g0)
        pred_un[:, j] = A_te @ c_un
        pred_sc[:, j] = A_te @ c_sc
        res[f"pump{j+1}"] = {
            "n_terms": len(sup),
            "max_total_degree": int(max(sum(m) for m in sup)),
            "flops_per_call": SY.flops(sup),
            "sign_violation_frac_unconstrained":
                SY.total_sign_violation(c_un, Fc, sup, j, regime),
            "sign_violation_frac_constrained":
                SY.total_sign_violation(c_sc, Fc, sup, j, regime),
            "qp_solved": bool(ok),
        }
        coefs.append(c_sc)
        supports.append(sup)
    m_un = metrics(pred_un, ds)
    m_sc = metrics(pred_sc, ds)
    log(f"[{regime}] symbolic REFIT      test nMAE={m_un['nmae_pct_of_range']:.2f}%  R2={m_un['r2']:.4f}")
    log(f"[{regime}] symbolic REFIT+SC   test nMAE={m_sc['nmae_pct_of_range']:.2f}%  R2={m_sc['r2']:.4f}"
        f"  (sign violations {res['pump1']['sign_violation_frac_unconstrained']:.3f}"
        f"/{res['pump2']['sign_violation_frac_unconstrained']:.3f} -> "
        f"{res['pump1']['sign_violation_frac_constrained']:.3f}"
        f"/{res['pump2']['sign_violation_frac_constrained']:.3f})")
    np.savez(os.path.join(MODELS, f"symbolic_law_full_{regime}.npz"),
             coef0=coefs[0], coef1=coefs[1],
             sup0=np.array(supports[0]), sup1=np.array(supports[1]))
    return res, m_un, m_sc, supports, coefs


def sparse_poly_baseline(ds, n_terms, degree, log, tag=""):
    """
    Same term budget as the symbolic KAN, but the support is chosen by
    orthogonal matching pursuit over the full degree-`degree` dictionary.
    """
    from itertools import combinations_with_replacement
    from sklearn.linear_model import OrthogonalMatchingPursuit

    sup_all = []
    for d in range(0, degree + 1):
        for comb in combinations_with_replacement(range(8), d):
            m = [0] * 8
            for i in comb:
                m[i] += 1
            sup_all.append(tuple(m))
    sup_all = sorted(set(sup_all))
    A_tr = SY.design(ds["train"][0], sup_all)
    A_te = SY.design(ds["test"][0], sup_all)
    pred = np.zeros_like(ds["test"][1])
    used = []
    for j in range(2):
        k = int(min(max(n_terms[j], 2), A_tr.shape[1]))
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=True)
        omp.fit(A_tr, ds["train"][1][:, j])
        pred[:, j] = omp.predict(A_te)
        used.append(int((omp.coef_ != 0).sum()))
    m = metrics(pred, ds)
    log(f"  sparse-poly OMP{tag} (deg<={degree}, {used} terms): nMAE={m['nmae_pct_of_range']:.2f}% R2={m['r2']:.4f}")
    m["n_terms"] = used
    m["dictionary_size"] = len(sup_all)
    m["degree"] = degree
    return m


# ------------------------------------------------------------------- MLP
class MLP(torch.nn.Module):
    def __init__(self, hidden=(32, 32)):
        super().__init__()
        layers, prev = [], 8
        for h in hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.Tanh()]
            prev = h
        layers += [torch.nn.Linear(prev, 2)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepONet(torch.nn.Module):
    """Branch/trunk operator network (Lu et al., 2021): the branch encodes the
    commanded reference, the trunk the measured state."""

    def __init__(self, p=32, hidden=64):
        super().__init__()
        self.branch = torch.nn.Sequential(
            torch.nn.Linear(4, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, 2 * p))
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(4, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, 2 * p), torch.nn.Tanh())
        self.p = p
        self.bias = torch.nn.Parameter(torch.zeros(2))

    def forward(self, f):
        x = f[:, :4]
        ref = f[:, :4] + f[:, 4:]
        b = self.branch(ref).view(-1, 2, self.p)
        t = self.trunk(x).view(-1, 2, self.p)
        return (b * t).sum(-1) + self.bias


def train_torch(model, ds, epochs=400, lr=1e-3, wd=1e-6, seed=0, log=print, name="mlp"):
    torch.manual_seed(seed)
    Xtr = torch.tensor(ds["train"][0], dtype=torch.float32)
    Ytr = torch.tensor(ds["train"][1], dtype=torch.float32)
    Xva = torch.tensor(ds["val"][0], dtype=torch.float32)
    Yva = torch.tensor(ds["val"][1], dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = float("inf"), None
    n = len(Xtr)
    t0 = time.time()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            torch.nn.functional.mse_loss(model(Xtr[idx]), Ytr[idx]).backward()
            opt.step()
        sched.step()
        with torch.no_grad():
            vl = torch.nn.functional.mse_loss(model(Xva), Yva).item()
        if vl < best:
            best, best_state = vl, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    dt = time.time() - t0
    log(f"  {name}: best val MSE {best:.3e}  ({dt:.0f} s)")
    return model, dt


def count_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def train_poly(ds, degree, alpha_grid=(1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0)):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    best, best_a, best_val = None, None, float("inf")
    for a in alpha_grid:
        pipe = make_pipeline(PolynomialFeatures(degree, include_bias=False), Ridge(alpha=a))
        pipe.fit(ds["train"][0], ds["train"][1])
        vl = float(np.mean((pipe.predict(ds["val"][0]) - ds["val"][1]) ** 2))
        if vl < best_val:
            best, best_a, best_val = pipe, a, vl
    n_feat = best.named_steps["polynomialfeatures"].n_output_features_
    return best, {"degree": degree, "alpha": best_a,
                  "n_terms_per_output": int(n_feat) + 1}


def main():
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    out = {}
    for regime in ("MP", "NMP"):
        log(f"================ regime {regime} ================")
        ds = load(regime)
        log(f"  n_train={len(ds['train'][0])} n_val={len(ds['val'][0])} n_test={len(ds['test'][0])}")
        r = {}

        k = train_kan(regime, ds, log)
        sym_res, m_un, m_sc, supports, coefs = symbolic_refit(regime, ds, k["formulas"], log)

        r["kan_spline"] = k["m_spline"]
        r["symbolic_raw"] = k["m_sym_raw"]
        r["symbolic_refit"] = m_un
        r["symbolic_refit_shape_constrained"] = m_sc
        r["symbolic_structure"] = sym_res
        r["edge_symbolic_r2"] = k["edge_r2"]
        r["edge_r2_min"] = float(np.nanmin([e["r2"] for e in k["edge_r2"]])) if k["edge_r2"] else None
        r["edge_r2_mean"] = float(np.nanmean([e["r2"] for e in k["edge_r2"]])) if k["edge_r2"] else None
        r["n_active_edges"] = len(k["edge_r2"])
        r["kan_train_time_s"] = k["train_time"]
        r["hyperparams"] = {
            "width": KAN_WIDTH, "grid": KAN_GRID, "k": KAN_K, "seed": KAN_SEED,
            "steps": [KAN_STEPS_1, KAN_STEPS_2], "lamb": KAN_LAMB,
            "lamb_entropy": KAN_LAMB_ENTROPY, "prune_edge_th": PRUNE_EDGE_TH,
            "prune_node_th": PRUNE_NODE_TH, "symbolic_lib": SYMB_LIB,
            "weight_simple": WEIGHT_SIMPLE, "optimizer": "LBFGS (full batch)",
        }
        # error decomposition requested by the reviewers
        r["error_decomposition"] = {
            "mpc_to_kan_nmae_pct": k["m_spline"]["nmae_pct_of_range"],
            "mpc_to_symbolic_raw_nmae_pct": k["m_sym_raw"]["nmae_pct_of_range"],
            "mpc_to_symbolic_refit_nmae_pct": m_un["nmae_pct_of_range"],
            "mpc_to_symbolic_final_nmae_pct": m_sc["nmae_pct_of_range"],
            "symbolisation_penalty_pct_points":
                m_sc["nmae_pct_of_range"] - k["m_spline"]["nmae_pct_of_range"],
        }
        with open(os.path.join(MODELS, f"symbolic_{regime}.json"), "w") as fh:
            json.dump({"pretty": [str(f) for f in k["formulas"]],
                       "support_sizes": [len(s) for s in supports]}, fh, indent=2)

        # --- neural baselines ---
        for tag, hid, seed in (("mlp", (32, 32), 1), ("mlp_small", (8,), 2)):
            m = MLP(hid)
            m, tt = train_torch(m, ds, seed=seed, log=log, name=f"MLP{hid}")
            with torch.no_grad():
                p = m(torch.tensor(ds["test"][0], dtype=torch.float32)).numpy()
            r[tag] = metrics(p, ds)
            r[tag].update(n_params=count_params(m), train_time_s=tt, hidden=list(hid))
            torch.save(m.state_dict(), os.path.join(MODELS, f"{tag}_{regime}.pt"))
            log(f"[{regime}] {tag}: nMAE={r[tag]['nmae_pct_of_range']:.2f}% R2={r[tag]['r2']:.4f}")

        don = DeepONet()
        don, tt = train_torch(don, ds, seed=3, log=log, name="DeepONet")
        with torch.no_grad():
            p = don(torch.tensor(ds["test"][0], dtype=torch.float32)).numpy()
        r["deeponet"] = metrics(p, ds)
        r["deeponet"].update(n_params=count_params(don), train_time_s=tt)
        torch.save(don.state_dict(), os.path.join(MODELS, f"deeponet_{regime}.pt"))
        log(f"[{regime}] deeponet: nMAE={r['deeponet']['nmae_pct_of_range']:.2f}% R2={r['deeponet']['r2']:.4f}")

        # --- direct polynomial regression ---
        for deg in (2, 3, 4):
            t0 = time.time()
            pipe, info = train_poly(ds, deg)
            r[f"poly{deg}"] = metrics(pipe.predict(ds["test"][0]), ds)
            r[f"poly{deg}"].update(info, train_time_s=time.time() - t0)
            np.savez(os.path.join(MODELS, f"poly{deg}_{regime}.npz"),
                     coef=pipe.named_steps["ridge"].coef_,
                     intercept=pipe.named_steps["ridge"].intercept_,
                     powers=pipe.named_steps["polynomialfeatures"].powers_)
            log(f"[{regime}] poly{deg} ({info['n_terms_per_output']} terms): "
                f"nMAE={r[f'poly{deg}']['nmae_pct_of_range']:.2f}% R2={r[f'poly{deg}']['r2']:.4f}")

        # --- sparse polynomial with the SAME budget as the symbolic KAN ---
        budget = [sym_res["pump1"]["n_terms"], sym_res["pump2"]["n_terms"]]
        maxdeg = max(sym_res["pump1"]["max_total_degree"], sym_res["pump2"]["max_total_degree"])
        r["sparse_poly_matched"] = sparse_poly_baseline(ds, budget, min(maxdeg, 4), log)
        r["sparse_poly_matched"]["budget"] = budget

        out[regime] = r

    with open(os.path.join(RES, "openloop_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(RES, "s02_log.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\nwrote", os.path.join(RES, "openloop_metrics.json"))


if __name__ == "__main__":
    main()
