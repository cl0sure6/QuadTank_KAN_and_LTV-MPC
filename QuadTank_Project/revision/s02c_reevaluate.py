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
Stage 2c -- re-evaluate the stored surrogates under the current policy skeleton.

Every open-loop number in Section 5.1 is an error on *decoded* pump commands, so
it depends on policyform.decode as well as on the trained model. When the
skeleton changes -- as it did when the level fade was added -- those numbers go
stale even though no model changed.

Retraining is the wrong way to refresh them. KAN training on this toolchain is
not bit-reproducible even at a fixed seed (a second run at seed 42 moves the
spline error by up to 0.27 percentage points, Section 5.2), so re-running stage 2
would move every baseline for reasons unrelated to the skeleton and would leave
the seed campaign of stage 9 describing networks that no longer exist on disk.

This stage instead reloads the stored artefacts and recomputes only what the
decoding affects:

    kan_spline, symbolic_raw                 models/kan_pred_{regime}.npz
    symbolic_refit, ..._shape_constrained    models/symbolic_law_full_{regime}.npz
    mlp, mlp_small, deeponet                 models/*.pt
    poly2, poly3, poly4                      models/poly{d}_{regime}.npz

Fields that do not pass through decode -- edge R^2 of the symbolic activation
fits, the extracted structure, parameter counts, training times -- are carried
over from the existing file unchanged, because they are properties of the
training run and this stage does not train.

Every recomputed value is checked against the stored one with the fade disabled
before anything is written: if the reconstruction cannot reproduce the previous
number exactly, the stage refuses to overwrite.

Outputs results/openloop_metrics.json (in place).
"""

from __future__ import annotations

import io
import json
import os
import shutil

import numpy as np
import torch

import policyform as PF
import symbolic as SY

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODELS = os.path.join(HERE, "models")
RES = os.path.join(HERE, "results")

U_RANGE = 12.0
TOL = 5e-4          # the stored file is written at full precision; this is slack


def load_split(regime):
    d = np.load(os.path.join(DATA, f"dataset_{regime}.npz"), allow_pickle=True)
    m = d["split"] == "test"
    return (d["feat"][m].astype(np.float64), d["x"][m], d["ref"][m], d["u"][m])


def metrics(pred_y, x, ref, u, regime, clip=True):
    """Identical in form to s02.metrics, on the decoded pump commands."""
    p = PF.decode(pred_y, x, ref, regime, clip=clip)
    t = np.asarray(u, float)
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


def predictions(regime, F):
    """Every stored surrogate's raw output on the test split, as s02 produced it."""
    from s02_train_models import MLP, DeepONet

    out = {}
    # Written by stage 2. The features are stored alongside the predictions and
    # checked here: an older file with a matching row count but a different
    # dataset would otherwise be silently misread, which is exactly what the
    # orphaned models/kan_pred_*.npz of an earlier pipeline would have caused.
    kp_path = os.path.join(MODELS, f"kan_testpred_{regime}.npz")
    if not os.path.exists(kp_path):
        raise SystemExit(
            "missing %s -- rerun stage 2 to produce it. The KAN is not otherwise\n"
            "persisted, so its two rows cannot be re-evaluated without retraining."
            % kp_path)
    kp = np.load(kp_path)
    if kp["feat_test"].shape != F.shape or not np.allclose(kp["feat_test"], F, atol=1e-9):
        raise SystemExit(
            "%s was produced from a different test split than data/dataset_%s.npz;\n"
            "rerun stage 2." % (kp_path, regime))
    out["kan_spline"] = kp["pred_spline"]
    out["symbolic_raw"] = kp["pred_sym_raw"]

    # Stage 2 persists only the shape-constrained coefficients. The unconstrained
    # ones are a plain regularised least-squares solve on the same stored support
    # and the same fixed training split, so they are reproduced here exactly
    # rather than approximated -- and the check below refuses to proceed if they
    # are not.
    #
    # The design matrix is weighted by the setpoint gate w(e) alone, deliberately
    # NOT by the level fade g(x). Folding the fade into the fit would let the
    # learned term grow to compensate for it near the limit, which is precisely
    # the behaviour the fade exists to suppress; the fit is therefore performed as
    # though g == 1 everywhere, exactly as the shape constraint is.
    full = np.load(os.path.join(MODELS, f"symbolic_law_full_{regime}.npz"),
                   allow_pickle=True)
    sup = [[tuple(int(v) for v in m) for m in full[f"sup{j}"]] for j in (0, 1)]

    d = np.load(os.path.join(DATA, f"dataset_{regime}.npz"), allow_pickle=True)
    tr = d["split"] == "train"
    F_tr = d["feat"][tr].astype(np.float64)
    Y_tr = PF.encode_targets(d["u"][tr], d["x"][tr], d["ref"][tr], regime)
    w_tr = PF.gate(F_tr)[:, None]

    pred_un = np.zeros((len(F), 2))
    pred_sc = np.zeros((len(F), 2))
    for j in (0, 1):
        A_tr = SY.design(F_tr, sup[j]) * w_tr
        c_un = SY.fit_unconstrained(A_tr, Y_tr[:, j])
        pred_un[:, j] = SY.design(F, sup[j]) @ c_un
        pred_sc[:, j] = SY.design(F, sup[j]) @ full[f"coef{j}"]
    out["symbolic_refit"] = pred_un
    out["symbolic_refit_shape_constrained"] = pred_sc

    Ft = torch.tensor(F, dtype=torch.float32)
    for tag, hid in (("mlp", (32, 32)), ("mlp_small", (8,))):
        m = MLP(hid)
        m.load_state_dict(torch.load(os.path.join(MODELS, f"{tag}_{regime}.pt")))
        m.eval()
        with torch.no_grad():
            out[tag] = m(Ft).numpy()
    don = DeepONet()
    don.load_state_dict(torch.load(os.path.join(MODELS, f"deeponet_{regime}.pt")))
    don.eval()
    with torch.no_grad():
        out["deeponet"] = don(Ft).numpy()

    # s02 evaluates the ridge pipelines on RAW features -- no clipping -- so this
    # reconstruction must not clip either, or it will not reproduce the stored value
    for deg in (2, 3, 4):
        p = os.path.join(MODELS, f"poly{deg}_{regime}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        sup = [tuple(int(v) for v in q) for q in d["powers"]]
        out[f"poly{deg}"] = SY.design(F, sup) @ d["coef"].T + d["intercept"]
    return out


def main():
    path = os.path.join(RES, "openloop_metrics.json")
    stored = json.load(io.open(path, encoding="utf-8"))

    recomputed, failures = {}, []
    for regime in ("MP", "NMP"):
        F, X, REF, U = load_split(regime)
        preds = predictions(regime, F)
        block = {}
        print(f"=== {regime} ===", flush=True)
        for tag, py in sorted(preds.items()):
            new = metrics(py, X, REF, U, regime)          # current skeleton
            PF.LEVEL_FADE = not PF.LEVEL_FADE
            alt = metrics(py, X, REF, U, regime)["nmae_pct_of_range"]
            PF.LEVEL_FADE = not PF.LEVEL_FADE

            old = stored[regime].get(tag, {}).get("nmae_pct_of_range")
            if old is None:
                failures.append(f"{regime}/{tag}: not in the stored file")
                continue
            if abs(new["nmae_pct_of_range"] - old) <= TOL:
                # already consistent with the current skeleton; nothing to do.
                # This is the idempotent case, and the case right after stage 2
                # has been run under the current skeleton.
                block[tag] = new
                print("    %-34s %8.4f  (already current)"
                      % (tag, old), flush=True)
                continue
            if abs(alt - old) > TOL:
                # the stored value matches neither setting, so the difference is
                # not the skeleton -- most likely the models on disk are not the
                # ones that produced the file. Refuse rather than guess.
                failures.append(
                    "%s/%s: stored %.4f matches neither the current skeleton "
                    "(%.4f) nor the alternative (%.4f)"
                    % (regime, tag, old, new["nmae_pct_of_range"], alt))
                continue
            block[tag] = new
            print("    %-34s %8.4f -> %8.4f  (%+0.4f)"
                  % (tag, old, new["nmae_pct_of_range"],
                     new["nmae_pct_of_range"] - old), flush=True)
        recomputed[regime] = block

    if failures:
        print("\nREFUSING TO WRITE -- reconstruction did not reproduce the stored file:")
        for f in failures:
            print("   " + f)
        raise SystemExit(1)

    shutil.copy2(path, path + ".pre_fade")
    for regime, block in recomputed.items():
        for tag, new in block.items():
            # keep every field that does not pass through decode: n_params,
            # train_time_s, hidden, degree, alpha, n_terms_per_output, ...
            stored[regime][tag].update(new)
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(stored, f, indent=2)
    print("\nwrote", path)
    print("previous file kept at", path + ".pre_fade")


if __name__ == "__main__":
    main()
