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
Stage 9 -- how much of the result depends on one draw of the training seed.

Every headline number elsewhere in this pipeline comes from a single trained
network per regime (KAN_SEED = 42). This stage retrains from several seeds and
reports the spread of the metrics the manuscript quotes, so that claims about
the read-out and about the KAN-versus-OMP comparison can be read with their
variability attached rather than as point estimates.

It also attaches Wilson score intervals to the Monte-Carlo robustness fractions,
which are binomial proportions over a finite sample and were previously reported
as bare percentages.

The stage is destructive to models/ and to two result files while it runs, so it
snapshots them first and restores them afterwards: the deployed law must remain
the seed-42 law that the rest of the manuscript describes.

Outputs results/seed_campaign.json.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MODELS = os.path.join(HERE, "models")

SEEDS = [42, 1, 7, 13, 23]
BASELINE_SEED = 42

# Written by s02 / s02b; snapshotted so the campaign cannot disturb the
# artefacts the rest of the pipeline depends on.
VOLATILE = ["openloop_metrics.json", "sparsity_sweep.json",
            "s02_log.txt", "s02b_log.txt"]

METRICS = [("kan_spline", "KAN before symbolic read-out"),
           ("symbolic_raw", "auto_symbolic output"),
           ("symbolic_refit", "coefficient refit"),
           ("symbolic_refit_shape_constrained", "refit + shape constraint"),
           ("mlp", "MLP 8-32-32-2")]


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion; k successes of n."""
    if n == 0:
        return [None, None]
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100.0 * max(0.0, c - h), 2), round(100.0 * min(1.0, c + h), 2)]


def snapshot(tmp):
    shutil.copytree(MODELS, os.path.join(tmp, "models"))
    for f in VOLATILE:
        p = os.path.join(RES, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(tmp, f))


def restore(tmp):
    shutil.rmtree(MODELS, ignore_errors=True)
    shutil.copytree(os.path.join(tmp, "models"), MODELS)
    for f in VOLATILE:
        p = os.path.join(tmp, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(RES, f))


def run_stage(script, seed):
    """Run one stage in a child process with the training seed overridden.

    Both ends of the pipe are pinned to UTF-8. Without this the child inherits
    the console's code page (cp1252 on a Western-locale Windows) while pykan's
    progress output contains bytes that page cannot represent, which kills the
    reader thread inside subprocess with a UnicodeDecodeError. The stage itself
    survives -- the return code is unaffected -- so the campaign completes and
    the damage is confined to the captured log, but a crash inside the harness
    that reports failures is not something to leave in place.
    """
    env = dict(os.environ, KAN_SEED=str(seed), PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, script], cwd=HERE, env=env,
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"    !! {script} failed (rc={r.returncode})", flush=True)
        # capture can still come back empty if the child died before writing;
        # this is the failure path, so it must not raise on its way out
        print("   ", (r.stdout or "")[-800:], flush=True)
        print("   ", (r.stderr or "")[-800:], flush=True)
    return r.returncode == 0


def harvest():
    """Headline metrics from whatever is currently in results/."""
    with open(os.path.join(RES, "openloop_metrics.json")) as f:
        m = json.load(f)
    with open(os.path.join(RES, "sparsity_sweep.json")) as f:
        sw = json.load(f)
    out = {}
    for regime in ("MP", "NMP"):
        row = {k: (round(float(m[regime][k]["nmae_pct_of_range"]), 3)
                   if k in m[regime] else None)
               for k, _ in METRICS}
        dep = sw[regime]["deployed"]
        row["deployed_k"] = int(dep["k"])
        row["deployed_nmae"] = round(float(dep["nmae"]), 3)
        row["deployed_flops"] = int(dep["flops"])
        # does the KAN support beat the OMP support, budget by budget?
        kan = {c["k"]: c["nmae"] for c in sw[regime]["kan_support_curve"]}
        pol = {c["k"]: c["nmae"] for c in sw[regime]["direct_polynomial_curve"]}
        common = sorted(set(kan) & set(pol))
        wins = sum(1 for k in common if kan[k] < pol[k] - 1e-9)
        row["budgets_compared"] = len(common)
        row["kan_wins"] = wins
        row["omp_wins"] = len(common) - wins
        row["max_abs_gap_pp"] = round(
            max((abs(kan[k] - pol[k]) for k in common), default=0.0), 3)
        out[regime] = row
    return out


def monte_carlo_intervals():
    """Wilson intervals on the robustness fractions already in stability.json."""
    p = os.path.join(RES, "stability.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        st = json.load(f)
    out = {}
    for regime in ("MP", "NMP"):
        mc = st.get(regime, {}).get("monte_carlo_robustness")
        if not mc:
            continue
        n = int(mc.get("n") or mc.get("n_samples") or 200)
        row = {"n_samples": n}
        for key in ("stable_fraction", "overflow_fraction", "dryout_fraction"):
            if key in mc:
                frac = float(mc[key])
                row[key + "_pct"] = round(100.0 * frac, 2)
                row[key + "_wilson95_pct"] = wilson(round(frac * n), n)
        out[regime] = row
    return out


def summarise(runs):
    """Spread of each metric across seeds."""
    import numpy as np
    keys = [k for k, _ in METRICS] + ["deployed_nmae", "deployed_k"]
    out = {}
    for regime in ("MP", "NMP"):
        r = {}
        for k in keys:
            v = [x[regime][k] for x in runs.values()
                 if x.get(regime, {}).get(k) is not None]
            if not v:
                continue
            v = np.array(v, float)
            r[k] = {"median": round(float(np.median(v)), 3),
                    "min": round(float(v.min()), 3),
                    "max": round(float(v.max()), 3),
                    "spread": round(float(v.max() - v.min()), 3),
                    "n_seeds": int(v.size)}
        kw = [x[regime]["kan_wins"] for x in runs.values() if regime in x]
        ow = [x[regime]["omp_wins"] for x in runs.values() if regime in x]
        r["kan_wins_per_seed"] = kw
        r["omp_wins_per_seed"] = ow
        r["kan_ever_dominates"] = bool(any(
            k > 0 and o == 0 for k, o in zip(kw, ow)))
        out[regime] = r
    return out


def main():
    tmp = tempfile.mkdtemp(prefix="seedcampaign_")
    print("snapshotting models/ and volatile results to", tmp, flush=True)
    snapshot(tmp)
    runs, failed = {}, []
    try:
        for seed in SEEDS:
            t0 = time.time()
            print(f"=== seed {seed} ===", flush=True)
            ok = run_stage("s02_train_models.py", seed)
            ok = ok and run_stage("s02b_sparsify.py", seed)
            if not ok:
                failed.append(seed)
                continue
            runs[str(seed)] = harvest()
            print(f"    done in {time.time() - t0:.0f}s  "
                  f"MP deployed nMAE={runs[str(seed)]['MP']['deployed_nmae']}  "
                  f"NMP={runs[str(seed)]['NMP']['deployed_nmae']}", flush=True)
    finally:
        print("restoring the seed-42 artefacts", flush=True)
        restore(tmp)
        shutil.rmtree(tmp, ignore_errors=True)

    result = {"config": {"seeds": SEEDS, "baseline_seed": BASELINE_SEED,
                         "failed_seeds": failed,
                         "note": "models/ restored to the baseline seed after the run"},
              "per_seed": runs,
              "summary": summarise(runs) if runs else {},
              "monte_carlo_wilson95": monte_carlo_intervals()}
    p = os.path.join(RES, "seed_campaign.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("\nwrote", p, flush=True)
    for regime in ("MP", "NMP"):
        s = result["summary"].get(regime, {})
        if "deployed_nmae" in s:
            d = s["deployed_nmae"]
            print(f"  {regime} deployed nMAE over {d['n_seeds']} seeds: "
                  f"median {d['median']}  range [{d['min']}, {d['max']}]", flush=True)
            print(f"  {regime} KAN wins per seed {s['kan_wins_per_seed']} "
                  f"vs OMP {s['omp_wins_per_seed']}", flush=True)


if __name__ == "__main__":
    main()
