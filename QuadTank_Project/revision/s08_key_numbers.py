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
Stage 8 -- every number quoted in the manuscript prose, collected in one place.

The tables and figures are generated automatically; the sentences are not, so
this script prints exactly the quantities the text refers to, and writes them to
results/key_numbers.json for checking.
"""

from __future__ import annotations

import json
import os

import numpy as np

import symbolic as SY

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")


def load(n):
    for d in (RES, DATA):
        p = os.path.join(d, n)
        if os.path.exists(p):
            return json.load(open(p))
    return None


def main():
    m = load("openloop_metrics.json")
    sw = load("sparsity_sweep.json")
    st = load("stability.json")
    sc = load("scenarios.json")
    sf = load("signflip_analysis.json")
    inv = load("invariance.json")
    ds = load("dataset_summary.json")
    out = {}

    o = out["imitation"] = {}
    for r in ("MP", "NMP"):
        o[r] = {
            "spline_kan": m[r]["kan_spline"]["nmae_pct_of_range"],
            "auto_symbolic_raw": m[r]["symbolic_raw"]["nmae_pct_of_range"],
            "refit": m[r]["symbolic_refit"]["nmae_pct_of_range"],
            "refit_shape_constrained": m[r]["symbolic_refit_shape_constrained"]["nmae_pct_of_range"],
            "deployed": sw[r]["deployed"]["nmae"],
            "deployed_k": sw[r]["deployed"]["k"],
            "deployed_flops": sw[r]["deployed"]["flops"],
            "mlp": m[r]["mlp"]["nmae_pct_of_range"],
            "mlp_small": m[r]["mlp_small"]["nmae_pct_of_range"],
            "deeponet": m[r]["deeponet"]["nmae_pct_of_range"],
            "poly2": m[r]["poly2"]["nmae_pct_of_range"],
            "poly3": m[r]["poly3"]["nmae_pct_of_range"],
            "poly4": m[r]["poly4"]["nmae_pct_of_range"],
            "sparse_poly_matched": m[r]["sparse_poly_matched"]["nmae_pct_of_range"],
            "edge_r2_min": m[r]["edge_r2_min"],
            "edge_r2_mean": m[r]["edge_r2_mean"],
            "n_active_edges": m[r]["n_active_edges"],
            "full_support_terms": m[r]["symbolic_structure"]["pump1"]["n_terms"],
            "max_degree": m[r]["symbolic_structure"]["pump1"]["max_total_degree"],
        }

    v = out["shape_constraint"] = {}
    for r in ("MP", "NMP"):
        cur = sw[r]["kan_support_curve"]
        vu = [c["sign_violation_unconstrained"] for c in cur]
        v[r] = {
            "violation_unconstrained_min_pct": 100 * min(min(x) for x in vu),
            "violation_unconstrained_max_pct": 100 * max(max(x) for x in vu),
            "violation_constrained_max_pct": 100 * max(max(c["sign_violation"]) for c in cur),
            "accuracy_cost_pp_at_deployed":
                sw[r]["deployed"]["nmae"] - sw[r]["deployed"]["nmae_unconstrained"],
            "full_support_cost_pp":
                m[r]["symbolic_refit_shape_constrained"]["nmae_pct_of_range"]
                - m[r]["symbolic_refit"]["nmae_pct_of_range"],
        }

    x = out["kan_vs_direct_polynomial"] = {}
    for r in ("MP", "NMP"):
        # one rule for every count the paper quotes; see symbolic.compare_supports
        x[r] = SY.compare_supports(sw[r]["kan_support_curve"],
                                   sw[r]["direct_polynomial_curve"])

    s = out["stability"] = {}
    for r in ("MP", "NMP"):
        s[r] = {
            "max_spectral_radius": st[r]["max_spectral_radius"],
            "all_locally_stable": st[r]["all_locally_stable"],
            "roa_pct": 100 * (st[r]["roa_estimate"] or {}).get("converged_fraction", float("nan")),
            "lyapunov_violation_pct": 100 * (st[r]["sampled_lyapunov"] or {}).get("violation_fraction", float("nan")),
            "mc_stable_pct": 100 * st[r]["monte_carlo_robustness"]["stable_fraction"],
            "mc_overflow_pct": 100 * st[r]["monte_carlo_robustness"]["overflow_fraction"],
            "mc_p95_err_h1_cm": st[r]["monte_carlo_robustness"]["h1_err_cm"]["p95_abs"],
            "invariance_P1": inv[r]["P1_holds"],
            "invariance_P2": inv[r]["P2_holds"],
            "invariance_rho_spread": inv[r]["spectral_radius_spread"],
        }

    c = out["closed_loop_S1"] = {}
    for r in ("MP", "NMP"):
        c[r] = {n: {"rmse": d["rmse_h12_cm"],
                    "ss": max(abs(d["steady_state_err_h1_cm"]), abs(d["steady_state_err_h2_cm"])),
                    "tv": d["total_variation_u_V"]}
                for n, d in sc[r]["S1_nominal"].items()}

    d3 = out["disturbance_S3_NMP"] = {
        n: {"peak": v["peak_deviation_h1_cm"], "recovery_s": v["recovery_time_s"],
            "tv": v["total_variation_u_V"]}
        for n, v in sc["NMP"]["S3_disturbance"].items()}

    out["handover"] = sc["handover"]["S5_handover"]
    out["constraints"] = {r: st[r]["constraint_satisfaction"] for r in ("MP", "NMP")}

    f = sf["original_dataset_conditioning"]
    j = sf["constrained_refit_on_revised_dataset"]["jitter_seed_sensitivity"]
    out["signflip"] = {
        "orig_rank": f["numerical_rank_of_9_columns"],
        "orig_cond": f["condition_number"],
        "orig_corr_e1e2": f["corr_e1_e2"],
        "new_rank": f["revised_dataset"]["numerical_rank_of_9_columns"],
        "new_cond": f["revised_dataset"]["condition_number"],
        "n_refs_new": f["revised_dataset"]["n_distinct_references"],
        "jitter_orig": j["ORIGINAL-single-trajectory"]["pump2"],
        "jitter_mp": j["MP"]["pump2"],
        "jitter_nmp": j["NMP"]["pump2"],
        "identifiable_change": sf["identifiability"]["change_in_identifiable_combination"],
        "extracted_mae_V": sf["fidelity_and_stability"]["extracted"]["imitation_mae_V"],
        "extracted_nmae_pct": sf["fidelity_and_stability"]["extracted"]["imitation_nmae_pct_of_range"],
        "repaired_mae_V": sf["fidelity_and_stability"]["repaired"]["imitation_mae_V"],
        "repaired_nmae_pct": sf["fidelity_and_stability"]["repaired"]["imitation_nmae_pct_of_range"],
        "extracted_diverged": sf["fidelity_and_stability"]["extracted"]["closed_loop_diverged"],
        "repaired_final_err": max(sf["fidelity_and_stability"]["repaired"]["closed_loop_h1_err_cm"],
                                  sf["fidelity_and_stability"]["repaired"]["closed_loop_h2_err_cm"]),
    }
    out["dataset"] = ds

    with open(os.path.join(RES, "key_numbers.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
