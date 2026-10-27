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
Stage 11 -- what the feature-saturation guard is worth.

Section 5.5 states how often the deployed law overflows a tank with and without
the guard that saturates its features to the training box before the polynomial
is evaluated. That figure previously came from an early experiment no stage
reproduced. This stage measures it on exactly the constraint campaign of
stage 4 (same 48 randomized set-point tasks, same 20 s settling window), with
the guard on (as deployed) and off. The level fade stays on in both cases, so
the comparison isolates the guard.

Outputs results/clipping_ablation.json.
"""

from __future__ import annotations

import json
import os

import controllers as C
import policyform as PF
import qtlib as Q
from s04_stability import constraint_study

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DT = 0.1


def main():
    out = {"config": {"campaign": "s04_stability.constraint_study (n=48, seed=2, T=100 s)",
                      "level_fade": PF.LEVEL_FADE}}
    saved = PF.CLIP_FEATURES
    try:
        for regime, reg in (("MP", Q.MP), ("NMP", Q.NMP)):
            out[regime] = {}
            for clip in (True, False):
                PF.CLIP_FEATURES = clip
                tag = "guard_on" if clip else "guard_off"
                print(f"=== {regime}: feature saturation {'ON' if clip else 'OFF'} ===", flush=True)
                res = constraint_study({"Symbolic KAN": C.SymbolicLaw(regime)}, reg, DT)
                out[regime][tag] = res["Symbolic KAN"]
        # The randomized campaign barely leaves the training box, so the guard
        # is rarely engaged there. The constraint-active fills of stage 10 are
        # where it acts -- on the in-box fill 38 % of the feature vector is
        # pinned at the box edge -- so the same comparison is repeated on them.
        from s10_constraint_robustness import SCENARIOS, constraint_scenario
        out["fills"] = {"source": "s10_constraint_robustness.SCENARIOS"}
        for regime, reg in (("MP", Q.MP), ("NMP", Q.NMP)):
            out["fills"][regime] = {}
            for tag, sc in SCENARIOS.items():
                out["fills"][regime][tag] = {}
                for clip in (True, False):
                    PF.CLIP_FEATURES = clip
                    key = "guard_on" if clip else "guard_off"
                    print(f"=== {regime} fill to {sc['ref_cm']} cm, guard {'ON' if clip else 'OFF'} ===", flush=True)
                    res = constraint_scenario(regime, reg, sc["ref_cm"], sc["x0"],
                                              pols={"Symbolic KAN": C.SymbolicLaw(regime)})
                    out["fills"][regime][tag][key] = res["controllers"]["Symbolic KAN"]
    finally:
        PF.CLIP_FEATURES = saved
    p = os.path.join(RES, "clipping_ablation.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", p, flush=True)
    for regime in ("MP", "NMP"):
        print("  %-4s randomized campaign overflow  guard on %.1f%%   guard off %.1f%%"
              % (regime, 100 * out[regime]["guard_on"]["overflow_fraction"],
                 100 * out[regime]["guard_off"]["overflow_fraction"]), flush=True)
        for tag in out["fills"][regime]:
            f = out["fills"][regime][tag]
            print("  %-4s %-11s fill  peak/violation  guard on %7.2f cm / %8.2f cm.s   guard off %7.2f cm / %8.2f cm.s"
                  % (regime, tag, f["guard_on"]["peak_level_cm"], f["guard_on"]["violation_area_cm_s"],
                     f["guard_off"]["peak_level_cm"], f["guard_off"]["violation_area_cm_s"]), flush=True)


if __name__ == "__main__":
    main()
