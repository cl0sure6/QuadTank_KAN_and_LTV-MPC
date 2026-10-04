# Structural guarantees in the approximation of model predictive control laws

**Author:** Adilkhan Salkimbayev — **Licence:** Apache-2.0 — **Status:** major revision under review at *Journal of Process Control*

Code and data for a study of which guarantees survive an offline
approximation of a linear time-varying MPC law, and which only appear to, on the Johansson quadruple-tank
benchmark in both its minimum-phase (MP) and non-minimum-phase (NMP) configurations.

Everything reported in the accompanying manuscript is produced by the numbered stage
scripts in [`QuadTank_Project/revision/`](QuadTank_Project/revision). Start there.

---

## What the study finds

The deployed controller is

```
u = sat[ u_eq(r)  +  K e  +  g(x) * w(e) * U_max * f(phi(x, r)) ]
     feed-forward   LQR core        learned correction
```

with a gate `w(e) = min(||e||^2 / s^2, 1)` that vanishes quadratically at the set-point,
and a level fade `g(x)` that hands authority back to the LQR core as any tank approaches
its 20 cm limit. Both multiply the learned term only, so the structural properties below
hold for any learned `f` and any bounded `g`.

| Finding | Evidence |
|---|---|
| The **symbolic read-out**, not the network, is a dominant error source | MP: a 2.39 % imitation error becomes 7.75 % under off-the-shelf `auto_symbolic`; a convex refit recovers it to 5.44 %. NMP: 4.60 % → 11.23 % → 9.00 % |
| An unconstrained read-out **violates negative feedback** on up to 11 % of realizable operating points | Driven to ≤ 0.6 % in both regimes by an affine inequality inside a convex least-squares fit. It costs 0.33 pp (MP) and 0.69 pp (NMP) on the full support; at the deployed 4-term MP budget it *improves* accuracy by 2.13 pp |
| The gate makes **zero steady-state offset and local stability structural** | Substituting random coefficients (σ up to 100) leaves the closed-loop spectral radius unchanged to ~1e-10 |
| KAN support selection is **not measurably better** than direct sparse regression | Across 11 term budgets × 2 regimes the two curves sit within a few tenths of a percentage point of each other, on every one of 5 training seeds |
| The benchmark **cannot justify distillation** | With a horizon that spans the inverse response the MPC beats a well-tuned gain-scheduled LQR by only 6–9 %, so there is little for any approximator to lose |

Deployed law: **4 terms/pump (MP, 38 multiply–accumulates, 6.40 % nMAE)** and
**48 terms/pump (NMP, 211, 9.60 %)**, certified locally exponentially stable
(ρ ≤ 0.9988) at every set-point tested, 96.5–100 % stable under ±20 % perturbation
of valve ratios and pump gains and 95–97 % with outlet areas, actuator delay and
measurement noise perturbed as well. Driven against its level limit, it stays clear
inside its training envelope — because of the level fade, added after the law was found
overflowing there — and overflows outside it, as does every controller in the comparison
except the online MPC. That last result is reported as a limitation, not smoothed over.

### Two benchmark properties the design turns on

Both are established by `s00_horizon_study.py` and Section 3 of the manuscript before any
distillation result is reported, because either one, got wrong, prevents the closed loop
from converging at all — and no approximation of a teacher that does not converge means
anything.

* **Not every commanded target is an equilibrium.** With two pumps the equilibrium set is
  two-dimensional: fixing `(h1, h2)` determines `h3, h4`. In the NMP regime the
  equilibrium at (10, 10) is `[10, 10, 4.16, 3.44]` cm, so a target such as
  `[10, 10, 2, 2]` is unreachable and *no* controller — the MPC included — can drive its
  cost to zero. Every reference used here is an exact solution of the equilibrium
  equations.
* **The prediction horizon must span the inverse response.** The right-half-plane zero
  has a 69 s time constant. At N = 30, a prediction step of Δt_p = 0.1 s gives a
  **3-second** horizon, and a myopic controller on a non-minimum-phase plant moves in the
  wrong direction: the loop drifts away from a reachable equilibrium and command activity
  rises to 439–714 V of total variation. Δt_p = 4 s spans 120 s and fixes both, with the
  control loop still running at 0.1 s.

---

## Repository layout

| Path | Status |
|---|---|
| `QuadTank_Project/revision/` | **Current.** The pipeline behind the manuscript — see its [README](QuadTank_Project/revision/README.md). |
| `QuadTank_Project/quad_tank_golden_reference_P_minus.csv` | **Current dependency.** A single closed-loop trajectory at a fixed reference, whose rank deficiency the identifiability analysis in `s03` diagnoses. |
| `QuadTank_Project/*.ipynb`, `Data/`, `Nucleo_MPC_GenFinal/`, other `*.csv` and `*.pdf` | **Superseded.** Retained for provenance only. |
| `legacy_photos/` | **Legacy.** Photographs of a hardware-in-the-loop bench and IDE from earlier exploratory work. Kept as a record; no result in the current paper depends on them. |

### On the superseded material

The notebooks, the generated OSQP C solver and the hardware-in-the-loop logs predate the
current pipeline. They are kept for provenance, not because they produce any current
result: nothing under `revision/` imports them, with the single exception of the
single-trajectory CSV noted above.

The STM32 firmware (`QT_HIL_Clean/`) has been deleted outright — nothing referenced it
once the hardware campaign left the paper. It remains recoverable from the git history
(`git log --all -- QT_HIL_Clean`).

The hardware campaign is not part of the study. Its latency and flash figures were
measured for a different symbolic law than the one now deployed, and the microcontroller
was not available for re-measurement, so retaining them would have mixed measured numbers
with carried-over ones. Computational cost is instead reported as an exact
multiply–accumulate count, which is a property of the control law rather than of a
processor. **Any speed-up figure appearing in the git history of this file is
withdrawn.**

---

## Changes after submission

The manuscript was submitted on 5 August 2026. The tag
[`submitted-jpc-2026-08-05`](https://github.com/cl0sure6/QuadTank_KAN_and_LTV-MPC/tree/submitted-jpc-2026-08-05)
marks the repository state at that moment.

That tag is provenance, not a reproduction target: at that commit the pipeline did
**not** produce every number in the submitted PDF. Auditing the manuscript's numeric
claims against the result files afterwards turned up one gap and three errors, all
corrected on `master`:

* **Added `s00_horizon_study.py`.** The prediction-horizon study behind Section 4.1 and
  the total-variation comparison in Section 6.2 were not produced by any script. They
  now are, along with the transmission zeros.
* **Corrected the 3-second-horizon total variation** to 439–714 V, the value the stage
  reproduces on the two tasks that section compares. The submitted PDF says 439–818 V.
* **Corrected two transmission zeros**, which had been rounded up: −0.0193 (MP) and
  +0.0145 (NMP), not −0.0194 and +0.0146. They are computed here, not taken from the
  benchmark's source, and the manuscript now says so.
* **Fixed the deployed term count** in Figure 3 and Table 3, which read "24 terms" and
  labelled a two-regime row with one regime's budget. It is 4 terms/pump (MP) and 48
  (NMP).

None of these changes a conclusion. They are recorded here so that anyone comparing the
released code against the submitted PDF can see precisely what differs and why.

### Revision (JPROCONT-D-26-00790R1)

The revised manuscript was submitted on 4 October 2026. The tag
[`submitted-jpc-r1-2026-10-04`](https://github.com/cl0sure6/QuadTank_KAN_and_LTV-MPC/tree/submitted-jpc-r1-2026-10-04)
marks the corresponding state, and unlike the first tag it **is** a reproduction target:
running the stages at that commit regenerates every number in the revised PDF, all of
them produced by a single pass of the pipeline. Relative to the first submission:

* **Level fade.** `policyform.py` multiplies the learned term by `g(x)`, which hands
  authority back to the LQR core as a tank approaches its limit. It was added after
  `s10` found the deployed law overflowing a tank inside its own training envelope. The
  structural properties (P1)/(P2) are unchanged and the coefficients are bit-identical;
  the operation count rises from 30 to 38 (MP) and 203 to 211 (NMP).
* **Cross-channel shape constraints** in the non-minimum-phase regime, where `s04c`
  shows the teacher's cross gains are large and sign-definite. They are left free in the
  minimum-phase regime, where they are neither.
* **Realizable constraint sampling.** `symbolic.constraint_points` draws `(x, r)` pairs
  with `r` an exact equilibrium. The previous independent draw implied references
  outside the tank for 33.9 % of points.
* **New stages.** `s04c` (own- versus cross-channel derivatives, law versus teacher),
  `s09` (five training seeds: the deployed law is identical on all of them), `s10`
  (state constraint made active; robustness widened to outlet areas, actuator delay and
  measurement noise) and `s02c` (re-scores stored models when the policy skeleton
  changes, without retraining).

One correction to the commit history, since commit messages are public: the message of
`f4542c5` states that KAN training is not reproducible at a fixed seed. That claim rests
on a measurement taken under concurrent load and was withdrawn in `7e5a04e`; on an idle
machine the seed-42 retrain reproduces the baseline to the precision of the result files.

---

## Reproducing

```bash
cd QuadTank_Project/revision
pip install -r ../../requirements.txt
python s00_horizon_study.py        # ~2 min   teacher's prediction horizon, transmission zeros
python s01_generate_dataset.py     # ~7 min   datasets, both regimes
python s02_train_models.py         # ~20 min  KAN, MLP, DeepONet, polynomial baselines
python s02c_reevaluate.py          # <1 min   re-score stored models under the current skeleton
python s02b_sparsify.py            # ~10 min  term-budget sweep, certified selection
python s03_signflip_analysis.py    # ~2 min   root cause of the sign anomaly
python s04_stability.py            # ~12 min  certificate, ROA, Monte-Carlo, constraints
python s04b_invariance.py          # ~1 min   structural properties (P1), (P2)
python s05_scenarios.py            # ~12 min  closed-loop scenario campaign
python s06_make_figures.py         # ~1 min   figures
python s07_make_tables.py          # <1 min   LaTeX tables
python s08_key_numbers.py          # <1 min   every number quoted in the prose
python s04c_cross_channel.py       # ~3 min   own- vs cross-channel derivatives, law vs teacher
python s09_seed_campaign.py        # ~2.5 h   how much the result depends on the training seed
python s10_constraint_robustness.py# ~20 min  active state constraint + widened robustness
```

Stage 0 runs first because every later stage inherits the horizon it settles; it is the
only stage that varies `dt_pred` away from the deployed `qtlib.DT_PRED = 4.0 s`.

Results land in `revision/results/`. `key_numbers.json` collects every quantity the
manuscript states in text, and `horizon_study.json` backs the horizon argument and the
transmission zeros.

Questions and issues: open a GitHub issue, or contact **@cl0sure6**.
