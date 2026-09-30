# Reproducible pipeline for the manuscript

Every number, figure and table in the paper is produced by the numbered stage scripts
in this directory. Run them in order from here; each writes into `data/`, `models/`,
`results/` and, where relevant, straight into the LaTeX tree
(`../../../els-cas-templates/figs` and `.../tables`).

```bash
python s00_horizon_study.py        # ~2 min   settles the teacher's prediction horizon
python s01_generate_dataset.py     # ~7 min   distillation datasets, both regimes
python s02_train_models.py         # ~20 min  KAN, MLP, DeepONet, polynomial baselines
python s02c_reevaluate.py          # <1 min   re-score stored models under the current skeleton
python s02b_sparsify.py            # ~10 min  term-budget sweep + certified selection
python s03_signflip_analysis.py    # ~2 min   root cause of the sign anomaly
python s04_stability.py            # ~12 min  certificate, ROA, Monte-Carlo, constraints
python s04b_invariance.py          # ~1 min   structural properties (P1), (P2)
python s05_scenarios.py            # ~12 min  closed-loop scenario campaign
python s06_make_figures.py         # ~1 min   all figures
python s07_make_tables.py          # <1 min   all LaTeX tables
python s08_key_numbers.py          # <1 min   every number quoted in the prose
python s04c_cross_channel.py       # ~3 min   own- vs cross-channel derivatives, law vs teacher
python s09_seed_campaign.py        # ~2.5 h   how much the result depends on the training seed
python s10_constraint_robustness.py# ~20 min  active state constraint + widened robustness
```

`results/key_numbers.json` is the one file to look at if you want to check the
manuscript's claims against the code: it collects each quantity the text states.
Stage 0 runs first because everything downstream inherits the horizon it settles; its
`results/horizon_study.json` backs the horizon argument in Section 4.1 and the
total-variation comparison in Section 6.2. It is the only stage that varies `dt_pred`
away from the deployed `qtlib.DT_PRED = 4.0 s`.

## Modules

| File | Contents |
|---|---|
| `qtlib.py` | Plant, EKF, LTV-MPC teacher, equilibrium map, transmission zeros, LQR, metrics, closed-loop driver |
| `policyform.py` | The deployed skeleton `u = sat[u_eq(r) + K e + w(e)·U_max·f(φ)]` and its structural properties |
| `symbolic.py` | Monomial support extraction, design matrices, the shape-constrained convex read-out, operation counting |
| `verify.py` | Closed-loop Jacobian, fixed points, local stability certificate |
| `controllers.py` | Uniform `u = π(x, ref)` interface for every controller compared |

## The controller

```
u = sat[ u_eq(r)  +  K e  +  g(x) · w(e) · U_max · f(φ(x, r)) ]

w(e) = min(‖e‖²/s², 1)
g(x) = clip((H_max − max_i x_i) / (H_max − H_on), 0, 1),   H_on = 17 cm
```

* `u_eq(r)` — the exact steady-state input, two square roots and four multiply–adds
* `K` — a deliberately detuned LQR gain (`R_CORE = 10·I`), of order 1 V/cm
* `w(e)` — a gate vanishing quadratically at the set-point
* `g(x)` — a level fade returning authority to the core as any tank nears its limit

Because `w` **and its gradient** vanish at every reachable set-point, two properties
hold *for any learned `f` whatsoever*:

* **(P1)** the commanded input at the set-point is exactly `u_eq(r)`, so the set-point
  is an exact closed-loop fixed point — zero steady-state offset;
* **(P2)** the closed-loop linearisation there is exactly `A(r) − BK`, so local
  exponential stability follows from LQR theory, not from the quality of the fit.

`s04b_invariance.py` checks this by substituting random coefficients (σ up to 100) for
the trained read-out: the spectral radius moves by less than 1e-10.

Multiplying by `g` costs nothing here. The argument above uses no property of the learned
term except that `w` and its gradient vanish at the set-point, so it survives *any*
bounded factor. And because `H_on` lies above every equilibrium level reachable from the
training reference box, `g ≡ 1` over the whole region the certificate covers.

## Design decisions worth knowing

* **The prediction horizon must span the inverse response.** The right-half-plane zero
  has a 69 s time constant. The original study used N = 30 at Δt_p = 0.1 s — a 3-second
  horizon — which makes the teacher myopic, and a myopic controller on a
  non-minimum-phase plant moves the wrong way. `qtlib.DT_PRED = 4.0` gives a 120 s
  horizon while the control loop still runs at 0.1 s. A teacher that does not converge
  cannot be distilled into a student that does.

* **References are always reachable equilibria.** With two pumps the equilibrium set is
  two-dimensional, so commanding `(h1, h2)` determines `h3, h4` through
  `qtlib.equilibrium`. The original target `[10, 10, 2, 2]` is not an equilibrium in the
  NMP regime — which is why the earlier cost converged to a nonzero value and why two of
  its scenario figures were indistinguishable.

* **Plant parameters** follow Johansson (2000) Table 1. The original notebooks used
  `k = [2.826, 2.961]`, which matched neither that table nor the earlier paper's own
  parameter table; the revision uses the published values so code and manuscript agree.

* **Splits are by trajectory and block, never by row**, so no test sample is temporally
  adjacent to a training sample.

* **The KAN contributes only the monomial support.** Coefficients are always
  re-estimated by the convex program in `symbolic.fit_shape_constrained`, which imposes
  `∂u_j/∂e_j ≥ 0` as affine inequalities — this replaces the manual sign correction of
  the earlier work with a procedure that has a unique solution and no expert input.
  The seed campaign (`s09`) shows why the support is the durable part: the network's
  own error spans 4 pp across seeds while the deployed law is identical on all of them.

* **Cross-channel constraints are imposed in NMP only.** `∂u_j/∂e_i` for `i ≠ j` is
  affine in the coefficients in exactly the same way, so adding it costs nothing
  structurally. Whether it *should* be added is a question about the plant, settled by
  `s04c`: the teacher's cross gains change sign in MP and are two orders of magnitude
  below its diagonal there, but are large and strictly positive in NMP.

* **Constraints are sampled at realizable operating points.** `φ` carries levels and
  errors together, so drawing them independently from the observed feature box implies
  references outside the tank — 33.9 % of points did. `symbolic.constraint_points` now
  generates `(x, r)` pairs with `r` an exact equilibrium.

* **The deployed term budget is chosen by the stability certificate**, not by accuracy
  alone: `s02b` takes the smallest budget certified locally exponentially stable at all
  six test set-points whose error is within 1 pp of the best certified candidate.

* **Features are clipped to the training box** before the polynomial is evaluated. A
  degree-4 polynomial extrapolates catastrophically outside the region it was fitted on;
  without this guard the law overflowed a tank in 6 % of randomised tasks, with it in
  none. Every reachable equilibrium is interior to the box, so (P1) and (P2) are
  unaffected.

* **Clipping keeps the read-out bounded but also freezes it, which is why the fade
  exists.** On an aggressive fill `s10` found the frozen correction holding +13.6 V while
  the LQR core had already reached −8.2 V to shut the pump, and the tank overflowed —
  inside the training envelope, the only one of eight approximants to do so. `g(x)`
  removes that. The coefficients are nonetheless fitted as though `g ≡ 1`: folding the
  fade into the fit would let the learned term grow to compensate for it near the limit,
  which is the behaviour it exists to suppress. The shape constraint is posed at `g = 1`
  for the same reason, and that is the conservative case (see `symbolic.total_shape_rows`).

* **`s02b` is idempotent**: it re-derives the KAN support from the stored symbolic
  formulas rather than from its own previous output.

## What the results say

| | MP | NMP |
|---|---|---|
| spline KAN → MPC policy | 2.43 % | 4.37 % |
| after `auto_symbolic` | 7.78 % | 11.24 % |
| after convex refit | 5.51 % | 9.11 % |
| deployed law | 6.40 % (4 terms, 38 MACs) | 9.60 % (48 terms, 211 MACs) |
| spectral radius / ROA / Monte-Carlo stable | 0.9984 / 100 % / 96.5 % | 0.9988 / 85 % / 100 % |
| negative-feedback violation, unconstrained → constrained | 11.4 % → 0.6 % | 4.2 % → 0.6 % |

nMAE is relative to the full actuator range (12 V), predictions clipped to the actuator
box before scoring.

Two results are negative and are reported as such in the paper: the KAN-selected support
is not measurably better than one chosen directly by sparse regression at any term
budget or on any of five training seeds, and on this benchmark the MPC beats a well-tuned gain-scheduled LQR
by only 7–9 %, so the plant cannot demonstrate the value of approximating an optimiser.

## Scope

Everything here is software-in-the-loop. The hardware campaign of the earlier work has
been dropped from the paper: its latency and flash figures were measured for a different
symbolic law and the microcontroller was not available for re-measurement, so keeping
them would have mixed measured with carried-over numbers. Computational cost is reported
as an exact operation count (`symbolic.flops` plus `policyform.CORE_FLOPS`), a property
of the control law rather than of a processor.

## External dependency

`s03_signflip_analysis.py` reads `../quad_tank_golden_reference_P_minus.csv`, the
original single-trajectory distillation set. That file is the subject of the analysis —
its regressor has rank 5 of 9 — so it must stay where it is. Nothing else in this
directory reads anything outside it.
