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
Symbolic read-out utilities.

The manuscript's methodological contribution lives here.  Rather than deploying
whatever pykan's `auto_symbolic` happens to return, the pipeline is split into
two clearly separated steps:

  (1) STRUCTURE SELECTION.  The trained KAN is pruned and each surviving
      univariate activation is replaced by the best symbolic candidate from a
      declared library.  Expanding the resulting composition gives a *sparse
      monomial support* S -- this is what the KAN contributes, and it is the
      only thing taken from the network.

  (2) SHAPE-CONSTRAINED COEFFICIENT FIT.  The coefficients on S are then
      re-estimated by a convex quadratic program

          min_c || A(S) c - u_mpc ||^2
          s.t.   d/de_j [ c^T m(phi) ] (phi^(l)) >= 0   for l = 1..L

      i.e. the closed-loop-relevant monotonicity (negative feedback) of the law
      is imposed as a *linear inequality* on the coefficients, evaluated on a
      dense sample of the operating box.  The program is convex, so its
      solution is unique and requires no expert intervention -- this replaces
      the manual sign correction of the original submission.

Both the unconstrained and the constrained fit are reported so that the price
of the guarantee is visible.
"""

from __future__ import annotations

import numpy as np
import sympy

NVAR = 8
XS = sympy.symbols("x_1:9")


def formula_support(expr, max_terms=4000):
    """Expanded monomial support (list of exponent tuples) of a sympy expression."""
    e = sympy.expand(sympy.sympify(expr))
    p = sympy.Poly(e, *XS)
    monoms = [tuple(int(v) for v in m) for m in p.monoms()]
    if len(monoms) > max_terms:
        raise ValueError(f"support too large: {len(monoms)}")
    return sorted(set(monoms))


def design(F, support):
    """Monomial design matrix; F is (n, 8)."""
    F = np.asarray(F, float)
    A = np.empty((len(F), len(support)))
    for c, m in enumerate(support):
        col = np.ones(len(F))
        for i, p in enumerate(m):
            if p:
                col = col * F[:, i] ** p
        A[:, c] = col
    return A


def d_design(F, support, var):
    """Partial derivative of the monomial design matrix w.r.t. feature `var`."""
    F = np.asarray(F, float)
    A = np.zeros((len(F), len(support)))
    for c, m in enumerate(support):
        p = m[var]
        if p == 0:
            continue
        col = float(p) * np.ones(len(F))
        for i, q in enumerate(m):
            e = q - 1 if i == var else q
            if e:
                col = col * F[:, i] ** e
        A[:, c] = col
    return A


def fit_unconstrained(A, y, ridge=1e-8):
    n = A.shape[1]
    return np.linalg.solve(A.T @ A + ridge * np.eye(n), A.T @ y)


def fit_shape_constrained(A, y, G=None, g0=None, Aeq=None, Wk=None, kt=None,
                          lam=0.0, ridge=1e-8, solver=None):
    """
    min ||A c - y||^2 + ridge||c||^2
    s.t.  G c >= 0        (negative-feedback shape constraint)
          Aeq c == 0      (equilibrium consistency, see below)

    Both constraint families are LINEAR in the coefficients, which is the whole
    point of reading the network out as a polynomial:

      * G collects rows d/de_j of the monomial basis at sample points, so
        G c >= 0 states that a positive level error never reduces the
        corresponding pump command;
      * Aeq collects rows m(phi) evaluated at (x_eq(r), r) for a set of
        reachable references, so Aeq c == 0 states that the learned *deviation*
        from the analytic feed-forward input vanishes exactly at every
        equilibrium.  Together with u = u_eq(r) + dpi this removes the
        steady-state offset that a static approximate policy would otherwise
        inherit from its imitation error.

    Neither constraint is expressible for a general multilayer perceptron: they
    are available precisely because the read-out is linear in its parameters.
    The programme is convex, so the solution is unique.
    """
    import cvxpy as cp
    n = A.shape[1]
    c = cp.Variable(n)
    expr = cp.sum_squares(A @ c - y) / max(len(A), 1) + ridge * cp.sum_squares(c)
    if lam > 0 and Wk is not None and len(Wk):
        expr = expr + lam * cp.sum_squares(Wk @ c - kt) / max(len(Wk), 1)
    obj = cp.Minimize(expr)
    cons = []
    if G is not None and len(G):
        cons.append(G @ c >= (0 if g0 is None else -np.asarray(g0).ravel()))
    if Aeq is not None and len(Aeq):
        cons.append(Aeq @ c == 0)
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=solver or cp.CLARABEL)
    except Exception:
        try:
            prob.solve(solver=cp.OSQP, max_iter=60000, eps_abs=1e-7, eps_rel=1e-7)
        except Exception:
            return fit_unconstrained(A, y, ridge), False
    if c.value is None:
        return fit_unconstrained(A, y, ridge), False
    return np.asarray(c.value).ravel(), True


def _eq_refs(reg, n_ref, box, seed):
    import qtlib as Q
    rng = np.random.default_rng(seed)
    out, tries = [], 0
    while len(out) < n_ref and tries < 20 * n_ref:
        tries += 1
        h1, h2 = rng.uniform(*box, 2)
        try:
            out.append(Q.equilibrium(h1, h2, reg))
        except ValueError:
            continue
    return out


def gain_rows(support, reg, dt, n_ref=24, box=(7.0, 15.0), seed=1):
    """
    Rows and targets that pin the LOCAL FEEDBACK GAIN of the symbolic law to the
    LQR gain of the same cost, at a set of reachable equilibria.

    The deployed law is  u = u_eq(r) + U_max * c^T m(phi(x, r)),  with
    phi_i = x_i / X_n and phi_{i+4} = (r_i - x_i) / X_n.  Hence

        d u_j / d x_i = (U_max / X_n) * c_j^T [ dm/dphi_i - dm/dphi_{i+4} ],

    which is linear in c_j.  Requiring d u_j / d x_i = -K_ji, with K the
    discrete LQR gain at that equilibrium, is therefore a linear condition, and
    a closed loop whose linearisation is A - BK is Schur stable by LQR theory.
    It is imposed as a weighted penalty rather than a hard equality so that the
    global imitation fit is not destroyed; the weight is chosen by the smallest
    value that certifies stability (stage 2b).

    Returns (W, k_target) for one output index at a time, stacked over
    references and states.
    """
    import qtlib as Q
    W = {0: [], 1: []}
    T = {0: [], 1: []}
    for x_eq, u_eq in _eq_refs(reg, n_ref, box, seed):
        try:
            K, _ = Q.gs_lqr_gain(reg, x_eq, dt)
        except Exception:
            continue
        F = Q.features(x_eq, x_eq)
        rows = [(d_design(F, support, i) - d_design(F, support, i + 4)).ravel()
                * (Q.NORM_U / Q.NORM_X) for i in range(4)]
        for j in (0, 1):
            for i in range(4):
                W[j].append(rows[i])
                T[j].append(-K[j, i])
    return ({j: np.array(W[j]) for j in (0, 1)},
            {j: np.array(T[j]) for j in (0, 1)})


def equilibrium_rows(support, reg, n_ref=120, box=(7.0, 15.0), seed=0):
    """
    Design rows m(phi) at (x_eq(r), r) for reachable references r, i.e. the
    states at which the learned deviation must vanish.
    """
    import qtlib as Q
    F = [Q.features(x_eq, x_eq).ravel() for x_eq, _ in _eq_refs(reg, n_ref, box, seed)]
    return design(np.array(F), support) if F else None


def constraint_points(F, rng, n_grid=800, reg=None, box=(7.0, 15.0)):
    """Points at which the shape constraint is imposed.

    Half are resampled training points. The other half used to be drawn i.i.d.
    from the observed feature box, which is a product set and therefore contains
    combinations the plant cannot occupy: phi carries the levels x/X_n and the
    errors (r-x)/X_n together, so an independent draw can pair a nearly full
    tank with a large positive error, implying a reference far outside the tank.
    Enforcing a sign condition at such a point costs imitation accuracy and buys
    nothing, because the closed loop never visits it.

    When `reg` is supplied the synthetic half is instead generated as realizable
    (x, r) pairs: r is an exact equilibrium of the plant and x is a state drawn
    around it, so every constraint point corresponds to an operating condition
    the controller can actually meet.
    """
    idx = rng.choice(len(F), size=min(n_grid, len(F)), replace=False)
    if reg is None:
        lo, hi = F.min(0), F.max(0)
        grid = rng.uniform(lo, hi, size=(n_grid, F.shape[1]))
        return np.vstack([F[idx], grid])

    import qtlib as Q
    lo, hi = F.min(0), F.max(0)
    rows, tries = [], 0
    while len(rows) < n_grid and tries < 40 * n_grid:
        tries += 1
        h1, h2 = rng.uniform(*box, 2)
        try:
            x_eq, _ = Q.equilibrium(h1, h2, reg)
        except ValueError:
            continue
        # tracking errors of the size the closed loop actually produces
        x = x_eq + np.concatenate([rng.uniform(-4.0, 4.0, 2),
                                   rng.uniform(-2.0, 2.0, 2)])
        x = np.clip(x, Q.H_MIN + 0.5, Q.H_MAX - 0.5)
        rows.append(Q.features(x, x_eq).ravel())
    grid = np.clip(np.array(rows), lo, hi)
    return np.vstack([F[idx], grid])


def flops(support):
    """
    Multiply-accumulate count for evaluating the monomial expansion with a
    shared power table (x_i^2..x_i^dmax computed once, then one multiply per
    factor per monomial plus one multiply-add per term).
    """
    dmax = [max((m[i] for m in support), default=0) for i in range(NVAR)]
    powers = sum(max(0, d - 1) for d in dmax)
    per_term = sum(max(0, sum(1 for p in m if p > 0) - 1) + 1 for m in support)
    return int(powers + per_term)


def sign_violation(c, F, support, var, thresh=0.0):
    """Fraction of sample points where d u / d e_var < thresh."""
    D = d_design(F, support, var)
    return float((D @ c < thresh).mean())


# --------------------------------------------------- joint stable read-out
def stability_data(subs, reg, dt, refs):
    """
    Everything needed to write the closed-loop linearisation as an affine
    function of the read-out coefficients, at each reference in `refs`.

    Returns a list of dicts with A_p (4x4), B_p (4x2), P (4x4, from the LQR
    Riccati equation at that equilibrium) and, for each output j and state i,
    the row w[j][i] such that  d u_j / d x_i = w[j][i] . c_j.
    """
    import qtlib as Q
    out = []
    for (h1, h2) in refs:
        try:
            x_eq, _ = Q.equilibrium(h1, h2, reg)
            _, P = Q.gs_lqr_gain(reg, x_eq, dt)
        except Exception:
            continue
        F = Q.features(x_eq, x_eq)
        w = {}
        for j in (0, 1):
            w[j] = [(d_design(F, subs[j], i) - d_design(F, subs[j], i + 4)).ravel()
                    * (Q.NORM_U / Q.NORM_X) for i in range(4)]
        # the LMI is homogeneous in P, so P is normalised purely for
        # conditioning: the Riccati solution has entries of order 1e3-1e5,
        # which otherwise swamps the coefficient scale in the conic solver
        P = P / np.linalg.norm(P, 2)
        out.append({"A_p": np.eye(4) + dt * Q.jac_A_cont(x_eq, reg),
                    "B_p": dt * Q.B_cont(reg), "P": P, "w": w, "ref": (h1, h2)})
    return out


def fit_joint_stable(A_tr, Y, G, Aeq, sdata, rho_max=0.999, ridge=1e-8,
                     solver=None):
    """
    Joint read-out for both pump commands under three families of constraints,
    all convex in the coefficients:

        min_{c_1,c_2}  sum_j ||A_j c_j - y_j||^2 / N  + ridge ||c||^2
        s.t.  G_j c_j >= 0                       (negative feedback)
              Aeq_j c_j = 0                      (equilibrium consistency)
              [[rho^2 P,  Acl' P],
               [ P Acl ,     P  ]] >= 0          (Schur stability at each ref)

    The last family is a linear matrix inequality: for a *fixed* Lyapunov
    matrix P the Schur complement of  Acl' P Acl <= rho^2 P  is linear in Acl,
    and Acl = A_p + B_p M(c) is affine in c because the law is a polynomial.
    Certified local exponential stability with decay rate rho is therefore
    obtained by construction rather than checked afterwards -- which is only
    possible because the controller is available in closed form.
    """
    import cvxpy as cp
    cs = [cp.Variable(A_tr[j].shape[1]) for j in (0, 1)]
    n = max(len(A_tr[0]), 1)
    obj = sum(cp.sum_squares(A_tr[j] @ cs[j] - Y[:, j]) / n for j in (0, 1))
    obj = obj + ridge * sum(cp.sum_squares(cs[j]) for j in (0, 1))
    cons = []
    for j in (0, 1):
        if G[j] is not None and len(G[j]):
            cons.append(G[j] @ cs[j] >= 0)
        if Aeq[j] is not None and len(Aeq[j]):
            cons.append(Aeq[j] @ cs[j] == 0)
    for d in sdata:
        rows = [cp.hstack([d["w"][j][i] @ cs[j] for i in range(4)]) for j in (0, 1)]
        M = cp.vstack(rows)                      # 2 x 4, affine in c
        Acl = d["A_p"] + d["B_p"] @ M            # 4 x 4, affine in c
        P = d["P"]
        blk = cp.bmat([[rho_max ** 2 * P, Acl.T @ P],
                       [P @ Acl, P]])
        cons.append(blk >> 1e-10 * np.eye(8))
    prob = cp.Problem(cp.Minimize(obj), cons)
    for slv in ([solver] if solver else [cp.CLARABEL, cp.SCS]):
        try:
            prob.solve(solver=slv)
            if cs[0].value is not None and cs[1].value is not None:
                return [np.asarray(c.value).ravel() for c in cs], True
        except Exception:
            continue
    return None, False



def total_shape_rows(F, support, j, regime, i=None):
    """
    Rows and offsets of the negative-feedback constraint for the FULL deployed
    law of policyform.py,

        u_j = u_eq,j(r) + (K e)_j + w(e) * U_max * c_j' m(phi),

    differentiated with respect to phi_{i+4} = e_i / X_n:

        d u_j / d phi_{i+4}
            = X_n K_ji  +  U_max [ (dw/dphi_{i+4}) m(phi)' c_j
                                 + w(phi) (dm/dphi_{i+4})' c_j ].

    The first term is a constant offset and the rest is linear in c_j, so the
    requirement  d u_j / d phi_{i+4} >= 0  is the affine inequality
    G c_j >= -g0.

    `i` defaults to `j`, giving the own-channel constraint. Passing i != j gives
    the cross-channel constraint, which is affine in exactly the same way.

    The level fade g(x) of policyform is deliberately NOT included. It depends on
    the levels only, so it scales this derivative by g in [0, 1] without touching
    its structure, and because the offset X_n K_ji is strictly positive on every
    channel constrained here, g0 + g (G c) >= min(g0, g0 + G c): imposing the
    constraint at g = 1 is the conservative case and implies it for all g.
    """
    import policyform as PF
    import qtlib as Q
    F = np.asarray(F, float).reshape(-1, NVAR)
    i = j if i is None else i
    v = 4 + i
    w = PF.gate(F)[:, None]
    dw = PF.dgate(F, v)[:, None]
    M = design(F, support)
    dM = d_design(F, support, v)
    G = Q.NORM_U * (dw * M + w * dM)
    g0 = np.full(len(F), Q.NORM_X * PF.lqr_gain(regime)[j, i])
    return G, g0


# Channels whose CROSS derivatives are additionally constrained, per regime.
# In the non-minimum-phase configuration the valve split sends most of each
# pump's flow to the *other* upper tank, so the teacher's cross response is
# large and strictly positive over the operating box (stage 4c). Constraining
# it there is physically justified and costs almost nothing, because the
# unconstrained read-out already nearly satisfies it. In the minimum-phase
# configuration the teacher's cross gains are two orders of magnitude smaller
# than its own-channel gains and change sign, so their sign carries no physical
# requirement and is deliberately left free.
CROSS_CONSTRAINED = {"MP": False, "NMP": True}


def shape_rows_for_output(F, support, j, regime):
    """Stacked shape-constraint rows for pump j: own channel always, cross
    channel too where the regime justifies it."""
    G, g0 = total_shape_rows(F, support, j, regime)
    if not CROSS_CONSTRAINED.get(regime, False):
        return G, g0
    blocks, offs = [G], [g0]
    for i in (0, 1):
        if i == j:
            continue
        Gi, gi = total_shape_rows(F, support, j, regime, i=i)
        blocks.append(Gi)
        offs.append(gi)
    return np.vstack(blocks), np.concatenate(offs)


def total_sign_violation(c, F, support, j, regime, i=None):
    G, g0 = total_shape_rows(F, support, j, regime, i=i)
    return float(((G @ c + g0) < 0).mean())
