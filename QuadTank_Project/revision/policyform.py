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

r"""
Policy parameterisation shared by every learned surrogate.

The deployed controller is

    u = sat[ u_eq(r)  +  K e  +  g(x) w(e) * U_max * f(phi(x, r)) ],
        \_______/     \___/     \___________________________________/
        feed-forward  LQR core        learned nonlinear correction

with  e = r - x,  phi = [x/X_n, e/X_n],  the set-point gate

    w(e) = min( ||e||^2 / s^2 , 1 )

and the level fade

    g(x) = clip( (H_max - max_i x_i) / (H_max - H_on), 0, 1 ).

Each term has a job and the split is what makes the controller certifiable:

* u_eq(r) is the exact steady-state input of the plant (qtlib.u_eq_of_ref) --
  two square roots and four multiply-accumulates.  It removes the steady-state
  offset that a static approximate policy would otherwise inherit from its
  imitation error.

* K is the discrete LQR gain of the *same* cost (Q, R) used by the MPC teacher,
  evaluated at the nominal equilibrium of the regime and then held constant.

* w(e) vanishes quadratically as e -> 0, so both the value and the Jacobian of
  the learned correction vanish at every reachable set-point, for any f
  whatsoever.  Consequently the closed-loop linearisation at a set-point is
  exactly A(r) - B K, independent of what was learned: local exponential
  stability is a *structural* property of the parameterisation, not something
  the fit has to be constrained into.  The learned term is free to model the
  predictive, constraint-aware behaviour that distinguishes MPC from LQR, which
  is precisely the behaviour that matters away from the set-point.

* g(x) hands authority back to the LQR core as any level approaches the overflow
  limit.  It is the second structural device in the parameterisation and it is
  argued for in exactly the same way as the first: because w and its gradient
  already vanish at the set-point, multiplying the learned term by ANY bounded
  factor leaves the two properties above untouched, so g buys a safety property
  without spending any of the guarantees.  It is not a constraint-satisfaction
  certificate -- there is no invariance argument here -- and the paper does not
  claim one.

Setting LQR_CORE = False and GATE = False recovers the plain "learn u directly"
formulation of the original submission; both are reported in the ablation.
"""

from __future__ import annotations

import numpy as np

import qtlib as Q

FEEDFORWARD = True
LQR_CORE = True
GATE = True
GATE_SCALE = 0.25          # s in w(e) = min(||e/X_n||^2 / s^2, 1)
LQR_REF = (10.0, 10.0)     # nominal equilibrium at which K is computed
DT_LQR = 0.1
# The core is deliberately DETUNED relative to the teacher's own R = 1e-3.
# With the teacher's weights the unconstrained LQR gain exceeds 100 V/cm and
# saturates on any appreciable error; even R = I saturates through the whole
# transient, which would leave the learned term with nothing to do and make
# every surrogate behave identically.  R_CORE = 10 I gives a gentle stabilising
# gain of order 1 V/cm: enough to own the linearisation at the set-point (and
# hence the stability certificate), small enough that the transient is governed
# by the learned term.
R_CORE = np.diag([10.0, 10.0])

REGIMES = {"MP": Q.MP, "NMP": Q.NMP}
_K_CACHE = {}
_BOX_CACHE = {}

# A polynomial extrapolates catastrophically outside the region it was fitted
# on, and the quadruple tank is easy to drive there (a large set-point change
# from a nearly full tank).  The deployed law is therefore evaluated on
# features saturated to the box spanned by its training data: outside that box
# the law is frozen at the boundary rather than extrapolated.  The box contains
# every reachable equilibrium in its interior, so the structural properties
# (P1) and (P2) are unaffected.  Bounded-activation networks (tanh MLP,
# DeepONet) are inherently bounded and need no such guard.
CLIP_FEATURES = True

# Clipping keeps the read-out bounded, but it freezes it: outside the box the
# polynomial holds its boundary value and stops responding to the state.  On a
# large fill that is a failure mode of its own -- the LQR core drives its command
# negative to shut the pump while the frozen correction holds a large positive
# value and overrides it, and a tank overflows.  The fade below returns authority
# to the core as any level approaches the overflow limit.
#
# It multiplies the learned term only, so it inherits the gate's argument
# wholesale: w(e) and its gradient already vanish at every reachable set-point,
# so for ANY bounded g the product g*w*f and its Jacobian vanish there too, and
# (P1)/(P2) are untouched.  H_FADE_ON is set above every equilibrium level
# reachable from the training reference box -- 15.0 cm at the box's own top,
# r = 15 cm -- so g == 1 on the whole region the stability certificate covers.
# The fade is inert there by construction rather than by tuning.
#
# It also cannot create a negative-feedback violation.  The constrained fit
# guarantees  d u_j / d e_i = X_n K_ji + U_max (G c)_ji >= 0  with X_n K_ji > 0
# on every constrained channel; the faded law gives X_n K_ji + g * U_max (G c)_ji,
# which is >= the former when (G c)_ji < 0 and >= X_n K_ji > 0 otherwise.
LEVEL_FADE = True
H_FADE_ON = 17.0           # g = 1 below this level, 0 at Q.H_MAX

# u_eq: 2 sqrt + 4 multiply-add;  K e: 8 multiply-add;  gate: 4 mul + 3 add + 1 div;
# level fade: 3 compare + 1 sub + 1 mul by a precomputed reciprocal + 2 clamp
# compares + 1 mul to fold it into the gate
CORE_FLOPS = 2 + 4 + 8 + 8 + 8


def lqr_gain(regime):
    """Constant discrete LQR gain of the teacher's cost at the nominal point."""
    if regime not in _K_CACHE:
        reg = REGIMES[regime]
        x_eq, _ = Q.equilibrium(LQR_REF[0], LQR_REF[1], reg)
        K, _ = Q.gs_lqr_gain(reg, x_eq, DT_LQR, R=R_CORE)
        _K_CACHE[regime] = K
    return _K_CACHE[regime]


def feature_box(regime):
    """Min/max of every policy feature over the regime's training split."""
    if regime not in _BOX_CACHE:
        import os as _os
        f = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                          "data", f"dataset_{regime}.npz")
        d = np.load(f, allow_pickle=True)
        F = d["feat"][d["split"] == "train"]
        _BOX_CACHE[regime] = (F.min(0), F.max(0))
    return _BOX_CACHE[regime]


def clip_features(feat, regime):
    if not CLIP_FEATURES:
        return np.asarray(feat, float).reshape(-1, 8)
    lo, hi = feature_box(regime)
    return np.clip(np.asarray(feat, float).reshape(-1, 8), lo, hi)


def gate(feat):
    """w(e), the factor that makes the learned correction vanish at e = 0."""
    feat = np.asarray(feat, float).reshape(-1, 8)
    if not GATE:
        return np.ones(len(feat))
    e = feat[:, 4:]
    return np.minimum(np.sum(e ** 2, axis=1) / GATE_SCALE ** 2, 1.0)


def dgate(feat, v):
    """d w / d phi_v  (zero for level features and in the saturated region)."""
    feat = np.asarray(feat, float).reshape(-1, 8)
    if not GATE or v < 4:
        return np.zeros(len(feat))
    e = feat[:, 4:]
    active = (np.sum(e ** 2, axis=1) / GATE_SCALE ** 2) < 1.0
    return np.where(active, 2.0 * feat[:, v] / GATE_SCALE ** 2, 0.0)


def level_fade(x):
    """g(x): 1 while every level is below H_FADE_ON, 0 at the overflow limit.

    Depends on the levels only, never on the tracking error, which is what makes
    it neutral for the shape constraint: d/de_i of the learned term is simply
    scaled by g >= 0, so no sign can be flipped.
    """
    x = np.asarray(x, float).reshape(-1, 4)
    if not LEVEL_FADE:
        return np.ones(len(x))
    return np.clip((Q.H_MAX - x.max(1)) / (Q.H_MAX - H_FADE_ON), 0.0, 1.0)


def _core(ref, x, regime):
    """u_eq(r) + K e, in volts."""
    ref = np.asarray(ref, float).reshape(-1, 4)
    x = np.asarray(x, float).reshape(-1, 4)
    reg = REGIMES[regime]
    u = np.zeros((len(ref), 2))
    if FEEDFORWARD:
        u = u + Q.u_eq_of_ref(ref, reg)
    if LQR_CORE:
        u = u + (ref - x) @ lqr_gain(regime).T
    return u


def encode_targets(u, x, ref, regime):
    """
    Learning target: the part of the MPC command that the core does not already
    explain, un-gated so that the surrogate learns f itself.
    """
    u = np.asarray(u, float).reshape(-1, 2)
    resid = (u - _core(ref, x, regime)) / Q.NORM_U
    return resid


def gate_of(x, ref):
    return gate(Q.features(x, ref))


def decode(y, x, ref, regime, clip=True):
    """Map a surrogate output back to a pump command.

    The fade is applied here, in the shared skeleton, rather than inside the
    symbolic law: every surrogate in the comparison is deployed through the same
    parameterisation and only f differs, so a safety device that belonged to one
    of them would make the comparison meaningless.
    """
    y = np.asarray(y, float).reshape(-1, 2)
    w = (gate(Q.features(x, ref)) * level_fade(x)).reshape(-1, 1)
    u = _core(ref, x, regime) + w * y * Q.NORM_U
    return np.clip(u, Q.U_MIN, Q.U_MAX) if clip else u
