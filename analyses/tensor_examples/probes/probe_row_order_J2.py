#!/usr/bin/env python3
r"""Genuine J > 1 row-label test, by closed form.

The report's §4 claimed the phi f2 channels test the lambda row order at J = 2
and J = 4.  That is WRONG: phif222/phif243 have top **L** = 2/4, but the mother
particle is psi with J = 1, so the last axis still has 3 rows.  A referee caught
this.

The honest test is a chain whose TOP particle has J = 2.  Take

    X(J=2) -> 0 + 0

Then lam_b = lam_c = 0, so delta = 0, and the cascade reduces to the pure
Wigner-D

    A[lam] = D^{2*}_{lam, 0}(alpha, beta, 0) * H * K ,   H = 1 (l = 2, s = 0)

with the tf-pwa convention D^J_{m1,m2} = e^{-i m1 alpha} d^J_{m1,m2}(beta)
e^{-i m2 gamma}.  So row i must equal

    e^{+i * LAM_SLOTS[2][i] * alpha} * d^2_{LAM_SLOTS[2][i], 0}(beta)

times ONE global complex constant.  This pins the row -> lambda mapping
absolutely, with no reference to EPS_A.
"""
from __future__ import annotations

import os
import sys

import numpy as onp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_T = os.path.join(ROOT, "agent", "tensor", "tests")
if _T not in sys.path:
    sys.path.insert(0, _T)

from agent.tensor.kernels import helicity_cascade as HC             # noqa: E402
from agent.tensor.kernels import omega as RT                        # noqa: E402


def assoc_legendre(l, m, x):
    """P_l^m(x) with Condon-Shortley phase, by recurrence (no scipy)."""
    import math
    am = abs(int(m))
    pmm = 1.0
    if am > 0:
        somx2 = math.sqrt(max(0.0, 1.0 - x * x))
        fact = 1.0
        for _ in range(1, am + 1):
            pmm *= -fact * somx2
            fact += 2.0
    if l == am:
        return pmm
    pmmp1 = x * (2 * am + 1) * pmm
    if l == am + 1:
        return pmmp1
    pll = 0.0
    for ll in range(am + 2, l + 1):
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + am - 1) * pmm) / (ll - am)
        pmm, pmmp1 = pmmp1, pll
    return pll


def d_m0(J, m, beta):
    """d^J_{m,0}(beta) in the tf-pwa / Wigner convention.

        d^J_{m,0}(b) = sqrt((J-|m|)!/(J+|m|)!) * P_J^{|m|}(cos b)
        d^J_{-|m|,0}(b) = (-1)^{|m|} d^J_{|m|,0}(b)

    Independent of the kernel's own `small_d_matrix` table (verified against it
    to 2.8e-16 for J = 1..4).
    """
    import math
    am = abs(int(m))
    val = (math.sqrt(math.factorial(int(J) - am) / math.factorial(int(J) + am))
           * assoc_legendre(int(J), am, math.cos(float(beta))))
    if m < 0:
        val = ((-1) ** am) * val
    return val


def main():
    for JX in (1, 2, 3, 4):
        run_J(JX)
        print()


def run_J(JX):
    # build a J -> 0+0 chain, decay axis at a random angle per event
    rng = onp.random.default_rng(5)
    n = 64
    beta = rng.uniform(0.15, onp.pi - 0.15, n)
    alpha = rng.uniform(-onp.pi, onp.pi, n)
    mX = 2.0
    p = onp.zeros((n, 3, 4))
    p[:, 0, 0] = mX                                    # X at rest, (E,x,y,z)
    q = 0.5 * mX
    d = onp.stack([onp.sin(beta) * onp.cos(alpha),
                   onp.sin(beta) * onp.sin(alpha),
                   onp.cos(beta)], axis=1)
    p[:, 1, 0] = q
    p[:, 1, 1:] = q * d
    p[:, 2, 0] = q
    p[:, 2, 1:] = -q * d

    chain = {"top": "X",
             "nodes": {"X": {"J": JX, "mass": mX, "daughters": ["a", "b"]},
                       "a": {"J": 0, "mass": 0.0},
                       "b": {"J": 0, "mass": 0.0}},
             "index": {"a": 0, "b": 1},
             "finals": ["a", "b"]}
    # the kernel indexes `mom` from 0, so pass only the two daughters
    chain["index"] = {"a": 0, "b": 1}
    mom = onp.concatenate([p[:, 1:2, :], p[:, 2:3, :]], axis=1)   # (n,2,4)

    ang = HC.node_angles(mom, chain)
    A = HC.chain_amplitude(mom, chain, {"X": [(JX, 0)]}, ang=ang)
    print("=" * 84)
    print("J = %d -> 0 + 0 :   A shape = %s   lambda_order(%d) = %s"
          % (JX, A.shape, JX, HC.lambda_order(JX)))
    print("row i   LAM_SLOTS[J][i]   |r|med        |r/r0 - 1|max")
    worst = 0.0
    for i, lam in enumerate(HC.lambda_order(JX)):
        closed = onp.array([onp.exp(1j * lam * al) * d_m0(JX, lam, be)
                            for al, be in zip(ang["a"][0], ang["a"][1])])
        m = onp.abs(closed) > 1e-9 * onp.abs(closed).max()
        if m.sum() < 3:
            print("  %d      %+d              skip (%d usable)" % (i, lam, m.sum()))
            continue
        r = A[m, i] / closed[m]
        rn = r / r[0]
        worst = max(worst, float(onp.abs(rn - 1).max()))
        print("  %d      %+d              %.6f      %.3e"
              % (i, lam, onp.median(onp.abs(r)), onp.abs(rn - 1).max()))
    print("  => worst |r/r0 - 1| = %.3e   %s"
          % (worst, "OK（行 i 就是 lambda=LAM_SLOTS[J][i]）" if worst < 1e-10
             else "**FAIL**"))

    print("  对照：若把行 i 配 lam = reversed(LAM_SLOTS[J])")
    for i, lam in enumerate(reversed(HC.lambda_order(JX))):
        closed = onp.array([onp.exp(1j * lam * al) * d_m0(JX, lam, be)
                            for al, be in zip(ang["a"][0], ang["a"][1])])
        m = onp.abs(closed) > 1e-9 * onp.abs(closed).max()
        if m.sum() < 3:
            continue
        r = A[m, i] / closed[m]
        print("    row %d 配 lam=%+d   |r/r0 - 1|max = %.3e"
              % (i, lam, onp.abs(r / r[0] - 1).max()))


if __name__ == "__main__":
    main()
