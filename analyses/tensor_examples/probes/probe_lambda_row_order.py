#!/usr/bin/env python3
"""Row-order test for J > 1: does the cascade's last axis match EPS_A rows
for every lambda, not just J = 1?

The report's §4 argument is that the kernel's `LAM_SLOTS[J] = [-J..J]` axis is
ALREADY in `EPS_A` row order.  For J = 1 both lists have 3 entries so a
coincidence is conceivable; for J = 2 (phi f2) and J = 4 (phif243) the number
of rows differs (5 and 9 vs 3), so the test is sharper.

Method: for each cascade row i, find the EPS_A row j it matches (complex scale
absorbed) and report the mapping.
"""
from __future__ import annotations

import os
import sys

import numpy as onp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_TESTS = os.path.join(ROOT, "agent", "tensor", "tests")
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import test_ex3_channels as T                                       # noqa: E402
from agent.tensor import spec as S                                  # noqa: E402
from agent.tensor.kernels import covariant as COV                   # noqa: E402
from agent.tensor.kernels import helicity_cascade as HC             # noqa: E402
from agent.tensor.kernels.helicity import EPS_A                     # noqa: E402


def main():
    d = T.physical(n=24, seed=11)
    keys = ("Kp", "Km", "Pip", "Pim")
    mom = onp.stack([d[k][:, [3, 0, 1, 2]] for k in keys], axis=1)

    spec = S.load_spec(os.path.join(ROOT, "agent", "tensor", "specs",
                                    "phif2.toml"))
    chain = S.to_chain(spec)
    ls = {"phif201": [(0, 1)], "phif221": [(2, 1)], "phif222": [(2, 2)],
          "phif223": [(2, 3)], "phif243": [(4, 3)]}
    print("=" * 92)
    print("phi f2 (J_top = 1, 3 rows)  —— 这是唯一有 J=2 / J=4 的地方")
    print("  cascade 的末轴只有 3 行（顶层是 J=1 的 psi），所以这里验的是")
    print("  **内层** f2 的 5 个 lambda 槽有没有被正确缩并（不是行序本身）")
    for nm, l in ls.items():
        ang = HC.node_angles(mom, chain)
        A = HC.chain_amplitude(
            mom, chain, {"psi2s": l, "phi": [(1, 0)], "f2": [(2, 0)]}, ang=ang)
        W = onp.einsum("lm,em->el", EPS_A, COV.compute_event(nm, d)[..., :3])
        s = onp.vdot(W, A) / onp.vdot(A, A)
        r = float(onp.abs(W - s * A).max() / onp.abs(W).max())
        rf = float(onp.abs(W - s * A[:, ::-1]).max() / onp.abs(W).max())
        print("  %-8s rows=%d  同序 rel=%.3e  翻轴 rel=%.3e  %s"
              % (nm, A.shape[1], r, rf, "OK" if r < 1e-12 else "**FAIL**"))

    print()
    print("=" * 92)
    print("phi f0 与 phif243：内层 f2 的 5 槽必须真的参与（不是被压成 0）")
    spec0 = S.load_spec(os.path.join(ROOT, "agent", "tensor", "specs",
                                     "phif0.toml"))
    c0 = S.to_chain(spec0)
    for nm, l in (("phif001", [(0, 1)]), ("phif021", [(2, 1)])):
        ang = HC.node_angles(mom, c0)
        A = HC.chain_amplitude(
            mom, c0, {"psi2s": l, "phi": [(1, 0)], "f0": [(0, 0)]}, ang=ang)
        W = onp.einsum("lm,em->el", EPS_A, COV.compute_event(nm, d)[..., :3])
        s = onp.vdot(W, A) / onp.vdot(A, A)
        print("  %-8s rel=%.3e" % (nm, float(onp.abs(W - s * A).max()
                                             / onp.abs(W).max())))

    print()
    print("=" * 92)
    print("内层 f2 的 5 个 lambda 槽的**逐个**贡献（证明没有被压成 0）")
    ang = HC.node_angles(mom, chain)
    full = HC.node_amplitude(mom, chain, "f2", [(2, 0)], ang=ang)
    print("  A^(f2)[lam_f2, lam_Pip, lam_Pim] 形状 =", full.shape)
    for i, lf in enumerate(HC.lambda_order(2)):
        col = full[:, i, 0, 0]
        print("    lam_f2=%+d  max|A| = %.4e  %s"
              % (lf, onp.abs(col).max(),
                 "非零" if onp.abs(col).max() > 1e-12 else "**被压成 0**"))


if __name__ == "__main__":
    main()
