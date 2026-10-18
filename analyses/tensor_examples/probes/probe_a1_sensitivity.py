#!/usr/bin/env python3
"""A1 是不是"自己跟自己比"？—— 逐项注入已知错误，看 A1 能不能抓住。

审阅者一定会问：级联与协变都用同一个 CG 表、同一个 Blatt-Weisskopf、
同一个 EPS_A，那 A1 会不会部分是套套逻辑？这里用"注入错误"回答：
把内核的某一块换掉，A1 必须失败，而且失败的**量级**要能定位到那一块。
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
from agent.tensor.kernels import omega as RT                        # noqa: E402
from agent.tensor.kernels.helicity import EPS_A                     # noqa: E402


def rel(a, b):
    s = onp.vdot(b, a) / onp.vdot(a, a)
    return float(onp.abs(b - s * a).max() / max(onp.abs(b).max(), 1e-300))


def main():
    d = T.physical(n=24, seed=11)
    keys = ("Kp", "Km", "Pip", "Pim")
    mom = onp.stack([d[k][:, [3, 0, 1, 2]] for k in keys], axis=1)
    spec = S.load_spec(os.path.join(ROOT, "analyses", "tensor_examples",
                                    "specs", "ex3_b1pi.toml"))
    chain0 = S.to_chain(spec)
    cc = [c for c in S.cascade_channels(spec) if c["name"] == "u_DD_1"][0]
    ch = HC.with_tree(chain0, cc["tree"])
    W = onp.einsum("lm,em->el", EPS_A,
                   COV.compute_event("u_DD_1", d)[..., :3])

    def A(**kw):
        ang = HC.node_angles(mom, ch)
        return HC.chain_amplitude(mom, ch, cc["ls"], ang=ang, **kw)

    print("=" * 94)
    print("基准：正确内核 vs 协变")
    print("   rel = %.3e" % rel(A(), W))

    print()
    print("注入 1：关掉 Chung Eq.52 协变项（covariant=False）")
    print("   rel = %.3e" % rel(A(covariant=False), W))

    print()
    print("注入 2：用 omega 的旧参考系（chain_angles 而不是 node_angles）")
    ang_old = RT.chain_angles(mom, ch)
    A_old = HC.chain_amplitude(mom, ch, cc["ls"], ang=ang_old)
    print("   rel = %.3e" % rel(A_old, W))

    print()
    print("注入 3：顶层 (L,S) 用错 —— 把 (2,1) 换成 (0,1)")
    ls_bad = {"psi2s": [(0, 1)], "b1": [(2, 1)], "phi": [(1, 0)]}
    A3 = HC.chain_amplitude(mom, ch, ls_bad, ang=HC.node_angles(mom, ch))
    print("   rel = %.3e" % rel(A3, W))

    print()
    print("注入 4：内层 (l,s) 用错 —— 把 (2,1) 换成 (0,1)")
    ls_bad2 = {"psi2s": [(2, 1)], "b1": [(0, 1)], "phi": [(1, 0)]}
    A4 = HC.chain_amplitude(mom, ch, ls_bad2, ang=HC.node_angles(mom, ch))
    print("   rel = %.3e" % rel(A4, W))

    print()
    print("注入 5：内层 s 用错 —— (2,1) -> (2,2)（三角允许但自旋耦合错）")
    ls_bad3 = {"psi2s": [(2, 1)], "b1": [(2, 2)], "phi": [(1, 0)]}
    try:
        A5 = HC.chain_amplitude(mom, ch, ls_bad3, ang=HC.node_angles(mom, ch))
        print("   rel = %.3e" % rel(A5, W))
    except Exception as e:                                       # noqa: BLE001
        print("   %s: %s" % (type(e).__name__, e))

    print()
    print("注入 6：phi 节点用错波 —— (1,0) -> (1,2)（自旋对不上，应为 0）")
    ls_bad4 = {"psi2s": [(2, 1)], "b1": [(2, 1)], "phi": [(1, 2)]}
    A6 = HC.chain_amplitude(mom, ch, ls_bad4, ang=HC.node_angles(mom, ch))
    print("   rel = %.3e   max|A| = %.3e" % (rel(A6, W), onp.abs(A6).max()))

    print()
    print("注入 7：把子粒子螺旋度压成 0（omega 的 _node_decay_d 口径）")
    # omega 的 `_node_decay_d` 对**每个**节点都只取 D^{J*}_{lam_a, 0-0}，
    # 即把子粒子的螺旋度固定成 0。逐节点照做：
    #   A_top[lam_psi, lam_b1, lam_Pim=0]
    #   A_b1 [lam_b1,  lam_phi=0, lam_Pip=0]
    #   A_phi[lam_phi=0, lam_K=0]
    ang = HC.node_angles(mom, ch)
    A_top7 = HC.node_amplitude(mom, ch, "psi2s", [(2, 1)], ang=ang)
    A_b17 = HC.node_amplitude(mom, ch, "b1", [(2, 1)], ang=ang)
    A_phi7 = HC.node_amplitude(mom, ch, "phi", [(1, 0)], ang=ang)
    print("   A_top shape = %s   A_b1 shape = %s   A_phi shape = %s"
          % (A_top7.shape, A_b17.shape, A_phi7.shape))
    # 顶层：保留 lam_psi 与 lam_b1，把 lam_Pim 压成 0
    Ztop = A_top7[:, :, :, 0]                    # (e, lam_psi, lam_b1)
    # b1 节点：保留 lam_b1，把 lam_phi / lam_Pip 压成 0
    Zb1 = A_b17[:, :, 0, 0]                      # (e, lam_b1)
    # phi 节点：把两个 K 的螺旋度压成 0（phi 自旋 1 -> 只剩 lam_phi 那一维）
    Zphi = A_phi7[:, :, 0, 0]                    # (e, lam_phi)
    A7 = onp.einsum("eab,eb->ea", Ztop, Zb1)
    A7 = A7 * Zphi[:, 1:2]                       # lam_phi = 0 那一槽
    print("   rel = %.3e   max|A| = %.3e" % (rel(A7, W), onp.abs(A7).max()))
    print("   （omega 时代就是这个量；它与协变差 O(1)，正是 §1.1/§1.2 的边界）")


if __name__ == "__main__":
    main()
