#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""A1 的**正负/相位**结构：逐族的相对符号是否与文档一致。

判据 A1 只要求"协变与级联差一个逐道复常数 $c_i$"，它**不**检查 $c_i$ 之间的
相对符号 —— 那正是置换求和里最容易写错、又最隐蔽的地方（例 1 的
`B12−B13+B23` 事件，每条道单看都自洽）。

本探针因此单独查三件事：

1. **逐族 $c_i$ 的相位**：同一族里各道的 $c_i$ 是否落在同一方向上
   （$\pm 1$ 或 $\pm i$）—— 不同方向意味着相对符号/相位不一致。
2. **置换协变性**：把某条道的末态动量重标号后，是否等于另一条道
   （$|c| = 1$）。这是"相对符号"的直接检验，不依赖文档下标。
3. **文档声明的符号型**：
   * 例 1 ρ：三项**同号**；
   * 例 1 ρ₃：**(+,−,−)**；
   * 例 2 K*（J=1/3/5）：两项**同号**（文档写 `-` 的那个负号是整体因子）。

用法：
    python analyses/tensor_examples/probes/probe_a1_signs.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)
os.chdir(_ROOT)

from agent.tensor import spec as S                                  # noqa: E402
from agent.tensor.kernels.helicity import EPS_A                     # noqa: E402
from probe_a1_full import KIN, cov_event, pick                      # noqa: E402

SPECS = os.path.join(_ROOT, "analyses", "tensor_examples", "specs")

# 每条族：spec、道名顺序、末态键顺序、置换伙伴 (i,j)、文档声明的符号型
#
# ⚠️ 置换的下标要按**配对**推，不能按道名后缀想当然：
#   ex1 的 keys = [Pip1, Pim2, Pi0]（slot 0/1/2）
#     u_rho_12 配对 (0,1) 旁观者 2  →  u_rho_13 配对 (0,2) 旁观者 1
#                                    ⇒ 换 slot **1↔2**
#     u_rho_12 → u_rho_23 配对 (1,2) 旁观者 0  ⇒ 换 slot **0↔2**
#   ex2 的 keys = [Kp1, Km2, Pi0]
#     u_kstar_13 配对 (0,2) → u_kstar_23 配对 (1,2)  ⇒ 换 slot 0↔1
FAM = [
    ("ex1_pipipi0_rho", ["u_rho_12", "u_rho_13", "u_rho_23"],
     ["Pip1", "Pim2", "Pi0"], [(1, 2), (0, 2)], "同号"),
    ("ex1_pipipi0_rho3", ["u_rho3_12", "u_rho3_13", "u_rho3_23"],
     ["Pip1", "Pim2", "Pi0"], [(1, 2), (0, 2)], "(+,−,−)"),
    ("ex2_kkpi0_kstar", ["u_kstar_13", "u_kstar_23"],
     ["Kp1", "Km2", "Pi0"], [(0, 1)], "同号"),
    ("ex2_kkpi0_kstar3", ["u_kstar3_13", "u_kstar3_23"],
     ["Kp1", "Km2", "Pi0"], [(0, 1)], "同号"),
    ("ex2_kkpi0_kstar5", ["u_kstar5_13", "u_kstar5_23"],
     ["Kp1", "Km2", "Pi0"], [(0, 1)], "同号"),
]


def fit(a, b):
    c = np.vdot(a.reshape(-1), b.reshape(-1)) / np.vdot(a.reshape(-1),
                                                         a.reshape(-1))
    r = float(np.abs(b - c * a).max() / max(np.abs(b).max(), 1e-300))
    return complex(c), r


def main():
    kin = {}
    print("=" * 100)
    print("A1 的相位/符号结构")
    print("=" * 100)
    allok = True
    for spec_name, names, keys, perms, declared in FAM:
        kind = pick(os.path.join(SPECS, spec_name + ".toml"))
        if kind not in kin:
            kin[kind] = KIN[kind]()
        momd, _k = kin[kind]
        sp = S.load_spec(os.path.join(SPECS, spec_name + ".toml"))
        cc = {c["name"]: c for c in S.cascade_channels(sp)}
        n = momd[keys[0]].shape[0]
        psi = np.broadcast_to(np.array([0.0, 0.0, 0.0, 3.686]), (n, 4))

        # 逐道 W（协变经 EPS_A）与级联 A
        W, A = {}, {}
        for nm in names:
            T = np.asarray(cov_event(sp, cc[nm], momd, psi))
            W[nm] = np.einsum("lm,em->el", EPS_A, T[..., :3])
            # 级联用同一个 tree
            from agent.tensor.kernels import helicity_cascade as HC
            mom = np.stack([momd[k] for k in keys], axis=1)[:, :, [3, 0, 1, 2]]
            ch = HC.with_tree(S.to_chain(sp), cc[nm]["tree"] or {})
            A[nm] = HC.chain_amplitude(mom, ch, cc[nm]["ls"],
                                       ang=HC.node_angles(mom, ch))

        print()
        print("--- %s（文档声明：%s）" % (spec_name, declared))
        cs = {}
        for nm in names:
            c, r = fit(W[nm], A[nm])
            cs[nm] = c
            good = r < 1e-12
            allok &= good
            print("    %-12s c = %+.6f%+.6fj   |c| = %.6f   phase = %+7.2f°"
                  "   rel = %.2e  %s"
                  % (nm, c.real, c.imag, abs(c), np.degrees(np.angle(c)), r,
                     "OK" if good else "**FAIL**"))

        # 1) 相位一致性：同一族里 c_i 的相位应当只差 0 或 180°
        ph = np.array([np.angle(cs[nm]) for nm in names])
        spread = float(np.degrees(np.abs(
            (ph - ph[0] + np.pi) % (2 * np.pi) - np.pi)).max())
        print("    相位离散度 = %.3f°（0 或 180 表示同轴；其他值=相位不一致）"
              % spread)
        # 允许 0 或 180，取最小
        dev = float(np.degrees(np.minimum(
            np.abs((ph - ph[0] + np.pi) % (2 * np.pi) - np.pi),
            np.abs(np.abs((ph - ph[0] + np.pi) % (2 * np.pi) - np.pi) - np.pi)
        )).max())
        print("    相对 0/180° 的最大偏离 = %.4f°  %s"
              % (dev, "OK" if dev < 1e-6 else "**不一致**"))
        allok &= dev < 1e-6

        # 2) 置换协变性：重标号后必须等于另一条道（|c| = 1）
        for other, (i, j) in zip(names[1:], perms):
            perm = list(range(len(keys)))
            perm[i], perm[j] = perm[j], perm[i]
            moved = {keys[t]: momd[keys[perm[t]]] for t in range(len(keys))}
            Tm = np.asarray(cov_event(sp, cc[names[0]], moved, psi))
            Wm = np.einsum("lm,em->el", EPS_A, Tm[..., :3])
            c, r = fit(Wm, W[other])
            good = r < 1e-12 and abs(abs(c) - 1) < 1e-9
            allok &= good
            print("    置换 %s: %s(重标号) vs %s  c = %+.6f%+.6fj  |c|=%.9f"
                  "  rel=%.2e  %s"
                  % (str((i, j)), names[0], other, c.real, c.imag, abs(c), r,
                     "OK" if good else "**FAIL**"))
    print()
    print("=> %s" % ("符号/相位结构一致" if allok else "**有符号/相位问题**"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
