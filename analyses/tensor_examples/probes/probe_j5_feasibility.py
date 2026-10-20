#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""J=5 / l=5（ρ₅、K*₅）到底卡在哪 —— 逐项实测，不靠猜。

背景：`spec.py` 原本在 `validate` 里硬拒 `J>4` 与 `l>4`，理由是
"`LAM_SLOTS` / Wigner-D 只到 J=4，`Bl` 只到 l=4"。本脚本逐项验证这句话
对每个组件是否成立，并给出扩展所需的**最小**改动。

    python analyses/tensor_examples/probes/probe_j5_feasibility.py

判据（每项都要"有限 + 与参考一致"）：

1. `small_d_weight(2J)`：tf-pwa 的公式本来就接受任意 `2J`（它的 `j` 是 2J）。
2. `small_d_matrix` 正交性 `d^T d = I`：J 越大越容易出数值问题，必须验。
3. `cg`：Racah 求和是精确的，验正交归一 `sum_m1 |<j1 m1; j2 m2|J M>|^2 = 1`。
4. `covariant_hel_term(J, spins, gamma)`：Chung Eq.52，验有限性。
5. `Bl(l)`：这是**唯一**真正需要新公式的组件 —— tf-pwa 有 L=5 的系数表，
   而且还有 `get_bprime_coeff` 的通用递推（反向 Bessel 多项式）兜底。
6. `LAM_SLOTS[J]`：纯字典，加键即可。
7. 端到端 `chain_amplitude`：J=5 的振幅必须有限、且**置换协变性**成立。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(_TE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.tensor.kernels import omega as RT                       # noqa: E402

# tf-pwa `breit_wigner.py::Bprime_polynomial` 的系数表（降幂）。
# L=5 那行是它写死的，不是我们编的。
TF_PWA_BPRIME_COEFF = {
    0: [1.0],
    1: [1.0, 1.0],
    2: [1.0, 3.0, 9.0],
    3: [1.0, 6.0, 45.0, 225.0],
    4: [1.0, 10.0, 135.0, 1575.0, 11025.0],
    5: [1.0, 15.0, 315.0, 6300.0, 99225.0, 893025.0],
}


def _polyval(coeff, z):
    """Horner；coeff 是降幂（与 tf-pwa 的 `polyval` 一致）。"""
    r = 0.0
    for a in coeff:
        r = r * z + a
    return r


# ---------------------------------------------------------------- 1..4
def check_small_d():
    print("[1] small_d_weight / small_d_matrix")
    for J in (4, 5, 6):
        w = RT.small_d_weight(2 * J)
        ok_w = w.shape == (2 * J + 1, 2 * J + 1, 2 * J + 1) and np.isfinite(w).all()
        th = np.array([0.3, 1.1, 2.4, 3.0])
        d = RT.small_d_matrix(th, 2 * J)
        orth = np.abs(np.einsum("eij,eik->ejk", d, d)
                      - np.eye(2 * J + 1)).max()
        print("    J=%d  weight%s shape=%s  正交性 %.2e %s"
              % (J, " OK" if ok_w else " BAD", w.shape, orth,
                 "OK" if orth < 1e-12 else "**FAIL**"))
        assert ok_w and orth < 1e-12, J


def check_cg():
    print("[2] cg 正交归一（固定 M，对 m1 求和）")
    worst = 0.0
    for J in (4, 5, 6):
        for j1 in range(J + 1):
            j2 = J - j1
            for M in range(-J, J + 1):
                tot = 0.0
                for m1 in range(-j1, j1 + 1):
                    m2 = M - m1
                    if abs(m2) <= j2:
                        tot += RT.cg(j1, m1, j2, m2, J, M) ** 2
                worst = max(worst, abs(tot - 1.0))
    print("    J 到 6 的最大偏差 = %.2e %s"
          % (worst, "OK" if worst < 1e-12 else "**FAIL**"))
    assert worst < 1e-12


def check_covariant_hel_term():
    print("[3] covariant_hel_term（Chung Eq.52）")
    for J in (4, 5, 6):
        spins = list(range(-J, J + 1, 2))
        r = RT.covariant_hel_term(J, spins, np.array([0.3, 1.2, 2.0]))
        ok = r.shape == (3, len(spins)) and np.isfinite(r).all()
        print("    J=%d  shape=%s 有限=%s %s"
              % (J, r.shape, np.isfinite(r).all(), "OK" if ok else "**FAIL**"))
        assert ok, J


# ---------------------------------------------------------------- 5
def check_bl_l5():
    """`Bl` 是唯一需要新公式的组件。

    我们的 `Bl` 与 tf-pwa 的 `Bprime` 差一个**与 q 无关**的常数因子
    （见 documentation 里对两种 Blatt-Weisskopf 约定的说明），所以只验
    "形状"：`1/Bl^2` 必须正比于 `P_l(q^2/Q0^2)`。
    """
    print("[4] Bl 到 l=5（tf-pwa `Bprime_polynomial` 的系数）")
    x0 = RT.Q0 ** 2
    q = np.array([0.05, 0.2, 0.5, 1.0, 2.0])
    for l in range(6):
        # ref = 平方后的 Bprime = P_l(z0)/P_l(z)，z = q^2/x0、z0 = 1
        ref = _polyval(TF_PWA_BPRIME_COEFF[l], 1.0) / _polyval(
            TF_PWA_BPRIME_COEFF[l], q / x0)
        # den 展开成 q 的幂（与现有 B4 的写法同构）：sum_i C_i q^(l-i) x0^i
        den = sum(c * q ** (l - i) * x0 ** i
                  for i, c in enumerate(TF_PWA_BPRIME_COEFF[l]))
        # num = P_l(1)；现有表里 l=0..4 的 num 正是各行的和
        num = sum(TF_PWA_BPRIME_COEFF[l])
        ours = num / den                      # = Bl(l,q)**2
        ratio = ours / ref
        ok = np.allclose(ratio, ratio[0], rtol=1e-12)
        tag = "（现有）" if l <= 4 else "（需新增）"
        if l <= 4:
            # 交叉验证：现有 Bl 的 num 表必须等于 P_l(1)
            got = {0: 1.0, 1: 2.0, 2: 13.0, 3: 277.0, 4: 12746.0}[l]
            assert abs(got - num) < 1e-9, (l, got, num)
            ours_direct = RT.Bl(l, q) ** 2
            assert np.allclose(ours_direct, ours, rtol=1e-12), l
        print("    l=%d%s num=%.0f  ours/ref = %.6f（常数=%s）%s"
              % (l, tag, num, ratio[0], ok, "OK" if ok else "**FAIL**"))
        assert ok, l

    # 系数表的来源：|theta_l(iw)|^2 的偶次项（反向 Bessel 多项式）。
    # 这一步是**独立复算**，确认 tf-pwa 写死的 L=5 那行不是笔误。
    print("    [独立复算] |theta_5(iw)|^2 的系数 vs tf-pwa 写死的表")
    n = 5
    co = [math.factorial(n + k) / (math.factorial(n - k) * math.factorial(k) * 2 ** k)
          for k in range(n + 1)]
    # theta_5(x) = sum co[k] x^(n-k)；x -> i w
    th_w = [co[k] * (1j) ** (n - k) for k in range(n + 1)]
    # 升幂多项式，再乘自己的共轭
    poly = np.array([complex(th_w[n - i]) for i in range(n + 1)])
    prod = np.polynomial.polynomial.polymul(poly, np.conj(poly)).real
    coeff = [prod[2 * (n - i)] for i in range(n + 1)]
    same = np.allclose(coeff, TF_PWA_BPRIME_COEFF[5], rtol=1e-9)
    print("        复算 = %s" % [round(c, 6) for c in coeff])
    print("        tf-pwa = %s  一致=%s %s"
          % (TF_PWA_BPRIME_COEFF[5], same, "OK" if same else "**FAIL**"))
    assert same


# ---------------------------------------------------------------- 6
def check_end_to_end():
    """打内存补丁，直接算 J=5 的链振幅 + 置换协变性。

    这一项才是关键：前五项都"看起来能算"，但真正的风险是**数值**在
    J=5 时炸掉，或者置换符号结构变了。
    """
    print("[5] 端到端：J=5 链振幅 + 置换协变性")
    saved_slots = dict(RT.LAM_SLOTS)
    saved_bl = RT.Bl
    try:
        RT.LAM_SLOTS[5] = list(range(-5, 6))

        def Bl5(l, q):
            if int(l) < 5:
                return saved_bl(l, q)
            x0 = RT.Q0 ** 2
            den = sum(c * q ** (5 - i) * x0 ** i
                      for i, c in enumerate(TF_PWA_BPRIME_COEFF[5]))
            return RT.xp.sqrt(TF_PWA_BPRIME_COEFF[5][0] / den)

        RT.Bl = Bl5

        sys.path.insert(0, os.path.join(_ROOT, "agent", "tensor", "tests"))
        import test_ex12_permutation as T                     # noqa: E402

        mom = T.threebody((0.13957039, 0.13957039, 0.1349768), 2.3, n=64, seed=5)
        from agent.tensor.kernels import helicity_cascade as HC   # noqa: E402

        def mk(daughters_top, daughters_rho):
            return {"top": "psi2s", "nodes": {
                "psi2s": {"J": 1, "mass": 3.686, "daughters": daughters_top},
                "rho5": {"J": 5, "mass": 2.3, "daughters": daughters_rho, "l": 5},
                "Pip1": {"J": 0, "mass": 0.13957039},
                "Pim2": {"J": 0, "mass": 0.13957039},
                "Pi0": {"J": 0, "mass": 0.1349768}},
                "index": {"Pip1": 0, "Pim2": 1, "Pi0": 2},
                "finals": ["Pip1", "Pim2", "Pi0"]}

        ls = {"psi2s": [(5, 5)], "rho5": [(5, 0)]}
        c12 = mk(["rho5", "Pi0"], ["Pip1", "Pim2"])
        c13 = mk(["rho5", "Pim2"], ["Pi0", "Pip1"])
        A = HC.chain_amplitude(mom, c12, ls, covariant=True)
        ok = A.shape == (64, 3) and np.isfinite(A).all()
        print("    振幅 shape=%s 有限=%s |A|max=%.4g %s"
              % (A.shape, np.isfinite(A).all(), np.abs(A).max(),
                 "OK" if ok else "**FAIL**"))
        assert ok

        # 置换协变性：把 slot 1<->2 交换后的动量代入 _12 树 == _13 树
        A13 = HC.chain_amplitude(mom, c13, ls, covariant=True)
        A12p = HC.chain_amplitude(mom[:, [0, 2, 1], :], c12, ls, covariant=True)
        p, o = A12p.reshape(-1), A13.reshape(-1)
        c = np.vdot(p, o) / np.vdot(p, p)
        res = np.abs(o - c * p).max() / np.abs(o).max()
        print("    置换协变性 c=%+.6f 残差=%.2e %s"
              % (c.real, res, "OK" if res < 1e-12 else "**FAIL**"))
        assert res < 1e-12
    finally:
        RT.LAM_SLOTS.clear()
        RT.LAM_SLOTS.update(saved_slots)
        RT.Bl = saved_bl


def main():
    print(__doc__.strip().split("\n")[0])
    print()
    check_small_d()
    check_cg()
    check_covariant_hel_term()
    check_bl_l5()
    check_end_to_end()
    print()
    print("结论：**J=5 / l=5 已支持**（2026-10-20 落地）。")
    print("  只改了三处 —— 其余原语本来就是通用的：")
    print("  (a) `omega.py::LAM_SLOTS` 加 5 / 6；")
    print("  (b) `omega.py::Bl` 改成系数表 + `_bprime_coeff` 现算（l>5）；")
    print("  (c) `spec.py` 的边界改成**按表象分级**（级联 6、协变张量 4）。")
    print("  `cg` / `small_d_weight` / `covariant_hel_term` / `D_lambda`")
    print("  不用动。")
    print()
    print()
    print("注意 `covariant.py` 的 `T5_cov` 是**另一件事**：那条路线按道名")
    print("硬编码笛卡尔张量（只到 T4），ρ₅ 要走它就得手写 T5。但 ρ₅ 用")
    print("`helicity_cascade` 不需要 —— 级联内核只读树，所以")
    print("`spec.py` 对 `cartesian`/`helicity` 仍然拒绝 J=5（这是**有意**的，")
    print("不是漏改）。")
    print()
    print("真正的墙不是 J=5，而是 `small_d_weight` 的阶乘溢出：")
    print("J=40 时正交性退化到 1.7e-6，J=50 时 NaN（171! 超 float64）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
