#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""三体节点缺陷的**数值**证据（2026-10-21）。

配套 `documentation/tensor_examples_feasibility.md` §2.5，以及
`test_spec.py::test_three_daughter_node_rejected`（那条钉的是 spec 层拒绝，
这条钉的是**为什么**必须拒绝）。

三体节点不能写成两体级联，因为 $\epsilon$ 会退化：

$$
\epsilon^{\mu\nu\lambda\sigma}p_{\rho\nu}p_{\pi\lambda}p_{\omega\sigma}=0
\qquad(p_\omega=p_\rho+p_\pi)
$$

而真三体的 $\epsilon^{\mu\nu\lambda\sigma}p_{1\nu}p_{2\lambda}p_{0\sigma}$
是 $O(1)$。**这是恒等式，不是数值精度问题。**

    python agent/tensor/tests/test_threebody_node.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.tensor.kernels.covariant import leci_four, omega_mu  # noqa: E402


def _kin(n=32, seed=5):
    """例 4 的五体相空间（`data/make_omega5.py`）。"""
    p = os.path.join(_ROOT, "analyses", "tensor_examples", "data",
                     "make_omega5.py")
    sp = importlib.util.spec_from_file_location("_mo5", p)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m.make(n=n, seed=seed)


def test_epsilon_degenerates_for_two_body_node():
    """两体节点的 ε 恒为零 —— 这就是"不能用两体树凑三体节点"的原因。

    取 ω→ρ+π⁰：$p_\\omega=p_\\rho+p_0$，三个动量线性相关，全反对称缩并为 0。
    量级 $10^{-16}$ 是四动量求和的浮点误差，不是"小但非零"。
    """
    d = _kin()
    Pip, Pim, Pi0 = d["Pip"], d["Pim"], d["Pi0"]
    omega = Pip + Pim + Pi0
    rho = Pip + Pim                         # 两体节点里的"中间态"
    L = leci_four()

    two_body = np.einsum("mnls,en,el,es->em", L, rho, Pi0, omega)
    assert np.abs(two_body).max() < 1e-13, (
        "两体节点的 ε 竟然非零（%.3e）—— 若真是这样，三体节点就能用两体树凑，"
        "feasibility §2.5 的前提要重写" % np.abs(two_body).max())

    # 真三体（三个**末态**动量）是 O(1)
    three_body = np.einsum("mnls,en,el,es->em", L, Pip, Pim, Pi0)
    assert np.abs(three_body).max() > 1e-2, (
        "真三体的 ε 是零（%.3e）—— 运动学退化，测试无意义"
        % np.abs(three_body).max())


def test_omega_mu_matches_document_formula():
    """`omega_mu` 与文档 423–427 行逐位相同（同一个 numpy 表达式，rel == 0）。"""
    from agent.tensor.kernels.covariant import B1

    d = _kin()
    Pip, Pim, Pi0 = d["Pip"], d["Pim"], d["Pi0"]
    omega = Pip + Pim + Pi0

    got = omega_mu(omega, Pi0, Pip, Pim)

    res12, res10, res20 = Pip + Pim, Pip + Pi0, Pim + Pi0
    b = (B1(omega, res12, Pi0) * B1(res12, Pip, Pim)
         + B1(omega, res10, Pim) * B1(res10, Pip, Pi0)
         + B1(omega, res20, Pip) * B1(res20, Pim, Pi0))
    want = np.einsum("mnls,en,el,es->em", leci_four(), Pip, Pim, Pi0) * b[:, None]

    assert np.abs(got - want).max() <= 1e-15 * max(np.abs(want).max(), 1.0), (
        "omega_mu 与文档公式不符：rel = %.3e"
        % (np.abs(got - want).max() / max(np.abs(want).max(), 1e-300)))

    # 三条置换道**都**在（少一条会让 |ω| 变小但不为零，容易漏）
    only_one = (np.einsum("mnls,en,el,es->em", leci_four(), Pip, Pim, Pi0)
                * (B1(omega, res12, Pi0) * B1(res12, Pip, Pim))[:, None])
    assert np.abs(got - only_one).max() > 1e-2, "只算了一条置换道"


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    bad = 0
    for name, fn in tests:
        try:
            fn()
            print("  OK   %s" % name)
        except Exception as e:                                  # noqa: BLE001
            bad += 1
            print("  FAIL %s: %s" % (name, e))
    print("%s: %d/%d" % ("PASS" if not bad else "**FAIL**",
                         len(tests) - bad, len(tests)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
