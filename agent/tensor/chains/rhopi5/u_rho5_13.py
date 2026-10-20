#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Eq.(U_{\rho_5}) 第 $(1\leftrightarrow3)$ 项：$\psi\to\rho_5(5^-)\pi^0$，$\rho_5\to\pi^+\pi^-$。

论文 `jpsipwa1.tex` 327–340 行（文档 249–263 行）：

    U_{\rho_5}^\mu = -i\epsilon_{\mu\nu\lambda\sigma}p_\psi^\sigma
                     \tilde T^{(5)\nu\alpha\beta\gamma\delta}_{(\rho_5 3)}
                     \tilde t^{(5)\lambda}_{(12)\alpha\beta\gamma\delta}
                     f^{(\rho_5)}_{(12)} - (1\leftrightarrow 3)
                     - (2\leftrightarrow 3)

本文件是**求和里的第二项**：把 (1,3) = $(\pi^+,\pi^0)$ 配成 $\rho_5$，
粒子 2 = $\pi^-$ 是旁观者（对应论文的 $-(1\leftrightarrow3)$）。

与例 3 的 `rhopi` 链不同，这里的 `_13`/`_23` 是**另外两对配对**，
不是"同一个共振换一个旁观者" —— 三条道合起来才是完整的置换求和。

两个张量都由 `kernels/covariant.py::t_l` 给出（论文 Eq.258 的通用构造）：
顶层 $L=5$、内层 $l=5$。在本次改动之前，协变内核只有 `T1_cov..T4_cov`，
所以这条道**根本写不出来**；现在 `t_l(5, ...)` 把它补齐了。

四动量约定 `(x,y,z,E)`（与 `Tensor/BASE.py` 一致）。
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import _einsum, leci_four, t_l


def u_rho5_13(psi, Pip1, Pim2, Pi0):
    r"""$\tilde T^{(5)}_{(\rho_5 3)}\tilde t^{(5)}_{(13)}$ 缩并后的 $\mu$ 分量。

    返回 `(N, 4)`：$\epsilon_{\mu\nu\lambda\sigma}p_\psi^\sigma
    \tilde T^{(5)\nu\cdots}\tilde t^{(5)\lambda}_{\ \ \cdots}$。
    """
    rho = Pi0 + Pip1
    t5_top = t_l(5, psi, rho, Pim2)        # \tilde T^{(5)}_{(\rho_5 3)}
    t5_in = t_l(5, rho, Pi0, Pip1)       # \tilde t^{(5)}_{(13)}
    # 内层 4 个指标 (alpha,beta,gamma,delta) 全缩，剩下 (nu, lambda)
    m_nl = _einsum("enABCD,elABCD->enl", t5_top, t5_in)
    # eps_{mu nu lam sig} p_psi^sig
    return _einsum("mnls,es,enl->em", leci_four(), psi, m_nl)
