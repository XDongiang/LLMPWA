#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 2（\rho' 家族）第 (1,2) 项：$\psi\to \rho' K$，$\rho'\to K\pi^0$。

文档《ψ衰变到介子的分波分析协变张量形式》274–304 行。

本文件是**求和里的一项**（手写协变道；`agent/tensor/emitter.py` 在生成时按
`def <name>` 取函数体内联）。把 (1,2) = $(K^+,K^-)$ 配成 $\rho'$，粒子 3 = $\pi^0$ 是旁观者。文档这一式**只有一项** —— $(1\leftrightarrow3)$ 会把 $K$ 换成 $\pi^0$，那是 $K^*$ 不是 $\rho'$（见 `ex2_kkpi0_kstar`）。

配对与子节点顺序**逐道**照抄 spec 的 `[[channel]].tree` —— `t_l` 里
$r = b - c$，交换两个子节点会让振幅整体翻号；级联侧的口径由同一个 `tree`
定，两边必须一致。详见
`analyses/tensor_examples/chains/notes/ex2_rho.notes.md`。

两个张量都由 `kernels/covariant.py::t_l` 给出（论文 Eq.(258) 的通用构造）。
四动量约定 `(x,y,z,E)`（与 `Tensor/BASE.py` 一致）。
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import _einsum, leci_four, t_l


def u_rho_K_12(psi, Kp1, Km2, Pi0):
    r"""$\epsilon_{\mu\nu\lambda\sigma}p_\psi^\sigma
    \tilde T^{(1)\nu}_{(\rho' Pi0)}
    \tilde t^{(1)\lambda}_{(1,2)}$。

    返回 `(N, 4)`：缩并后的 $\mu$ 分量（配对 = (1,2)，旁观者 = Pi0）。
    """
    res = Kp1 + Km2
    t_top = t_l(1, psi, res, Pi0)
    t_in = t_l(1, res, Kp1, Km2)
    m_nl = _einsum("en,el->enl", t_top, t_in)
    return _einsum("mnls,es,enl->em", leci_four(), psi, m_nl)
