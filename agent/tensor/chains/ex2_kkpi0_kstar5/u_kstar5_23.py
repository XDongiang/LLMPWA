#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 2（K^*_5 家族）第 (2,3) 项：$\psi\to K^*_5 K$，$K^*_5\to K\pi^0$。

文档《ψ衰变到介子的分波分析协变张量形式》274–304 行。

本文件是**求和里的一项**（手写协变道；`agent/tensor/emitter.py` 在生成时按
`def <name>` 取函数体内联）。把 (2,3) = $(K^-,\pi^0)$ 配成 $K^*$，粒子 1 = $K^+$ 是旁观者。 文档写的是 $- (1\leftrightarrow2)$，但实测两条道在置换下**同号** —— 那个负号来自 $\tilde T^{(J)}$ 缩并的整体因子，不是道与道之间的相对符号（与 J=1 的 $U_{K^*}$ 一致）。

配对与子节点顺序**逐道**照抄 spec 的 `[[channel]].tree` —— `t_l` 里
$r = b - c$，交换两个子节点会让振幅整体翻号；级联侧的口径由同一个 `tree`
定，两边必须一致。详见
`analyses/tensor_examples/chains/notes/ex2_kstar5.notes.md`。

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


def u_kstar5_23(psi, Kp1, Km2, Pi0):
    r"""$\epsilon_{\mu\nu\lambda\sigma}p_\psi^\sigma
    \tilde T^{(5)\nuABCD}_{(K^*_5 Kp1)}
    \tilde t^{(5)\lambdaABCD}_{(2,3)}$。

    返回 `(N, 4)`：缩并后的 $\mu$ 分量（配对 = (2,3)，旁观者 = Kp1）。
    """
    res = Km2 + Pi0
    t_top = t_l(5, psi, res, Kp1)
    t_in = t_l(5, res, Km2, Pi0)
    m_nl = _einsum("enABCD,elABCD->enl", t_top, t_in)
    return _einsum("mnls,es,enl->em", leci_four(), psi, m_nl)
