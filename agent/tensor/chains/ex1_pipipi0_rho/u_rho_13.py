#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 1（\rho 家族）第 (1,3) 项：$\psi\to \rho\pi^0$，$\rho\to\pi^+\pi^-$。

文档《ψ衰变到介子的分波分析协变张量形式》200–273 行。

本文件是**求和里的一项**（手写协变道；`agent/tensor/emitter.py` 在生成时按
`def <name>` 取函数体内联）。把 (1,3) = $(\pi^+,\pi^0)$ 配成共振，粒子 2 = $\pi^-$ 是旁观者（对应文档的 $-(1\leftrightarrow3)$ 项）。 文档的 $U_\rho^\mu$ 三项**同号**相加。

配对与子节点顺序**逐道**照抄 spec 的 `[[channel]].tree` —— `t_l` 里
$r = b - c$，交换两个子节点会让振幅整体翻号；级联侧的口径由同一个 `tree`
定，两边必须一致。详见
`analyses/tensor_examples/chains/notes/ex1_rho.notes.md`。

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


def u_rho_13(psi, Pip1, Pim2, Pi0):
    r"""$\epsilon_{\mu\nu\lambda\sigma}p_\psi^\sigma
    \tilde T^{(1)\nu}_{(\rho Pim2)}
    \tilde t^{(1)\lambda}_{(1,3)}$。

    返回 `(N, 4)`：缩并后的 $\mu$ 分量（配对 = (1,3)，旁观者 = Pim2）。
    """
    res = Pi0 + Pip1
    t_top = t_l(1, psi, res, Pim2)
    t_in = t_l(1, res, Pi0, Pip1)
    m_nl = _einsum("en,el->enl", t_top, t_in)
    return _einsum("mnls,es,enl->em", leci_four(), psi, m_nl)
