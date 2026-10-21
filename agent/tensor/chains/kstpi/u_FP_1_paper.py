#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""$U^\mu_{K^*_2}$ FP（左）：顶层 $F$，内层 $P$ —— **历史形式（未修正）**

文档《ψ衰变到介子的分波分析协变张量形式》431–623 行（$\psi\to K^+\pi^-K^-\pi^+$）。
参考实现 `Tensor/FUNC.py::_u_FP_l`（jax），本文件是**逐式移植**（走 `_backend.xp`，
jax / numpy 两后端通用）：`jax.numpy` 换成后端命名空间、`self.psi2s` 提成显式
参数 `psi`，公式一个数字都没动。

⚠️ **这是历史形式，不用于生产。** 它与 PP 族**同病**：内层是 P 波，
$1\otimes1\to2$ 的无迹化被隐式交给 $P^{(2)}(p_\psi)$（$\psi$ 系），
而耦合发生在 $K^*_2$ 系。A1 残差 $6.55\times10^{-3}$。

* **生产用**：同目录的 [`u_FP_1.py`](u_FP_1.py)（修正版）。
* **对照说明**：[`CORRECTION.md`](CORRECTION.md)（公式 + 代码逐行对照）。

在 spec 里用道名 `u_FP_1_paper` 即可生成这一种形式。

粒子编号（与文档一致）：$K^+=1$，$K^-=2$，$\pi^+=3$，$\pi^-=4$
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import (
    _einsum, _metric_axis, _raise_last, Gt_cov, P2, T1_cov, T3_cov
)


def u_FP_1_paper(psi, phi, f2, Kp, Km, Pip, Pim):
    kst2 = Kp + Km + Pim
    p2psi_cov = P2(psi)
    gt_ks_con = _raise_last(Gt_cov(kst2), 2)
    t3_top_cov = T3_cov(psi, kst2, Pip)       # $\tilde T^{(3)}_{(K^*_2 4)}$
    t3_top_con = _raise_last(t3_top_cov, 3)
    t1_in_cov = T1_cov(kst2, phi, Pim)
    t1_in_con = _metric_axis(t1_in_cov, 1)
    t1_12_cov = T1_cov(phi, Kp, Km)            # 注意：**协变**（不升）
    return _einsum("eijk,ejklm,el,emn,en->ei",
                   t3_top_con, p2psi_cov, t1_in_con, gt_ks_con, t1_12_cov)
