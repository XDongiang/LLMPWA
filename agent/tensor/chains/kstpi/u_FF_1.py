#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""$U^\mu_{K^*_2}$ FF（左）：顶层 $F$，内层 $F$

文档《ψ衰变到介子的分波分析协变张量形式》431–623 行（$\psi\to K^+\pi^-K^-\pi^+$）。
参考实现 `Tensor/FUNC.py::_u_FF_l`（jax），本文件是**逐式移植**（走 `_backend.xp`，
jax / numpy 两后端通用）：`jax.numpy` 换成后端命名空间、`self.psi2s` 提成显式
参数 `psi`，公式一个数字都没动。

粒子编号（与文档一致）：$K^+=1$，$K^-=2$，$\pi^+=3$，$\pi^-=4$

拓扑（**深度 3**，与 b₁ 同构）：$\psi\to K^*\pi \to (\phi\pi)\pi \to (K^+K^-)(\pi\pi)$

本项：顶层 $F$ 波、内层 $F$ 波：两个 $\tilde t^{(3)}$ 都升指标，中间用**协变**的 $\tilde g_{K^*_2}$，$\tilde t^{(1)}_{(12)}$ 用**逆变**。

⚠️ 与文档 Eq 的逐条对应**尚未完全敲定**：文档 431–623 行给的是
$\langle K^*K^*_J|LS\rangle$ 一族的公式，而参考实现这 10 个函数按它自己的
$P/F$（顶层/内层轨道角动量）与 $l/r$（旁观者 π）分类命名（见
`FUNC.py::Modifiers`）。本文件以**参考实现为准**移植 —— 它是可运行的权威，
文档行号只作背景索引。
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


def u_FF_1(psi, phi, f2, Kp, Km, Pip, Pim):
    kst2 = Kp + Km + Pim
    p2psi_cov = P2(psi)
    gt_ks_cov = Gt_cov(kst2)                   # 注意：**协变**（不升）
    t3_top_cov = T3_cov(psi, kst2, Pip)
    t3_top_con = _raise_last(t3_top_cov, 3)
    t3_in_cov = T3_cov(kst2, phi, Pim)
    t3_in_con = _raise_last(t3_in_cov, 3)
    t1_12_cov = T1_cov(phi, Kp, Km)
    t1_12_con = _metric_axis(t1_12_cov, 1)
    return _einsum("eijk,ejklm,elmn,eno,eo->ei",
                   t3_top_con, p2psi_cov, t3_in_con, gt_ks_cov, t1_12_con)
