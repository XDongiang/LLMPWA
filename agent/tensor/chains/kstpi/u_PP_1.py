#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""$U^\mu_{K^*_2}$ PP（左）：顶层 $P$，内层 $P$

文档《ψ衰变到介子的分波分析协变张量形式》431–623 行（$\psi\to K^+\pi^-K^-\pi^+$）。
参考实现 `Tensor/FUNC.py::_u_PP_l`（jax），本文件是**逐式移植**（走 `_backend.xp`，
jax / numpy 两后端通用）：`jax.numpy` 换成后端命名空间、`self.psi2s` 提成显式
参数 `psi`，公式一个数字都没动。

粒子编号（与文档一致）：$K^+=1$，$K^-=2$，$\pi^+=3$，$\pi^-=4$

拓扑（**深度 3**，与 b₁ 同构）：$\psi\to K^*\pi \to (\phi\pi)\pi \to (K^+K^-)(\pi\pi)$

本项：顶层 $P$ 波、内层 $P$ 波：$P^{(2)}_{\psi}$ 横向投影 × $\tilde T^{(1)}_{(K^*_2 n)}$ × $\tilde t^{(1)}_{(\phi n)}$ × $\tilde g_{K^*_2}$（**协变**）× $\tilde t^{(1)}_{(12)}$（**逆变**）。

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
    _einsum, _metric_axis, _raise_last, Gt_cov, P2, T1_cov
)


def u_PP_1(psi, phi, f2, Kp, Km, Pip, Pim):
    kst2 = Kp + Km + Pim
    p2psi_cov = P2(psi)                     # $P^{(2)}_{\psi}$
    p2psi_con = _raise_last(p2psi_cov, 4)
    gt_ks_cov = Gt_cov(kst2)                # $\tilde g_{K^*_2}$（协变）
    t1_top_cov = T1_cov(psi, kst2, Pip)    # $\tilde T^{(1)}_{(K^*_2 4)}$
    t1_in_cov = T1_cov(kst2, phi, Pim)     # $\tilde t^{(1)}_{(\phi 4)}$
    t1_12_cov = T1_cov(phi, Kp, Km)         # $\tilde t^{(1)}_{(12)}$
    t1_12_con = _metric_axis(t1_12_cov, 1)
    return _einsum("eijkl,ej,ek,elm,em->ei",
                   p2psi_con, t1_top_cov, t1_in_cov, gt_ks_cov, t1_12_con)
