#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""$U^\mu_{K^*_0}$（左）：$\psi\to K^*_0\pi$，$K^*_0\to\phi\pi$

文档《ψ衰变到介子的分波分析协变张量形式》431–623 行（$\psi\to K^+\pi^-K^-\pi^+$）。
参考实现 `Tensor/FUNC.py::_u_kst0_l`（jax），本文件是**逐式移植**（走 `_backend.xp`，
jax / numpy 两后端通用）：`jax.numpy` 换成后端命名空间、`self.psi2s` 提成显式
参数 `psi`，公式一个数字都没动。

粒子编号（与文档一致）：$K^+=1$，$K^-=2$，$\pi^+=3$，$\pi^-=4$

拓扑（**深度 3**，与 b₁ 同构）：$\psi\to K^*\pi \to (\phi\pi)\pi \to (K^+K^-)(\pi\pi)$

本项：$K^*_0$ 由 $(1,2,4)=\phi\pi^-$ 重建，$\pi^+$（粒子 3）是旁观者。三个 $\tilde t^{(1)}$ 依次缩并成矢量（顶层、$K^*_0\to\phi\pi^-$、$\phi\to K^+K^-$），中间**没有**投影算符。顶层 $(l,s)=(1,0)$，内层 $(l,s)=(1,1)$。

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
    _einsum, _metric_axis, T1_cov
)


def u_kst0_l(psi, phi, f2, Kp, Km, Pip, Pim):
    kst0 = Kp + Km + Pim                    # 旁观者 Pip 之外的三个粒子
    t1_top_cov = T1_cov(psi, kst0, Pip)     # \tilde T^{(1)}_{(K^*_0 4)}
    t1_top_con = _metric_axis(t1_top_cov, 1)
    t1_in_cov = T1_cov(kst0, phi, Pim)      # \tilde t^{(1)}_{(\phi 4)}
    t1_in_con = _metric_axis(t1_in_cov, 1)
    t1_12_cov = T1_cov(phi, Kp, Km)          # \tilde t^{(1)}_{(12)}
    return _einsum("ei,ej,ej->ei", t1_top_con, t1_in_con, t1_12_cov)
