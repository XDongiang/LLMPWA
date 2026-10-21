#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""$U^\mu_{K^*_2}$ PP（左）：顶层 $P$，内层 $P$ —— **修正版**

文档《ψ衰变到介子的分波分析协变张量形式》431–623 行（$\psi\to K^+\pi^-K^-\pi^+$）。

⚠️ **本文件不是 `Tensor/FUNC.py` 的逐式移植**，而是对它的**修正**。
原式（= `FUNC.py::_u_PP_l`）把 $1\otimes1\to2$ 的无迹化交给了
$P^{(2)}(p_\psi)$ —— 那是 $\psi$ 系的度规，而耦合发生在 $K^*_2$ 系。
修正后内层在 $K^*_2$ 系里无迹，A1（对级联）从 $6.99\times10^{-3}$ 降到
$\sim7\times10^{-15}$。

**历史形式见 [`u_PP_1_paper.py`](u_PP_1_paper.py)**（逐式移植，可与 jax 对拍）。
两种形式的完整对照、公式与代码见
[`CORRECTION.md`](CORRECTION.md)。

粒子编号（与文档一致）：$K^+=1$，$K^-=2$，$\pi^+=3$，$\pi^-=4$

拓扑（**深度 3**，与 b₁ 同构）：$\psi\to K^*\pi \to (\phi\pi)\pi \to (K^+K^-)(\pi\pi)$

本项：顶层 $P$ 波、内层 $P$ 波。

    修正前（论文 / `FUNC.py` / `u_PP_1_paper.py`）：
      $P^{(2)}_{\psi}{}^{ijkl}\,
        \tilde T^{(1)}_j\,\tilde t^{(1)}_k\,
        \tilde g_{K^*_2}{}^{lm}\,\tilde t^{(1)}_m$
      —— 内层 $\tilde t^{(1)}_k\tilde g^{lm}\tilde t^{(1)}_m$ 是**秩 2 但不无迹**
      （无迹化被隐式交给 $P^{(2)}_{\psi}$，用了错的系）。

    修正后（本文件）：
      $P^{(2)}_{\psi}{}^{ijkl}\,
        \tilde T^{(1)}_j\,
        T^{(2)}_{K^*_2}{}_{kl}$
      其中 $T^{(2)}_{K^*_2}$ 由 `T2_from_pair(kst2, ...)` 在 $K^*_2$ 系里无迹化。
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import (
    _einsum, _metric_axis, _raise_last, Gt_cov, P2, T1_cov, T2_from_pair
)


def u_PP_1(psi, phi, f2, Kp, Km, Pip, Pim):
    kst2 = Kp + Km + Pim
    p2psi_con = _raise_last(P2(psi), 4)          # $P^{(2)}_{\psi}$（顶层投影）
    gt_ks_cov = Gt_cov(kst2)                     # $\tilde g_{K^*_2}$（**协变**）
    t1_top_cov = T1_cov(psi, kst2, Pip)          # $\tilde T^{(1)}_{(K^*_2 4)}$
    t1_in_cov = T1_cov(kst2, phi, Pim)           # $\tilde t^{(1)}_{(\phi 4)}$
    t1_12_cov = T1_cov(phi, Kp, Km)              # $\tilde t^{(1)}_{(12)}$
    # 内层的第二个秩-1 因子：$\tilde g_{K^*_2}\cdot\tilde t^{(1)}_{(12)}$
    t1_12_con = _metric_axis(t1_12_cov, 1)
    gphi = _einsum("elm,em->el", gt_ks_cov, t1_12_con)
    # 在 $K^*_2$ 系里无迹 —— 这是修正的那一步
    t2_in_cov = T2_from_pair(kst2, t1_in_cov, gphi)
    return _einsum("eijkl,ej,ekl->ei", p2psi_con, t1_top_cov, t2_in_cov)
