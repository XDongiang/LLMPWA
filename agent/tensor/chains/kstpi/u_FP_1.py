#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""$U^\mu_{K^*_2}$ FP（左）：顶层 $F$，内层 $P$ —— **修正版**

文档《ψ衰变到介子的分波分析协变张量形式》431–623 行（$\psi\to K^+\pi^-K^-\pi^+$）。

⚠️ **本文件不是 `Tensor/FUNC.py` 的逐式移植**，而是对它的**修正**（与
`u_PP_1` 同一机制）。原式把内层 $1\otimes1\to2$ 的无迹化交给了
$P^{(2)}(p_\psi)$ —— 那是 $\psi$ 系的度规，而耦合发生在 $K^*_2$ 系。
修正后 A1（对级联）从 $6.55\times10^{-3}$ 降到 $\sim8\times10^{-15}$。

**历史形式见 [`u_FP_1_paper.py`](u_FP_1_paper.py)**；
完整对照见 [`CORRECTION.md`](CORRECTION.md)。

粒子编号（与文档一致）：$K^+=1$，$K^-=2$，$\pi^+=3$，$\pi^-=4$

拓扑（**深度 3**，与 b₁ 同构）：$\psi\to K^*\pi \to (\phi\pi)\pi \to (K^+K^-)(\pi\pi)$

本项：顶层 $F$ 波（$\tilde T^{(3)}$）、内层 $P$ 波。

    修正前（论文 / `FUNC.py` / `u_FP_1_paper.py`）：
      $\tilde T^{(3)}{}^{ijk}_{\psi}\,
        P^{(2)}_{\psi}{}_{jklm}\,
        \tilde t^{(1)}_{(\phi 4)}{}^{l}\,
        \tilde g_{K^*_2}{}^{mn}\,\tilde t^{(1)}_{(12)n}$
      —— 内层 $\tilde t^{(1)}{}^{l}\tilde g^{mn}\tilde t^{(1)}_n$ 秩 2 但不无迹。

    修正后（本文件）：
      $\tilde T^{(3)}{}^{ijk}_{\psi}\,
        P^{(2)}_{\psi}{}_{jklm}\,
        T^{(2)}_{K^*_2}{}^{lm}$
      其中 $T^{(2)}_{K^*_2}$ 由 `T2_from_pair(kst2, ...)` 在 $K^*_2$ 系里无迹化，
      再把两个指标升上去。
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import (
    _einsum, _metric_axis, _raise_last, Gt_cov, P2, T1_cov, T2_from_pair, T3_cov
)


def u_FP_1(psi, phi, f2, Kp, Km, Pip, Pim):
    kst2 = Kp + Km + Pim
    p2psi_cov = P2(psi)                          # $P^{(2)}_{\psi}$
    t3_top_con = _raise_last(T3_cov(psi, kst2, Pip), 3)   # $\tilde T^{(3)}{}^{ijk}$
    gt_ks_cov = Gt_cov(kst2)                     # $\tilde g_{K^*_2}$（**协变**）
    t1_in_cov = T1_cov(kst2, phi, Pim)           # $\tilde t^{(1)}_{(\phi 4)}$
    t1_12_cov = T1_cov(phi, Kp, Km)              # $\tilde t^{(1)}_{(12)}$
    t1_12_con = _metric_axis(t1_12_cov, 1)
    gphi = _einsum("emn,en->em", gt_ks_cov, t1_12_con)
    # 在 $K^*_2$ 系里无迹 —— 这是修正的那一步
    t2_in_cov = T2_from_pair(kst2, t1_in_cov, gphi)
    t2_in_con = _raise_last(t2_in_cov, 2)
    return _einsum("eijk,ejklm,elm->ei", t3_top_con, p2psi_cov, t2_in_con)
