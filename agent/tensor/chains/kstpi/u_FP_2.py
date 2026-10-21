#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""$U^\mu_{K^*_2}$ FP（右）：顶层 $F$，内层 $P$ —— **修正版**

与 `u_FP_1` 互为旁观者 π 对调（$3\leftrightarrow 4$）。
修正的机制、公式对照见 [`CORRECTION.md`](CORRECTION.md)；
历史形式见 [`u_FP_2_paper.py`](u_FP_2_paper.py)。

粒子编号（与文档一致）：$K^+=1$，$K^-=2$，$\pi^+=3$，$\pi^-=4$
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


def u_FP_2(psi, phi, f2, Kp, Km, Pip, Pim):
    kst2 = Kp + Km + Pip
    p2psi_cov = P2(psi)
    t3_top_con = _raise_last(T3_cov(psi, kst2, Pim), 3)   # $\tilde T^{(3)}{}^{ijk}$
    gt_ks_cov = Gt_cov(kst2)
    t1_in_cov = T1_cov(kst2, phi, Pip)           # $\tilde t^{(1)}_{(\phi 3)}$
    t1_12_cov = T1_cov(phi, Kp, Km)
    t1_12_con = _metric_axis(t1_12_cov, 1)
    gphi = _einsum("emn,en->em", gt_ks_cov, t1_12_con)
    t2_in_cov = T2_from_pair(kst2, t1_in_cov, gphi)
    t2_in_con = _raise_last(t2_in_cov, 2)
    return _einsum("eijk,ejklm,elm->ei", t3_top_con, p2psi_cov, t2_in_con)
