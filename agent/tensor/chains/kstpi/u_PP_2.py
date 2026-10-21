#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""$U^\mu_{K^*_2}$ PP（右）：顶层 $P$，内层 $P$ —— **修正版**

与 `u_PP_1` 互为旁观者 π 对调（$3\leftrightarrow 4$）。
修正的机制、公式对照见 [`CORRECTION.md`](CORRECTION.md)；
历史形式见 [`u_PP_2_paper.py`](u_PP_2_paper.py)。

粒子编号（与文档一致）：$K^+=1$，$K^-=2$，$\pi^+=3$，$\pi^-=4$
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


def u_PP_2(psi, phi, f2, Kp, Km, Pip, Pim):
    kst2 = Kp + Km + Pip
    p2psi_con = _raise_last(P2(psi), 4)
    gt_ks_cov = Gt_cov(kst2)
    t1_top_cov = T1_cov(psi, kst2, Pim)          # $\tilde T^{(1)}_{(K^*_2 3)}$
    t1_in_cov = T1_cov(kst2, phi, Pip)           # $\tilde t^{(1)}_{(\phi 3)}$
    t1_12_cov = T1_cov(phi, Kp, Km)
    t1_12_con = _metric_axis(t1_12_cov, 1)
    gphi = _einsum("elm,em->el", gt_ks_cov, t1_12_con)
    t2_in_cov = T2_from_pair(kst2, t1_in_cov, gphi)
    return _einsum("eijkl,ej,ekl->ei", p2psi_con, t1_top_cov, t2_in_cov)
