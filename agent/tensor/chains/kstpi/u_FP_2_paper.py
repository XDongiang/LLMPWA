#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""$U^\mu_{K^*_2}$ FP（右）：顶层 $F$，内层 $P$ —— **历史形式（未修正）**

与 `u_FP_1_paper` 互为旁观者 π 对调（$3\leftrightarrow 4$）。
参考实现 `Tensor/FUNC.py::_u_FP_r` 的逐式移植。

⚠️ **历史形式，不用于生产** —— 见 [`u_FP_1_paper.py`](u_FP_1_paper.py) 的说明
与 [`CORRECTION.md`](CORRECTION.md)。
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


def u_FP_2_paper(psi, phi, f2, Kp, Km, Pip, Pim):
    kst2 = Kp + Km + Pip
    p2psi_cov = P2(psi)
    gt_ks_con = _raise_last(Gt_cov(kst2), 2)
    t3_top_cov = T3_cov(psi, kst2, Pim)       # $\tilde T^{(3)}_{(K^*_2 3)}$
    t3_top_con = _raise_last(t3_top_cov, 3)
    t1_in_cov = T1_cov(kst2, phi, Pip)
    t1_in_con = _metric_axis(t1_in_cov, 1)
    t1_12_cov = T1_cov(phi, Kp, Km)
    return _einsum("eijk,ejklm,el,emn,en->ei",
                   t3_top_con, p2psi_cov, t1_in_con, gt_ks_con, t1_12_cov)
