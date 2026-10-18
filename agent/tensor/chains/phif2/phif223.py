#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eq.(44)：<phi f_2|23>。

本文件是 **协变张量内核的一条道**，按 `chain.<name>/chain.<channel>.py` 的
约定独立存放；`agent/tensor/emitter.py` 在生成时按道表把它们组装起来。
依赖 `..kernels.covariant` 里的原语，公式与 `Tensor/FUNC.py` 逐式对应。
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import (  # noqa: F401
    Gt_con, P3, T1_cov, T2_cov, T4_cov, dot, leci_four, metric,
    _leci_con, _metric_axis, _raise_last, _s1_vector,
)


def phif223(psi, phi, f2, Kp, Km, Pip, Pim):
    r"""Eq.(44)：<\phi f_2|23>。"""
    p3_cov = P3(psi)
    p3_con = _raise_last(p3_cov, 6)
    t2_phif2 = T2_cov(psi, phi, f2)
    t2_34 = T2_cov(f2, Pip, Pim)
    t1_12 = T1_cov(phi, Kp, Km)
    return xp.einsum("eijklmn,ejk,elm,en->ei", p3_con, t2_phif2, t2_34, t1_12)
