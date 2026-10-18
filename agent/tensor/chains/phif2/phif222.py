#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eq.(43)：<phi f_2|22>。

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
    _einsum, _leci_con, _metric_axis, _raise_last, _s1_vector,
)


def phif222(psi, phi, f2, Kp, Km, Pip, Pim):
    r"""Eq.(43)：<\phi f_2|22>。"""
    leci_cov = leci_four()
    leci_con = _leci_con()
    psi2s_cov = _metric_axis(psi, 1)
    t2_phif2 = _metric_axis(T2_cov(psi, phi, f2), 2)
    t2_34 = _metric_axis(T2_cov(f2, Pip, Pim), 1)
    t1_con = _metric_axis(T1_cov(phi, Kp, Km), 1)
    # 这两条 7 操作数收缩是整条道的全部代价：默认路径 0.10 s / 128 事例，
    # `_einsum` 的 optimal 路径 0.0001 s —— 差 1000x，结果逐位相同。
    return (_einsum("ijkl,ej,emk,lnop,enm,eo,ep->ei",
                    leci_con, psi2s_cov, t2_phif2, leci_cov, t2_34, psi, t1_con)
            + _einsum("ijkl,ej,emk,mnop,enl,eo,ep->ei",
                      leci_con, psi2s_cov, t2_phif2, leci_cov, t2_34, psi, t1_con))
