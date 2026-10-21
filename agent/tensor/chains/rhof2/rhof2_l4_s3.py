#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""假想链 $\psi(2S)\to\rho f_2$ 的 `rhof2_l4_s3`（与 `phif2::phif243` 同式）

本文件是 `chains/phif2/phif243.py` 的**逐式改名**（$\phi\to\rho$、
$K^\pm\to\pi^\pm$），公式一个数字都没动 —— `rhof2_generic.toml` 与
`phif2.toml` **拓扑完全相同**（同 $(L,S)$、同 $J$），只是末态换了粒子。
对拍见 `probes/probe_missing_covariant.py`（rel ~ 1e-15）。

$[chain].chain_name = "rhof2"$，道表是**自动枚举**的（没有 `[[channel]]`），
所以文件名必须与 `spec.resolve_channels` 生成的道名一致。
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


def rhof2_l4_s3(psi, rho, f2, Pip, Pim, Pip2, Pim2):
    r"""Eq.(45)：<\rho f_2|43>。"""
    t4_cov = T4_cov(psi, rho, f2)
    t4_con = _raise_last(t4_cov, 4)
    t2_cov = T2_cov(f2, Pip2, Pim2)
    t1_cov = T1_cov(rho, Pip, Pim)
    return xp.einsum("eijkl,ej,ekl->ei", t4_con, t1_cov, t2_cov)
