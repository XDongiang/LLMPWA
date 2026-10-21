#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""假想链 $\psi(2S)\to\rho f_2$ 的 `rhof2_l2_s2`（与 `phif2::phif222` 同式）

本文件是 `chains/phif2/phif222.py` 的**逐式改名**（$\phi\to\rho$、
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
    _einsum, _leci_con, _metric_axis, _raise_last, _s1_vector,
)


def rhof2_l2_s2(psi, rho, f2, Pip, Pim, Pip2, Pim2):
    r"""Eq.(43)：<\rho f_2|22>。"""
    leci_cov = leci_four()
    leci_con = _leci_con()
    psi2s_cov = _metric_axis(psi, 1)
    t2_rhof2 = _metric_axis(T2_cov(psi, rho, f2), 2)
    t2_34 = _metric_axis(T2_cov(f2, Pip2, Pim2), 1)
    t1_con = _metric_axis(T1_cov(rho, Pip, Pim), 1)
    # 这两条 7 操作数收缩是整条道的全部代价：默认路径 0.10 s / 128 事例，
    # `_einsum` 的 optimal 路径 0.0001 s —— 差 1000x，结果逐位相同。
    return (_einsum("ijkl,ej,emk,lnop,enm,eo,ep->ei",
                    leci_con, psi2s_cov, t2_rhof2, leci_cov, t2_34, psi, t1_con)
            + _einsum("ijkl,ej,emk,mnop,enl,eo,ep->ei",
                      leci_con, psi2s_cov, t2_rhof2, leci_cov, t2_34, psi, t1_con))
