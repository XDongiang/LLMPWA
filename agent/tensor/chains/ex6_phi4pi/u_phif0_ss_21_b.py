#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 6：\(<\phi f_0|21>_{(\sigma\sigma)}\) 的第二个配对 $(36)(45)$。

与 `u_phif0_ss_21` 函数体相同（$f_0$ 是标量，配对只影响 $\sigma$ 的
传播子，而传播子在道外）。
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import T1_cov, T2_cov, _raise_last  # noqa: F401


def u_phif0_ss_21_b(psi, phi, f0, Kp, Km, Pip, Pim, Pip2, Pim2):
    r"""\(<\phi f_0|21>_{(\sigma\sigma)}\)，配对 (36)(45)。"""
    t2_con = _raise_last(T2_cov(psi, phi, f0), 2)
    return xp.einsum("eij,ej->ei", t2_con, T1_cov(phi, Kp, Km))
