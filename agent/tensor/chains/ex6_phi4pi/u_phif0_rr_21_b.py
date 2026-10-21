#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 6：$<\phi f_0|21>_{(\rho\rho)}$，配对 (36)(45)。

文档 687–694 行。与 `01` 的区别是多一个顶层 $L=2$：
$$
<\phi f_0|21>_{(\rho\rho)}=\tilde T^{(2)\mu\nu}_{(\phi f_0)}
\Bigl[\tilde g(p_{f_0})^{\alpha\beta}\tilde t^{(1)}_{(34)\alpha}
\tilde t^{(1)}_{(56)\beta}\Bigr]\tilde t^{(1)}_{(12)\nu}
$$
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import (  # noqa: F401
    S_rr, T1_cov, T2_cov, _einsum, _raise_last,
)


def u_phif0_rr_21_b(psi, phi, f0, Kp, Km, Pip, Pim, Pip2, Pim2):
    r"""配对 (36)(45)"""
    s_rr = S_rr(f0, Pip, Pim2, Pip2, Pim)
    w = s_rr[:, None] * T1_cov(phi, Kp, Km)
    t2_con = _raise_last(T2_cov(psi, phi, f0), 2)
    return _einsum("eij,ej->ei", t2_con, w)
