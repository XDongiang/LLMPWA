#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 4：\(<\omega f_2|21>\)。

文档 418–430 行 + 例 3 的 Eq.(42)：

$$
<\omega f_2|21> = \tilde T^{(2)\mu\alpha}_{(\omega f_2)}
\tilde t^{(2)}_{(34)\alpha\nu}\omega^\nu
$$
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import (  # noqa: F401
    T2_cov, _einsum, _raise_last, omega_mu,
)


def u_omega_f2_21(psi, omega, f2, Kp, Km, Pi0, Pip, Pim):
    r"""$<\omega f_2|21>$。"""
    w = omega_mu(omega, Pi0, Pip, Pim)
    t2_top = _raise_last(T2_cov(psi, omega, f2), 2)
    t2_34 = _raise_last(T2_cov(f2, Kp, Km), 2)
    return _einsum("eij,ejk,ek->ei", t2_top, t2_34, w)
