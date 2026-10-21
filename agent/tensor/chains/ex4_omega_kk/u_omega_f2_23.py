#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 4：\(<\omega f_2|23>\)。

文档 418–430 行 + 例 3 的 Eq.(44)：

$$
<\omega f_2|23> = P^{(3)\mu\alpha\beta\gamma\delta\nu}(p_\psi)
\tilde T^{(2)}_{(\omega f_2)\alpha\beta}\tilde t^{(2)}_{(34)\gamma\delta}
\omega_\nu
$$

$J=2\otimes2$ 耦合到 $J=3$ 用 $P^{(3)}$（完全对称、无迹、横向）。
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import (  # noqa: F401
    P3, T2_cov, _raise_last, omega_mu,
)


def u_omega_f2_23(psi, omega, f2, Kp, Km, Pi0, Pip, Pim):
    r"""$<\omega f_2|23>$。"""
    p3_con = _raise_last(P3(psi), 6)
    t2_top = T2_cov(psi, omega, f2)
    t2_34 = T2_cov(f2, Kp, Km)
    w = omega_mu(omega, Pi0, Pip, Pim)
    return xp.einsum("eijklmn,ejk,elm,en->ei", p3_con, t2_top, t2_34, w)
