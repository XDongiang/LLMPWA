#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 4：\(<\omega f_0|21>\)。

文档 418–430 行 + 例 3 的 Eq.(37)：

$$
<\omega f_0|21> = \tilde T^{(2)\mu\nu}_{(\omega f_0)}\omega_\nu
$$

与 `u_omega_f0_01` 的区别：多一个顶层 $L=2$ 算符
$\tilde T^{(2)}_{(\omega f_0)}$，它来自 $\psi$ 与 $\omega f_0$ 之间的
轨道角动量。
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import T2_cov, _raise_last, omega_mu  # noqa: F401


def u_omega_f0_21(psi, omega, f0, Kp, Km, Pi0, Pip, Pim):
    r"""$<\omega f_0|21>$。"""
    w = omega_mu(omega, Pi0, Pip, Pim)
    t2_con = _raise_last(T2_cov(psi, omega, f0), 2)
    return xp.einsum("eij,ej->ei", t2_con, w)
