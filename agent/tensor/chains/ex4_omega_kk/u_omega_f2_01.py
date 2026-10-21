#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 4：\(<\omega f_2|01>\)。

文档 418–430 行 + 例 3 的 Eq.(41)：

$$
<\omega f_2|01> = \tilde t^{(2)\mu\nu}_{(34)}\omega_\nu
$$

$f_2$ 由 $K^+K^-$（粒子 3、4）重建，$\tilde t^{(2)}_{(34)}$ 是它的
$J=2$ 张量；$\omega^\mu$ 顶在 $\mu$ 上。
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import T2_cov, _raise_last, omega_mu  # noqa: F401


def u_omega_f2_01(psi, omega, f2, Kp, Km, Pi0, Pip, Pim):
    r"""$<\omega f_2|01>$。"""
    t2_34 = _raise_last(T2_cov(f2, Kp, Km), 2)   # $\tilde t^{(2)\mu}{}_\nu$
    w = omega_mu(omega, Pi0, Pip, Pim)
    return xp.einsum("eij,ej->ei", t2_34, w)
