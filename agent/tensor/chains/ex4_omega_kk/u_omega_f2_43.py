#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 4：\(<\omega f_2|43>\)。

文档 418–430 行 + 例 3 的 Eq.(45)：

$$
<\omega f_2|43> = \tilde T^{(4)\mu\nu\lambda\sigma}_{(\omega f_2)}
\omega_\nu\tilde t^{(2)}_{(34)\lambda\sigma}
$$

顶层 $L=4$（秩 4 算符），$f_2$ 的 $J=2$ 与 $\omega$ 的 $J=1$ 组成
$S=3$，$L=4$ 与 $S=3$ 给出 $\psi$ 的 $J=1$。
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import (  # noqa: F401
    T2_cov, T4_cov, _raise_last, omega_mu,
)


def u_omega_f2_43(psi, omega, f2, Kp, Km, Pi0, Pip, Pim):
    r"""$<\omega f_2|43>$。"""
    t4_con = _raise_last(T4_cov(psi, omega, f2), 4)
    t2_34 = T2_cov(f2, Kp, Km)
    w = omega_mu(omega, Pi0, Pip, Pim)
    return xp.einsum("eijkl,ej,ekl->ei", t4_con, w, t2_34)
