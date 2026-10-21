#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 4：\(<\omega f_2|22>\)。

文档 418–430 行 + 例 3 的 Eq.(43)：

$$
<\omega f_2|22> = \epsilon^{\mu\alpha\beta\gamma}p_{\psi\alpha}
\tilde T^{(2)\delta}_{(\omega f_2)\beta}
[\epsilon_{\gamma\lambda\sigma\nu}\tilde t^{(2)\lambda}_{(34)\delta}
+\epsilon_{\delta\lambda\sigma\nu}\tilde t^{(2)\lambda}_{(34)\gamma}]
p_\psi^\sigma \omega^\nu
$$

$J=2$ 与 $J=2$ 耦合成 $\psi$ 的 $J=1$，需两个 Levi-Civita
（一个把 $2\otimes2\to1$，一个把 $1\otimes1\to1$）。
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import (  # noqa: F401
    T2_cov, _einsum, _leci_con, _metric_axis, leci_four, omega_mu,
)


def u_omega_f2_22(psi, omega, f2, Kp, Km, Pi0, Pip, Pim):
    r"""$<\omega f_2|22>$。"""
    w = omega_mu(omega, Pi0, Pip, Pim)
    leci_cov = leci_four()
    leci_con = _leci_con()
    psi2s_cov = _metric_axis(psi, 1)
    t2_top = _metric_axis(T2_cov(psi, omega, f2), 2)
    t2_34 = _metric_axis(T2_cov(f2, Kp, Km), 1)
    w_con = _metric_axis(w, 1)
    return (_einsum("ijkl,ej,emk,lnop,enm,eo,ep->ei",
                    leci_con, psi2s_cov, t2_top, leci_cov, t2_34, psi, w_con)
            + _einsum("ijkl,ej,emk,mnop,enl,eo,ep->ei",
                      leci_con, psi2s_cov, t2_top, leci_cov, t2_34, psi, w_con))
