#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 6：$<\phi f_2|22>_{(\sigma\sigma)}$，配对 (36)(45)。

文档 646–653 行（两个 $\epsilon$ 的 $J=2\otimes2\to1$ 耦合）：

$$
<\phi f_2|22>_{(\sigma\sigma)} = \epsilon^{\mu\alpha\beta\gamma}
p_{\psi\alpha}\tilde T^{(2)\delta}_{(\phi f_2)\beta}
[\epsilon_{\gamma\lambda\sigma\nu}\tilde T^{(f_2)\lambda}_{(\sigma\sigma)\delta}
+\epsilon_{\delta\lambda\sigma\nu}\tilde T^{(f_2)\lambda}_{(\sigma\sigma)\gamma}]
p_\psi^\sigma \tilde t^{(1)\nu}_{(12)} f^{(\phi)}_{(12)}f^{(f_2)}_{(\sigma\sigma)}
$$
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import (  # noqa: F401
    P3, T1_cov, T2_cov, T2_ss, T4_cov, _einsum, _leci_con,
    _metric_axis, _raise_last, leci_four, s1_from_T2,
)


def u_phif2_ss_22_b(psi, phi, f2, Kp, Km, Pip, Pim, Pip2, Pim2):
    r"""配对 (36)(45)"""
    leci_cov, leci_con = leci_four(), _leci_con()
    psi_cov = _metric_axis(psi, 1)
    t2_top = _metric_axis(T2_cov(psi, phi, f2), 2)
    t1_con = _metric_axis(T1_cov(phi, Kp, Km), 1)
    t2_ss_c = _metric_axis(T2_ss(f2, Pip, Pim2, Pip2, Pim), 1)
    return (_einsum("ijkl,ej,emk,lnop,enm,eo,ep->ei",
                    leci_con, psi_cov, t2_top, leci_cov, t2_ss_c, psi, t1_con)
            + _einsum("ijkl,ej,emk,mnop,enl,eo,ep->ei",
                      leci_con, psi_cov, t2_top, leci_cov, t2_ss_c, psi, t1_con))
