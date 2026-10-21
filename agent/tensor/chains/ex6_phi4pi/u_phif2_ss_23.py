#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 6：$<\phi f_2|23>_{(\sigma\sigma)}$，配对 (34)(56)。

文档 654–658 行：

$$
<\phi f_2|23>_{(\sigma\sigma)} = P^{(3)\mu\alpha\beta\gamma\delta\nu}(p_\psi)
\tilde T^{(2)}_{(\phi f_2)\alpha\beta}
\tilde T^{(f_2)}_{(\sigma\sigma)\gamma\delta}\tilde t^{(1)}_{(12)\nu}
f^{(\phi)}_{(12)}f^{(f_2)}_{(\sigma\sigma)}
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
    _metric_axis, _raise_last, s1_from_T2,
)


def u_phif2_ss_23(psi, phi, f2, Kp, Km, Pip, Pim, Pip2, Pim2):
    r"""配对 (34)(56)"""
    p3_con = _raise_last(P3(psi), 6)
    return _einsum("eijklmn,ejk,elm,en->ei", p3_con,
                   T2_cov(psi, phi, f2), T2_ss(f2, Pip, Pim, Pip2, Pim2),
                   T1_cov(phi, Kp, Km))
