#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 6：$<\phi f_2|43>_{(\sigma\sigma)}$，配对 (36)(45)。

文档 659–662 行：

$$
<\phi f_2|43>_{(\sigma\sigma)} = \tilde T^{(4)\mu\nu\lambda\sigma}_{(\phi f_2)}
\tilde t^{(1)}_{(12)\nu} T^{(f_2)}_{(\sigma\sigma)\lambda\sigma}
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


def u_phif2_ss_43_b(psi, phi, f2, Kp, Km, Pip, Pim, Pip2, Pim2):
    r"""配对 (36)(45)"""
    t4_con = _raise_last(T4_cov(psi, phi, f2), 4)
    return _einsum("eijkl,ej,ekl->ei", t4_con,
                   T1_cov(phi, Kp, Km), T2_ss(f2, Pip, Pim2, Pip2, Pim))
