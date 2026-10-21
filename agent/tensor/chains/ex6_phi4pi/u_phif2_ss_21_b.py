#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 6：$<\phi f_2|21>_{(\sigma\sigma)}$，配对 (36)(45)。

文档 642–645 行：

$$
<\phi f_2|21>_{(\sigma\sigma)} = \tilde T^{(2)\mu\alpha}_{(\phi f_2)}
\tilde T^{(f_2)}_{(\sigma\sigma)\alpha\nu}\tilde t^{(1)\nu}_{(12)}
f^{(\phi)}_{(12)} f^{(f_2)}_{(\sigma\sigma)}
$$
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import (  # noqa: F401
    T1_cov, T2_cov, T2_ss, _einsum, _raise_last, s1_from_T2,
)


def u_phif2_ss_21_b(psi, phi, f2, Kp, Km, Pip, Pim, Pip2, Pim2):
    r"""配对 (36)(45)"""
    w = s1_from_T2(psi, T2_ss(f2, Pip, Pim2, Pip2, Pim), T1_cov(phi, Kp, Km))
    t2_con = _raise_last(T2_cov(psi, phi, f2), 2)
    return _einsum("eij,ej->ei", t2_con, w)
