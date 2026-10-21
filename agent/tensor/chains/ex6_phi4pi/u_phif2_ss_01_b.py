#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 6：$<\phi f_2|01>_{(\sigma\sigma)}$，配对 (36)(45)。

文档 639–641 行：

$$
<\phi f_2|01>_{(\sigma\sigma)} = T^{(f_2)\mu\nu}_{(\sigma\sigma)}
\tilde t^{(1)}_{(12)\nu} f^{(\phi)}_{(12)}f^{(f_2)}_{(\sigma\sigma)}
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


def u_phif2_ss_01_b(psi, phi, f2, Kp, Km, Pip, Pim, Pip2, Pim2):
    r"""配对 (36)(45)"""
    t2_ss = T2_ss(f2, Pip, Pim2, Pip2, Pim)
    return s1_from_T2(psi, t2_ss, T1_cov(phi, Kp, Km))
