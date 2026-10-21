#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 6：\(<\phi f_0|21>_{(\sigma\sigma)}\)。

文档 634–638 行：

$$
<\phi f_0|21>_{(\sigma\sigma)} = \tilde T^{(2)\mu\nu}_{(\phi f_0)}
\tilde t^{(1)}_{(12)\nu}f^{(\phi)}_{(12)}f^{(f_0)}_{(\sigma\sigma)}
\Bigl[f^{(\sigma)}_{(34)}f^{(\sigma)}_{(56)}
+f^{(\sigma)}_{(36)}f^{(\sigma)}_{(45)}\Bigr]
$$
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import T1_cov, T2_cov, _raise_last  # noqa: F401


def u_phif0_ss_21(psi, phi, f0, Kp, Km, Pip, Pim, Pip2, Pim2):
    r"""\(<\phi f_0|21>_{(\sigma\sigma)}\)，配对 (34)(56)。"""
    t2_con = _raise_last(T2_cov(psi, phi, f0), 2)
    return xp.einsum("eij,ej->ei", t2_con, T1_cov(phi, Kp, Km))
