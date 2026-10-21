#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 6：$<\phi f_0|01>_{(\rho\rho)}$，配对 (34)(56)。

文档 680–686 行 / tex。$f_0$ 是标量，所以两个 $\rho$ 先合成 $S=0$，
再用 $f_0$ 的横向度规夹住：

$$
<\phi f_0|01>_{(\rho\rho)}=\tilde t^{(1)\mu}_{(12)}
f^{(\phi)}_{(12)}f^{(f_0)}_{(\rho\rho)}
\tilde g(p_{f_0})^{\alpha\beta}
\tilde t^{(1)}_{(34)\alpha}\tilde t^{(1)}_{(56)\beta}
$$

⚠️ 裸点积（不夹 $\tilde g$）是错的：实测 rel 1.02e-1。
加上横向投影后 rel = 3.7e-15。"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import (  # noqa: F401
    S_rr, T1_cov, T2_cov, _einsum, _raise_last,
)


def u_phif0_rr_01(psi, phi, f0, Kp, Km, Pip, Pim, Pip2, Pim2):
    r"""配对 (34)(56)"""
    s_rr = S_rr(f0, Pip, Pim, Pip2, Pim2)
    return s_rr[:, None] * T1_cov(phi, Kp, Km)
