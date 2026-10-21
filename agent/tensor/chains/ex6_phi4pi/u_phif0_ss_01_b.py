#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 6：\(<\phi f_0|01>_{(\sigma\sigma)}\) 的**第二个**配对：$(\pi_3\pi_6)(\pi_5\pi_4)$。

文档 632–633 行的第二个方括号项
$f^{(\sigma)}_{(36)}f^{(\sigma)}_{(45)}$。

⚠️ 这一项在 $<\phi f_0|01>$ 里是**标量**（$\tilde t^{(1)}_{(12)}$ 是
$\phi$ 的分析器，与 $\pi$ 的配对无关），所以本道与 `u_phif0_ss_01`
**函数体完全相同**。两条道的区别只在 $g_i$（以及级联侧的配对树）。
保留成两条道是为了让 `(l,s)` 表与文档的两项一一对应。
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import T1_cov  # noqa: F401


def u_phif0_ss_01_b(psi, phi, f0, Kp, Km, Pip, Pim, Pip2, Pim2):
    r"""\(<\phi f_0|01>_{(\sigma\sigma)}\)，配对 (36)(45)。"""
    return T1_cov(phi, Kp, Km)
