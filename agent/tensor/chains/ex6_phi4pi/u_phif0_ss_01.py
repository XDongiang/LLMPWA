#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 6：$<\phi f_0|01>_{(\sigma\sigma)}$。

文档 630–633 行：

$$
<\phi f_0|01>_{(\sigma\sigma)} = \tilde t^{(1)\mu}_{(12)}
f^{(\phi)}_{(12)}f^{(f_0)}_{(\sigma\sigma)}
\Bigl[f^{(\sigma)}_{(34)}f^{(\sigma)}_{(56)}
+f^{(\sigma)}_{(36)}f^{(\sigma)}_{(45)}\Bigr]
$$

本道只给**一个** $\sigma\sigma$ 配对：$(\pi_3\pi_4)$ 与 $(\pi_5\pi_6)$。
另一个配对 $(\pi_3\pi_6)(\pi_5\pi_4)$ 是**另一条道**
（`u_phif0_ss_01_b`），两者的 $g_i$ 由拟合定。

注意：文档把两个配对写在同一个分波振幅里、共用一个耦合。框架里
"同一份 spec 多条道 + 逐道 $g_i$" 已经能表达"两项相加"，而若真要让
两项共用一个 $g$，把两个 $g_i$ 约束成相等即可 —— 这是拟合层的事。
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import T1_cov  # noqa: F401


def u_phif0_ss_01(psi, phi, f0, Kp, Km, Pip, Pim, Pip2, Pim2):
    r"""$<\phi f_0|01>_{(\sigma\sigma)}$，配对 (34)(56)。"""
    return T1_cov(phi, Kp, Km)
