#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 4：$<\omega f_0|01>$。

文档 418–430 行：把例 3（$\psi\to\phi\pi^+\pi^-$）各公式里的
$\tilde t^{(1)\mu}_{(12)}$ 换成 $\omega^\mu$、$f^{(\phi)}_{(12)}$ 换成
$f^{(\omega)}_{(012)}$，即得本道。

$$
<\omega f_0|01> = \omega^\mu f^{(\omega)}_{(012)}f^{(f_0)}_{(34)}
$$

协变张量路线里"$\omega^\mu$ 是一个整体分析器"这件事**只能用
`kernels/covariant.py::omega_mu` 表达** —— 它把 $\pi^0\pi^+\pi^-$ 三
个末态动量同时塞进 $\epsilon^{\mu\nu\lambda\sigma}$，没有对应的两体节点。
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import omega_mu  # noqa: F401


def u_omega_f0_01(psi, omega, f0, Kp, Km, Pi0, Pip, Pim):
    r"""$<\omega f_0|01>=\omega^\mu$（$f$ 因子都在道外）。"""
    return omega_mu(omega, Pi0, Pip, Pim)
