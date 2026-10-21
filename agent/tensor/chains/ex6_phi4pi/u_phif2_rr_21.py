#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 6：$<\phi f_2|21>_{(\rho\rho)}$，配对 (34)(56)。

文档 697–700 行。

⚠️ **本道与螺旋度级联对不上**（A1 必然失败）。逐式直译文档 697 行就是本
实现；问题不在抄错，而在 $1\otimes1\to2$ 的无迹化参考系 ——
$P^{(2)}(p_{f_2})$ 作用在**实验室系**张量上，缺少"先 boost 到 $f_2$ 系、
去迹、再 boost 回来"的等价物。见
`analyses/tensor_examples/chains/notes/ex6_phi4pi.notes.md`。
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import (  # noqa: F401
    T1_cov, T2_cov, T2_f2s, T2_rr, _einsum, _raise_last, s1_from_T2,
)


def u_phif2_rr_21(psi, phi, f2, Kp, Km, Pip, Pim, Pip2, Pim2):
    r"""配对 (34)(56)"""
    w = s1_from_T2(psi, T2_rr(f2, Pip, Pim, Pip2, Pim2), T1_cov(phi, Kp, Km))
    t2_con = _raise_last(T2_cov(psi, phi, f2), 2)
    return _einsum("eij,ej->ei", t2_con, w)
