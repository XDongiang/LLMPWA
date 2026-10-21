#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""$U^\mu_{K^*_2}$ PP（左）：顶层 $P$，内层 $P$ —— **历史形式（未修正）**

文档《ψ衰变到介子的分波分析协变张量形式》431–623 行（$\psi\to K^+\pi^-K^-\pi^+$）。
参考实现 `Tensor/FUNC.py::_u_PP_l`（jax），本文件是**逐式移植**（走 `_backend.xp`，
jax / numpy 两后端通用）：`jax.numpy` 换成后端命名空间、`self.psi2s` 提成显式
参数 `psi`，公式一个数字都没动。

⚠️ **这是历史形式，不用于生产。** 它把 $1\otimes1\to2$ 的无迹化交给了
$P^{(2)}(p_\psi)$ —— 那是 $\psi$ 系的度规，而耦合发生在 $K^*_2$ 系，
所以对级联的 A1 残差停在 $6.99\times10^{-3}$。

* **生产用**：同目录的 [`u_PP_1.py`](u_PP_1.py)（修正版）。
* **对照说明**：[`CORRECTION.md`](CORRECTION.md)（公式 + 代码逐行对照）。
* **为什么保留**：这是与 jax 参考实现对拍的落点（A5 要求逐位一致），
  也是"论文形式 → 修正形式"的 diff 基线。

要在产物里生成**这一种**形式，在 spec 里把道名写成 `u_PP_1_paper`：

```toml
[[channel]]
name = "u_PP_1_paper"
l = 1
s = 2
ls = { psi2s = [[1, 2]], Kstar = [[1, 1]], phi = [[1, 0]] }
tree = { psi2s = ["Kstar", "Pip"], Kstar = ["phi", "Pim"] }
```

粒子编号（与文档一致）：$K^+=1$，$K^-=2$，$\pi^+=3$，$\pi^-=4$

拓扑（**深度 3**，与 b₁ 同构）：$\psi\to K^*\pi \to (\phi\pi)\pi \to (K^+K^-)(\pi\pi)$
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import (
    _einsum, _metric_axis, _raise_last, Gt_cov, P2, T1_cov
)


def u_PP_1_paper(psi, phi, f2, Kp, Km, Pip, Pim):
    kst2 = Kp + Km + Pim
    p2psi_cov = P2(psi)                     # $P^{(2)}_{\psi}$
    p2psi_con = _raise_last(p2psi_cov, 4)
    gt_ks_cov = Gt_cov(kst2)                # $\tilde g_{K^*_2}$（协变）
    t1_top_cov = T1_cov(psi, kst2, Pip)    # $\tilde T^{(1)}_{(K^*_2 4)}$
    t1_in_cov = T1_cov(kst2, phi, Pim)     # $\tilde t^{(1)}_{(\phi 4)}$
    t1_12_cov = T1_cov(phi, Kp, Km)         # $\tilde t^{(1)}_{(12)}$
    t1_12_con = _metric_axis(t1_12_cov, 1)
    return _einsum("eijkl,ej,ek,elm,em->ei",
                   p2psi_con, t1_top_cov, t1_in_cov, gt_ks_cov, t1_12_con)
