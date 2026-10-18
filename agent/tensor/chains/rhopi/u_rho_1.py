#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Eq.(46)：$U^\mu_{\rho'}$ 的第一项 —— $\psi\to\rho'\pi^+$，$\rho'\to\phi\pi^-$

文档《ψ衰变到介子的分波分析协变张量形式》374–387 行。
参考实现 `Tensor/FUNC.py::_u_rho_1`（jax），本文件是**逐式移植**
（走 `_backend.xp`，jax / numpy 两后端通用）：把 `jax.numpy` 换成后端命名空间、
把 `self.psi2s` 提成显式参数 `psi`，公式一个数字都没动。

粒子编号（与文档一致）：K+=1，K-=2，pi+=3，pi-=4

本项：$\rho'$ 由 $(1,2,4)=\phi\pi^-$ 重建（`rho = Kp + Km + Pim`），
$\pi^+$（粒子 3）是旁观者，故 $\tilde T^{(1)}_{(\rho'3)}$ 的第三态用 `Pip`。
姊妹项 `u_rho_2` 把 3 <-> 4 对调。

顶层 (l,s) = (1,1)
"""
from __future__ import annotations

try:                      # 包内（正常路径）
    from ...kernels import _backend as _B
except ImportError:       # 被 emitter 内联进产物脚本后
    import _backend as _B

xp = _B.xp

from ...kernels.covariant import (
    Gt_con, T1_cov, _einsum, _leci_con, _metric_axis, leci_four, metric,

)


def u_rho_1(psi, phi, f2, Kp, Km, Pip, Pim):
    leci_cov = leci_four()
    leci_con = _leci_con()
    # 参考实现写的是 np.einsum("ijkl,in->njkl", leci_cov, G)（对角度规升第 0 个指标）。
    # 度规是对角的，所以这与 _metric_axis(leci_cov, 0) **逐位相同** —— 见
    # kernels/covariant.py 顶部关于"用 G 升指标退化成乘符号外积"的说明。
    leci_mu = _metric_axis(leci_cov, 0)
    psi_cov = _metric_axis(psi, 1)          # \P_{\psi \alpha} 协变

    rho = Kp + Km + Pim
    t1_rho_cov = T1_cov(psi, rho, Pip)      # \tilde T^{(1)}_{(\rho'3)}
    t1_rho_con = _metric_axis(t1_rho_cov, 1)
    t1_phi4_cov = T1_cov(rho, phi, Pim)     # \tilde t^{(1)}_{(\phi 4)}
    t1_12_cov = T1_cov(phi, Kp, Km)         # \tilde t^{(1)}_{(12)}

    # \epsilon^{\mu}_{\alpha\beta\gamma} p_\psi^\alpha
    #   \tilde T^{(1)\beta}_{(\rho'3)}
    #   \epsilon^{\gamma\delta\sigma\lambda} p_{\psi\delta}
    #   \tilde t^{(1)}_{(\phi 4)\sigma} \tilde t^{(1)}_{(12)\lambda}
    return _einsum("ijkl,ej,ek,lmno,em,en,eo->ei",
                   leci_mu, psi, t1_rho_con, leci_con, psi_cov,
                   t1_phi4_cov, t1_12_cov)
