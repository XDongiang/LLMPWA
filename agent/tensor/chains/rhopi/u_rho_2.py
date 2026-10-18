#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Eq.(46)：$U^\mu_{\rho'}$ 的第二项 —— $\psi\to\rho'\pi^-$，$\rho'\to\phi\pi^+$

文档《ψ衰变到介子的分波分析协变张量形式》374–387 行。
参考实现 `Tensor/FUNC.py::_u_rho_2`（jax），本文件是**逐式 numpy 移植** ——
只把 `jax.numpy` 换成 `numpy`、把 `self.psi2s` 提成显式参数 `psi`，
公式一个数字都没动。

粒子编号（与文档一致）：K+=1，K-=2，pi+=3，pi-=4

本项：$\rho'$ 由 $(1,2,3)=\phi\pi^+$ 重建（`rho = Kp + Km + Pip`），
$\pi^-$（粒子 4）是旁观者，故 $\tilde T^{(1)}_{(\rho'4)}$ 的第三态用 `Pim`。

顶层 (l,s) = (1,1)
"""
from __future__ import annotations

import numpy as np

from ...kernels.covariant import (
    T1_cov, _einsum, _leci_con, _metric_axis, leci_four,

)


def u_rho_2(psi, phi, f2, Kp, Km, Pip, Pim):
    leci_cov = leci_four()
    leci_con = _leci_con()
    leci_mu = _metric_axis(leci_cov, 0)
    psi_cov = _metric_axis(psi, 1)

    rho = Kp + Km + Pip
    t1_rho_cov = T1_cov(psi, rho, Pim)      # \tilde T^{(1)}_{(\rho'4)}
    t1_rho_con = _metric_axis(t1_rho_cov, 1)
    t1_phi3_cov = T1_cov(rho, phi, Pip)     # \tilde t^{(1)}_{(\phi 3)}
    t1_12_cov = T1_cov(phi, Kp, Km)

    return _einsum("ijkl,ej,ek,lmno,em,en,eo->ei",
                   leci_mu, psi, t1_rho_con, leci_con, psi_cov,
                   t1_phi3_cov, t1_12_cov)
