#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Eq.(47) 第二项：$U^\mu_{b_1 SS}$ —— 把粒子 3 与 4 对调。

文档《ψ衰变到介子的分波分析协变张量形式》391–416 行。
参考实现 `Tensor/FUNC.py::_u_SS_2`（jax），本文件是逐式 numpy 移植。

粒子编号（与文档一致）：K+=1，K-=2，pi+=3，pi-=4

本项：$b_1$ 由 $(1,2,4)=\phi\pi^-$ 重建（`b1 = Kp + Km + Pim`），
$\pi^+$（粒子 3）是旁观者。

顶层 (l,s) = (0,1)
"""
from __future__ import annotations

import numpy as np

from ...kernels.covariant import Gt_con, T1_cov


def u_SS_2(psi, phi, f2, Kp, Km, Pip, Pim):
    b1 = Kp + Km + Pim
    gt_con = Gt_con(b1)                 # \tilde g^{\mu\nu}_{(124)}
    t1_cov = T1_cov(phi, Kp, Km)        # \tilde t^{(1)}_{(12)\nu}
    return np.einsum("eij,ej->ei", gt_con, t1_cov)
