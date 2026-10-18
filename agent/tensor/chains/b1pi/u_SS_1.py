#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Eq.(47) 第一项：$U^\mu_{b_1 SS}$ —— $\psi\to b_1\pi^-$ 与 $b_1\to\phi\pi^+$ 都是 S 波。

文档《ψ衰变到介子的分波分析协变张量形式》391–416 行。
参考实现 `Tensor/FUNC.py::_u_SS_1`（jax），本文件是逐式 numpy 移植。

粒子编号（与文档一致）：K+=1，K-=2，pi+=3，pi-=4

本项：$b_1$ 由 $(1,2,3)=\phi\pi^+$ 重建（`b1 = Kp + Km + Pip`），
$\pi^-$（粒子 4）是旁观者。$\psi\to b_1\pi$ 的 S 波给出
$\tilde g^{\mu\nu}_{(123)}$，与 $\tilde t^{(1)}_{(12)\nu}$ 缩并即得。

顶层 (l,s) = (0,1)
"""
from __future__ import annotations

import numpy as np

from ...kernels.covariant import Gt_con, T1_cov


def u_SS_1(psi, phi, f2, Kp, Km, Pip, Pim):
    b1 = Kp + Km + Pip
    gt_con = Gt_con(b1)                 # \tilde g^{\mu\nu}_{(123)}
    t1_cov = T1_cov(phi, Kp, Km)        # \tilde t^{(1)}_{(12)\nu}
    return np.einsum("eij,ej->ei", gt_con, t1_cov)
