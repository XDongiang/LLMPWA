#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Eq.(48) 第一项：$U^\mu_{b_1 SD}$ —— $\psi\to b_1\pi$ S 波、$b_1\to\phi\pi$ D 波。

文档《ψ衰变到介子的分波分析协变张量形式》391–416 行。
参考实现 `Tensor/FUNC.py::_u_SD_1`（jax），本文件是逐式 numpy 移植。

粒子编号（与文档一致）：K+=1，K-=2，pi+=3，pi-=4

本项：$b_1$ 由 $(1,2,3)=\phi\pi^+$ 重建，D 波给出
$\tilde t^{(2)\mu\nu}_{(\phi 3)}$；与 $\tilde t^{(1)}_{(12)\nu}$ 缩并。

顶层 (l,s) = (2,1)
"""
from __future__ import annotations

import numpy as np

from ...kernels.covariant import T1_cov, T2_cov, _raise_last


def u_SD_1(psi, phi, f2, Kp, Km, Pip, Pim):
    b1 = Kp + Km + Pip
    t2_cov = T2_cov(b1, phi, Pip)       # \tilde t^{(2)}_{(\phi 3)\mu\nu}
    t2_con = _raise_last(t2_cov, 2)
    t1_cov = T1_cov(phi, Kp, Km)        # \tilde t^{(1)}_{(12)\nu}
    return np.einsum("eij,ej->ei", t2_con, t1_cov)
