#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eq.(45)：<phi f_2|43>。

本文件是 **协变张量内核的一条道**，按 `chain.<name>/chain.<channel>.py` 的
约定独立存放；`agent/tensor/emitter.py` 在生成时按道表把它们组装起来。
依赖 `..kernels.covariant` 里的原语，公式与 `Tensor/FUNC.py` 逐式对应。
"""
from __future__ import annotations

import numpy as np

from ...kernels.covariant import (  # noqa: F401
    Gt_con, P3, T1_cov, T2_cov, T4_cov, dot, leci_four, metric,
    _leci_con, _metric_axis, _raise_last, _s1_vector,
)


def phif243(psi, phi, f2, Kp, Km, Pip, Pim):
    r"""Eq.(45)：<\phi f_2|43>。"""
    t4_cov = T4_cov(psi, phi, f2)
    t4_con = _raise_last(t4_cov, 4)
    t2_cov = T2_cov(f2, Pip, Pim)
    t1_cov = T1_cov(phi, Kp, Km)
    return np.einsum("eijkl,ej,ekl->ei", t4_con, t1_cov, t2_cov)
