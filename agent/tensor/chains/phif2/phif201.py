#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eq.(41) 修正版：<phi f_2|01>。

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


def phif201(psi, phi, f2, Kp, Km, Pip, Pim):
    r"""Eq.(41) 修正版：<\phi f_2|01>。"""
    return _s1_vector(psi, phi, f2, Kp, Km, Pip, Pim)
