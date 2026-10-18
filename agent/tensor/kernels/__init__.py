#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tensor 内核。

* `omega.py`     —— 与 tf-pwa 无关的螺旋度级联运行时（CG / Wigner-D /
                   级联角 / Chung Eq.52）。纯 numpy，已与 tf-pwa 对拍 2.5e-15。
* `covariant.py` —— Chung 协变张量的 numpy 移植（A1 方案），与 jax 原版
                   `Tensor/{BASE,FUNC}.py` 逐式对拍 rel < 1.3e-15。
* `helicity.py`  —— `EPS_A ∘ covariant`，即螺度表象内核。

`covariant` 与 `omega` 是**两条独立实现**，它们的重合是框架最强的回归判据
（见 `agent/tensor/tests/test_cross_basis.py`）。
"""
from __future__ import annotations

__all__ = ["omega", "covariant", "helicity"]
