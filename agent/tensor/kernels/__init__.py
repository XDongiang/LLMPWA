#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tensor 内核。

* `_backend.py`  —— 数值后端选择：jax 优先，无 jax 时回退 numpy
                   （环境变量 `TENSOR_BACKEND` = auto / jax / numpy）。
* `omega.py`     —— 与 tf-pwa 无关的螺旋度级联运行时（CG / Wigner-D /
                   级联角 / Chung Eq.52）。已与 tf-pwa 对拍 2.5e-15。
* `covariant.py` —— Chung 协变张量（原 jax 版的逐式移植），与
                   `Tensor/{BASE,FUNC}.py` 对拍 rel < 1.3e-15。
* `helicity.py`  —— `EPS_A ∘ covariant`，即螺度表象内核。

三个内核都通过 `_backend.xp` 取数组命名空间，所以在 jax 下可被 `jit` 追踪、
在无 jax 环境下也能用 numpy 跑；两条后端的一致性由
`agent/tensor/tests/test_backend.py` 锁住（rel < 1e-12）。

`covariant` 与 `omega` 是**两条独立实现**，它们的重合是框架最强的回归判据
（见 `agent/tensor/tests/test_cross_basis.py`）。
"""
from __future__ import annotations

__all__ = ["omega", "covariant", "helicity"]
