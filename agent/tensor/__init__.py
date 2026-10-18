#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`agent.tensor` —— 框架级 tensor 缓存代码生成。

层级比 analysis 高一层：作为 `LLMPWA/agent` 的一部分，提供给**跨分析**使用；
一份 TOML 同时产出**协变张量**与**螺度**两种 tensor 的生成代码。

    agent/tensor/
      spec.py            统一 TOML schema：粒子/级联 + 道表 + [[basis]] 选择
      basis.py           4 个表象钩子（precompute/amplitude/intensity/env）
      chains/            chain.<name>/chain.<channel>.py 存储层（生成时组装）
      kernels/
        _backend.py      数值后端：jax 优先，无 jax 时回退 numpy
        omega.py         tf-pwa 无关的螺旋度级联运行时（CG / Wigner-D / Eq.52）
        covariant.py     Chung 协变张量（原 jax 版的逐式移植）
        helicity.py      EPS_A ∘ covariant
      emitter.py         按 basis 产出自足、**jax 优先**的生成脚本
      cli.py             python -m agent.tensor show|check|emit
"""
from __future__ import annotations

__all__ = ["spec", "basis", "emitter", "chains"]
