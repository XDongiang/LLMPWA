#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""表象层（basis layer）—— 4 个表象钩子的注册表。

框架把"预计算什么、拟合时怎么用"这件事抽成四个钩子，每条路线只是这四件事的
一组选择：

    CACHE_BASIS      生成器写进 cache 的是哪个表象
                     "cartesian"（实数 T_i(e,mu)） | "helicity"（复数 W_i(e,lambda)）
                     | "omega"（与 (l,s) 无关的角向对象）
    ROTATE_AT_FIT    拟合时是否再做一次 EPS_A 旋转
    HELICITY_LAYER   强度是否只取 lambda = +-1
    N_COMPONENTS     cache 里存几个洛伦兹分量（2 -> (x,y)，3 -> (x,y,z)）
    precompute(T)    生成器要用的一步
    amplitude(blk)   拟合时对"耦合 x 传播子缩并完"的振幅块做的事
    intensity(blk)   强度
    env()            跑 run/ 里的共享 likelihood 时要设的环境变量

本模块是**表象层的规范定义**：`analyses/*/run/base_functions.py` 里那 255 行
表象层（`EPS_A` / `apply_helicity` / `intensity_from_amplitude` /
`_assemble_omega_cache` …）目前不在任何模板里，任何一次全量重跑生成器都会让
`likelihood_function.py` ImportError。框架提供这一层的模板，由分析内 agent 实例化。

四条路线（`4_三条路线/method{1..4}/basis.py`）逐条对应下面的注册项，数值已用
四路线拟合基准交叉验证（`four_route_fit_bench.md`）。
"""
from __future__ import annotations

import numpy as np

from .kernels import helicity as _hel

__all__ = ["BASES", "get", "names", "env_for", "describe_all", "check_consistency"]


class Basis:
    """一个表象的四个钩子 + 元数据。"""

    def __init__(self, name, title, cache_basis, rotate_at_fit, helicity_layer,
                 n_components, precompute, amplitude, intensity, env, describe):
        self.name = name
        self.title = title
        self.cache_basis = cache_basis
        self.rotate_at_fit = rotate_at_fit
        self.helicity_layer = helicity_layer
        self.n_components = n_components
        self._precompute = precompute
        self._amplitude = amplitude
        self._intensity = intensity
        self._env = env
        self._describe = describe

    def precompute(self, T):
        return self._precompute(T)

    def amplitude(self, contracted):
        return self._amplitude(contracted)

    def intensity(self, amplitude_block):
        return self._intensity(amplitude_block)

    def env(self):
        return dict(self._env())

    def describe(self):
        return self._describe()

    def __repr__(self):
        return "<Basis %s cache=%s rotate=%s hel=%s k=%d>" % (
            self.name, self.cache_basis, self.rotate_at_fit,
            self.helicity_layer, self.n_components)


# --------------------------------------------------------------- 路线 1
def _cartesian_precompute(T):
    """笛卡尔表象就是张量本身。"""
    return np.asarray(T, dtype=np.float64)[..., :3]


def _identity(contracted):
    return contracted


def _cartesian_intensity(blk):
    """I = |A_x|^2 + |A_y|^2。dplex 布局 (2, ..., k)，轴 0 = (Re, Im)。"""
    abs2 = blk[0] ** 2 + blk[1] ** 2
    return abs2[..., _hel.CARTESIAN_TRANSVERSE].sum(axis=-1)


# --------------------------------------------------------------- 路线 2
def _rot_amplitude(contracted):
    """把 dplex 布局 (2,...,k) 的笛卡尔块旋到 (2,...,3)。"""
    return _hel.amplitude(contracted)


def _hel_intensity(blk):
    return _hel.intensity(blk)


def _env(cache_basis, hel):
    return {"TENSOR_BASIS": cache_basis, "HELICITY_LAYER": "1" if hel else "0"}


BASES = {
    "cartesian": Basis(
        name="cartesian",
        title="协变张量（笛卡尔表象）",
        cache_basis="cartesian",
        rotate_at_fit=False,
        helicity_layer=False,
        n_components=3,
        precompute=_cartesian_precompute,
        amplitude=_identity,
        intensity=_cartesian_intensity,
        env=lambda: _env("cartesian", False),
        describe=lambda: (
            "预计算: T_i(e,mu) 实数 (x,y[,z])\n"
            "拟合  : I = |T_x|^2 + |T_y|^2\n"
            "注    : 螺旋度求和被吞掉了；lambda = 0 分量即使存了也不参与"),
    ),
    "cov_to_helicity": Basis(
        name="cov_to_helicity",
        title="协变张量 -> 拟合时旋到螺旋度表象",
        cache_basis="cartesian",     # 与 cartesian 逐位相同的 cache
        rotate_at_fit=True,
        helicity_layer=True,
        n_components=3,              # 要看到 lambda = 0 就必须存 z
        precompute=_cartesian_precompute,
        amplitude=_rot_amplitude,
        intensity=_hel_intensity,
        env=lambda: _env("cartesian", True),
        describe=lambda: (
            "预计算: T_i(e,mu) 实数 (x,y[,z])   <-- 与 cartesian 同一份文件\n"
            "拟合  : W = EPS_A . T，I = sum_{lambda=+-1} |W_lambda|^2\n"
            "代价  : 每次迭代多一次逐事例旋转"),
    ),
    "helicity": Basis(
        name="helicity",
        title="一开始就在螺旋度表象下计算",
        cache_basis="helicity",
        rotate_at_fit=False,
        helicity_layer=True,
        n_components=3,
        precompute=_hel.precompute,
        amplitude=_identity,
        intensity=_hel_intensity,
        env=lambda: _env("helicity", True),
        describe=lambda: (
            "预计算: W_i(e,lambda) = EPS_A . T_i，复数 (lambda=+1,0,-1)\n"
            "拟合  : I = sum_{lambda=+-1} |W_lambda|^2，不再有洛伦兹指标\n"
            "代价  : cache 变大，缩并从 2 个实 einsum 变 4 个"),
    ),
    "omega": Basis(
        name="omega",
        title="Omega 原生：预计算与道无关的角向对象，拟合时装配 H.D",
        cache_basis="omega",
        rotate_at_fit=False,
        helicity_layer=True,
        n_components=3,
        precompute=None,             # Omega 不走 T -> W 这条路
        amplitude=_identity,
        intensity=_hel_intensity,
        env=lambda: _env("omega", True),
        describe=lambda: (
            "预计算: Omega(e, lam_A, lam_phi, lam_f)  —— 与 (l,S) 无关\n"
            "拟合  : W_i = g_i * K_i(e) * (M_i . Omega)，即 tf-pwa 的 H.D\n"
            "代价  : cache 大 3 倍；循环里多一次 45 维缩并（不是为速度）"),
    ),
}


def names():
    """全部表象名。"""
    return sorted(BASES)


def get(name):
    """按名取表象；未知则给出可用列表。"""
    if name not in BASES:
        raise KeyError("未知表象 %r；可用：%s" % (name, names()))
    return BASES[name]


def env_for(name):
    """跑 run/ 里的共享 likelihood 时要设的环境变量。"""
    return get(name).env()


def describe_all():
    """一页摘要，CLI `show` 用。"""
    out = []
    for n in names():
        b = BASES[n]
        out.append("### %s —— %s\n%s\n%s" % (n, b.title, b.describe(), repr(b)))
    return "\n\n".join(out)


def check_consistency(seed=7, n=6):
    """跨表象恒等式自检（最强回归）。

    `|W_{+1}|^2 + |W_{-1}|^2 == |T_x|^2 + |T_y|^2` 是**严格**等式，
    与 `4_三条路线/method3_helicity_native/gen_tensor.py` 的判据一致。
    """
    rng = np.random.default_rng(seed)
    T = rng.normal(size=(n, 3))
    W = BASES["helicity"].precompute(T)
    lhs = np.abs(W[:, 0]) ** 2 + np.abs(W[:, 2]) ** 2
    rhs = T[:, 0] ** 2 + T[:, 1] ** 2
    d1 = float(np.abs(lhs - rhs).max())

    # dplex 布局下的旋转：把 T 当成 (2, n, 3) 的 (Re, Im)，旋转后应逐位等于 W
    blk = np.stack([T, np.zeros_like(T)], axis=0)      # (2, n, 3)
    rot = BASES["cov_to_helicity"].amplitude(blk)      # (2, n, 3)
    d2 = float(np.abs((rot[0] + 1j * rot[1]) - W).max())

    # 两条路线的强度必须逐位相同
    i1 = BASES["cartesian"].intensity(blk)
    i2 = BASES["cov_to_helicity"].intensity(rot)
    d3 = float(np.abs(i1 - i2).max())
    return {"transverse_identity": d1, "rotate_matches_precompute": d2,
            "intensity_cartesian_vs_helicity": d3}
