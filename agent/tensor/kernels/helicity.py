#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""螺度表象内核：`W_i = EPS_A · T_i`，即协变张量经 psi(2S) 横向极化投影。

数学
----
psi(2S) 由 e+e- 产生，初始态横向极化，所以洛伦兹指标 mu 上的求和不是
"全部三个笛卡尔分量"，而是先旋到螺旋度基底、再只保留 lambda = +-1：

    W_i(e, lambda) = sum_mu EPS_A[lambda, mu] * T_i(e, mu)

    EPS_A = [[ 1/sqrt2, -i/sqrt2, 0],     lambda = +1
             [ 0,        0,       1],     lambda =  0
             [-1/sqrt2, -i/sqrt2, 0]]     lambda = -1

**行序是承重约定**（`4_三条路线/eps_a_provenance.md`）：把第三行取正号、
或换成 (1, +i) 那一支，逐事例与 tf-pwa 的重叠度会从 1.000000 掉到 0.62–0.67。
`specgen` 的 `LAM_SLOTS[1] = [-1, 0, 1]` 与这里的行序一致，**不需要翻 lambda 轴**。

与 omega 的关系
---------------
`kernels/omega.py` 已经在螺旋度表象下直接构造角向对象；本模块是**从协变
张量一侧**到达同一表象的另一条路。两者互为交叉验证：

    W_i(e,lambda) == c_i * (EPS_A · T_i(e,mu))_lambda      |c_i| = 3.464102 / 6.531973

这条恒等式实测逐事例离散 1.26e-09（float32 cache 噪声底），作为回归测试
比"重算一遍自己"强得多 —— 它能抓住 EPS_A 约定写错这类错误。
"""
from __future__ import annotations

import numpy as np

# 行 = lambda(+1, 0, -1)，列 = mu(x, y, z)。取法承重，见模块 docstring。
EPS_A = np.array(
    [
        [1.0 / np.sqrt(2.0), -1.0j / np.sqrt(2.0), 0.0],
        [0.0, 0.0, 1.0],
        [-1.0 / np.sqrt(2.0), -1.0j / np.sqrt(2.0), 0.0],
    ],
    dtype=np.complex128,
)

# 强度里保留哪几个 lambda 槽（k=3 -> (0, 2) 即 +1 与 -1；k=2 -> (0, 1)）
TRANSVERSE = {2: (0, 1), 3: (0, 2)}

# 初始态横向极化只落在这两个笛卡尔方向上（k=3 时必须显式排除 z）
CARTESIAN_TRANSVERSE = (0, 1)


def precompute(T):
    """生成器里的那一步：T -> W。EPS_A 只在这里出现一次。

    T : (..., 3) 实数笛卡尔张量（mu = x, y, z）
    返回 (..., 3) complex128 的螺旋度振幅（lambda = +1, 0, -1）
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape[-1] < 3:
        raise ValueError(
            "螺度表象需要 3 个洛伦兹分量 (x,y,z)，收到 %d；"
            "旧 cache 丢掉了 z，EPS_A 的 lambda=0 行无法复现" % T.shape[-1])
    return np.einsum("lm,...m->...l", EPS_A, T[..., :3])


def amplitude(contracted):
    """拟合时把 dplex 布局 (2, ..., k) 的笛卡尔块旋到 (2, ..., 3)。

        a_{+1} = ( A_x - i A_y) / sqrt2
        a_{ 0} =   A_z
        a_{-1} = (-A_x - i A_y) / sqrt2
    """
    re, im = contracted[0], contracted[1]
    rx, ry = re[..., 0], re[..., 1]
    ix, iy = im[..., 0], im[..., 1]
    r2 = 1.0 / np.sqrt(2.0)
    if re.shape[-1] >= 3:
        re0, im0 = re[..., 2], im[..., 2]
    else:
        re0 = np.zeros_like(rx)
        im0 = np.zeros_like(rx)
    return np.stack(
        [
            np.stack([(rx + iy) * r2, (ix - ry) * r2], axis=0),   # lambda = +1
            np.stack([re0, im0], axis=0),                         # lambda =  0
            np.stack([(iy - rx) * r2, (-ix - ry) * r2], axis=0),  # lambda = -1
        ],
        axis=-1,
    )


def intensity(amplitude_block):
    """只对 lambda = +-1 求和；恒等于 |T_x|^2 + |T_y|^2。"""
    abs2 = amplitude_block[0] ** 2 + amplitude_block[1] ** 2
    return abs2[..., TRANSVERSE[abs2.shape[-1]]].sum(axis=-1)


def cartesian_intensity(amplitude_block):
    """I = |A_x|^2 + |A_y|^2（笛卡尔表象的强度）。"""
    abs2 = amplitude_block[0] ** 2 + amplitude_block[1] ** 2
    return abs2[..., CARTESIAN_TRANSVERSE].sum(axis=-1)


def env():
    return {"TENSOR_BASIS": "helicity", "HELICITY_LAYER": "1"}


def describe():
    return (
        "预计算: W_i(e,lambda) = EPS_A . T_i，复数 (lambda=+1,0,-1)\n"
        "拟合  : I = sum_{lambda=+-1} |W_lambda|^2，不再有洛伦兹指标"
    )
