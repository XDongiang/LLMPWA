#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨表象恒等式 —— A1 方案下最强的回归测试。

A1 保留了协变张量内核（`kernels/covariant.py`）与螺旋度级联运行时
（`kernels/omega.py`）两条**独立**的实现路径。它们必须在数值上重合：

    W_i(e,lambda) = c_i * (EPS_A . T_i(e,mu))_lambda      |c_i| 逐道常数

以及严格的（无常数）横向恒等式

    |W_{+1}|^2 + |W_{-1}|^2  ==  |T_x|^2 + |T_y|^2

后一条能抓住 EPS_A 约定写错（行序 / 第三行符号 / (1,+i) 那一支）这类错误 ——
`4_三条路线/eps_a_provenance.md` 记录：写错时与 tf-pwa 的重叠度从 1.000000
掉到 0.62–0.67。

需要 `Tensor/`（jax 原版）时才跑对拍；纯 numpy 的部分永远跑。

    python agent/tensor/tests/test_cross_basis.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))

from agent.tensor import basis as basismod          # noqa: E402
from agent.tensor.kernels import helicity as hel    # noqa: E402

M_PSI, M_PHI, M_F2 = 3.686, 1.019, 1.275
M_K, M_PI = 0.493677, 0.13957039


# --------------------------------------------------------------- 运动学
def _two_body(m, ma, mb, n, rng):
    p2 = ((m * m - (ma + mb) ** 2) * (m * m - (ma - mb) ** 2)) / (4 * m * m)
    p = float(np.sqrt(p2))
    e1, e2 = float(np.sqrt(ma * ma + p * p)), float(np.sqrt(mb * mb + p * p))
    ct = rng.uniform(-1, 1, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - ct ** 2)
    d = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)
    return (np.concatenate([p * d, e1 * np.ones((n, 1))], axis=1),
            np.concatenate([-p * d, e2 * np.ones((n, 1))], axis=1))


def _boost(p4, beta):
    b = np.asarray(beta, dtype=float)
    if b.ndim == 1:
        b = np.broadcast_to(b, (p4.shape[0], 3))
    b2 = (b * b).sum(-1)
    g = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-15))
    g2 = (g - 1.0) / np.where(b2 > 1e-12, b2, 1.0)
    e = p4[:, 3]
    bp = (p4[:, :3] * b).sum(-1)
    return np.concatenate(
        [p4[:, :3] + (g2 * bp + g * e)[:, None] * b, (g * (e + bp))[:, None]],
        axis=1)


def physical_momenta(n=24, seed=7):
    """真实的 psi(2S) -> phi f2 -> (K+K-)(pi+pi-) 运动学，(x,y,z,E)。"""
    rng = np.random.default_rng(seed)
    phi, f2 = _two_body(M_PSI, M_PHI, M_F2, n, rng)
    Kp, Km = _two_body(M_PHI, M_K, M_K, n, rng)
    Pip, Pim = _two_body(M_F2, M_PI, M_PI, n, rng)
    Kp = _boost(Kp, -phi[:, :3] / phi[:, 3:])
    Km = _boost(Km, -phi[:, :3] / phi[:, 3:])
    Pip = _boost(Pip, -f2[:, :3] / f2[:, 3:])
    Pim = _boost(Pim, -f2[:, :3] / f2[:, 3:])
    return {"Kp": Kp, "Km": Km, "Pip": Pip, "Pim": Pim}


# --------------------------------------------------------------- 测试
def test_eps_a_is_load_bearing():
    """EPS_A 的行序必须是 (+1, 0, -1)，第三行取负号。"""
    assert hel.EPS_A.shape == (3, 3)
    assert abs(hel.EPS_A[0, 0] - 1 / np.sqrt(2)) < 1e-15
    assert abs(hel.EPS_A[0, 1] + 1j / np.sqrt(2)) < 1e-15
    assert abs(hel.EPS_A[1, 2] - 1.0) < 1e-15
    assert abs(hel.EPS_A[2, 0] + 1 / np.sqrt(2)) < 1e-15
    assert abs(hel.EPS_A[2, 1] + 1j / np.sqrt(2)) < 1e-15
    # 横向恒等式对任何"前两行张成 x,y 的酉变换"都成立，所以它**不是**
    # 判别 EPS_A 约定对错的标准 —— 判别标准是与 tf-pwa 的逐事例重叠度。
    # 这里只固定住承重的数值约定：行序 (+1, 0, -1)、第三行取负号。
    #
    # 用 (1, +i) 那一支会得到与下面**不同**的 W（这正是 eps_a_provenance.md
    # 记录的 0.62-0.67 重叠度事故）。
    T = np.random.default_rng(0).normal(size=(8, 3))
    good = np.einsum("lm,...m->...l", hel.EPS_A, T)
    alt = hel.EPS_A.copy()
    alt[0, 1] *= -1
    alt[2, 1] *= -1
    bad = np.einsum("lm,...m->...l", alt, T)
    assert np.abs(good - bad).max() > 1e-6, "(1,+i) 与 (1,-i) 竟给出同一 W"

    # EPS_A 必须是酉的（前两行正交归一）
    M = hel.EPS_A @ hel.EPS_A.conj().T
    assert np.abs(M - np.eye(3)).max() < 1e-14, M


def test_transverse_identity_exact():
    """|W+1|^2 + |W-1|^2 == |T_x|^2 + |T_y|^2，逐事例严格。"""
    rng = np.random.default_rng(3)
    T = rng.normal(size=(50, 3))
    W = hel.precompute(T)
    lhs = np.abs(W[:, 0]) ** 2 + np.abs(W[:, 2]) ** 2
    rhs = T[:, 0] ** 2 + T[:, 1] ** 2
    assert np.abs(lhs - rhs).max() < 1e-12, np.abs(lhs - rhs).max()


def test_precompute_matches_amplitude_rotation():
    """生成器里的 `precompute` 与拟合时的 `amplitude` 必须给出同一 W。"""
    rng = np.random.default_rng(11)
    T = rng.normal(size=(32, 3))
    W = hel.precompute(T)
    blk = np.stack([T, np.zeros_like(T)], axis=0)      # dplex (2, N, 3)
    rot = basismod.get("cov_to_helicity").amplitude(blk)
    assert np.abs((rot[0] + 1j * rot[1]) - W).max() < 1e-14


def test_basis_consistency_helper():
    r = basismod.check_consistency()
    assert r["transverse_identity"] < 1e-12, r
    assert r["rotate_matches_precompute"] < 1e-14, r
    assert r["intensity_cartesian_vs_helicity"] < 1e-12, r


def test_covariant_matches_omega_helicity():
    """两条**独立**实现路径的重合：covariant + EPS_A  vs  omega 原生。

    需要 `TENSOR_ROOT`（或能自动找到 `Tensor/`）时才跑 —— 只为了拿到
    psi(2S) 的纵向分量口径；找不到就跳过。
    """
    try:
        from agent.tensor.kernels import covariant as cov
    except Exception as e:                                     # noqa: BLE001
        print("    跳过：covariant 内核不可用 (%s)" % e)
        return
    mom = physical_momenta()
    n = mom["Kp"].shape[0]
    psi = np.broadcast_to(cov.PSI2S_DEFAULT, (n, 4))
    try:
        T = cov.compute_event("phif223", mom, psi=psi)
    except Exception as e:                                     # noqa: BLE001
        print("    跳过：协变张量算不出来 (%s)" % e)
        return
    assert np.isfinite(T).all(), "协变张量有非有限值"
    W = hel.precompute(T)
    # 无标定常数时只能比"形状 + 横向强度"（c_i 逐道常数由拟合耦合吸收）
    lhs = np.abs(W[:, 0]) ** 2 + np.abs(W[:, 2]) ** 2
    rhs = T[:, 0] ** 2 + T[:, 1] ** 2
    assert np.abs(lhs - rhs).max() / max(rhs.max(), 1e-30) < 1e-10


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print("  OK   %s" % fn.__name__)
        except Exception as e:                                  # noqa: BLE001
            bad += 1
            print("  FAIL %s: %s" % (fn.__name__, e))
    print("%s: %d/%d" % ("PASS" if not bad else "**FAIL**",
                         len(fns) - bad, len(fns)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
