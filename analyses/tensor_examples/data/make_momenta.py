#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合成 psi(2S) -> phi pi+ pi- -> (K+K-)(pi+pi-) 四动量，供例子回归用。

为什么需要合成数据
------------------
`agent/tensor/chains/` 里的道函数（例 3 的 rho'/b1、例 5 的 K*_0/K*_2）是
**按道名硬编码的公式**，对任意满足能量-动量守恒的四动量都成立。所以验证它们
不需要真实数据，只需要一批**物理的**四动量：

  * 每个末态在壳（`m^2 = -p_x^2 - p_y^2 - p_z^2 + E^2`，约定 `(x,y,z,E)`）；
  * 四动量守恒；
  * 每个两体子系统的不变质量在阈值之上 —— 否则 `Qabc` 变负，
    `B1/B2` 的 `sqrt` 出 NaN。**原版 jax 与本内核都会 NaN**，那不是移植错误，
    但会让对拍失去意义。

生成方式（顺序两体，严格守恒）
------------------------------
1. psi 静止系里均匀抽 `m(pi+pi-)`，取遍 `[2 m_pi, m_psi - m_phi]`；
2. `psi -> phi + X(m_pp)` 两体衰变；
3. `X -> pi+ pi-` 在 X 静止系里两体衰变，boost 回 psi 静止系；
4. `phi -> K+ K-` 在 phi 静止系里两体衰变，boost 回 psi 静止系。

第 1 步的均匀分布让 `m(phi pi+)` / `m(phi pi-)`（即 rho'/b1 的不变质量）
自然扫过一个宽区间，覆盖 `Qabc` 可能踩到的所有形状。

用法
----
    python data/make_momenta.py --out data/ex3_phipipi.npz --n 2000 --seed 7
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# 质量口径与 `specs/*.toml` 保持一致
M_PSI = 3.686
M_PHI = 1.019
M_K = 0.493677
M_PI = 0.13957039


def _decay_momentum(m_parent, m1, m2):
    """两体衰变的动量模长（母粒子静止系）。"""
    lam = ((m_parent ** 2 - (m1 + m2) ** 2)
           * (m_parent ** 2 - (m1 - m2) ** 2))
    if lam <= 0:
        raise ValueError("m_parent=%.6g 低于阈值 %.6g" % (m_parent, m1 + m2))
    return np.sqrt(lam) / (2.0 * m_parent)


def _directions(rng, n):
    """n 个各向同性的单位三维矢量。"""
    ct = rng.uniform(-1.0, 1.0, n)
    st = np.sqrt(np.maximum(1.0 - ct ** 2, 0.0))
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)


def _pair(m_parent, m1, m2, rng, n):
    """母粒子静止系下的两体衰变：返回 (p1, p2)，各 (n,4)，序 (x,y,z,E)。"""
    p = _decay_momentum(m_parent, m1, m2)
    d = _directions(rng, n)
    e1 = np.sqrt(m1 ** 2 + p ** 2)
    e2 = np.sqrt(m2 ** 2 + p ** 2)
    p1 = np.concatenate([p * d, np.full((n, 1), e1)], axis=1)
    p2 = np.concatenate([-p * d, np.full((n, 1), e2)], axis=1)
    return p1, p2


def _pair_varmass(m_parent, m1, m2, rng, n):
    """两体衰变，但 m1 逐事例不同（这里只用 m2 在壳）。

    `m_parent -> (质量 m1) + (质量 m2)`，动量模长逐事例按 m1 算。
    返回 (p1, p2)；p1 的能量由 m1 定，p2 的由 m2 定。
    """
    p = np.sqrt(np.maximum(
        ((m_parent ** 2 - (m1 + m2) ** 2)
         * (m_parent ** 2 - (m1 - m2) ** 2)), 0.0)) / (2.0 * m_parent)
    d = _directions(rng, n)
    p1 = np.concatenate([p[:, None] * d,
                         np.sqrt(m1 ** 2 + p ** 2)[:, None]], axis=1)
    p2 = np.concatenate([-p[:, None] * d,
                         np.sqrt(m2 ** 2 + p ** 2)[:, None]], axis=1)
    return p1, p2


def boost(p, beta):
    """把四动量 p 按 3-速度 beta boost（分量序 (x,y,z,E)）。"""
    b2 = (beta ** 2).sum(-1)
    g = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-300))
    bp = (beta * p[:, :3]).sum(-1)
    g2 = np.where(b2 > 1e-12, (g - 1.0) / np.where(b2 > 1e-12, b2, 1.0), 0.0)
    E = g * (p[:, 3] + bp)
    v = p[:, :3] + (g2 * bp + g * p[:, 3])[:, None] * beta
    return np.concatenate([v, E[:, None]], axis=1)


def make(n=2000, seed=7, m_lo=2.0 * M_PI + 1e-3, m_hi=M_PSI - M_PHI - 1e-3):
    """返回 {"Kp","Km","Pip","Pim"}，各 (n,4)，序 (x,y,z,E)。"""
    rng = np.random.default_rng(seed)

    # 1) psi 静止系里均匀抽 m(pi+pi-)；phi 的质量固定
    m_pp = rng.uniform(m_lo, m_hi, n)

    # 2) psi -> phi + X(m_pp)
    phi, x = _pair_varmass(M_PSI, M_PHI, m_pp, rng, n)

    # 3) X -> pi+ pi-（在 X 静止系），boost 回 psi 静止系
    pip0, pim0 = _pair_varmass(m_pp, M_PI, M_PI, rng, n)
    bx = x[:, :3] / x[:, 3:4]
    pip, pim = boost(pip0, bx), boost(pim0, bx)

    # 4) phi -> K+ K-（在 phi 静止系），boost 回 psi 静止系
    kp0, km0 = _pair(M_PHI, M_K, M_K, rng, n)
    bphi = phi[:, :3] / phi[:, 3:4]
    return {"Kp": boost(kp0, bphi), "Km": boost(km0, bphi),
            "Pip": pip, "Pim": pim}


def _mass2(p):
    return -p[:, 0] ** 2 - p[:, 1] ** 2 - p[:, 2] ** 2 + p[:, 3] ** 2


def check(mom, tol=1e-9):
    """守恒 + 在壳自检，返回 (ok, 报告行列表)。"""
    lines = []
    ok = True
    # 守恒：末态四动量之和 == psi 静止系的 (0,0,0,m_psi)
    want = np.array([0.0, 0.0, 0.0, M_PSI])
    res = np.abs(sum(mom.values()) - want).max()
    lines.append("四动量守恒 max|sum p - (0,0,0,%.4f)| = %.3e  %s"
                 % (M_PSI, res, "OK" if res < tol else "**FAIL**"))
    ok &= bool(res < tol)
    for k, m in (("Kp", M_K), ("Km", M_K), ("Pip", M_PI), ("Pim", M_PI)):
        d = np.abs(np.sqrt(np.maximum(_mass2(mom[k]), 0.0)) - m).max()
        lines.append("  %-4s 在壳 max|m - %.6g| = %.3e  %s"
                     % (k, m, d, "OK" if d < tol else "**FAIL**"))
        ok &= bool(d < tol)
    # 每个两体子系统的不变质量都要过阈值，否则 Qabc < 0 -> B1/B2 NaN。
    # 阈值用 `>=`：phi 的质量是**固定**的，min 恰好等于阈值是正常的。
    for nm, keys, m_thr in (("phi=Kp+Km", ("Kp", "Km"), M_PHI),
                            ("X(pi pi)=Pip+Pim", ("Pip", "Pim"), 2 * M_PI),
                            ("phi+Pip", ("Kp", "Km", "Pip"), M_PHI + M_PI),
                            ("phi+Pim", ("Kp", "Km", "Pim"), M_PHI + M_PI)):
        s = sum(mom[k] for k in keys)
        m = np.sqrt(np.maximum(_mass2(s), 0.0))
        good = bool(m.min() >= m_thr - 1e-9)
        lines.append("  %-18s m in [%.4f, %.4f]  阈值 %.4f  %s"
                     % (nm, m.min(), m.max(), m_thr,
                        "OK" if good else "**低于阈值**"))
        ok &= good
    return ok, lines


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    mom = make(args.n, args.seed)
    ok, lines = check(mom)
    print("\n".join(lines))
    if not ok:
        raise SystemExit("生成的相空间自检未通过")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **mom)
    print("saved %s  %d 事例  keys=%s" % (args.out, args.n, sorted(mom)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
