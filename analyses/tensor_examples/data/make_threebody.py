#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""例 1 / 例 2 的三体相空间（`ψ → a b c`），写成 spec 的 `[input].layout`。

例 1 `ψ→π⁺π⁻π⁰`、例 2 `ψ→K⁺K⁻π⁰` 都是**真三体**（不是"两体 + 一个旁观者"），
所以不能像 `data/make_momenta.py` 那样用 ψ→X+Y 的两体学构造。这里按
"ψ → R + spectator，R → pair" 的**运动学**造 —— 这只是为了生成满足
四动量守恒的事例，**不代表**我们假设了某个共振（道表里三条配对都试）。

注意 $m_R$ 取什么不影响"是否为合法三体相空间"，但会影响各节点的 $q^2$。
取远高于阈值的值（ρ 取 1.5，ρ₃ 取 2.2）以保证所有配对下 $q^2>0$、
$B_l$ 不出现 NaN。

    python analyses/tensor_examples/data/make_threebody.py ex1 --n 64
    # -> data/ex1_pipipi0.npz   （Pip1, Pim2, Pi0，布局 x,y,z,E）
"""
from __future__ import annotations

import argparse
import os

import numpy as np

M_PSI = 3.686
M_PI, M_PI0 = 0.13957039, 0.1349768
M_K = 0.493677

# 每个例子的末态粒子名、质量、"母共振"质量（只用于造相空间）
CASES = {
    "ex1": {"keys": ("Pip1", "Pim2", "Pi0"),
            "mass": (M_PI, M_PI, M_PI0),
            "resonance_mass": 1.5,
            "files": "ex1_pipipi0.npz"},
    "ex1_r3": {"keys": ("Pip1", "Pim2", "Pi0"),
               "mass": (M_PI, M_PI, M_PI0),
               "resonance_mass": 2.2,
               "files": "ex1_pipipi0_rho3.npz"},
    "ex1_r5": {"keys": ("Pip1", "Pim2", "Pi0"),
               "mass": (M_PI, M_PI, M_PI0),
               "resonance_mass": 2.5,
               "files": "ex1_pipipi0_rho5.npz"},
    "ex2": {"keys": ("Kp1", "Km2", "Pi0"),
            "mass": (M_K, M_K, M_PI0),
            "resonance_mass": 1.9,
            "files": "ex2_kkpi0.npz"},
}


def _pair(m0, m1, m2, rng, n):
    """母粒子静止系两体衰变，(x,y,z,E)。"""
    lam = (m0**2 - (m1 + m2) ** 2) * (m0**2 - (m1 - m2) ** 2)
    p = np.sqrt(max(lam, 0.0)) / (2.0 * m0)
    ct = rng.uniform(-1.0, 1.0, n)
    st = np.sqrt(np.maximum(1.0 - ct**2, 0.0))
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    d = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)
    e1 = np.sqrt(m1**2 + p**2)
    e2 = np.sqrt(m2**2 + p**2)
    return (np.concatenate([p * d, np.full((n, 1), e1)], axis=1),
            np.concatenate([-p * d, np.full((n, 1), e2)], axis=1))


def _boost(p, beta):
    """boost 到速度为 beta 的系；p 与返回值都是 (x,y,z,E)。"""
    e, v = p[:, 3], p[:, :3]
    b2 = (beta**2).sum(-1)
    g = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-300))
    bp = (beta * v).sum(-1)
    g2 = np.where(b2 > 1e-12, (g - 1.0) / np.where(b2 > 1e-12, b2, 1.0), 0.0)
    return np.concatenate(
        [v + (g2 * bp + g * e)[:, None] * beta, (g * (e + bp))[:, None]], axis=1)


def make(case, n=64, seed=5):
    cfg = CASES[case]
    rng = np.random.default_rng(seed)
    m_res = cfg["resonance_mass"]
    m1, m2, m3 = cfg["mass"]

    # ψ → R + spectator(3)
    res, spec3 = _pair(M_PSI, m_res, m3, rng, n)
    # R → 1 + 2
    p1, p2 = _pair(m_res, m1, m2, rng, n)
    b = res[:, :3] / res[:, 3:4]
    p1 = _boost(p1, b)
    p2 = _boost(p2, b)

    return {cfg["keys"][0]: p1, cfg["keys"][1]: p2, cfg["keys"][2]: spec3}


def main():
    ap = argparse.ArgumentParser(description="例 1/2 的三体相空间")
    ap.add_argument("case", choices=sorted(CASES))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mom = make(args.case, n=args.n, seed=args.seed)
    # 守恒自检
    e = sum(mom[k][:, 3] for k in mom)
    px = sum(mom[k][:, 0] for k in mom)
    assert np.allclose(e, M_PSI, atol=1e-10), e[:3]
    assert np.allclose(px, 0.0, atol=1e-10), px[:3]

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), CASES[args.case]["files"])
    np.savez(out, **mom)
    print("%s: %d 事例 -> %s" % (args.case, args.n, out))
    print("  ΣE = %.10f（应 %.4f）  Σpx = %.2e（应 0）"
          % (e[0], M_PSI, abs(px).max()))


if __name__ == "__main__":
    main()
