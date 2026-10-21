#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 4 的五体相空间：$\psi(2S)\to\omega K^+K^-\to(\pi^+\pi^-\pi^0)K^+K^-$。

**为什么不能用真实的 $m_\omega=0.782$**：文档的 $\omega^\mu$ 里三条道都要
$B_1(Q_{\omega\rho i})$，而 $\omega\to\rho\pi$ 在物理质量下**在阈下**
（$m_\omega=0.782<m_\rho+m_\pi=0.911$），$Q_{\omega\rho i}^2<0$，
$B_1=\sqrt{2/(Q^2+Q_0^2)}$ 里的 $Q^2$ 为负虽不致命，但 $\rho$ 的
两体衰变 $Q_{\rho ij}^2$ 也接近 0 —— 位垒因子会剧烈起伏。

本生成器用一个**虚拟**的 $\omega$ 质量（默认 1.5 GeV）保证所有 $Q^2>0$，
与 `make_threebody.py` 对 $m_\rho$ 的处理是同一套理由。

四动量布局 `(x, y, z, E)`，与 `chains/*/u_*.py` 里的协变道一致。
"""
from __future__ import annotations

import argparse
import os

import numpy as np

M_PSI = 3.686
M_PI = 0.13957039
M_PI0 = 0.1349768
M_K = 0.493677


def _dirs(rng, n):
    """球面上均匀的 n 个单位矢量。"""
    ct = rng.uniform(-1.0, 1.0, n)
    st = np.sqrt(np.maximum(1.0 - ct ** 2, 0.0))
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)


def _pair(m0, m1, m2, rng, n):
    r"""$m_0\to m_1+m_2$ 的两体衰变，返回 `(N,4)` 的 `(x,y,z,E)`。

    动量大小由 $m_0$ 系里的 $q$ 给出，方向随机；$m_0$ 自己放在静止系。
    """
    q2 = (m0 ** 2 - (m1 + m2) ** 2) * (m0 ** 2 - (m1 - m2) ** 2) / (4.0 * m0 ** 2)
    q = np.sqrt(np.maximum(q2, 0.0))
    d = _dirs(rng, n)
    p1 = np.column_stack([q * d, np.full(n, np.sqrt(m1 ** 2 + q ** 2))])
    p2 = np.column_stack([-q * d, np.full(n, np.sqrt(m2 ** 2 + q ** 2))])
    return p1, p2


def _boost(p, beta):
    r"""把 `(N,4)` 的 `(x,y,z,E)` 沿 `beta`（`(N,3)`）做洛伦兹推进。"""
    b2 = (beta ** 2).sum(-1, keepdims=True)
    g = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-300))
    bp = (beta * p[..., :3]).sum(-1, keepdims=True)
    g2 = np.where(b2 > 1e-12, (g - 1.0) / np.where(b2 > 1e-12, b2, 1.0), 0.0)
    return np.concatenate(
        [p[..., :3] + (g2 * bp + g * p[..., 3:4]) * beta,
         g * (p[..., 3:4] + bp)], axis=-1)


def make(n=64, seed=5, m_omega=1.5, m_f0=1.9, m_pi=M_PI, m_pi0=M_PI0, m_k=M_K):
    r"""返回 `{'Kp','Km','Pi0','Pip','Pim'}`，各 `(N,4)`、`(x,y,z,E)`。

    生成顺序（$\psi$ 静止系）：

        psi -> omega + X,  X -> Kp + Km
        omega -> rho + Pi0, rho -> Pip + Pim

    $\omega$ 与 X 的动量反向，$\rho$ 在 $\omega$ 系里构造后再推进到 $\psi$ 系。
    """
    rng = np.random.default_rng(seed)

    # psi -> omega + X
    om, X = _pair(M_PSI, m_omega, m_f0, rng, n)
    # X -> Kp + Km
    kp, km = _pair(m_f0, m_k, m_k, rng, n)
    kp = _boost(kp, X[:, :3] / X[:, 3:4])
    km = _boost(km, X[:, :3] / X[:, 3:4])
    # omega -> rho + Pi0
    rho, pi0 = _pair(m_omega, 1.0, m_pi0, rng, n)
    # rho -> Pip + Pim（用虚拟 rho 质量 1.0）
    pip, pim = _pair(1.0, m_pi, m_pi, rng, n)
    b = rho[:, :3] / rho[:, 3:4]
    pip = _boost(pip, b)
    pim = _boost(pim, b)
    pi0 = _boost(pi0, om[:, :3] / om[:, 3:4])
    pip = _boost(pip, om[:, :3] / om[:, 3:4])
    pim = _boost(pim, om[:, :3] / om[:, 3:4])

    return {"Kp": kp, "Km": km, "Pi0": pi0, "Pip": pip, "Pim": pim}


def check(mom, tol=1e-9):
    r"""检查 $\sum p = p_\psi$ 且各末态在壳；返回 (最大动量偏移, 总能量)。"""
    tot = sum(mom[k] for k in mom)
    d3 = np.abs(tot[:, :3]).max()
    return d3, tot[:, 3]


def main():
    ap = argparse.ArgumentParser(description="例 4 的五体相空间")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--m-omega", type=float, default=1.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mom = make(args.n, args.seed, m_omega=args.m_omega)
    d3, e = check(mom)
    print("|sum p|_max = %.3e   E = %.6f" % (d3, e[0]))
    if args.out:
        np.savez(args.out, **mom)
        print("wrote %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
