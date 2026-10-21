#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 6 的六体相空间：$\psi(2S)\to\phi\pi^+\pi^-\pi^+\pi^-\to K^+K^-\pi^+\pi^-\pi^+\pi^-$。

文档 624–795 行。末态编号照文档：$K^+\to1$、$K^-\to2$、
$\pi^+\to3$、$\pi^-\to4$、$\pi^+\to5$、$\pi^-\to6$。

生成顺序（$\psi$ 静止系）：

    psi -> phi + f2
    phi -> Kp + Km
    f2  -> sigma + sigma   (或 rho + rho / f2(1270) + sigma，见 branch)
    sigma -> Pip + Pim

**为什么 $m_{f_2}$ 默认取 1.9 而不是 1.275**：$\rho\rho$ 要求
$m_{f_2}>2m_\rho=1.55$，否则 $Q_{f_2\rho\rho}^2<0$、位垒因子取到负开方。
$f_2(1270)\sigma$ 同理（$f_2(1270)$ 自己也要有 $Q^2>0$）。
$\sigma\sigma$ 分支在物理 $1.275$ 下没问题，但为了让同一个 `.npz` 能喂给
**全部三条子分支**，默认用虚拟的 1.9。

四动量布局 `(x, y, z, E)`。
"""
from __future__ import annotations

import argparse
import os

import numpy as np

M_PSI = 3.686
M_PI = 0.13957039
M_K = 0.493677
M_PHI = 1.019461
M_RHO = 0.775
M_SIG = 0.5


def _dirs(rng, n):
    ct = rng.uniform(-1.0, 1.0, n)
    st = np.sqrt(np.maximum(1.0 - ct ** 2, 0.0))
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)


def _pair(m0, m1, m2, rng, n):
    q2 = (m0 ** 2 - (m1 + m2) ** 2) * (m0 ** 2 - (m1 - m2) ** 2) / (4.0 * m0 ** 2)
    q = np.sqrt(np.maximum(q2, 0.0))
    d = _dirs(rng, n)
    p1 = np.column_stack([q * d, np.full(n, np.sqrt(m1 ** 2 + q ** 2))])
    p2 = np.column_stack([-q * d, np.full(n, np.sqrt(m2 ** 2 + q ** 2))])
    return p1, p2


def _boost(p, beta):
    b2 = (beta ** 2).sum(-1, keepdims=True)
    g = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-300))
    bp = (beta * p[..., :3]).sum(-1, keepdims=True)
    g2 = np.where(b2 > 1e-12, (g - 1.0) / np.where(b2 > 1e-12, b2, 1.0), 0.0)
    return np.concatenate(
        [p[..., :3] + (g2 * bp + g * p[..., 3:4]) * beta,
         g * (p[..., 3:4] + bp)], axis=-1)


def make(n=64, seed=5, m_f2=1.9, m_sig=M_SIG, m_rho=M_RHO, m_phi=M_PHI, m_k=M_K):
    r"""返回 `{'Kp','Km','Pip','Pim','Pip2','Pim2'}`，各 `(N,4)`、`(x,y,z,E)`。"""
    rng = np.random.default_rng(seed)

    phi, f2 = _pair(M_PSI, m_phi, m_f2, rng, n)
    kp, km = _pair(m_phi, m_k, m_k, rng, n)
    bp = phi[:, :3] / phi[:, 3:4]
    kp = _boost(kp, bp)
    km = _boost(km, bp)

    # f2 -> sigma + sigma
    sa, sb = _pair(m_f2, m_sig, m_sig, rng, n)
    pa, ma = _pair(m_sig, M_PI, M_PI, rng, n)
    pb, mb = _pair(m_sig, M_PI, M_PI, rng, n)
    bf = f2[:, :3] / f2[:, 3:4]
    bsa = sa[:, :3] / sa[:, 3:4]
    bsb = sb[:, :3] / sb[:, 3:4]
    pa = _boost(_boost(pa, bsa), bf)
    ma = _boost(_boost(ma, bsa), bf)
    pb = _boost(_boost(pb, bsb), bf)
    mb = _boost(_boost(mb, bsb), bf)

    return {"Kp": kp, "Km": km, "Pip": pa, "Pim": ma, "Pip2": pb, "Pim2": mb}


def check(mom, tol=1e-9):
    tot = sum(mom[k] for k in mom)
    return np.abs(tot[:, :3]).max(), tot[:, 3]


def main():
    ap = argparse.ArgumentParser(description="例 6 的六体相空间")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--m-f2", type=float, default=1.9)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    mom = make(args.n, args.seed, m_f2=args.m_f2)
    d3, e = check(mom)
    print("|sum p|_max = %.3e   E = %.6f" % (d3, e[0]))
    if args.out:
        np.savez(args.out, **mom)
        print("wrote %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
