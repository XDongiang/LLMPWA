#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复现 documentation/helicity_amplitude_roadmap.md §7 的五个脚本。

跑法（仓库根）：
    analyses/kk_dis/.venv/bin/python analyses/tensor_examples/_work/repro_roadmap_7.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "agent", "tensor", "tests"))

import test_ex3_channels as T                                   # noqa: E402
from agent.tensor import spec as S                              # noqa: E402
from agent.tensor.kernels import covariant as COV               # noqa: E402
from agent.tensor.kernels import omega as RT                    # noqa: E402
from agent.tensor.kernels.helicity import EPS_A                 # noqa: E402


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def rot3(v, axis, ang):
    ax = axis / np.linalg.norm(axis, axis=1, keepdims=True)
    c, s = np.cos(ang), np.sin(ang)
    d = (v * ax).sum(1, keepdims=True)
    return v * c + np.cross(ax, v) * s + ax * d * (1 - c)


def script1():
    print("\n### 脚本 1 —— 最内层敏感性（§1.3）")
    momd = T.physical(n=16, seed=3)
    s = S.load_spec(os.path.join(ROOT, "analyses/tensor_examples/specs/ex3_b1pi.toml"))
    chain = S.to_chain(s)
    mom_o = np.stack([momd[k][:, [3, 0, 1, 2]] for k in ["Kp", "Km", "Pip", "Pim"]],
                     axis=1)
    ang = np.deg2rad(37.0)

    phi3 = mom_o[:, 0, 1:4] + mom_o[:, 1, 1:4]
    r = mom_o[:, 0, 1:4] - mom_o[:, 1, 1:4]
    r2 = rot3(r, phi3, ang)
    mom_r = mom_o.copy()
    mom_r[:, 0, 1:4] = (phi3 + r2) / 2
    mom_r[:, 1, 1:4] = (phi3 - r2) / 2

    Om0 = RT.omega_from_momenta(mom_o, chain, covariant=True)
    Om1 = RT.omega_from_momenta(mom_r, chain, covariant=True)
    print("  Omega 变化 rel = %.3e   (文档 0.000e+00)" % rel(Om1, Om0))

    T0 = COV.compute_event("u_DD_1", momd)
    momd_r = {k: v.copy() for k, v in momd.items()}
    phi3c = momd["Kp"][:, :3] + momd["Km"][:, :3]
    rc = momd["Kp"][:, :3] - momd["Km"][:, :3]
    r2c = rot3(rc, phi3c, ang)
    momd_r["Kp"][:, :3] = (phi3c + r2c) / 2
    momd_r["Km"][:, :3] = (phi3c - r2c) / 2
    print("  T     变化 rel = %.3e   (文档 4.877e-01)"
          % rel(COV.compute_event("u_DD_1", momd_r), T0))


def true_amp(momd, nm):
    Tc = COV.compute_event(nm, momd)
    return np.einsum("lm,em->el", EPS_A, Tc[..., :3]).ravel()


def resid(A, basis):
    B = np.stack(basis, axis=1)
    c, *_ = np.linalg.lstsq(B, A, rcond=None)
    return float(np.linalg.norm(A - B @ c) / np.linalg.norm(A))


def script2():
    print("\n### 脚本 2 —— 最小二乘可表示性（§2.1）")
    momd = T.physical(n=48, seed=11)
    mom2 = np.stack([momd[k][:, [3, 0, 1, 2]] for k in ["Kp", "Km", "Pip", "Pim"]],
                    axis=1)

    phif2 = os.path.join(
        ROOT, "analyses/kk_dis/task/20261017_多表象拟合流程标准化_多GPU/"
        "bundle/route4/tensor/specs/phif2.toml")
    s = S.load_spec(phif2)
    ch = S.to_chain(s)
    Om = RT.omega_from_momenta(mom2, ch, covariant=True)
    ls = [(0, 1), (2, 1), (2, 2), (2, 3), (4, 3)]
    M = RT.cg_matrix(ls, 1, 1, 2, [-1, 0, 1], [-2, -1, 0, 1, 2])
    K = RT.node_factors(mom2, ch, ls)
    W = RT.assemble(M, Om, np.ones(5), K)
    for nm in ["phif201", "phif221", "phif222", "phif223", "phif243"]:
        print("  phi f2  %-9s 残差 = %.3e  (文档 1e-15 量级)"
              % (nm, resid(true_amp(momd, nm), [W[i].ravel() for i in range(5)])))

    s2 = S.load_spec(os.path.join(ROOT,
                                  "analyses/tensor_examples/specs/ex3_b1pi.toml"))
    ch2 = S.to_chain(s2)
    Om2 = RT.omega_from_momenta(mom2, ch2, covariant=True)
    ls2 = [(0, 1), (2, 1)]
    M2 = RT.cg_matrix(ls2, 1, 1, 0, [-1, 0, 1], [0])
    K2 = RT.node_factors(mom2, ch2, ls2)
    W2 = RT.assemble(M2, Om2, np.ones(2), K2)
    for nm in ["u_SS_1", "u_SD_1", "u_DS_1", "u_DD_1"]:
        print("  b1     %-9s 残差 = %.3e  (文档 9.8e-01..9.9e-01)"
              % (nm, resid(true_amp(momd, nm), [W2[i].ravel() for i in range(2)])))


def allowed_ls(Ja, Pa, Jb, Pb, Jc, Pc, lmax=4):
    return [(L, S) for L in range(lmax + 1)
            if Pa == Pb * Pc * (-1) ** L
            for S in range(abs(Jb - Jc), Jb + Jc + 1)
            if abs(L - S) <= Ja <= L + S]


def script3():
    print("\n### 脚本 3 —— 逐节点枚举与计数律（§2.2）")
    for nm, nodes in [
        ("rho'", [("psi->rho'pi", 1, -1, 1, -1, 0, -1),
                  ("rho'->phipi", 1, -1, 1, -1, 0, -1)]),
        ("b1", [("psi->b1pi", 1, -1, 1, +1, 0, -1),
                ("b1->phipi", 1, +1, 1, -1, 0, -1)]),
        ("phif0", [("psi->phif0", 1, -1, 1, -1, 0, +1),
                   ("f0->pipi", 0, +1, 0, -1, 0, -1)]),
        ("phif2", [("psi->phif2", 1, -1, 1, -1, 2, +1),
                   ("f2->pipi", 2, +1, 0, -1, 0, -1)]),
    ]:
        prod = 1
        print("  %s" % nm)
        for label, *q in nodes:
            ls = allowed_ls(*q)
            prod *= len(ls)
            print("     %-14s %d 个: %s" % (label, len(ls),
                                            "  ".join("(%d,%d)" % t for t in ls)))
        print("     => 独立振幅数 = %d" % prod)


def script4():
    print("\n### 脚本 4 —— 逐节点组合的秩（§4.3）")
    from agent.tensor.kernels.omega import cg_matrix
    LAM = {0: [0], 1: [-1, 0, 1], 2: [-2, -1, 0, 1, 2]}
    H1 = cg_matrix([(0, 1), (2, 1)], 1, 1, 0, LAM[1], LAM[0])[:, :, 0]
    H2 = cg_matrix([(0, 1), (2, 1)], 1, 1, 0, LAM[1], LAM[0])[:, :, 0]
    M4 = np.stack([np.outer(H1[a], H2[b]).ravel()
                   for a in range(2) for b in range(2)])
    print("  逐节点组合张成的秩 = %d   (文档 4)"
          % np.linalg.matrix_rank(M4))
    M2 = np.stack([np.outer(H1[a], H2[a]).ravel() for a in range(2)])
    print("  只按顶层分组      秩 = %d   (文档 2)"
          % np.linalg.matrix_rank(M2))


def script5():
    print("\n### 脚本 5 —— 逐事件比值判据（§4.4）")
    momd = T.physical(n=64, seed=11)
    mom = np.stack([momd[k][:, [3, 0, 1, 2]] for k in ["Kp", "Km", "Pip", "Pim"]],
                   axis=1)
    s = S.load_spec(os.path.join(
        ROOT, "analyses/kk_dis/task/20261017_多表象拟合流程标准化_多GPU/"
        "bundle/route4/tensor/specs/phif2.toml"))
    ch = S.to_chain(s)
    ls = [(0, 1), (2, 1), (2, 2), (2, 3), (4, 3)]
    M = RT.cg_matrix(ls, 1, 1, 2, [-1, 0, 1], [-2, -1, 0, 1, 2])
    K = RT.node_factors(mom, ch, ls)
    for nm, i in [("phif201", 0), ("phif221", 1), ("phif222", 2)]:
        Wc = np.einsum("lm,em->el", EPS_A, COV.compute_event(nm, momd)[..., :3])
        for cov in (True, False):
            Om = RT.omega_from_momenta(mom, ch, covariant=cov)
            W = RT.assemble(M, Om, np.ones(5), K)[i]
            mask = np.abs(Wc).max(1) > 1e-6
            r = Wc[mask] / W[mask]
            r = r / r[0]
            print("  %-9s covariant=%-5s 比值散布 = %.3e"
                  % (nm, cov, np.abs(r - 1).max()))


if __name__ == "__main__":
    script1()
    script2()
    script3()
    script4()
    script5()
