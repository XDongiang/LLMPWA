#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计：哪些"螺度振幅已支持"的态还缺协变公式？

## 结论（2026-10-21）

**没有一个是"表达不出来"的 —— 18 个道全是"没人写手写函数"。**

`agent/tensor/chains/<chain>/<channel>.py` 里放的**不是**新公式，而是
**逐道的手工搬运**：把论文/`Tensor/FUNC.py` 里那一条道的缩并写成代码，
`emitter.py` 在生成时按 `def <name>` 取函数体内联。所以"缺"意味着
`emit` 到那一步会 `LookupError`，不代表协变路线算不了这个态。

本探针用**通用构造**（`t_l` + ε 缩并，即论文 Eq.(258) + Eq.(349)）直接算出
每一个"缺公式"的道，与**级联路线**逐道对拍。全部到机器精度，说明缺的只是
搬运工作。

## 三类缺口

| 类别 | 缺的道 | 为什么缺 | 补的代价 |
|---|---|---|---|
| **A. 置换族** | `ex1` 的 ρ/ρ₃/ρ₅ 共 9 条；`ex2` 的 K*/K*₃/K*₅ 共 6 条 | 三条道是**三对配对**（不是"同一共振换旁观者"），要逐道写 | 通用公式已验证，抄一遍即可 |
| **B. 纯改名** | `rhof2` 的 5 条 | 与 `phif2` **拓扑完全相同**（同 LS、同 J），只是粒子名不同 | 改名即可（本探针实测 rel ~1e-15） |
| **C. 单道** | `ex2_kkpi0_rho` 的 1 条 | 单道，与 `rhopi` 同构 | 一行 |

## 为什么当初只做了 `ex1_rho5`

`rhopi5`（ρ₅）是**唯一**在补 `t_l` 之前**根本写不出来**的 —— 其余 17 条在
`T1_cov..T4_cov` 时代就能写，只是没人写。所以那次改动的意义是"打通 J=5"，
不是"补齐所有道"。

⚠️ 动量布局：协变内核 `CO` 要 `(x,y,z,E)`；级联内核 `HC` 要 `(E,x,y,z)`。
本探针两套都备着（`mom` / `mom_e`），别混。
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "agent", "tensor", "tests"))
os.chdir(_ROOT)

import numpy as np                                                     # noqa: E402

import test_ex3_channels as T                                          # noqa: E402
from agent.tensor import spec as S                                     # noqa: E402
from agent.tensor.kernels import covariant as CO                       # noqa: E402
from agent.tensor.kernels import helicity_cascade as HC                # noqa: E402
from agent.tensor.kernels.helicity import EPS_A                        # noqa: E402

_IX = "ABCDEFGH"


def u_generic(mom, psi, pair, spec, J):
    r"""$\epsilon_{\mu\nu\lambda\sigma}p_\psi^\sigma
    \tilde T^{(J)\nu\cdots}_{(\rho 3)}\tilde t^{(J)\lambda}_{(12)\cdots}$。

    `mom` 为 (N, 4, 4) 的 (x,y,z,E)；`pair=(a,b)` 配成共振，`spec` 是旁观者。
    """
    a, b = pair
    rho = mom[:, a] + mom[:, b]
    idx = _IX[:J - 1]          # 第 0 个指标留给 ε 的 μ，其余 J-1 个全缩
    t_top = CO.t_l(J, psi, rho, mom[:, spec])
    t_in = CO.t_l(J, rho, mom[:, a], mom[:, b])
    m_nl = np.einsum("en%s,el%s->enl" % (idx, idx), t_top, t_in)
    return np.einsum("mnls,es,enl->em", CO.leci_four(), psi, m_nl)


def _tree_from_nodes(sp):
    """4 体 spec 的 `tree` 是空的 —— 从粒子表重建。"""
    out = {}
    for name, p in sp["particles"].items():
        d = p.get("daughters") if isinstance(p, dict) else None
        if d:
            out[name] = list(d)
    return out


def _rel_to_cascade(spec_path, mom_xyzE, mom_Exyz, psi, idx_of, J):
    """逐道：通用公式 vs 级联，返回 [(道名, rel), ...]。"""
    sp = S.load_spec(spec_path)
    base = S.to_chain(sp)
    out = []
    for ch in S.cascade_channels(sp):
        tree = ch["tree"] or _tree_from_nodes(sp)
        res, spec = tree["psi2s"]
        pair = tuple(idx_of[x] for x in tree[res])
        u = u_generic(mom_xyzE, psi, pair, idx_of[spec], J)
        w = np.einsum("lm,em->el", EPS_A, u[:, :3])
        ref = np.asarray(HC.chain_amplitude(mom_Exyz, HC.with_tree(base, tree),
                                            ch["ls"]))
        c = np.vdot(w.reshape(-1), ref.reshape(-1)) \
            / np.vdot(w.reshape(-1), w.reshape(-1))
        out.append((ch["name"],
                    float(np.abs(ref - c * w).max() / np.abs(ref).max())))
    return out


def main():
    ok = True

    # ---------------------------------------------------- A. 置换族（ex1）
    d = np.load("analyses/tensor_examples/data/ex1_pipipi0.npz")
    keys = ["Pip1", "Pim2", "Pi0"]
    mom = np.stack([d[k] for k in keys], 1)
    mom_e = mom[:, :, [3, 0, 1, 2]]
    n = mom.shape[0]
    psi = np.broadcast_to(np.array([0.0, 0.0, 0.0, 3.686]), (n, 4))
    idx_of = {"Pip1": 0, "Pim2": 1, "Pi0": 2}
    print("=== A. 置换族：ex1（ψ→π⁺π⁻π⁰），通用公式 vs 级联 ===")
    for name, J in (("ex1_pipipi0_rho", 1), ("ex1_pipipi0_rho3", 3),
                    ("ex1_pipipi0_rho5", 5)):
        p = "analyses/tensor_examples/specs/%s.toml" % name
        for cname, rel in _rel_to_cascade(p, mom, mom_e, psi, idx_of, J):
            good = rel < 1e-10
            ok &= good
            print("  %-18s %-12s J=%d rel=%.2e %s"
                  % (name, cname, J, rel, "OK" if good else "FAIL"))

    # ---------------------------------------------------- A'. 置换族（ex2）
    d2 = np.load("analyses/tensor_examples/data/ex2_kkpi0.npz")
    keys2 = ["Kp1", "Km2", "Pi0"]
    mom2 = np.stack([d2[k] for k in keys2], 1)
    mom2_e = mom2[:, :, [3, 0, 1, 2]]
    n2 = mom2.shape[0]
    psi2 = np.broadcast_to(np.array([0.0, 0.0, 0.0, 3.686]), (n2, 4))
    idx2 = {"Kp1": 0, "Km2": 1, "Pi0": 2}
    print()
    print("=== A'. 置换族：ex2（ψ→K⁺K⁻π⁰）===")
    for name, J in (("ex2_kkpi0_kstar", 1), ("ex2_kkpi0_kstar3", 3),
                    ("ex2_kkpi0_kstar5", 5), ("ex2_kkpi0_rho", 1)):
        p = "analyses/tensor_examples/specs/%s.toml" % name
        for cname, rel in _rel_to_cascade(p, mom2, mom2_e, psi2, idx2, J):
            good = rel < 1e-10
            ok &= good
            print("  %-18s %-12s J=%d rel=%.2e %s"
                  % (name, cname, J, rel, "OK" if good else "FAIL"))

    # ---------------------------------------------------- B. 纯改名（rhof2）
    print()
    print("=== B. rhof2 与 phif2 拓扑相同 —— 把 phif2 的函数改名 ===")
    from agent.tensor.chains.phif2 import (phif201, phif221, phif222,
                                           phif223, phif243)
    fns = [phif201, phif221, phif222, phif223, phif243]
    pair, boost = T._pair, T._boost
    MPI, MPSI = 0.13957039, 3.686
    rng = np.random.default_rng(7)
    nn = 32
    rho, f2 = pair(MPSI, 0.775, 1.275, rng, nn)
    a0, b0 = pair(0.775, MPI, MPI, rng, nn)
    c0, d0 = pair(1.275, MPI, MPI, rng, nn)
    br = rho[:, :3] / rho[:, 3:4]
    bf = f2[:, :3] / f2[:, 3:4]
    Pip, Pim = boost(a0, br), boost(b0, br)
    Pip2, Pim2 = boost(c0, bf), boost(d0, bf)
    psi3 = np.broadcast_to(np.array([0.0, 0.0, 0.0, MPSI]), (nn, 4))
    sp3 = S.load_spec("agent/tensor/specs/rhof2_generic.toml")
    base3 = S.to_chain(sp3)
    arr = {"Pip": Pip, "Pim": Pim, "Pip2": Pip2, "Pim2": Pim2}
    mom3 = np.stack([arr[k] for k in sp3["input_keys"]], 1)[:, :, [3, 0, 1, 2]]
    for i, ch in enumerate(S.cascade_channels(sp3)):
        o = np.asarray(fns[i](psi3, Pip + Pim, Pip2 + Pim2, Pip, Pim, Pip2, Pim2))
        w = np.einsum("lm,em->el", EPS_A, o[:, :3])
        ref = np.asarray(HC.chain_amplitude(mom3, HC.with_tree(base3, {}),
                                            ch["ls"]))
        c = np.vdot(w.reshape(-1), ref.reshape(-1)) \
            / np.vdot(w.reshape(-1), w.reshape(-1))
        rel = float(np.abs(ref - c * w).max() / np.abs(ref).max())
        good = rel < 1e-10
        ok &= good
        print("  %-12s <- %-9s rel=%.2e %s"
              % (ch["name"], fns[i].__name__, rel, "OK" if good else "FAIL"))

    print()
    print("=>", "18 个道全部可用通用构造复现 —— 缺的只是手写搬运"
          if ok else "**有实质缺口（通用构造表达不了）**")


if __name__ == "__main__":
    main()
