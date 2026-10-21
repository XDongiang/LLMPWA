#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""`rhopi`（例 3 的 $\rho'$）为什么不是级联？—— 同一机制的检验

## 已知

`u_rho_1`/`u_rho_2` 的 A1 rel ≈ 0.24 / 0.28，随样本变；三条独立证据
（自由复系数最小二乘、逐事例 $c(e)$ 离散度、子空间主角）都指向"两个 2 维
子空间的**方向**不同"，不是"少了几维"。

## 本探针的假设

与 `ex5_kst2` 的 `u_PP_*`、以及 `phif2` 的 (41)(42) **同型**：

> 两个 $\epsilon$ 张起来的结构里，**中间态没有投影算符** ——
> 也就是"两个子系分析器在各自子系里定义横向，却在母系里耦合"，
> 少了母系自旋投影那一步。

`b1pi` 的 Eq.(47)–(50) 含 $\tilde g(p_{b_1})$ / $T^{(2)}(p_{b_1})$，
所以天然是级联；`rhopi` 的 Eq.(46) 内层 $\epsilon$ 缩并的是 $p_\psi$ 而不是
$p_{\rho'}$，中间没有 $\tilde g(p_{\rho'})$。

## 检验方式

把 Eq.(46) 里的 $p_\psi$（内层那个 $\epsilon$ 的缩并矢量）换成 $p_{\rho'}$，
再插入 $\tilde g(p_{\rho'})$ 投影 —— 逐项试，看哪一项能把 rel 打到 1e-12。
`README` 说第一项（换 $p_{\rho'}$）试过、没改善（0.29→0.36）；本探针把
**投影算符**那一支也穷举，并给出机制的量级。
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "agent", "tensor", "tests"))
os.chdir(_ROOT)

import test_ex3_channels as T                                       # noqa: E402
from agent.tensor import spec as S                                  # noqa: E402
from agent.tensor.kernels import covariant as COV                   # noqa: E402
from agent.tensor.kernels import helicity_cascade as HC             # noqa: E402
from agent.tensor.kernels.helicity import EPS_A                     # noqa: E402
from agent.tensor.kernels.covariant import (                        # noqa: E402
    Gt_cov, Gt_con, T1_cov, _einsum, _leci_con, _metric_axis,
    leci_four)

M_PSI, M_PHI, M_K, M_PI = 3.686, 1.019, 0.493677, 0.13957039
SPEC = "analyses/tensor_examples/specs/ex3_rhopi.toml"


def mom():
    d = T.physical(n=48, seed=11)
    return d, np.stack([d[k][:, [3, 0, 1, 2]]
                        for k in ("Kp", "Km", "Pip", "Pim")], axis=1)


def Wcov(name, d):
    return np.einsum("lm,em->el", EPS_A, COV.compute_event(name, d)[..., :3])


def casc(sp, name, m):
    cc = {c["name"]: c for c in S.cascade_channels(sp)}[name]
    ch = HC.with_tree(S.to_chain(sp), cc["tree"] or {})
    return HC.chain_amplitude(m, ch, cc["ls"], ang=HC.node_angles(m, ch))


def fit(a, b):
    c = np.vdot(a.reshape(-1), b.reshape(-1)) / np.vdot(a.reshape(-1),
                                                         a.reshape(-1))
    return complex(c), float(np.abs(b - c * a).max() / max(np.abs(b).max(), 1e-300))


def main():
    d, m = mom()
    sp = S.load_spec(SPEC)
    n = d["Kp"].shape[0]
    psi = np.broadcast_to(np.array([0.0, 0.0, 0.0, M_PSI]), (n, 4))
    Kp, Km, Pip, Pim = d["Kp"], d["Km"], d["Pip"], d["Pim"]
    phi = Kp + Km

    print("=" * 100)
    print("基准：原式 Eq.(46) vs 级联")
    print("=" * 100)
    for nm in ("u_rho_1", "u_rho_2"):
        c, r = fit(Wcov(nm, d), casc(sp, nm, m))
        print("  %-10s c=%+.5f%+.5fj  rel=%.3e  %s"
              % (nm, c.real, c.imag, r, "OK" if r < 1e-12 else "**FAIL**"))

    print()
    print("=" * 100)
    print("穷举：把 Eq.(46) 的装置逐项改掉，看哪一支落回级联")
    print("=" * 100)
    leci_cov = leci_four()
    leci_con = _leci_con()
    leci_mu = _metric_axis(leci_cov, 0)
    psi_cov = _metric_axis(psi, 1)

    for nm, spec_pi, in_pi in (("u_rho_1", Pip, Pim), ("u_rho_2", Pim, Pip)):
        rho = phi + in_pi
        t1_rho = T1_cov(psi, rho, spec_pi)
        t1_rho_con = _metric_axis(t1_rho, 1)
        t1_in = T1_cov(rho, phi, in_pi)
        t1_12 = T1_cov(phi, Kp, Km)

        def rel_of(tag, W):
            # 必须经 EPS_A 投到螺旋度（W 是 (N,4)，级联是 (N,3)）
            w = np.einsum("lm,em->el", EPS_A, np.asarray(W)[..., :3])
            c, r = fit(w, casc(sp, nm, m))
            print("    %-46s c=%+.4f%+.4fj  rel=%.3e  %s"
                  % (tag, c.real, c.imag, r, "OK" if r < 1e-12 else "—"))
            return r

        print()
        print("  --- %s（rho' = phi + %s，旁观者 %s）"
              % (nm, "Pim" if in_pi is Pim else "Pip",
                 "Pip" if spec_pi is Pip else "Pim"))
        base = _einsum("ijkl,ej,ek,lmno,em,en,eo->ei",
                       leci_mu, psi, t1_rho_con, leci_con, psi_cov,
                       t1_in, t1_12)
        rel_of("原式（内层 eps 缩并 p_psi）", base)

        # (a) 内层 eps 缩并 p_rho' 而不是 p_psi
        rho_cov = _metric_axis(rho, 1)
        a = _einsum("ijkl,ej,ek,lmno,em,en,eo->ei",
                    leci_mu, psi, t1_rho_con, leci_con, rho_cov, t1_in, t1_12)
        rel_of("(a) 内层 eps 缩并 p_rho'", a)

        # (b) 插入 g̃(p_rho') 到内层：t1_in -> g̃(rho)·t1_in
        gt_rho = Gt_cov(rho)
        t1_in_proj = _einsum("eij,ej->ei", gt_rho, t1_in)
        b = _einsum("ijkl,ej,ek,lmno,em,en,eo->ei",
                    leci_mu, psi, t1_rho_con, leci_con, psi_cov,
                    t1_in_proj, t1_12)
        rel_of("(b) 内层 t1 -> g̃(rho')·t1", b)

        # (c) 插入 g̃(p_rho') 到顶层：t1_rho -> g̃(rho')·t1_rho
        t1_rho_proj = _einsum("eij,ej->ei", gt_rho, t1_rho)
        t1_rho_proj_con = _metric_axis(t1_rho_proj, 1)
        c_ = _einsum("ijkl,ej,ek,lmno,em,en,eo->ei",
                     leci_mu, psi, t1_rho_proj_con, leci_con, psi_cov,
                     t1_in, t1_12)
        rel_of("(c) 顶层 T1 -> g̃(rho')·T1", c_)

        # (d) 两者都插
        dd = _einsum("ijkl,ej,ek,lmno,em,en,eo->ei",
                     leci_mu, psi, t1_rho_proj_con, leci_con, psi_cov,
                     t1_in_proj, t1_12)
        rel_of("(d) 顶层与内层都插 g̃(rho')", dd)

        # (g) 内层 eps 缩并 p_rho' + 插入 g̃(rho') 到内层
        g_ = _einsum("ijkl,ej,ek,lmno,em,en,eo->ei",
                     leci_mu, psi, t1_rho_con, leci_con, rho_cov,
                     t1_in_proj, t1_12)
        rel_of("(g) 内层 eps·p_rho' + 内层 g̃(rho')", g_)

        # (h) 内层 eps 缩并 p_rho' + 顶层 g̃(rho')
        h_ = _einsum("ijkl,ej,ek,lmno,em,en,eo->ei",
                     leci_mu, psi, t1_rho_proj_con, leci_con, rho_cov,
                     t1_in, t1_12)
        rel_of("(h) 内层 eps·p_rho' + 顶层 g̃(rho')", h_)


if __name__ == "__main__":
    main()
