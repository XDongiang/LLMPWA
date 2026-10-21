#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""`ex5_kst2` 的 `u_PP_*` / `u_FP_*` 为什么不是级联？—— **定位到投影所在的系**

## 结论（先写在这里，下面是机检）

与 `phif2` 的 (41)(42) 缺陷**同型**（见
`analyses/kk_dis/task/20260929_螺旋度耦合振幅推导/3_报告/ZB_psi_to_phif2_原式缺陷与修正.md`
§4「母系自旋投影错配」）：

> 内层是 **P 波**（$l=1$）时，"两个秩-1 分析器耦合成 $K^*_2$ 的自旋 2"这一步
> 需要一次**无迹化**，而原式用的是 $\tilde g(p_\psi)$（$\psi$ 系）——
> 应该在 $\tilde g(p_{K^*_2})$（$K^*_2$ 系）里做。

把内层换成用 $\tilde g(p_{K^*_2})$ 无迹化的 $T^{(2)}$，
`u_PP_1` 从 rel = 6.99e-03 掉到 **7.17e-15**（落回级联子空间）。

## 为什么只有内层 P 波出问题

| 内层 | 原式装置 | 无迹化在哪个系 | 命中 |
|---|---|---|---|
| $l=1$（`u_PP_*`/`u_FP_*`） | $P^{(2)}(p_\psi)$ 同时管内层 | **$\psi$ 系** ❌ | ❌ |
| $l=3$（`u_PF_*`/`u_FF_*`） | `T3_cov` 自带 $rrr-\frac15\tilde r^2(\tilde g\tilde r+\ldots)$ | **$K^*_2$ 系** ✅ | ✅ |

`T3_cov(kst2, phi, Pim)` 里的 $\tilde g$ 是 $\tilde g(p_{K^*_2})$ —— 内层自己
就把无迹化做在正确的系里了，所以 $P^{(2)}(p_\psi)$ 只剩顶层的作用，不再引入错配。

## 为什么 `ex5_kst0` 的内层也是 P 波却命中

$J_{K^*}=0$ ⇒ 内层 $(l,s)=(1,1)$ 耦合成 **0**，是**标量**，
根本没有"秩-2 无迹化"这一步 —— 也就没有"在哪个系里去迹"的问题。

## 复现

    python analyses/tensor_examples/probes/probe_ex5_why.py
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

from agent.tensor.kernels import covariant as COV                    # noqa: E402
from agent.tensor.kernels import helicity_cascade as HC              # noqa: E402
from agent.tensor.kernels.helicity import EPS_A                      # noqa: E402
from agent.tensor.kernels.covariant import (                         # noqa: E402
    Gt_cov, P2, T1_cov, T3_cov, _einsum, _metric_axis, _raise_last)

M_PSI, M_PHI, M_K, M_PI = 3.686, 1.019, 0.493677, 0.13957039


def _pair(m_parent, m1, m2, rng, n):
    lam = ((m_parent ** 2 - (m1 + m2) ** 2)
           * (m_parent ** 2 - (m1 - m2) ** 2))
    p = np.sqrt(max(lam, 0.0)) / (2.0 * m_parent)
    ct = rng.uniform(-1.0, 1.0, n)
    st = np.sqrt(np.maximum(1.0 - ct ** 2, 0.0))
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    d = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)
    return (np.concatenate([p * d, np.full((n, 1), np.sqrt(m1 ** 2 + p ** 2))], 1),
            np.concatenate([-p * d, np.full((n, 1), np.sqrt(m2 ** 2 + p ** 2))], 1))


def _boost(p, beta):
    b2 = (beta ** 2).sum(-1)
    g = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-300))
    bp = (beta * p[:, :3]).sum(-1)
    g2 = np.where(b2 > 1e-12, (g - 1.0) / np.where(b2 > 1e-12, b2, 1.0), 0.0)
    return np.concatenate(
        [p[:, :3] + (g2 * bp + g * p[:, 3])[:, None] * beta,
         (g * (p[:, 3] + bp))[:, None]], axis=1)


def physical(n=48, seed=11, m_kst=1.43):
    rng = np.random.default_rng(seed)
    kst, spec_pi = _pair(M_PSI, m_kst, M_PI, rng, n)
    phi, pi_in = _pair(m_kst, M_PHI, M_PI, rng, n)
    kp0, km0 = _pair(M_PHI, M_K, M_K, rng, n)
    bk = kst[:, :3] / kst[:, 3:4]
    bph = phi[:, :3] / phi[:, 3:4]
    return {"Kp": _boost(_boost(kp0, bph), bk), "Km": _boost(_boost(km0, bph), bk),
            "Pip": _boost(pi_in, bk), "Pim": spec_pi}


# --------------------------------------------------------------- 级联基底
BASE_NODES = {
    "psi2s": {"J": 1, "mass": M_PSI, "daughters": ["Kstar", "Pim"]},
    "Kstar": {"J": 2, "mass": 1.43, "daughters": ["phi", "Pip"]},
    "phi": {"J": 1, "mass": M_PHI, "daughters": ["Kp", "Km"]},
    "Kp": {"J": 0, "mass": M_K}, "Km": {"J": 0, "mass": M_K},
    "Pip": {"J": 0, "mass": M_PI}, "Pim": {"J": 0, "mass": M_PI},
}
TREES = {"l": {"psi2s": ["Kstar", "Pip"], "Kstar": ["phi", "Pim"]},
         "r": {"psi2s": ["Kstar", "Pim"], "Kstar": ["phi", "Pip"]}}
INDEX = {"Kp": 0, "Km": 1, "Pip": 2, "Pim": 3}


def allowed(jp, j1, j2, e_p, e1, e2, lmax=4):
    out = []
    for l in range(0, lmax + 1):
        if e_p != e1 * e2 * (-1) ** l:
            continue
        for sp in range(abs(j1 - j2), j1 + j2 + 1):
            if abs(l - sp) <= jp <= l + sp:
                out.append((l, sp))
    return out


def cascade_basis(mom, side, jk=2, ek=-1):
    c = {"top": "psi2s", "index": INDEX,
         "nodes": {k: dict(v) for k, v in BASE_NODES.items()}}
    c["nodes"]["Kstar"]["J"] = jk
    c = HC.with_tree(c, TREES[side])
    ang = HC.node_angles(mom, c)
    TOP = allowed(1, jk, 0, -1, ek, -1)
    INNER = allowed(jk, 1, 0, ek, -1, -1)
    amps, tags = [], []
    for tl in TOP:
        for il in INNER:
            try:
                A = HC.chain_amplitude(
                    mom, c, {"psi2s": [tl], "Kstar": [il], "phi": [(1, 0)]},
                    ang=ang)
            except Exception:                                    # noqa: BLE001
                continue
            if np.all(np.isfinite(A)) and float(np.abs(A).max()) > 0:
                amps.append(A)
                tags.append((tl, il))
    return amps, tags


def _fitc(a, b):
    c = np.vdot(a.reshape(-1), b.reshape(-1)) / np.vdot(a.reshape(-1),
                                                         a.reshape(-1))
    return complex(c), float(np.abs(b - c * a).max() / max(np.abs(b).max(), 1e-300))


def sub_res(B, w):
    coef, *_ = np.linalg.lstsq(B, w, rcond=None)
    return float(np.abs(w - B @ coef).max() / max(np.abs(w).max(), 1e-300))


# ------------------------------------------------- 修正后的内层（K*₂ 系无迹化）
def main():
    d = physical(n=48, seed=11)
    mom = np.stack([d[k][:, [3, 0, 1, 2]]
                    for k in ("Kp", "Km", "Pip", "Pim")], axis=1)
    psi = np.broadcast_to(np.array([0.0, 0.0, 0.0, M_PSI]),
                          (d["Kp"].shape[0], 4))

    def W(name):
        return np.einsum("lm,em->el", EPS_A,
                         COV.compute_event(name, d)[..., :3])

    def W4_show(tag, W4, B):
        w = np.einsum("lm,em->el", EPS_A, np.asarray(W4)[..., :3]).ravel()
        return "  %-40s max|W|=%.3e  级联 rel=%.3e" % (
            tag, float(np.abs(w).max()), sub_res(B, w))

    print("=" * 100)
    print("A) 四条未命中道：原式 vs「内层在 K*₂ 系无迹化」")
    print("=" * 100)
    for ch, side in (("u_PP_1", "l"), ("u_PP_2", "r"),
                     ("u_FP_1", "l"), ("u_FP_2", "r")):
        amps, tags = cascade_basis(mom, side)
        B = np.stack([a.ravel() for a in amps], axis=1)
        print()
        print("--- %s（树 %s）" % (ch, side))
        # ⚠️ 2026-10-21 起生产文件 `u_PP_1` 已是**修正版**，所以"原式"要
        #    从 `*_paper.py`（历史形式 / FUNC.py 逐式移植）取。
        #    见 `agent/tensor/chains/kstpi/CORRECTION.md`。
        print(W4_show("原式（`*_paper.py`）",
                      COV.compute_event(ch + "_paper", d), B))

        Kp, Km, Pip, Pim = d["Kp"], d["Km"], d["Pip"], d["Pim"]
        phi = Kp + Km
        if side == "l":
            ks, spec_in, spec_top = Kp + Km + Pim, Pim, Pip
        else:
            ks, spec_in, spec_top = Kp + Km + Pip, Pip, Pim
        p2psi = _raise_last(P2(psi), 4)
        gt_ks = Gt_cov(ks)
        gt_ks_inv = _raise_last(gt_ks, 2)
        t1_in = T1_cov(ks, phi, spec_in)
        t1_12 = T1_cov(phi, Kp, Km)
        t1_12_con = _metric_axis(t1_12, 1)
        gphi = _einsum("elm,em->el", gt_ks, t1_12_con)

        # 修正：内层秩-2 在 K*₂ 系无迹
        t2_in = (_einsum("ek,el->ekl", t1_in, gphi)
                 - (1.0 / 3) * _einsum("ekl,emn,em,en->ekl", gt_ks, gt_ks_inv,
                                       t1_in, gphi))

        # 顶层按道名选波：`u_P*` 是 P 波（T1），`u_F*` 是 F 波（T3）
        if ch.startswith("u_P"):
            fixed = _einsum("eijkl,ej,ekl->ei",
                            p2psi, T1_cov(psi, ks, spec_top), t2_in)
        else:
            # F 顶层：T3(psi, ks, spec_top) 的 3 个指标里，第 1 个给 P2、
            # 后 2 个给内层 t2 —— 对应 `u_FP_*` 的指标块
            t3_top = T3_cov(psi, ks, spec_top)
            t3_top_con = _raise_last(t3_top, 3)
            fixed = _einsum("eijk,ejklm,elm->ei", t3_top_con, p2psi, t2_in)
        print(W4_show("修正：内层用 g̃(K*₂) 无迹", fixed, B))

        # 与**具体那条**级联道（不是子空间）比：确认 (l,s) 组合也对上
        tl_expect = (1, 2) if ch.startswith("u_P") else (3, 2)
        il_expect = (1, 1)
        A_exp = HC.chain_amplitude(
            mom, HC.with_tree({"top": "psi2s", "index": INDEX,
                               "nodes": {k: dict(v) for k, v in BASE_NODES.items()}},
                              TREES[side]),
            {"psi2s": [tl_expect], "Kstar": [il_expect], "phi": [(1, 0)]},
            ang=HC.node_angles(mom, HC.with_tree(
                {"top": "psi2s", "index": INDEX,
                 "nodes": {k: dict(v) for k, v in BASE_NODES.items()}},
                TREES[side])))
        wf = np.einsum("lm,em->el", EPS_A, np.asarray(fixed)[..., :3])
        c, r = _fitc(wf, A_exp)
        print("      对**具体级联道** top%s inner%s：c=%+.5f%+.5fj rel=%.3e"
              % (tl_expect, il_expect, c.real, c.imag, r))

    print()
    print("=" * 100)
    print("B) 阳性对照：两条命中道（内层 F 波）—— 修正不该改变它们")
    print("=" * 100)
    for ch, side in (("u_PF_1", "l"), ("u_FF_1", "l")):
        amps, tags = cascade_basis(mom, side)
        B = np.stack([a.ravel() for a in amps], axis=1)
        print(W4_show("%s（原式）" % ch, COV.compute_event(ch, d), B))

    print()
    print("=" * 100)
    print("C) 反证：K*₀ 内层也是 P 波却命中（因为 J=0 是标量，没有无迹化）")
    print("=" * 100)
    amps0, tags0 = cascade_basis(mom, "l", jk=0, ek=-1)
    B0 = np.stack([a.ravel() for a in amps0], axis=1)
    print(W4_show("u_kst0_l（J=0，子空间 %d 维）" % len(tags0),
                  COV.compute_event("u_kst0_l", d), B0))

    print()
    print("=" * 100)
    print("D) 错配的量级：g̃(p_ψ) 与 g̃(p_K*₂) 差多少")
    print("=" * 100)
    Kp, Km, Pip, Pim = d["Kp"], d["Km"], d["Pip"], d["Pim"]
    ks = Kp + Km + Pim
    print("  max|g̃(K*₂) - g̃(ψ)| = %.4f"
          % float(np.abs(np.asarray(Gt_cov(ks)) - np.asarray(Gt_cov(psi))).max()))
    print("  两系不同的原因：K*₂ 在 ψ 静止系里是动的（|p| ≈ %.2f GeV）"
          % float(np.linalg.norm(np.asarray(ks)[:, :3], axis=1).mean()))


if __name__ == "__main__":
    main()
