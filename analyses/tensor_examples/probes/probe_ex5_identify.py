#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 5 判定：K*₀/K*₂ 的 10 条道是不是"逐节点级联"。

## 为什么先做这一步

ρ′ 的教训（见 `chains/rhopi.md`）：Eq.(46) 满足 omega 的计数律、深度也够，
**看起来该成功**，但它不是级联（A1 rel ≈ 0.32）。所以"能算"不等于"是级联"。

例 5 的公式（文档 437–623 行）与 Eq.(46) 结构相似 —— 也含裸 $\epsilon$ 张量、
也缩并 $p_\psi$。所以**必须先判定**，再决定移植策略：

* 命中（rel ≲ 1e-12）⇒ 例 5 是级联的第二个验收场，照 b₁ 流程移植；
* 不命中 ⇒ 与 ρ′ 同类，只能作为"协变路线可算"的例子。

## 判据

对每条协变道 $W_{\rm cov}$，穷举「树 × 顶层 $(l,s)$ × 内层 $(l,s)$」，
用级联内核算出 $A$，看是否存在一组使 $\|W-sA\|/\|W\| < 10^{-12}$。

    python analyses/tensor_examples/probes/probe_ex5_identify.py

需要 jax 参考实现（`Tensor/FUNC.py`）来计算例 5 的协变道 —— 它们**还没移植**
到本仓库。用 `analyses/kk_dis/.venv/bin/python` 跑。
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from agent.tensor.kernels import helicity_cascade as HC             # noqa: E402
from agent.tensor.kernels.helicity import EPS_A                     # noqa: E402

TENSOR = os.path.join(ROOT, "analyses", "kk_dis", "task",
                      "20260929_螺旋度耦合振幅推导", "1_运行代码",
                      "tensor_example")
M_PSI, M_PHI, M_K, M_PI = 3.686, 1.019, 0.493677, 0.13957039

# 例 5 的 10 条道（参考实现 Tensor/FUNC.py 的**公开**名，见 Modifiers()）
EX5 = [("u_kst0_l", "l"), ("u_kst0_r", "r"),
       ("u_PP_1", "l"), ("u_PP_2", "r"),
       ("u_PF_1", "l"), ("u_PF_2", "r"),
       ("u_FP_1", "l"), ("u_FP_2", "r"),
       ("u_FF_1", "l"), ("u_FF_2", "r")]


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


def physical(n=32, seed=11, m_kst=1.43):
    r"""$\psi\to K^*\pi$，$K^*\to\phi\pi$，$\phi\to K^+K^-$ 的物理事例。

    每个 `Qabc` 的 `a -> b + c` 都是真两体衰变，所以 $q^2>0$、$B_l$ 不 NaN。
    """
    rng = np.random.default_rng(seed)
    kst, spec_pi = _pair(M_PSI, m_kst, M_PI, rng, n)
    phi, pi_in = _pair(m_kst, M_PHI, M_PI, rng, n)
    kp0, km0 = _pair(M_PHI, M_K, M_K, rng, n)
    bk = kst[:, :3] / kst[:, 3:4]
    bph = phi[:, :3] / phi[:, 3:4]
    # Kp/Km 先在 phi 静止系 -> 再 boost 到 K* 系 -> 再 boost 到 psi 系（实验室）。
    # 少一层会让四动量不守恒、Qabc 出负值、B_l 变 NaN。
    kp_kst, km_kst = _boost(kp0, bph), _boost(km0, bph)
    return {"Kp": _boost(kp_kst, bk), "Km": _boost(km_kst, bk),
            "Pip": _boost(pi_in, bk), "Pim": spec_pi}


def rel(a, b):
    na, nb = float(np.vdot(a, a).real), float(np.abs(b).max())
    if not np.isfinite(na) or na < 1e-300 or nb < 1e-300:
        return np.inf          # a 恒零（组合被 CG 禁止）或 b 非有限
    s = np.vdot(b, a) / np.vdot(a, a)
    return float(np.abs(b - s * a).max() / nb)


def load_jax():
    t = os.path.join(TENSOR, "Tensor")
    if not os.path.isdir(t):
        return None
    if t not in sys.path:
        sys.path.insert(0, t)
    try:
        import jax
        # **必须在导入 FUNC 之前**打开 x64，否则 jax 全程 float32，
        # 对拍只能到 1e-7（被误判成"未命中"）。见 test_covariant_vs_jax。
        jax.config.update("jax_enable_x64", True)
        import FUNC
    except Exception as e:                                       # noqa: BLE001
        print("跳过：导入 FUNC 失败（%s: %s）" % (type(e).__name__, e))
        return None
    f = FUNC.FUNCTION()
    f.Modifiers()
    return f


def main():
    if "--control" in sys.argv:
        control()
        return 0
    n = 32
    d = physical(n=n, seed=11)
    mom = np.stack([d[k][:, [3, 0, 1, 2]]
                    for k in ("Kp", "Km", "Pip", "Pim")], axis=1)
    from agent.tensor.kernels import covariant as COV

    # 拓扑（K* 的质量取 1.43，K*₀(1430) / K*₂(1430) 附近）
    base_nodes = {
        "psi2s": {"J": 1, "mass": M_PSI, "daughters": ["Kstar", "Pim"]},
        "Kstar": {"J": 2, "mass": 1.43, "daughters": ["phi", "Pip"]},
        "phi": {"J": 1, "mass": M_PHI, "daughters": ["Kp", "Km"]},
        "Kp": {"J": 0, "mass": M_K}, "Km": {"J": 0, "mass": M_K},
        "Pip": {"J": 0, "mass": M_PI}, "Pim": {"J": 0, "mass": M_PI},
    }
    TREES = {
        "l": {"psi2s": ["Kstar", "Pip"], "Kstar": ["phi", "Pim"]},
        "r": {"psi2s": ["Kstar", "Pim"], "Kstar": ["phi", "Pip"]},
    }
    # 候选 (l,s) **按宇称 + 三角系统枚举**，不硬编码 —— 第一版手写的清单漏掉了
    # u_kst0 真正需要的顶层 (1,0)，导致 0/10 的假阴性。
    PAR = {"psi2s": -1, "Kstar": None, "phi": -1,
           "Kp": -1, "Km": -1, "Pip": -1, "Pim": -1}

    def allowed(jp, j1, j2, e_p, e1, e2, lmax=4):
        """宇称 eta_p = eta_1 eta_2 (-1)^l + 三角 |l-s| <= Jp <= l+s。"""
        out = []
        for l in range(0, lmax + 1):
            if e_p != e1 * e2 * (-1) ** l:
                continue
            for sp in range(abs(j1 - j2), j1 + j2 + 1):
                if abs(l - sp) <= jp <= l + sp:
                    out.append((l, sp))
        return out

    # K* 候选：0^-（u_kst0 那族）与 2^-（u_PP/PF/FP/FF 那族，K*₂(1770)）
    KSTAR = [(0, -1), (2, -1)]
    JCAND = [(jk, ek) for jk, ek in KSTAR] + [(1, -1), (3, -1), (2, 1), (1, 1)]

    # 10 条道已移植到 `agent/tensor/chains/kstpi/`，所以这里**不需要 jax** ——
    # 直接调本仓库的协变内核（它是逐式复刻，已与 jax 对拍通过，见
    # probes/verify_ex5_port.py）。
    INDEX = {"Kp": 0, "Km": 1, "Pip": 2, "Pim": 3}

    print("例 5 逐道穷举（rel < 1e-12 视为命中）")
    print("%-10s %-3s %-46s %s" % ("道", "族", "判据一（单个 (l,s) 精确匹配）",
                                   "判据二（子空间）"))
    print("-" * 100)
    nhit = 0
    tally = {}          # 按链分组统计 —— 单链数与全局数**不是**一回事
    for ch, side in EX5:
        try:
            W4 = COV.compute_event(ch, d)
            # (N,4) 四矢量 -> (N,3) 螺旋度振幅，与 test_ex3_channels 同一约定
            W = np.einsum("lm,em->el", EPS_A, W4[..., :3])
        except Exception as e:                                   # noqa: BLE001
            print("%-10s **协变内核调用失败** %s" % (ch, e))
            continue
        best = None
        # 同一 (J, 宇称) 才算同一假设；不同宇称的 K* 不能混在一个子空间里
        groups = {}
        for jk, ek in JCAND:
            # 内核从 0 起给 mom 的末态轴编号（Kp,Km,Pip,Pim 的顺序）
            c = {"top": "psi2s", "index": INDEX,
                 "nodes": {k: dict(v) for k, v in base_nodes.items()}}
            c["nodes"]["Kstar"]["J"] = jk
            c = HC.with_tree(c, TREES[side])
            ang = HC.node_angles(mom, c)
            TOP = allowed(1, jk, 0, -1, ek, -1)
            INNER = allowed(jk, 1, 0, ek, -1, -1)
            for tl in TOP:
                for il in INNER:
                    try:
                        A = HC.chain_amplitude(
                            mom, c, {"psi2s": [tl], "Kstar": [il],
                                     "phi": [(1, 0)]}, ang=ang)
                    except Exception:                            # noqa: BLE001
                        continue
                    if np.all(np.isfinite(A)) and float(np.abs(A).max()) > 0:
                        groups.setdefault((jk, ek), []).append(A)
                    r = rel(A, W)
                    if best is None or r < best[0]:
                        best = (r, tl, il, jk)
        if best is None:
            print("%-10s %-4s **无任何组合可算**" % (ch, side))
            continue
        # 判据二：W 是否落在"同一 K* 自旋下全部允许 (l,s) 组合"张成的子空间里。
        # 这才是计数律 prod_n N_n 的意思 —— 一个节点可以同时带多个分波。
        sub_best = (np.inf, None, 0)
        for key, amps in groups.items():
            Bm = np.stack([a.ravel() for a in amps], axis=1)     # (N*3, n_basis)
            w = W.ravel()
            coef, *_ = np.linalg.lstsq(Bm, w, rcond=None)
            res = float(np.abs(w - Bm @ coef).max() / max(np.abs(w).max(), 1e-300))
            if res < sub_best[0]:
                sub_best = (res, key, Bm.shape[1])
        exact = sub_best[0] < 1e-12
        nhit += exact
        which = "ex5_kst0" if ch.startswith("u_kst0") else "ex5_kst2"
        tally.setdefault(which, [0, 0])
        tally[which][0] += bool(exact)
        tally[which][1] += 1
        print("%-10s %-3s 单组合 %-12s %-10s %.2e   |  子空间 %d 维 %s  %.2e  %s   |W|max=%.3e"
              % (ch, side, str(best[1]), str(best[3]), best[0],
                 sub_best[2], sub_best[1], sub_best[0],
                 "命中" if exact else "未命中", float(np.abs(W).max())))
    print("-" * 72)
    print("命中 %d / %d（判据：W 落在该 K* 自旋的全部 (l,s) 组合张成的子空间里）"
          % (nhit, len(EX5)))
    print()
    print("按链拆分（**引用单链的数字时用这一行**）：")
    for k in sorted(tally):
        a, b = tally[k]
        print("  %-10s %d / %d" % (k, a, b))
    print("  %-10s %d / %d   <-- 全部例 5 道（跨两份 spec）"
          % ("合计", nhit, len(EX5)))
    return 0




# ------------------------------------------------------------------ 对照
def control():
    r"""**探针自检**：同一套机器跑 b₁ 的 8 条道（已知级联，已移植）。

    b₁ 的协变函数在 `agent/tensor/chains/b1pi/` 里、已与 jax 对拍过，所以
    它们是"级联一定应该命中"的**阳性对照**。若这里不给出 8/8，说明探针
    （而不是例 5）有问题 —— 那么例 5 的结论也不能信。
    """
    sys.path.insert(0, os.path.join(ROOT, "agent", "tensor", "tests"))
    import test_ex3_channels as T

    d = T.physical(n=32, seed=11)
    mom = np.stack([d[k][:, [3, 0, 1, 2]]
                    for k in ("Kp", "Km", "Pip", "Pim")], axis=1)
    from agent.tensor.kernels import covariant as COV

    base_nodes = {
        "psi2s": {"J": 1, "mass": M_PSI, "daughters": ["b1", "Pim"]},
        "b1": {"J": 1, "mass": 1.23, "daughters": ["phi", "Pip"]},
        "phi": {"J": 1, "mass": M_PHI, "daughters": ["Kp", "Km"]},
        "Kp": {"J": 0, "mass": M_K}, "Km": {"J": 0, "mass": M_K},
        "Pip": {"J": 0, "mass": M_PI}, "Pim": {"J": 0, "mass": M_PI},
    }
    TREES = {"1": {"psi2s": ["b1", "Pim"], "b1": ["phi", "Pip"]},
             "2": {"psi2s": ["b1", "Pip"], "b1": ["phi", "Pim"]}}
    print("对照：b₁ 的 8 条道（已知级联）")
    print("%-10s %-3s %-12s %-12s %s" % ("道", "族", "顶层(l,s)", "内层(l,s)", "rel"))
    print("-" * 60)
    ok = 0
    for ch in ["u_SS_1", "u_SD_1", "u_DS_1", "u_DD_1",
               "u_SS_2", "u_SD_2", "u_DS_2", "u_DD_2"]:
        side = ch[-1]
        W = np.einsum("lm,em->el", EPS_A,
                      COV.compute_event(ch, d)[..., :3])
        best = None
        for tl in [(0, 1), (2, 1)]:
            for il in [(0, 1), (2, 1)]:
                c = {"top": "psi2s",
                     "index": {"Kp": 0, "Km": 1, "Pip": 2, "Pim": 3},
                     "nodes": {k: dict(v) for k, v in base_nodes.items()},
                     "finals": []}
                c = HC.with_tree(c, TREES[side])
                ang = HC.node_angles(mom, c)
                A = HC.chain_amplitude(
                    mom, c, {"psi2s": [tl], "b1": [il], "phi": [(1, 0)]},
                    ang=ang)
                r = rel(A, W)
                if best is None or r < best[0]:
                    best = (r, tl, il)
        ok += best[0] < 1e-12
        print("%-10s %-3s %-12s %-12s %.3e  %s"
              % (ch, side, str(best[1]), str(best[2]), best[0],
                 "命中" if best[0] < 1e-12 else "**未命中**"))
    print("-" * 60)
    print("对照结果 %d / 8" % ok)
    return ok


if __name__ == "__main__":
    sys.exit(main())
