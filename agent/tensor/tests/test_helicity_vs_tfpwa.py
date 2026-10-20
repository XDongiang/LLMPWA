#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把本仓库的**螺度级联**内核与上游 **tf-pwa** 对拍（A5 判据）。

与相邻两个文件的分工：

* `test_covariant_vs_jax.py` —— 与**项目自己的 jax 参考实现**对拍，回答
  "从 Tensor/ 移植过来有没有改数字"；需要 jax，缺了就跳过。
* `test_ex3_channels.py`    —— 只用本仓库代码，回答"搬进来之后还自洽吗"。
* **本文件**                 —— 与**上游 tf-pwa**对拍，回答"我们的螺度约定
  和社区标准实现是不是同一个东西"。用冻结的 fixture，**不需要 TF/tf-pwa**，
  永远能跑。

fixture：`tfpwa_b1pi_golden.json`（由 `make_tfpwa_golden.py` 在装了 tf-pwa 的
环境里生成，来源与 flags 记在 fixture 的 `convention` 字段里）。

--------------------------------------------------------------------------
比什么、怎么比
--------------------------------------------------------------------------

三层，从下往上，任一层错都能定位：

1. **CG 矩阵**（`cg_matrix`）—— tf-pwa `HelicityDecay.get_cg_matrix()`，
   逐元素比。这是最底层，错了后面全错。
2. **逐节点欧拉角**（`node_angles`）—— tf-pwa `cal_angle_from_momentum` 的
   `(alpha, beta)`。这一层确认"螺旋度参考系"选得一样。
3. **链振幅**（`chain_amplitude`）—— tf-pwa `get_amp3`。

第 3 层**必须把位垒因子除掉**：tf-pwa 用 `d = 3` 的 Blatt-Weisskopf 参数化，
本仓库用 `Q0 = 0.197321`，两者是不同的约定（实测连 q² 形状都不同，不是差
一个常数）。所以 fixture 里存的是 tf-pwa `has_barrier_factor=False` 的振幅，
测试端把本仓库的位垒乘子除掉再比 —— 比的是 D 矩阵 + CG + 链缩并这一层。

比法用"一个**全局**复常数"：`W_ours ≈ c · W_tfpwa`，`c` 用最小二乘拟合，
然后要求残差相对误差 < 1e-12。**不逐事例、不逐道**各拟合一个常数 —— 那样
会把"只对得上形状"的错也放过。整体常数是允许的：tf-pwa 的 `g_ls` 默认归一
与我们的 `g=1` 差一个约定因子。

    python agent/tensor/tests/test_helicity_vs_tfpwa.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.tensor import spec as specmod                       # noqa: E402
from agent.tensor.kernels import helicity_cascade as HC        # noqa: E402
from agent.tensor.kernels import omega as RT                   # noqa: E402

GOLDEN = os.path.join(_HERE, "tfpwa_b1pi_golden.json")
SPEC = os.path.join(_ROOT, "analyses", "tensor_examples", "specs",
                    "ex3_b1pi.toml")
TOL = 1e-12

# fixture 里的角是按 tf-pwa 的**节点名**索引的，映射到我们的粒子名
NODE_OF_PARTICLE = {
    "Pim": "psi->(Km, Kp, Pip)+Pim",
    "Pip": "(Km, Kp, Pip)->(Km, Kp)+Pip",
    "Kp": "(Km, Kp)->Kp+Km",
    "Km": "(Km, Kp)->Kp+Km",
}


def _golden():
    with open(GOLDEN, encoding="utf-8") as fh:
        return json.load(fh)


def _momenta():
    """fixture 里冻结的四动量，形状 `(N, 4, 4)` = (事例, 粒子, (E,x,y,z))。

    fixture 存的就是 tf-pwa 布局 `(T,X,Y,Z)`，与本仓库的 `(E,x,y,z)` 同序，
    不需要换轴。顺序固定为 `("Kp","Km","Pip","Pim")`。
    """
    g = _golden()
    mom = g["momenta"]
    order = ("Kp", "Km", "Pip", "Pim")
    arr = np.stack([np.asarray(mom[k], dtype=np.float64) for k in order], axis=1)
    assert arr.shape == (g["n_ev"], 4, 4), arr.shape
    return arr


def _rel_global(a, b):
    """`b ≈ c·a`（单个全局复常数 c）时的最大相对残差。"""
    A = np.asarray(a).reshape(-1)
    B = np.asarray(b).reshape(-1)
    c = np.vdot(A, B) / np.vdot(A, A)
    return float(np.abs(B - c * A).max() / np.abs(B).max())


# ------------------------------------------------------------------ 1) CG
def test_cg_matrix_matches_tfpwa():
    """`omega.cg_matrix` 与 tf-pwa `get_cg_matrix` 逐元素一致。"""
    g = _golden()
    ls_list = [(0, 1), (2, 1)]
    lam_b, lam_c = RT.LAM_SLOTS[1], RT.LAM_SLOTS[0]
    for key, ja, jb, jc in (("top_J1_1_0", 1, 1, 0),
                            ("inner_J1_1_0", 1, 1, 0)):
        want = np.asarray(g["cg"][key], dtype=np.float64)      # (nls, nb, nc)
        got = RT.cg_matrix(ls_list, ja, jb, jc, lam_b, lam_c)
        assert got.shape == want.shape, (key, got.shape, want.shape)
        d = np.abs(got - want).max()
        assert d < TOL, "%s: max|diff| = %.3e" % (key, d)


# ------------------------------------------------------------- 2) 逐节点角
def test_node_angles_match_tfpwa():
    """`node_angles` 的 (alpha, beta) 与 tf-pwa 逐节点一致。"""
    g = _golden()
    mom = _momenta()
    sp = specmod.load_spec(SPEC)
    base = specmod.to_chain(sp)
    ch = [c for c in specmod.cascade_channels(sp) if c["name"] == "u_SS_1"][0]
    chain = HC.with_tree(base, ch["tree"])
    ang = HC.node_angles(mom, chain)

    checked = 0
    for pname, node in NODE_OF_PARTICLE.items():
        want = g["angles"][node][pname]
        for i, comp in enumerate(("alpha", "beta")):
            w = np.asarray(want[comp], dtype=np.float64)
            got = np.asarray(ang[pname][i], dtype=np.float64)
            assert got.shape == w.shape, (pname, comp, got.shape, w.shape)
            d = np.abs(got - w).max()
            assert d < 1e-12, "%s.%s: max|diff| = %.3e" % (pname, comp, d)
            checked += 1
    assert checked == 8, checked


# --------------------------------------------------------------- 3) 链振幅
def _barrier_product(mom, chain, ls_map):
    """本仓库的位垒乘子 `prod_n q^l B_l(q^2)`（l=0 时为 1）。

    只有在**每个节点恰好一个 (l,s)** 时，位垒才是整条链的一个公因子、可以
    直接除掉。本 fixture 的 8 条道都满足（顶层单 ls、b1 单 ls、phi 恒 (1,0)），
    这里显式断言，避免以后加了多 (l,s) 的道之后这条测试**静默**变成错的。
    """
    nodes = chain["nodes"]
    out = np.ones(mom.shape[0])
    for n, ls_list in ls_map.items():
        ds = nodes[n].get("daughters")
        if not ds:
            continue
        assert len(ls_list) == 1, (
            "节点 %s 有 %d 个 (l,s)，位垒不再是公因子，不能整体除掉"
            % (n, len(ls_list)))
        pa = RT._node_momentum(mom, chain, n)
        pb = RT._node_momentum(mom, chain, ds[0])
        pc = RT._node_momentum(mom, chain, ds[1])
        Q = RT.q2_relative(pa, pb, pc)
        for (l, _s) in ls_list:
            out = out * np.asarray(RT.node_barrier(int(l), Q))
    return out


def test_chain_amplitude_matches_tfpwa():
    """8 条 b1pi 道的链振幅（除掉位垒后）与 tf-pwa 一致到 1e-12。"""
    g = _golden()
    mom = _momenta()
    sp = specmod.load_spec(SPEC)
    base = specmod.to_chain(sp)

    worst = 0.0
    for ch in specmod.cascade_channels(sp):
        nm = ch["name"]
        chain = HC.with_tree(base, ch["tree"])
        ls_map = {k: [tuple(x) for x in v] for k, v in ch["ls"].items()}
        ours = HC.chain_amplitude(mom, chain, ls_map, covariant=True)
        ours = ours / _barrier_product(mom, chain, ls_map)[:, None]
        blk = g["channels"][nm]
        want = (np.asarray(blk["re"], dtype=np.float64)
                + 1j * np.asarray(blk["im"], dtype=np.float64))
        assert ours.shape == want.shape, (nm, ours.shape, want.shape)
        r = _rel_global(ours, want)
        worst = max(worst, r)
        assert r < TOL, "%s: 全局常数拟合后残差 = %.3e" % (nm, r)
    assert worst < TOL, worst


def test_barrier_convention_is_documented():
    """守住"位垒约定不同"这个前提：fixture 必须是无位垒 + 开协变项。

    如果哪天 tf-pwa 的默认改了、或者有人重新生成 fixture 时忘了关位垒，
    上面的链振幅测试会以"莫名其妙差 20%"的方式失败。这条把它变成一句
    明确的报错。
    """
    g = _golden()
    conv = g["convention"]
    assert conv["has_barrier_factor"] is False, conv
    assert conv["add_covariant_term"] is True, conv


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
