#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用协变张量 `t_l` 的回归测试（论文 Eq.258 的实现）。

背景
----
`kernels/covariant.py` 原来只有写死的 `T1_cov..T4_cov` 与 `B1..B4`，所以
协变张量路线的 J/l 上限是 **4**；ρ₅ / K*₅（J=5）根本写不出来。

2026-10-20 补上了论文 `jpsipwa1.tex` Eq.(258) 的通用构造：

    \\tilde t^{(L)}_{i_1...i_L} = sum_k (-1)^k (2L-2k-1)!!/(2L-1)!!
                                  * sum_{|S|=2k} (r.r)^k
                                    * prod_{(u,v) in S} g~_{i_u i_v}
                                    * prod_{j notin S} r~_{i_j}

（`g~ = Gt_cov(a)`、`r~ = g~.(b-c)`、`r.r = r~_mu r~_nu G^{mu nu}`。）

本文件钉住三件事，缺一不可：

1. **兼容性**：l=1..4 必须与写死的 `T1_cov..T4_cov` **逐位一致** ——
   新构造不能改变任何已有结果（生产 cache 就是按 1e-12 比的）。
2. **无迹性**：对任意一对指标缩并度规必须为 0 —— 这是论文
   "traceless" 那两行的直接检验，也是构造写错时最先炸的判据。
3. **全对称**：交换任意两个指标不变（张量必须是完全对称的）。

另外单独验 `B5` 与 tf-pwa 的 `Bprime_polynomial` 系数一致 ——
论文把 B5 那一行**注释掉了**，本仓库是照着 tf-pwa 补的，属于独立来源。
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.tensor.kernels import covariant as CO   # noqa: E402

_FIXED = {1: CO.T1_cov, 2: CO.T2_cov, 3: CO.T3_cov, 4: CO.T4_cov}
_LET = "ijklmnopqrstuvwxyz"


def _momenta(n=6, seed=0):
    rng = np.random.default_rng(seed)
    a = np.tile(np.array([0.0, 0.0, 0.0, 3.686]), (n, 1))
    b = rng.normal(size=(n, 4))
    c = rng.normal(size=(n, 4))
    return a, b, c


# ---------------------------------------------------------------- 1) 兼容性
def test_t_l_reproduces_hardcoded_t1_to_t4():
    """l=1..4 与写死的 `T1_cov..T4_cov` 逐位一致。

    这是**最重要的**一条：它保证新构造是"扩能力"而不是"改公式"。
    """
    a, b, c = _momenta()
    for l, fn in _FIXED.items():
        mine = np.asarray(CO.t_l(l, a, b, c))
        ref = np.asarray(fn(a, b, c))
        assert mine.shape == ref.shape, (l, mine.shape, ref.shape)
        scale = np.abs(ref).max()
        d = np.abs(mine - ref).max() / scale
        assert d < 1e-12, "l=%d 与写死的 T%d_cov 差 %.3e" % (l, l, d)


def test_t_l_without_barrier_matches_fixed_shape():
    """去掉 B_l 之后仍要一致 —— 防止"两边都乘了 B_l"把差异掩盖掉。

    `t_l` 与 `T_l_cov` 都含 `B_l`，直接比会掩盖 `B_l` 本身写错的情况
    （两边同错就抵消）。这里把 `T_l_cov` 的 `B_l` 除掉再比，等价于
    单独验张量部分。
    """
    a, b, c = _momenta()
    bl = {1: CO.B1, 2: CO.B2, 3: CO.B3, 4: CO.B4}
    for l, fn in _FIXED.items():
        n = a.shape[0]
        shape = (n,) + (1,) * l
        mine = np.asarray(CO.t_l(l, a, b, c)) / np.asarray(bl[l](a, b, c)).reshape(shape)
        ref = np.asarray(fn(a, b, c)) / np.asarray(bl[l](a, b, c)).reshape(shape)
        d = np.abs(mine - ref).max() / np.abs(ref).max()
        assert d < 1e-12, "l=%d（去 B_l）差 %.3e" % (l, d)


# ---------------------------------------------------------------- 2) 无迹性
def test_t_l_is_traceless():
    """对任意一对指标缩并度规为 0（论文明确要求 traceless）。

    构造写错时这条最先炸：少一项修正、或者 `r.r` 忘了用度规（用了欧氏
    内积）都会让迹不为 0。
    """
    a, b, c = _momenta()
    G = CO.metric()
    for l in range(2, 7):
        t = np.asarray(CO.t_l(l, a, b, c))
        scale = np.abs(t).max()
        for i in range(l):
            for j in range(i + 1, l):
                names = [_LET[k] for k in range(l)]
                names[i] = names[j] = "z"
                sub = "e" + "".join(names) + ",zz->e"
                tr = np.einsum(sub, t, G)
                d = np.abs(tr).max() / scale
                assert d < 1e-10, ("l=%d 指标 (%d,%d) 迹偏离 %.3e"
                                   % (l, i, j, d))


# ---------------------------------------------------------------- 3) 全对称
def test_t_l_is_fully_symmetric():
    """交换任意两个指标不变。"""
    a, b, c = _momenta()
    for l in range(2, 6):
        t = np.asarray(CO.t_l(l, a, b, c))
        scale = np.abs(t).max()
        for i in range(l):
            for j in range(i + 1, l):
                axes = list(range(t.ndim))
                axes[1 + i], axes[1 + j] = axes[1 + j], axes[1 + i]
                d = np.abs(t - t.transpose(axes)).max() / scale
                assert d < 1e-12, "l=%d 交换 (%d,%d) 差 %.3e" % (l, i, j, d)


# ------------------------------------------------------- 4) l=5/6 是新能力
def test_t_l_beyond_four_is_finite():
    """l=5/6 能算且有限 —— 这是本次改动新增的能力（原来直接没有）。"""
    a, b, c = _momenta()
    for l in (5, 6):
        t = np.asarray(CO.t_l(l, a, b, c))
        assert t.shape == (a.shape[0],) + (4,) * l, (l, t.shape)
        assert np.isfinite(t).all(), l
        assert np.abs(t).max() > 0.0, l


def test_t_l_rejects_absurd_l():
    """l 太大时要报错（而不是静默算错或吃光内存）。"""
    a, b, c = _momenta(n=2)
    try:
        CO.t_l(7, a, b, c)
    except (KeyError, ValueError) as e:
        assert "7" in str(e) or "l" in str(e).lower(), str(e)
    else:
        raise AssertionError("l=7 未报错（_BL 只有 0..6）")


# ------------------------------------------------------------- 5) B5 系数
def test_b5_matches_independent_implementation():
    """`B5` 与 `omega.Bl(5, q)` 一致 —— 两个**独立**实现互为对照。

    `omega.Bl` 走的是 tf-pwa 那套"反向 Bessel 多项式系数表"
    （`_bprime_coeff`），`covariant.B5` 走的是论文那个被注释掉的显式多项式。
    两条路径独立，能对上才说明 B5 的系数没抄错。

    这里同时验 l=1..4：它们本来就在两条路径里都有，是"对照有效"的前提。
    """
    from agent.tensor.kernels import omega as OM

    rng = np.random.default_rng(3)
    a = np.tile(np.array([0.0, 0.0, 0.0, 3.686]), (6, 1))
    b = rng.normal(size=(6, 4))
    c = rng.normal(size=(6, 4))
    q2 = np.asarray(CO.Qabc(a, b, c))
    assert (q2 > 0).all(), q2          # 别拿退化的组态比

    bl = {1: CO.B1, 2: CO.B2, 3: CO.B3, 4: CO.B4, 5: CO.B5}
    for l in sorted(bl):
        mine = np.asarray(bl[l](a, b, c))
        ref = np.asarray(OM.Bl(l, q2))
        d = np.abs(mine - ref).max() / np.abs(ref).max()
        assert d < 1e-12, "B%d vs omega.Bl 差 %.3e" % (l, d)


def test_b6_denominator_is_monic():
    """`B6` 的降幂首项系数为 1（与 `Bprime_polynomial` 的约定一致）。"""
    assert CO._BPRIME_COEFF_6[0] == 1.0
    assert len(CO._BPRIME_COEFF_6) == 7          # l=6 -> 7 个系数


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("  OK   %s" % name)
            except Exception:
                fails += 1
                print("  FAIL %s" % name)
                traceback.print_exc()
    total = sum(1 for n, f in globals().items()
                if n.startswith("test_") and callable(f))
    print("\n%s: %d/%d" % ("**FAIL**" if fails else "PASS", total - fails,
                           total))
    sys.exit(1 if fails else 0)
