#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`kernels/helicity_cascade.py` 的框架级回归。

三层，从"这是不是 tf-pwa 的公式"到"能力边界有没有真的推开"：

1. **逐节点角度参考系**（§1.1 的根因）：`node_angles` 必须逐位复刻
   "每个节点在**母节点**静止系里量角"的构造。判据是它让 b₁ 的 4 个独立
   振幅对上协变路线 —— `chain_angles`（omega 的口径）对不上。
2. **判据 A1**：级联 vs `kernels/covariant.py` 逐道对拍（经 `EPS_A`）。
   协变内核是手写公式、与 omega/级联完全独立，所以这条能同时抓住 CG 约定
   错、螺旋度槽错、缩并顺序错。
3. **判据 A2/A3**：逐节点组合的秩 = ∏_n N_n，且道两两不重合（omega 时代
   b₁ 的 8 条道里 12/28 对逐位重合）。

跑法：`python agent/tensor/tests/test_helicity_cascade.py`
"""
from __future__ import annotations

import os
import sys

import numpy as onp

HERE = os.path.dirname(os.path.abspath(__file__))
# HERE = <repo>/agent/tensor/tests  ->  ROOT = <repo>
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import test_ex3_channels as T                                       # noqa: E402
from agent.tensor import spec as S                                  # noqa: E402
from agent.tensor.kernels import covariant as COV                   # noqa: E402
from agent.tensor.kernels import helicity_cascade as HC             # noqa: E402
from agent.tensor.kernels import omega as RT                        # noqa: E402
from agent.tensor.kernels.helicity import EPS_A                     # noqa: E402

EX3 = os.path.join(ROOT, "analyses", "tensor_examples", "specs")
PHIF2 = os.path.join(ROOT, "agent", "tensor", "specs", "phif2.toml")
N = 24


def _mom(n=N, seed=11):
    d = T.physical(n=n, seed=seed)
    return d, onp.stack([d[k][:, [3, 0, 1, 2]]
                         for k in ("Kp", "Km", "Pip", "Pim")], axis=1)


def _rel(a, b):
    s = onp.vdot(b, a) / onp.vdot(a, a)
    return float(onp.abs(b - s * a).max() / max(onp.abs(b).max(), 1e-300))


def _wcov(name, momd):
    return onp.einsum("lm,em->el", EPS_A,
                      COV.compute_event(name, momd)[..., :3])


def _cascade(spec, cc, mom):
    c = HC.with_tree(S.to_chain(spec), cc["tree"] or {})
    return HC.chain_amplitude(mom, c, cc["ls"], ang=HC.node_angles(mom, c))


# ------------------------------------------------------------------ 1
def test_node_angles_differ_from_chain_angles_when_nested():
    """b₁ 的 `b1` 子节点自己还衰变，两种参考系构造必须**不同**。

    φf₂ 里 `phi` 是末态，两种构造逐位相同 —— 这正是这个 bug 到现在才被
    发现的原因（§1.1）。
    """
    momd, mom = _mom()
    sp = S.load_spec(os.path.join(EX3, "ex3_b1pi.toml"))
    chain = S.to_chain(sp)
    a1 = HC.node_angles(mom, chain)
    a2 = RT.chain_angles(mom, chain)
    d = max(float(onp.abs(onp.asarray(a1[k][i]) - onp.asarray(a2[k][i])).max())
            for k in a1 for i in range(3))
    assert d > 1e-3, "b1 链上两种参考系构造应当不同，实测 max diff = %.3e" % d

    sp2 = S.load_spec(PHIF2)
    c2 = S.to_chain(sp2)
    b1 = HC.node_angles(mom, c2)
    b2 = RT.chain_angles(mom, c2)
    d2 = max(float(onp.abs(onp.asarray(b1[k][i]) - onp.asarray(b2[k][i])).max())
             for k in b1 for i in range(3))
    assert d2 < 1e-12, "phif2 上两种构造应当逐位相同，实测 %.3e" % d2


# ------------------------------------------------------------------ 2 (A1)
def test_b1_a1_matches_covariant_all_eight():
    """判据 A1：b₁ 的 8 条道，级联 vs 协变（经 EPS_A）rel < 1e-12。"""
    momd, mom = _mom()
    sp = S.load_spec(os.path.join(EX3, "ex3_b1pi.toml"))
    worst = 0.0
    for cc in S.cascade_channels(sp):
        r = _rel(_cascade(sp, cc, mom), _wcov(cc["name"], momd))
        worst = max(worst, r)
        assert r < 1e-12, "%s rel = %.3e" % (cc["name"], r)
    assert worst < 1e-12


def test_phif2_a1_matches_covariant_all_five():
    """φf₂ 不能退化（这个修正对它是无操作）。"""
    momd, mom = _mom()
    sp = S.load_spec(PHIF2)
    for cc in S.cascade_channels(sp):
        r = _rel(_cascade(sp, cc, mom), _wcov(cc["name"], momd))
        assert r < 1e-12, "%s rel = %.3e" % (cc["name"], r)


def test_omega_cannot_represent_b1_where_cascade_does():
    """把 §1.1/§2.1 的边界钉成断言：omega 的基底**表示不了** b₁。

    判据是最小二乘可表示性（自由复系数，任何整体常数都被吸收）：
    用 omega 的两条道（顶层 (L,S) = (0,1) / (2,1)）张成的基底去拟合协变的
    `u_DD_1`，残差必须是 O(1) —— 而级联的残差是机器精度。

    如果哪天 omega 也能表示，说明这条链被"意外修好"，那时才该删掉
    helicity_cascade —— 这条断言是防静默退化的。
    """
    momd, mom = _mom()
    sp = S.load_spec(os.path.join(EX3, "ex3_b1pi.toml"))
    chain = S.to_chain(sp)
    nodes = chain["nodes"]
    ja = nodes["psi2s"]["J"]
    jb, jc = nodes["b1"]["J"], nodes["Pim"]["J"]
    lam_b, lam_c = RT.LAM_SLOTS[jb], RT.LAM_SLOTS[jc]
    om = RT.omega_from_momenta(mom, chain, covariant=True)
    # omega 的道表只由**顶层** (L,S) 决定 -> 基底只有 2 维
    basis = []
    for (l, s) in ((0, 1), (2, 1)):
        M = RT.cg_matrix([(l, s)], ja, jb, jc, lam_b, lam_c)
        basis.append(onp.einsum("enab,ab->en", om, M[:, :, 0]))
    B = onp.stack([b.ravel() for b in basis], 1)
    W = _wcov("u_DD_1", momd).ravel()
    c, *_ = onp.linalg.lstsq(B, W, rcond=None)
    resid = float(onp.linalg.norm(W - B @ c) / onp.linalg.norm(W))
    assert resid > 0.1, "omega 居然能表示 b1（残差 %.3e）—— 边界记录需要更新" % resid

    # 对照：级联基底（逐节点 (l,s) 的 4 个组合）必须能精确表示
    cc = [x for x in S.cascade_channels(sp) if x["name"] == "u_DD_1"][0]
    c2 = HC.with_tree(chain, cc["tree"])
    ang = HC.node_angles(mom, c2)
    cb = []
    for tl in ((0, 1), (2, 1)):
        for il in ((0, 1), (2, 1)):
            A = HC.chain_amplitude(
                mom, c2, {"psi2s": [tl], "b1": [il], "phi": [(1, 0)]}, ang=ang)
            cb.append(A.ravel())
    B2 = onp.stack(cb, 1)
    c3, *_ = onp.linalg.lstsq(B2, W, rcond=None)
    resid2 = float(onp.linalg.norm(W - B2 @ c3) / onp.linalg.norm(W))
    assert resid2 < 1e-12, "级联应当能精确表示 u_DD_1（残差 %.3e）" % resid2


# ------------------------------------------------------------------ 3 (A2/A3)
def test_a2_rank_equals_product_of_node_counts():
    """判据 A2：逐节点组合张成的秩 = ∏_n N_n（b₁：2×2 = 4）。"""
    momd, mom = _mom()
    sp = S.load_spec(os.path.join(EX3, "ex3_b1pi.toml"))
    chain = S.to_chain(sp)
    tree = {"psi2s": ["b1", "Pim"], "b1": ["phi", "Pip"]}
    c = HC.with_tree(chain, tree)
    ang = HC.node_angles(mom, c)
    # 4 个逐节点组合 -> 4 个向量；把它们拉平后必须秩 4
    basis = []
    for tl in ((0, 1), (2, 1)):
        for il in ((0, 1), (2, 1)):
            A = HC.chain_amplitude(
                mom, c, {"psi2s": [tl], "b1": [il], "phi": [(1, 0)]}, ang=ang)
            basis.append(A.ravel())
    B = onp.stack(basis, axis=1)
    rank = int(onp.linalg.matrix_rank(B, tol=1e-8 * onp.abs(B).max()))
    assert rank == 4, "逐节点组合的秩应为 4，实测 %d" % rank
    # 只按顶层分组只有 2 维 —— 这就是 omega 丢掉的那 2 维
    top_only = [basis[0] + basis[1], basis[2] + basis[3]]
    B2 = onp.stack(top_only, axis=1)
    rank2 = int(onp.linalg.matrix_rank(B2, tol=1e-8 * onp.abs(B2).max()))
    assert rank2 == 2, "只按顶层分组应为秩 2，实测 %d" % rank2


def test_a3_b1_eight_channels_pairwise_distinct():
    """判据 A3：8 条道两两不重合（omega 时代 12/28 对逐位重合）。"""
    momd, mom = _mom()
    sp = S.load_spec(os.path.join(EX3, "ex3_b1pi.toml"))
    cache = {cc["name"]: _cascade(sp, cc, mom) for cc in S.cascade_channels(sp)}
    ks = list(cache)
    assert len(ks) == 8
    worst = 1.0
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            worst = min(worst, _rel(cache[ks[i]], cache[ks[j]]))
    assert worst > 1e-6, "最接近的一对 rel = %.3e（应 > 1e-6）" % worst


# ------------------------------------------------------------------ 4 (A4)
def test_a4_depth3_sensitive_to_innermost_plane():
    """判据 A4：绕 φ 方向转 φ→K⁺K⁻ 的衰变平面，级联必须有反应。

    omega 的 `omega_from_momenta` 只算 `D0*D1*D2`，对这种深度 3 的链
    **完全不敏感**（§1.3）。这里同时验两条：级联敏感、omega 不敏感。
    """
    momd, mom = _mom(n=16, seed=3)
    phi3 = momd["Kp"][:, :3] + momd["Km"][:, :3]
    r = momd["Kp"][:, :3] - momd["Km"][:, :3]
    ax = phi3 / onp.linalg.norm(phi3, axis=1, keepdims=True)
    ang = onp.deg2rad(37.0)
    c_, s_ = onp.cos(ang), onp.sin(ang)
    d_ = (r * ax).sum(1, keepdims=True)
    r2 = r * c_ + onp.cross(ax, r) * s_ + ax * d_ * (1 - c_)
    mr = {k: v.copy() for k, v in momd.items()}
    mr["Kp"][:, :3] = (phi3 + r2) / 2
    mr["Km"][:, :3] = (phi3 - r2) / 2
    momr = onp.stack([mr[k][:, [3, 0, 1, 2]]
                      for k in ("Kp", "Km", "Pip", "Pim")], axis=1)

    # 自检：p_phi 与总四动量逐位不变
    assert _rel(mr["Kp"] + mr["Km"], momd["Kp"] + momd["Km"]) < 1e-12
    t0 = sum(momd[k] for k in ("Kp", "Km", "Pip", "Pim"))
    t1 = sum(mr[k] for k in ("Kp", "Km", "Pip", "Pim"))
    assert _rel(t1, t0) < 1e-12

    sp = S.load_spec(os.path.join(EX3, "ex3_b1pi.toml"))
    cc = [c for c in S.cascade_channels(sp) if c["name"] == "u_DD_1"][0]
    A0 = _cascade(sp, cc, mom)
    A1 = _cascade(sp, cc, momr)
    assert _rel(A1, A0) > 0.1, "级联对最内层角分布不敏感（rel = %.3e）" % _rel(A1, A0)

    # omega 口径：应当**毫无反应**
    chain = S.to_chain(sp)
    o0 = RT.omega_from_momenta(mom, chain, covariant=True)
    o1 = RT.omega_from_momenta(momr, chain, covariant=True)
    assert _rel(o1, o0) < 1e-12, "omega 居然对最内层敏感了（rel = %.3e）" % _rel(o1, o0)


# ------------------------------------------------------------------ 5
def test_lambda_row_order_matches_eps_a():
    """落盘行序必须与 `EPS_A` 的行序逐位一致（承重约定，不要翻轴）。"""
    momd, mom = _mom()
    sp = S.load_spec(PHIF2)
    chain = S.to_chain(sp)
    ang = HC.node_angles(mom, chain)
    for name, ls in (("phif201", [(0, 1)]), ("phif221", [(2, 1)]),
                     ("phif243", [(4, 3)])):
        A = HC.chain_amplitude(
            mom, chain, {"psi2s": ls, "phi": [(1, 0)], "f2": [(2, 0)]}, ang=ang)
        W = _wcov(name, momd)
        # 逐行同号：第 i 行配第 i 行，不需要任何翻转
        assert _rel(A, W) < 1e-12, "%s 行序不对（rel = %.3e）" % (name, _rel(A, W))
        assert _rel(A[:, ::-1], W) > 0.1, "%s 翻了轴反而对上了？" % name


def test_covariant_term_is_required():
    """§4.4：Chung Eq.52 的 `f^J_{|m|}(gamma)` 不能省。

    关掉后连"逐事件比值是常数"都不成立 —— 它不是一个可吸收的整体归一化。
    """
    momd, mom = _mom(n=16, seed=11)
    sp = S.load_spec(PHIF2)
    chain = S.to_chain(sp)
    ang = HC.node_angles(mom, chain)
    for name, ls in (("phif201", [(0, 1)]), ("phif221", [(2, 1)]),
                     ("phif222", [(2, 2)])):
        lm = {"psi2s": ls, "phi": [(1, 0)], "f2": [(2, 0)]}
        A = HC.chain_amplitude(mom, chain, lm, covariant=True, ang=ang)
        B = HC.chain_amplitude(mom, chain, lm, covariant=False, ang=ang)
        W = _wcov(name, momd)
        assert _rel(A, W) < 1e-12, "%s 打开协变项应当精确" % name
        # 关掉后：逐事件比值必须**不是**常数
        r = onp.array([onp.vdot(B[e], W[e]) / onp.vdot(B[e], B[e])
                       for e in range(B.shape[0])])
        spread = float(onp.abs(r / r[0] - 1).max())
        assert spread > 1e-3, "%s 关掉协变项后比值仍是常数（%.3e）" % (name, spread)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in tests:
        try:
            fn()
            print("  OK   %s" % fn.__name__)
        except AssertionError as e:
            bad += 1
            print("  FAIL %s: %s" % (fn.__name__, e))
        except Exception as e:                       # noqa: BLE001
            bad += 1
            print("  ERR  %s: %s: %s" % (fn.__name__, type(e).__name__, e))
    print("%s: %d/%d" % ("PASS" if not bad else "**FAIL**",
                         len(tests) - bad, len(tests)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
