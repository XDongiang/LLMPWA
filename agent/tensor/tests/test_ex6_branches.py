#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 6 三条子分支的**状态**钉死（2026-10-21）。

例 6 拆成三份 spec（一份一个子分支）：

| spec | 子分支 | 道数 | A1 |
|---|---|---|---|
| `ex6_phi4pi_ss` | $f_J\to\sigma\sigma$ | 14 | ✅ 7e-15 |
| `ex6_phi4pi_rr` | $f_J\to\rho\rho$ | 14 | f_0 的 4 条 ✅ 5.7e-15；f_2 的 10 条 ⚠️ 0.47–0.86 |
| `ex6_phi4pi_f2s` | $f_2\to f_2(1270)\sigma$ | 10 | ⚠️ 0.47–1.08 |

ρρ / f₂σ 的两条断言是**互补**的：

* `test_primitives_match_document` —— 保证 `T2_rr`/`T2_f2s` 确实是文档
  公式的直译（若有人"顺手改对"，这条会先失败）；
* `test_rr_and_f2s_fail_a1` —— 保证 $f_2$ 那 20 条与级联**仍然**对不上。
  哪天真的修好了（无迹化落到 $f_2$ 系），这条会失败并提醒：
  删掉 `A1_KNOWN`/`A1_EXEMPT` 里的条目、更新 feasibility §2.3.1。

另外 `test_f0_rho_rho_passes_a1` 单独钉住 $f_0\to\rho\rho$ —— 它是
$\rho\rho$ 家族里**能过**的那一半，靠 $\tilde g(p_{f_0})$ 横向投影而非
$P^{(2)}$，所以没有无迹化参考系问题。

    python agent/tensor/tests/test_ex6_branches.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.tensor.kernels.covariant import (                     # noqa: E402
    P2, T1_cov, T2_cov, T2_f2s, T2_rr,
)
from agent.tensor.kernels.helicity import EPS_A                  # noqa: E402

_G = np.diag([-1.0, -1.0, -1.0, 1.0])


def _kin(n=32, seed=5):
    p = os.path.join(_ROOT, "analyses", "tensor_examples", "data",
                     "make_phi4pi.py")
    sp = importlib.util.spec_from_file_location("_mp4", p)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m.make(n=n, seed=seed)


def _gt_cov(a):
    r"""$\tilde G_{\mu\nu}$ 的**独立**实现（协变指标）。

    `_atleast2` 的口径是 $a_\mu$ 直接取值，所以先把上指标降下来再外积。
    """
    ac = a @ _G
    a2 = np.einsum("ei,ei->e", a, ac)
    return _G[None] - np.einsum("ei,ej->eij", ac, ac) / a2[:, None, None]


def _P2(a):
    g = _gt_cov(a)
    return (0.5 * (np.einsum("eij,ekl->eikjl", g, g)
                   + np.einsum("eil,ekj->eikjl", g, g))
            - (1.0 / 3.0) * np.einsum("eik,ejl->eikjl", g, g))


def test_primitives_match_document():
    r"""`T2_rr` / `T2_f2s` 必须与文档 723–729 / 764–774 行的直译逐位相同。

    这条防的是"有人为了让 A1 过而改公式" —— 那会让本测试失败，从而强制
    改动者先去 feasibility §2.3.1 说明为什么。
    """
    d = _kin()
    Pip, Pim = d["Pip"], d["Pim"]
    Pip2, Pim2 = d["Pip2"], d["Pim2"]
    f2 = Pip + Pim + Pip2 + Pim2

    p2 = _P2(f2)
    assert np.abs(p2 - np.asarray(P2(f2))).max() < 1e-13 * np.abs(p2).max(), \
        "P2 的独立实现与内核不一致 —— 独立检查本身失效"

    # 文档 723–729：P2(f2)[ t1(34)_a t1(56)_b ]
    want_rr = np.einsum(
        "eijab,ea,eb->eij", p2,
        T1_cov(Pip + Pim, Pip, Pim),
        T1_cov(Pip2 + Pim2, Pip2, Pim2))
    got_rr = np.asarray(T2_rr(f2, Pip, Pim, Pip2, Pim2))
    rel = np.abs(got_rr - want_rr).max() / max(np.abs(want_rr).max(), 1e-300)
    assert rel < 1e-13, "T2_rr 不是文档直译：rel = %.3e" % rel

    # 文档 764–774：P2(f2)[ t2(34) + t2(56) ]
    want_f2s = np.einsum(
        "eijab,eab->eij", p2,
        T2_cov(Pip + Pim, Pip, Pim) + T2_cov(Pip2 + Pim2, Pip2, Pim2))
    got_f2s = np.asarray(T2_f2s(f2, Pip, Pim, Pip2, Pim2))
    rel = np.abs(got_f2s - want_f2s).max() / max(np.abs(want_f2s).max(), 1e-300)
    assert rel < 1e-13, "T2_f2s 不是文档直译：rel = %.3e" % rel


def _spec_of(channel):
    """道名 -> 它在哪份拆分后的 spec 里。"""
    if "_ss_" in channel:
        return "ex6_phi4pi_ss.toml"
    if "_rr_" in channel:
        return "ex6_phi4pi_rr.toml"
    return "ex6_phi4pi_f2s.toml"


def _a1(channel, momd):
    """跑单条道，返回（螺度 vs 级联）的 A1 残差。"""
    from agent.tensor import spec as S
    from agent.tensor.kernels import helicity_cascade as HC

    sys.path.insert(0, os.path.join(_ROOT, "analyses", "tensor_examples",
                                    "probes"))
    from probe_a1_full import cov_event                            # noqa: E402

    sp = S.load_spec(os.path.join(_ROOT, "analyses", "tensor_examples",
                                  "specs", _spec_of(channel)))
    base = S.to_chain(sp)
    keys = list(momd)
    n = momd[keys[0]].shape[0]
    mom = np.stack([momd[k] for k in keys], axis=1)[:, :, [3, 0, 1, 2]]
    psi = np.broadcast_to(np.array([0.0, 0.0, 0.0, 3.686]), (n, 4))

    for c in S.cascade_channels(sp):
        if c["name"] != channel:
            continue
        ch = HC.with_tree(base, c["tree"] or {})
        A = HC.chain_amplitude(mom, ch, c["ls"], ang=HC.node_angles(mom, ch))
        w = np.einsum("lm,em->el", EPS_A,
                      np.asarray(cov_event(sp, c, momd, psi))[..., :3])
        a = w.reshape(n, -1)
        b = np.asarray(A).reshape(n, -1)
        coef = np.vdot(a.ravel(), b.ravel()) / np.vdot(a.ravel(), a.ravel())
        return float(np.abs(b - coef * a).max() / max(np.abs(b).max(), 1e-300))
    raise AssertionError("spec 里没有道 %r" % channel)


def test_sigma_sigma_passes_a1():
    """σσ 子分支必须过 A1（这是它可用的唯一理由）。"""
    d = _kin()
    for ch in ("u_phif2_ss_01", "u_phif2_ss_22", "u_phif0_ss_21"):
        r = _a1(ch, d)
        assert r < 1e-12, "%s 的 A1 残差 %.3e —— σσ 分支本该精确" % (ch, r)


def test_f0_rho_rho_passes_a1():
    r"""$f_0\to\rho\rho$ 必须过 A1。

    它是 $\rho\rho$ 家族里能过的**那一半**：$f_0$ 是标量，两个 $\rho$ 先
    合成 $S=0$（$S_rr$ 用 $\tilde g(p_{f_0})$ 横向夹住），**不涉及**
    $P^{(2)}$，所以没有 $1\otimes1\to2$ 的无迹化参考系问题。

    ⚠️ 裸点积（不夹 $\tilde g$）实测 rel 1.02e-1 —— 这条测试同时钉住
    "必须夹横向度规"。
    """
    d = _kin()
    for ch in ("u_phif0_rr_01", "u_phif0_rr_21", "u_phif0_rr_01_b"):
        r = _a1(ch, d)
        assert r < 1e-12, "%s 的 A1 残差 %.3e —— f_0→ρρ 本该精确" % (ch, r)


def test_rr_and_f2s_fail_a1():
    """ρρ 与 f₂(1270)σ **必须**失败 —— 这是已知缺陷，不是待办。

    哪天修好了（无迹化落到 $f_2$ 系），本测试会失败。那时要做的是：
    删 `probes/verify_all_specs.py::A1_KNOWN` 与
    `tests/test_handwritten_complete.py::A1_EXEMPT` 里的 20 条，
    并把 feasibility §2.3.1 从"通不过"改成"已修"。
    """
    d = _kin()
    bad = []
    for ch in ("u_phif2_rr_01", "u_phif2_rr_22",
               "u_phif2_f2s_01", "u_phif2_f2s_43"):
        r = _a1(ch, d)
        if r < 1e-6:
            bad.append((ch, r))
    assert not bad, (
        "这些道现在**过**了 A1：%s —— 说明无迹化参考系已修好。"
        "请删掉 A1_KNOWN / A1_EXEMPT 里的 20 条并更新 feasibility §2.3.1。"
        % bad)


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    bad = 0
    for name, fn in tests:
        try:
            fn()
            print("  OK   %s" % name)
        except Exception as e:                                  # noqa: BLE001
            bad += 1
            print("  FAIL %s: %s" % (name, e))
    print("%s: %d/%d" % ("PASS" if not bad else "**FAIL**",
                         len(tests) - bad, len(tests)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
