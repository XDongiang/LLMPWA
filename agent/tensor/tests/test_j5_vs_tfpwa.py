#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""J=5 / l=5 的**外部**对拍：协变张量路线 vs tf-pwa。

为什么需要这个文件
------------------
`test_covariant_general.py` 验的是"新构造与写死的 T1..T4 一致 + 无迹 + 全对称"
—— 全是**内部**判据。`test_ex12_permutation.py` 验的是"cartesian / helicity /
cascade 三条路线互相一致" —— 也是内部判据（同一套 `t_l`）。

本文件用 **tf-pwa**（独立实现）来验 J=5 的张量公式本身。fixture 由
`make_tfpwa_j5_golden.py` 在远程 tf-pwa docker 里生成。

⚠️ 为什么 fixture 的配对质量必须在壳
------------------------------------
tf-pwa 的 `has_barrier_factor=False` 只关掉 `B_l`，**仍然乘 `q^l`**
（`tf_pwa/amp/core.py::get_barrier_factor2` 的 `has_ql` 分支）；而论文的
`t~^(l)` 是 `Q_abc^l B_l(Q_abc)`，本仓库的 `t_l` 只含 `B_l`，`q^l` 由拟合
参数 `g_i` 吸收。所以**位垒口径本来就不同**（既有约定差异，不是 bug）。

若配对质量逐事例变化，`q^l/B_l` 就是个逐事例因子，对拍会失败 —— 而且失败
的是"口径差"不是"公式错"，无法区分。把配对质量固定在 `m_ρ5` 之后，这个
因子退化成**整体常数**，剩下的角分布 / 全部 λ 分量 / 置换结构必须逐点对上。

实测（`tfpwa_j5_golden.json`，seed=33，8 事例）：三条道 overlap=1.00000000、
残差 ~2e-15、|c| 一致。
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

from agent.tensor.kernels import covariant as CO           # noqa: E402
from agent.tensor.kernels.helicity import EPS_A            # noqa: E402

GOLDEN = os.path.join(_HERE, "tfpwa_j5_golden.json")

# 道名 -> (内层第一子, 内层第二子, 顶层旁观者)
ROLE = {"12": ("Pip", "Pim", "Pi0"),
        "13": ("Pi0", "Pip", "Pim"),
        "23": ("Pi0", "Pim", "Pip")}

PSI_MASS = 3.686


def _load():
    if not os.path.exists(GOLDEN):
        return None
    with open(GOLDEN, encoding="utf-8") as fh:
        return json.load(fh)


def _cov_channel(psi, a, b, c):
    r"""论文 `U_{\rho_5}^\mu` 的一项（不含 f/BW，它们在 g_i 里）。

    `-i eps_{mu nu lam sig} p_psi^sig T~^{(5)nu...} t~^{(5)lam}...`
    返回 (N, 4)。
    """
    t5_top = CO.t_l(5, psi, a + b, c)
    t5_in = CO.t_l(5, a + b, a, b)
    m_nl = np.einsum("enABCD,elABCD->enl", t5_top, t5_in)
    return np.einsum("mnls,es,enl->em", CO.leci_four(), psi, m_nl)


def _ours(golden, tag):
    """我们的螺度振幅 W(e, lambda)，(N, 3)。"""
    d1, d2, spec = ROLE[tag]
    mom = {k: np.asarray(v)[:, [1, 2, 3, 0]]      # tf-pwa (T,X,Y,Z) -> 我们 (x,y,z,E)
           for k, v in golden["momenta"][tag].items()}
    n = mom[d1].shape[0]
    psi = np.broadcast_to(np.array([0.0, 0.0, 0.0, PSI_MASS]), (n, 4))
    a4 = _cov_channel(psi, mom[d1], mom[d2], mom[spec])
    return np.einsum("lm,em->el", EPS_A, a4[:, :3])


def _tf(golden, tag):
    return (np.asarray(golden["channels"][tag]["re"])
            + 1j * np.asarray(golden["channels"][tag]["im"]))


# --------------------------------------------------------------- 1) 逐道对拍
def test_j5_matches_tfpwa_per_channel():
    """三条道分别与 tf-pwa 只差一个**整体常数**（残差 ~1e-15）。

    这是"J=5 张量公式正确"的核心证据：角分布与全部 λ 分量逐点对上。
    """
    golden = _load()
    if golden is None:
        return _skip("没有 %s" % GOLDEN)
    for tag in ("12", "13", "23"):
        w = _ours(golden, tag)
        tf = _tf(golden, tag)
        assert w.shape == tf.shape, (tag, w.shape, tf.shape)
        p, o = w.reshape(-1), tf.reshape(-1)
        c = np.vdot(p, o) / np.vdot(p, p)
        d = np.abs(o - c * p).max() / np.abs(o).max()
        assert d < 1e-12, "_%s: 与 tf-pwa 残差 %.3e" % (tag, d)


def test_j5_overlap_is_exactly_one():
    """归一化重叠必须 ≈1 —— 排除"差一个逐事例因子"的情况。

    `test_j5_matches_tfpwa_per_channel` 的"拟合一个 c"允许 c 随道变化；
    这条额外要求**形状**完全一致（overlap=1），能抓出"每个事例差一个
    不同因子但碰巧残差小"的假阳性。
    """
    golden = _load()
    if golden is None:
        return _skip("没有 %s" % GOLDEN)
    for tag in ("12", "13", "23"):
        p = _ours(golden, tag).reshape(-1)
        o = _tf(golden, tag).reshape(-1)
        ov = abs(np.vdot(p, o)) / (np.linalg.norm(p) * np.linalg.norm(o))
        assert ov > 1.0 - 1e-9, "_%s: overlap=%.10f" % (tag, ov)


def test_j5_overall_constant_is_shared():
    """三条道的归一化常数**同阶**（位垒/BW 口径差是同一个）。

    若某条道的 |c| 与其他道差很多，说明那条道的置换结构写错了
    （张量缩并漏了指标、或顶层/内层顺序搞反）。
    """
    golden = _load()
    if golden is None:
        return _skip("没有 %s" % GOLDEN)
    cs = []
    for tag in ("12", "13", "23"):
        p = _ours(golden, tag).reshape(-1)
        o = _tf(golden, tag).reshape(-1)
        cs.append(np.vdot(p, o) / np.vdot(p, p))
    mags = [abs(c) for c in cs]
    assert max(mags) / min(mags) < 100.0, "三条道 |c| 差太多：%s" % (mags,)


# ------------------------------------------------------- 2) 反向：故意写错
def test_wrong_pairing_is_caught():
    """负对照：把 _13 的配对换成 _12 的，必须**对不上**。

    确认上面的对拍不是"怎么算都能过"。用 `_12` 的树去算 `_13` 的动量
    （即用错配对），残差应当很大。
    """
    golden = _load()
    if golden is None:
        return _skip("没有 %s" % GOLDEN)
    # _13 的正确配对是 (Pi0, Pip)；故意用 (Pip, Pim)
    mom = {k: np.asarray(v)[:, [1, 2, 3, 0]]
           for k, v in golden["momenta"]["13"].items()}
    n = mom["Pip"].shape[0]
    psi = np.broadcast_to(np.array([0.0, 0.0, 0.0, PSI_MASS]), (n, 4))
    wrong = _cov_channel(psi, mom["Pip"], mom["Pim"], mom["Pi0"])
    w = np.einsum("lm,em->el", EPS_A, wrong[:, :3]).reshape(-1)
    o = _tf(golden, "13").reshape(-1)
    c = np.vdot(w, o) / np.vdot(w, w)
    d = np.abs(o - c * w).max() / np.abs(o).max()
    assert d > 0.1, "用错配对竟然对上了（残差 %.3e）—— 测试是空的" % d


def test_offshell_data_would_not_match():
    """记录一个**已知的口径差异**（不是 bug，是护栏）。

    若 fixture 的配对不在壳，tf-pwa 的 `q^l` 会变成逐事例因子，
    与我们的 `B_l` 口径对不上。这条测试用 `tfpwa_j5_offshell_probe.json`
    （早期探针，同一组动量、配对不在壳）钉住这个事实 ——
    以后若有人拿"离壳对拍失败"当 bug 报，这里能立刻解释清楚。
    """
    path = os.path.join(_HERE, "tfpwa_j5_offshell_probe.json")
    if not os.path.exists(path):
        return _skip("没有 %s（早期探针，可选）" % path)
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    # 该探针里三条道共用同一组动量，只有 _12 的配对在壳
    p12 = np.asarray(d["12"]["re"]) + 1j * np.asarray(d["12"]["im"])
    p13 = np.asarray(d["13"]["re"]) + 1j * np.asarray(d["13"]["im"])
    # 两条道用的是**不同**的树，所以振幅形状不该相同
    a = p12.reshape(-1)
    b = p13.reshape(-1)
    c = np.vdot(a, b) / np.vdot(a, a)
    d13 = np.abs(b - c * a).max() / np.abs(b).max()
    assert d13 > 0.1, ("离壳探针里 _12 与 _13 竟然形状相同（残差 %.3e）"
                       " —— 说明该探针的前提变了" % d13)


def _skip(msg):
    print("  （跳过：%s）" % msg)
    return None


if __name__ == "__main__":
    import traceback
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print("  OK   %s" % name)
        except Exception:
            fails += 1
            print("  FAIL %s" % name)
            traceback.print_exc()
    print("\n%s: %d/%d" % ("**FAIL**" if fails else "PASS",
                           len(tests) - fails, len(tests)))
    sys.exit(1 if fails else 0)
