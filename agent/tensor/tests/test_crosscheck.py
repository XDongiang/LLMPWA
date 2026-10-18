#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨算法一致性检验（`crosscheck`）的回归测试。

核心恒等式（逐事例、逐道成立）：

    W_i^{cov}(e,lam) = c_i * g_i * K_i(e) * (M_i . Omega)_{e,lam}

c_i 是逐道的**复常数**（耦合 + 归一化 + 相位，由拟合吸收），所以检验的是
"c_i 与事例无关"。

这个测试同时验证**正例**（合成的一致数据必须通过）与**负例**（结构性错误
必须被抓到）—— 负例才是这个检验存在的理由。

    python agent/tensor/tests/test_crosscheck.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)

from agent.tensor import crosscheck as cc       # noqa: E402

BUNDLE = os.path.join(_ROOT, "analyses", "kk_dis", "tmp", "bundledata")
F2_NAMES = ["phif201", "phif221", "phif222", "phif223", "phif243"]
F0_NAMES = ["phif001", "phif021"]


def _skip(msg):
    print("    跳过：%s" % msg)
    return None


def _synth(n_ch=3, N=64, seed=5):
    """合成一组**严格满足**恒等式的数据。"""
    rng = np.random.default_rng(seed)
    nA, nb, nc = 3, 3, 5
    Omega = rng.normal(size=(N, nA, nb, nc)) + 1j * rng.normal(size=(N, nA, nb, nc))
    M = rng.normal(size=(n_ch, nb, nc))
    K = np.abs(rng.normal(size=(n_ch, N))) + 0.5
    g = np.abs(rng.normal(size=n_ch)) + 0.5
    c = rng.normal(size=n_ch) + 1j * rng.normal(size=n_ch)
    P = np.einsum("iab,enab->ien", M, Omega)
    W = c[:, None, None] * g[:, None, None] * K[:, :, None] * P
    # 反旋回笛卡尔：W_l = sum_m EPS_A[l,m] T_m  =>  T_m = sum_l conj(EPS_A[l,m]) W_l
    T = np.einsum("lm,...l->...m", cc.EPS_A.conj(), W)
    return T, Omega, K, M, g, c


def test_synthetic_roundtrip_passes():
    """合成数据（严格满足恒等式）必须通过，且还原出的 c_i 要对。"""
    T, Om, K, M, g, c = _synth()
    names = ["c%d" % i for i in range(len(c))]
    r = cc.check_arrays(T, Om, K, M, g, names=names, tol=1e-9)
    assert r["ok"], cc.report(r)
    for i, nm in enumerate(names):
        assert abs(r[nm]["c"] - c[i]) / abs(c[i]) < 1e-9, (nm, r[nm]["c"], c[i])


def test_shape_mismatch_is_reported():
    """形状对不上要报可读的 CrossCheckError，而不是 numpy 的广播错误。"""
    T, Om, K, M, g, _c = _synth()
    for label, kw in (("K 列数不符", {"K": K[:, :-1]}),
                      ("M 首轴不符", {"M": M[:-1]}),
                      ("Omega nA 不符", {"Omega": Om[:, :2]})):
        args = {"T": T, "Omega": Om, "K": K, "M": M, "g": g}
        args.update(kw)
        try:
            cc.check_arrays(args["T"], args["Omega"], args["K"], args["M"],
                            args["g"])
        except cc.CrossCheckError:
            pass
        else:
            raise AssertionError("%s 竟然没报错" % label)


def test_event_dependent_errors_are_caught():
    """**事件相关**的结构性错误必须全部抓到（这是检验存在的理由）。"""
    T, Om, K, M, g, _c = _synth(N=48)
    cases = {
        "K 隔事例缩放": {"K": K * np.where(np.arange(K.shape[1]) % 2, 1.5, 1.0)},
        "Omega 隔事例缩放": {"Omega": Om * np.where(np.arange(Om.shape[0]) % 2,
                                                1.3, 1.0)[:, None, None, None]},
        "T 隔事例缩放": {"T": T * np.where(np.arange(T.shape[1]) % 3, 1.2, 1.0)[None, :, None]},
        "T 的 y 分量取反": {"T": T * np.array([1.0, -1.0, 1.0])},
        "Omega 事例轴错位": {"Omega": np.roll(Om, 1, axis=0)},
        "K 事例轴错位": {"K": np.roll(K, 1, axis=1)},
        "单道 K 随事例漂移": {"K": np.concatenate(
            [(K[0] * np.linspace(1, 3, K.shape[1]))[None], K[1:]], axis=0)},
    }
    for label, kw in cases.items():
        args = {"T": T, "Omega": Om, "K": K, "M": M, "g": g}
        args.update(kw)
        r = cc.check_arrays(args["T"], args["Omega"], args["K"], args["M"],
                            args["g"])
        assert not r["ok"], "%s 竟然通过了（max_rel=%.3e）" % (label, r["max_rel"])


def test_per_channel_constants_are_absorbed():
    """逐道常数按定义会被 c_i 吸收 —— 这**不是** bug，写下来防止误判。"""
    T, Om, K, M, g, _c = _synth(N=48)
    cases = {
        "K 逐道常数偏移": {"K": K * np.array([1.0, 1.5, 2.0])[:, None]},
        "g 全部改掉": {"g": np.array([2.0, 3.0, 4.0])},
        "单道整体偏移": {"T": T * np.array([1.0, 1.05, 1.0])[:, None, None]},
    }
    for label, kw in cases.items():
        args = {"T": T, "Omega": Om, "K": K, "M": M, "g": g}
        args.update(kw)
        r = cc.check_arrays(args["T"], args["Omega"], args["K"], args["M"],
                            args["g"])
        assert r["ok"], "%s 本应被吸收，却判失败了" % label


def test_production_cache_passes_both_families():
    """生产 cache（若在）必须通过 —— 这条是真正的验收。"""
    cart = os.path.join(BUNDLE, "route1_cartesian", "real_data")
    omg = os.path.join(BUNDLE, "route4_omega", "real_data")
    if not os.path.exists(os.path.join(cart, "phif2_kk.npy")):
        return _skip("没有 %s" % cart)
    for chain, suf, names in (("phif2", "", F2_NAMES), ("phif0", "0", F0_NAMES)):
        r = cc.check_dirs(cart, omg, chain=chain, suffix=suf, names=names)
        assert r["ok"], cc.report(r)
        print("    %-6s 后缀 %-2r max_rel=%.3e  c=%s"
              % (chain, suf, r["max_rel"],
                 np.round([r[n]["c"].real for n in names], 4)))


def test_already_rotated_flag_avoids_double_rotation():
    """`_hel.npy` 已经是 W，不能再乘一次 EPS_A。"""
    T, Om, K, M, g, _c = _synth(N=32)
    W = np.einsum("lm,...m->...l", cc.EPS_A, T)
    r_bad = cc.check_arrays(W, Om, K, M, g)                       # 又乘一次
    r_ok = cc.check_arrays(W, Om, K, M, g, already_rotated=True)
    assert not r_bad["ok"], "重复旋转竟然通过了"
    assert r_ok["ok"], cc.report(r_ok)


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
