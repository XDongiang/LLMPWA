#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""例 3 的 rho'/b1 十条道 + 深度分级的纯 numpy 回归（不需要 jax / Tensor/）。

与 `test_covariant_vs_jax.py` 的分工：

* 那个文件回答"移植得对不对"（需要 jax 原版，缺了就跳过）；
* 本文件回答"搬进来之后还自洽吗"，只用本仓库的代码，**永远能跑**：
  - `chains/` 里的 10 个道文件能被发现、导入、调用；
  - 每条道对**物理**四动量返回 (N,4) 且有限；
  - rho' 与 b1 的 `_1`/`_2` 确实是"旁观者对调"（把 3 <-> 4 交换后两条道互换）；
  - `b1` 的 SS/SD/DS/DD 是四个**线性独立**的振幅（否则道表冗余）；
  - 深度分级：三层树只出 cartesian/helicity 时放行，出 omega 时拒绝。

    python agent/tensor/tests/test_ex3_channels.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.tensor import chains as chainsmod            # noqa: E402
from agent.tensor import spec as specmod                # noqa: E402
from agent.tensor.kernels import covariant as COV       # noqa: E402

M_PSI, M_PHI, M_K, M_PI = 3.686, 1.019, 0.493677, 0.13957039

RHO_NAMES = ["u_rho_1", "u_rho_2"]
B1_NAMES = ["u_SS_1", "u_SS_2", "u_SD_1", "u_SD_2",
            "u_DS_1", "u_DS_2", "u_DD_1", "u_DD_2"]


# --------------------------------------------------------------- 运动学
def _pair(m_parent, m1, m2, rng, n):
    """母粒子静止系两体衰变，(x,y,z,E)。"""
    lam = ((m_parent ** 2 - (m1 + m2) ** 2)
           * (m_parent ** 2 - (m1 - m2) ** 2))
    p = np.sqrt(max(lam, 0.0)) / (2.0 * m_parent)
    ct = rng.uniform(-1.0, 1.0, n)
    st = np.sqrt(np.maximum(1.0 - ct ** 2, 0.0))
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    d = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)
    e1 = np.sqrt(m1 ** 2 + p ** 2)
    e2 = np.sqrt(m2 ** 2 + p ** 2)
    return (np.concatenate([p * d, np.full((n, 1), e1)], axis=1),
            np.concatenate([-p * d, np.full((n, 1), e2)], axis=1))


def _boost(p, beta):
    b2 = (beta ** 2).sum(-1)
    g = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-300))
    bp = (beta * p[:, :3]).sum(-1)
    g2 = np.where(b2 > 1e-12, (g - 1.0) / np.where(b2 > 1e-12, b2, 1.0), 0.0)
    return np.concatenate(
        [p[:, :3] + (g2 * bp + g * p[:, 3])[:, None] * beta,
         (g * (p[:, 3] + bp))[:, None]], axis=1)


def physical(n=16, seed=3):
    """psi -> phi f2 -> (K+K-)(pi+pi-) 的物理事例。

    这样构造是因为 rho'/b1 的两体学**恰好**成立（见 test_covariant_vs_jax.py
    里那段注释）：`phi + Pip + Pim == psi`，所以每个 `Qabc` 的 `a -> b + c`
    都是真两体衰变，`q^2 > 0`，`B1/B2` 不会 NaN。
    """
    rng = np.random.default_rng(seed)
    phi, f2 = _pair(M_PSI, M_PHI, 1.275, rng, n)
    kp0, km0 = _pair(M_PHI, M_K, M_K, rng, n)
    pip0, pim0 = _pair(1.275, M_PI, M_PI, rng, n)
    bphi = phi[:, :3] / phi[:, 3:4]
    bf2 = f2[:, :3] / f2[:, 3:4]
    return {"Kp": _boost(kp0, bphi), "Km": _boost(km0, bphi),
            "Pip": _boost(pip0, bf2), "Pim": _boost(pim0, bf2)}


def _psi(mom):
    return np.broadcast_to(np.array([0.0, 0.0, 0.0, M_PSI]),
                           (mom["Kp"].shape[0], 4))


def swap34(mom):
    """把粒子 3 (Pip) 与 4 (Pim) 对调。"""
    return {"Kp": mom["Kp"], "Km": mom["Km"],
            "Pip": mom["Pim"], "Pim": mom["Pip"]}


def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


# ------------------------------------------------------------------ 测试
def test_chains_discovered():
    """`chains/` 能发现 rhopi / b1pi，且道数与道名符合预期。"""
    allc = chainsmod.all_channels()
    for nm in RHO_NAMES:
        assert allc.get(nm) == "rhopi", (nm, allc.get(nm))
    for nm in B1_NAMES:
        assert allc.get(nm) == "b1pi", (nm, allc.get(nm))
    assert chainsmod.list_channels("rhopi") == ["u_rho_1", "u_rho_2"]
    assert chainsmod.list_channels("b1pi") == sorted(B1_NAMES)


def test_channels_callable_and_finite():
    """10 条道都能调用，返回 (N,4)，且对物理四动量有限。"""
    mom = physical()
    psi = _psi(mom)
    for nm in RHO_NAMES + B1_NAMES:
        T = COV.compute_event(nm, mom, psi=psi)
        assert T.shape == (mom["Kp"].shape[0], 4), (nm, T.shape)
        assert np.isfinite(T).all(), "%s 有非有限值" % nm
        assert np.abs(T).max() > 0, "%s 恒为 0" % nm


def test_compute_event_falls_back_to_chains():
    """`channel_fn` 对未在 `_CHANNELS` 里的道名回退到 `chains/`。"""
    for nm in RHO_NAMES + B1_NAMES:
        assert nm not in COV._CHANNELS
        assert callable(COV.channel_fn(nm))
    assert "u_rho_1" in COV.available_channels()
    assert "phif001" in COV.available_channels()
    try:
        COV.channel_fn("no_such_channel_at_all")
    except KeyError as e:
        assert "no_such_channel_at_all" in str(e)
    else:
        raise AssertionError("未知道名应该 KeyError")


def test_rho_pair_are_spectator_swaps():
    """rho 的 `_1`/`_2` 是 3 <-> 4 对调：T1(p) == T2(swap(p))。"""
    mom = physical()
    psi = _psi(mom)
    a = COV.compute_event("u_rho_1", mom, psi=psi)
    b = COV.compute_event("u_rho_2", swap34(mom), psi=psi)
    assert _rel(a, b) < 1e-12, _rel(a, b)
    # 而且两条道本身**不相等**（否则道表冗余）
    c = COV.compute_event("u_rho_2", mom, psi=psi)
    assert _rel(a, c) > 1e-3


def test_b1_pairs_are_spectator_swaps():
    """b1 的每组 `_1`/`_2` 同样是 3 <-> 4 对调。"""
    mom = physical()
    psi = _psi(mom)
    for fam in ("SS", "SD", "DS", "DD"):
        a = COV.compute_event("u_%s_1" % fam, mom, psi=psi)
        b = COV.compute_event("u_%s_2" % fam, swap34(mom), psi=psi)
        assert _rel(a, b) < 1e-12, (fam, _rel(a, b))


def test_b1_four_amplitudes_are_independent():
    """SS/SD/DS/DD 必须是四个线性独立的振幅（否则道表冗余）。

    把 8 条道堆成 (8, N*4) 的矩阵，奇异值分解：rank 应为 8。
    """
    mom = physical(n=64, seed=5)
    psi = _psi(mom)
    M = np.stack([COV.compute_event(nm, mom, psi=psi).ravel() for nm in B1_NAMES])
    sv = np.linalg.svd(M, compute_uv=False)
    assert sv.min() / sv.max() > 1e-8, "b1 的 8 条道线性相关，最小奇异值比 %.3e" \
        % (sv.min() / sv.max())


def test_depth_grading():
    """深度分级：三层树在 cartesian/helicity 下放行，在 omega 下拒绝。"""
    path = os.path.join(_ROOT, "analyses", "tensor_examples", "specs",
                        "ex3_rhopi.toml")
    if not os.path.exists(path):
        print("    跳过：%s 不在" % path)
        return
    s = specmod.load_spec(path)
    assert s["depth"] == 3, s["depth"]

    raw = open(path, encoding="utf-8").read()
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_tmp_depth_omega.toml")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(raw + '\n[[basis]]\nname = "omega"\nkernel = "omega"\n')
    try:
        specmod.load_spec(tmp)
    except specmod.SpecError as e:
        assert "深度" in str(e) and "omega" in str(e)
    else:
        raise AssertionError("三层树 + omega 应该被拒绝")
    finally:
        os.unlink(tmp)


def test_specs_load_and_emit():
    """两份 spec 能读入、道表顺序正确，且能 emit 出自足脚本。"""
    import tempfile

    from agent.tensor import emitter
    base = os.path.join(_ROOT, "analyses", "tensor_examples", "specs")
    for tree, names in (("ex3_rhopi", RHO_NAMES), ("ex3_b1pi", B1_NAMES)):
        p = os.path.join(base, "%s.toml" % tree)
        if not os.path.exists(p):
            print("    跳过：%s 不在" % p)
            continue
        s = specmod.load_spec(p)
        assert [c[0] for c in specmod.resolve_channels(s)] == names
        with tempfile.TemporaryDirectory() as td:
            res = emitter.emit(s, out_dir=td)
            assert len(res["scripts"]) == 2
            assert res["fallback"] == [], res["fallback"]
            for e in res["scripts"]:
                assert os.path.exists(e["script"])
                src = open(e["script"], encoding="utf-8").read()
                for nm in names:
                    assert "def %s(" % nm in src, (tree, nm)
                # 产物自足：不 import agent.tensor
                assert "import agent" not in src


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
