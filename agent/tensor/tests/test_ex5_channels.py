#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 5（$\psi\to K^+\pi^-K^-\pi^+$）的 10 条 K*₀/K*₂ 道 —— 纯 numpy 回归。

与 `test_ex3_channels.py` 的分工相同：

* `test_covariant_vs_jax.py` / `probes/verify_ex5_port.py` 回答"移植得对不对"
  （需要 jax 原版，缺了就跳过）；
* 本文件回答"搬进来之后还自洽吗"，只用本仓库代码，**永远能跑**。

覆盖：

1. `chains/kstpi/` 的 10 条道能被发现、导入、调用，对物理四动量返回 (N,4) 且有限；
2. `_1`/`_2` 确实是"旁观者 π 对调"（把 3 <-> 4 交换后两条道互换）；
3. 两份 spec（`ex5_kst2` / `ex5_kst0`）能 `check`，且深度 3 让 omega 被拒；
4. 例 5 的拓扑与 b₁ 同构 —— 内层允许**多个** (l,s)（P 与 F），
   所以它是级联内核的第二个验收场。

    python agent/tensor/tests/test_ex5_channels.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from agent.tensor import chains as chainsmod            # noqa: E402
from agent.tensor import spec as specmod                # noqa: E402
from agent.tensor.kernels import covariant as COV       # noqa: E402

M_PSI, M_PHI, M_K, M_PI = 3.686, 1.019, 0.493677, 0.13957039

KST2_NAMES = ["u_PP_1", "u_PP_2", "u_PF_1", "u_PF_2",
              "u_FP_1", "u_FP_2", "u_FF_1", "u_FF_2"]
KST0_NAMES = ["u_kst0_l", "u_kst0_r"]
EX5_NAMES = KST0_NAMES + KST2_NAMES

SPECS = os.path.join(_ROOT, "analyses", "tensor_examples", "specs")


# --------------------------------------------------------------- 运动学
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


def physical(n=16, seed=7, m_kst=1.43):
    r"""$\psi\to K^*\pi\to(\phi\pi)\pi\to(K^+K^-)(\pi\pi)$ 的物理事例。

    每个 `Qabc` 的 `a -> b + c` 都是真两体衰变（$q^2>0$，$B_l$ 不 NaN）。
    Kp/Km 需要**两层** boost（φ 系 -> K* 系 -> ψ 系）。
    """
    rng = np.random.default_rng(seed)
    kst, spec_pi = _pair(M_PSI, m_kst, M_PI, rng, n)
    phi, pi_in = _pair(m_kst, M_PHI, M_PI, rng, n)
    kp0, km0 = _pair(M_PHI, M_K, M_K, rng, n)
    bk = kst[:, :3] / kst[:, 3:4]
    bph = phi[:, :3] / phi[:, 3:4]
    return {"Kp": _boost(_boost(kp0, bph), bk),
            "Km": _boost(_boost(km0, bph), bk),
            "Pip": _boost(pi_in, bk), "Pim": spec_pi}


def swap34(mom):
    """把粒子 3 (Pip) 与 4 (Pim) 对调。"""
    return {"Kp": mom["Kp"], "Km": mom["Km"],
            "Pip": mom["Pim"], "Pim": mom["Pip"]}


def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


# ------------------------------------------------------------------ 测试
def test_chains_discovered():
    """`chains/` 能发现 kstpi 的 10 条道，且道名符合预期。"""
    allc = chainsmod.all_channels()
    for nm in EX5_NAMES:
        assert allc.get(nm) == "kstpi", (nm, allc.get(nm))
    assert chainsmod.list_channels("kstpi") == sorted(EX5_NAMES)


def test_all_channels_finite_on_physical_momenta():
    """10 条道对物理四动量都返回 (N,4) 且有限。"""
    d = physical(n=16, seed=7)
    for nm in EX5_NAMES:
        a = COV.compute_event(nm, d)
        assert a.shape == (16, 4), (nm, a.shape)
        assert np.all(np.isfinite(a)), nm
        assert np.abs(a).max() > 0, nm


def test_swap34_swaps_the_pair():
    """`_1`/`_2`（以及 `_l`/`_r`）是"旁观者 π 对调"。"""
    d = physical(n=16, seed=7)
    ds = swap34(d)
    for a, b in [("u_kst0_l", "u_kst0_r"),
                 ("u_PP_1", "u_PP_2"), ("u_PF_1", "u_PF_2"),
                 ("u_FP_1", "u_FP_2"), ("u_FF_1", "u_FF_2")]:
        x = COV.compute_event(a, ds)
        y = COV.compute_event(b, d)
        assert _rel(x, y) < 1e-12, (a, b, _rel(x, y))


def test_specs_check_and_omega_is_rejected():
    """两份 spec 都能 `check`；深度 3 让 omega 被拒（不是静默出错）。"""
    for fn in ("ex5_kst2.toml", "ex5_kst0.toml"):
        p = os.path.join(SPECS, fn)
        sp = specmod.load_spec(p)
        assert sp["depth"] == 3, (fn, sp["depth"])
        assert "omega" not in [b["name"] for b in sp["bases"]]
        # 追加 omega 必须被拒
        raw = open(p, encoding="utf-8").read()
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False,
                                         encoding="utf-8") as f:
            f.write(raw + '\n[[basis]]\nname = "omega"\nkernel = "omega"\n')
            tmp = f.name
        try:
            try:
                specmod.load_spec(tmp)
            except specmod.SpecError:
                pass
            else:
                raise AssertionError("%s 加了 omega 竟然通过了" % fn)
        finally:
            os.unlink(tmp)


def test_inner_node_has_multiple_partial_waves():
    """K*₂ 的内层允许 P 与 F 两个 (l,s) —— 这是例 5 作为级联验收场的理由。

    `allowed_ls` 对 K*₂ -> phi + pi 应给出 l=1 与 l=3（s=1）。
    """
    sp = specmod.load_spec(os.path.join(SPECS, "ex5_kst2.toml"))
    ls = specmod.allowed_ls(sp, "Kstar")
    lset = sorted({l for l, _ in ls})
    assert lset == [1, 3], lset
    # 顶层只有 2 组 (l,s)，而道有 8 条 => (l,s) 不是完备标签
    top = specmod.allowed_ls(sp, "psi2s")
    assert len(top) == 2, top


def test_channel_ls_matches_spec_metadata():
    """spec 里每条道的逐节点 (l,s) 与 `cascade_channels` 给的一致。"""
    sp = specmod.load_spec(os.path.join(SPECS, "ex5_kst2.toml"))
    cc = {c["name"]: c for c in specmod.cascade_channels(sp)}
    assert set(cc) == set(KST2_NAMES)
    for nm, c in cc.items():
        assert c["ls"]["psi2s"] == [(1, 2)] if nm.startswith("u_P") \
            else c["ls"]["psi2s"] == [(3, 2)], (nm, c["ls"])
        assert c["ls"]["phi"] == [(1, 0)], nm
    # PF/FF 的内层是 F 波 (3,1)，PP/FP 是 P 波 (1,1)
    assert cc["u_PF_1"]["ls"]["Kstar"] == [(3, 1)]
    assert cc["u_PP_1"]["ls"]["Kstar"] == [(1, 1)]
    assert cc["u_FF_1"]["ls"]["Kstar"] == [(3, 1)]
    assert cc["u_FP_1"]["ls"]["Kstar"] == [(1, 1)]


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    ok = 0
    for n, f in tests:
        try:
            f()
            print("  OK   %s" % n)
            ok += 1
        except AssertionError as e:
            print("  FAIL %s: %s" % (n, e))
        except Exception as e:                                   # noqa: BLE001
            print("  ERR  %s: %s: %s" % (n, type(e).__name__, e))
    print("%s: %d/%d" % ("PASS" if ok == len(tests) else "**FAIL**",
                         ok, len(tests)))
    return 0 if ok == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
