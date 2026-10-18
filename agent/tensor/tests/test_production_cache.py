#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对**生产 cache** 的回归：这是 A1 方案最硬的验收判据。

如果 `analyses/kk_dis/tmp/bundledata/` 下有参照 cache，就重算并与它比：

    route1_cartesian/real_data/phif{0,2}_kk.npy   <- kernels/covariant.py
    route4_omega/real_data/{Omega,K,M}.npy        <- kernels/omega.py

⚠️ 真实样本的 psi 四动量必须用 `psi_mass = 3.685782956`（不是硬编码的 3.686）：
`psi_mass_residual.md` §6b 记录了这个口径差，用错会让 phif0 的 rel 从 1.7e-15
掉到 4.2e-04 —— 数值仍然"看着像样"，所以这个测试是有价值的。

没有参照 cache 时**跳过**，不算失败。

    python agent/tensor/tests/test_production_cache.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)

from agent.tensor.kernels import covariant as cov      # noqa: E402
from agent.tensor.kernels import omega as RT           # noqa: E402

BUNDLE = os.path.join(_ROOT, "analyses", "kk_dis", "tmp", "bundledata")
PSI_MASS_REAL = 3.685782956       # 真实样本；MC 交付样本用 3.686
TOL = 1e-12

_CHAIN = {
    "top": "psi2s",
    "nodes": {
        "psi2s": {"J": 1, "mass": 3.686, "daughters": ["phi", "f2"]},
        "phi": {"J": 1, "mass": 1.019, "daughters": ["Kp", "Km"], "l": 1},
        "f2": {"J": 2, "mass": 1.275, "daughters": ["Pip", "Pim"], "l": 2},
        "Kp": {"J": 0, "mass": 0.493677}, "Km": {"J": 0, "mass": 0.493677},
        "Pip": {"J": 0, "mass": 0.13957039},
        "Pim": {"J": 0, "mass": 0.13957039},
    },
    "index": {"Kp": 0, "Km": 1, "Pip": 2, "Pim": 3},
    "finals": ["Kp", "Km", "Pip", "Pim"],
}
_LS = [(0, 1), (2, 1), (2, 2), (2, 3), (4, 3)]


def _skip(msg):
    print("    跳过：%s" % msg)
    return None


def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def test_covariant_reproduces_production_cache():
    src = os.path.join(BUNDLE, "route1_cartesian", "real_data")
    npz = os.path.join(src, "Momentum_kk.npz")
    if not os.path.exists(npz):
        return _skip("没有 %s" % npz)
    z = np.load(npz)
    mom = {k: np.asarray(z[k], dtype=float) for k in ("Kp", "Km", "Pip", "Pim")}
    n = mom["Kp"].shape[0]
    psi = np.broadcast_to(np.array([0.0, 0.0, 0.0, PSI_MASS_REAL]), (n, 4))
    for names, fname in ((cov.F0_NAMES, "phif0_kk.npy"),
                         (cov.F2_NAMES, "phif2_kk.npy")):
        ref_p = os.path.join(src, fname)
        if not os.path.exists(ref_p):
            _skip("没有 %s" % ref_p)
            continue
        ref = np.load(ref_p)
        blocks = []
        for beg in range(0, n, 128):
            end = min(beg + 128, n)
            sub = {k: v[beg:end] for k, v in mom.items()}
            blocks.append(np.stack([cov.compute_event(nm, sub, psi=psi[beg:end])
                                    for nm in names]))
        mine = np.concatenate(blocks, axis=1)[..., :ref.shape[-1]]
        rel = _rel(mine, ref)
        assert rel < TOL, "%s rel=%.3e >= %.0e" % (fname, rel, TOL)
        print("    %-14s rel=%.3e  (ref max %.6g)" % (fname, rel, np.abs(ref).max()))


def test_omega_reproduces_production_cache():
    src = os.path.join(BUNDLE, "route4_omega", "real_data")
    npz = os.path.join(src, "Momentum_kk.npz")
    if not os.path.exists(npz):
        return _skip("没有 %s" % npz)
    mom = RT.momenta_from_npz(npz, ["Kp", "Km", "Pip", "Pim"], order="x,y,z,E")
    Om = RT.omega_from_momenta(mom, _CHAIN)
    K = RT.node_factors(mom, _CHAIN, _LS)
    M = RT.cg_matrix(_LS, 1, 1, 2, RT.LAM_SLOTS[1], RT.LAM_SLOTS[2])
    for fname, mine in (("Omega.npy", Om), ("K.npy", K), ("M.npy", M)):
        p = os.path.join(src, fname)
        if not os.path.exists(p):
            _skip("没有 %s" % p)
            continue
        ref = np.load(p)
        rel = _rel(mine, ref)
        assert rel < TOL, "%s rel=%.3e >= %.0e" % (fname, rel, TOL)
        print("    %-14s rel=%.3e" % (fname, rel))


def test_generated_script_reproduces_production_cache():
    """端到端：emit 出来的脚本（不是内核直调）也要复现生产 cache。

    这条抓的是**产物侧**的接线错误 —— 曾经在这里照抄了 omega 的
    `(E,x,y,z)` 置换，而协变内核的原生约定是 `(x,y,z,E)`，结果差 0.98。
    内核本身是对的，错的是生成物的输入口径。
    """
    import subprocess
    import tempfile

    from agent.tensor import emitter, spec as specmod

    src = os.path.join(BUNDLE, "route1_cartesian", "real_data")
    npz = os.path.join(src, "Momentum_kk.npz")
    if not os.path.exists(npz):
        return _skip("没有 %s" % npz)
    spec_p = os.path.join(_ROOT, "agent", "tensor", "specs", "phif0.toml")
    if not os.path.exists(spec_p):
        return _skip("没有 %s" % spec_p)
    sp = specmod.load_spec(spec_p)
    sp["_spec_path"] = spec_p
    sp["raw"]["chain"]["psi_mass"] = PSI_MASS_REAL      # 真实样本口径
    sp["bases"] = [{"name": "cartesian", "kernel": "covariant", "files": [],
                    "env": {}, "suffix_map": {}}]
    with tempfile.TemporaryDirectory() as tmp:
        r = emitter.emit(sp, out_dir=tmp)
        out = os.path.join(tmp, "data")
        rc = subprocess.call([sys.executable, r["script"], "--momenta", npz,
                              "--out", out, "--n", "400", "--batch", "128"])
        assert rc == 0, "生成脚本退出码 %d" % rc
        mine = np.load(os.path.join(out, "phif0_kk.npy"))
    ref = np.load(os.path.join(src, "phif0_kk.npy"))[:, :mine.shape[1], :]
    rel = _rel(mine, ref)
    assert rel < TOL, "端到端 rel=%.3e >= %.0e" % (rel, TOL)
    print("    端到端 phif0_kk rel=%.3e" % rel)


def test_wrong_psi_mass_is_detectable():
    """用错 psi 质量（3.686 vs 3.685782956）必须能被这个测试抓到。"""
    src = os.path.join(BUNDLE, "route1_cartesian", "real_data")
    npz = os.path.join(src, "Momentum_kk.npz")
    ref_p = os.path.join(src, "phif0_kk.npy")
    if not (os.path.exists(npz) and os.path.exists(ref_p)):
        return _skip("没有参照 cache")
    z = np.load(npz)
    mom = {k: np.asarray(z[k], dtype=float) for k in ("Kp", "Km", "Pip", "Pim")}
    n = mom["Kp"].shape[0]
    ref = np.load(ref_p)
    psi = np.broadcast_to(np.array([0.0, 0.0, 0.0, 3.686]), (n, 4))
    blocks = []
    for beg in range(0, n, 128):
        end = min(beg + 128, n)
        sub = {k: v[beg:end] for k, v in mom.items()}
        blocks.append(np.stack([cov.compute_event(nm, sub, psi=psi[beg:end])
                                for nm in cov.F0_NAMES]))
    mine = np.concatenate(blocks, axis=1)[..., :3]
    rel = _rel(mine, ref)
    assert rel > 1e-6, "用错 psi 质量竟然也吻合（rel=%.3e）—— 测试没有区分力" % rel
    print("    用错质量的 rel=%.3e（> 1e-6，符合预期）" % rel)


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
