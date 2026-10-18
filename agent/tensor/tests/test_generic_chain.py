#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端 smoke：一个**与 KK 无关**的链也要能跑通（审计 §2.4）。

`specs/rhof2_generic.toml` 是 `psi(2S) -> rho + f2 -> (pi+pi-)(pi+pi-)`：
顶层 daughter 自旋是 1 与 2，道表**完全自动枚举**，末态键名也不是 Kp/Km。
它证明生成器没有写死 KK 道。

这里用**合成四动量**（不需要数据文件）跑一遍 Omega 路径并检查形状/有限性。

    python agent/tensor/tests/test_generic_chain.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)

from agent.tensor import spec as specmod        # noqa: E402
from agent.tensor.kernels import omega as RT    # noqa: E402

SPEC = os.path.join(_ROOT, "agent", "tensor", "specs", "rhof2_generic.toml")


def _two_body(m, ma, mb, n, rng):
    """m -> a + b 各向同性，返回 (E,x,y,z) 约定（omega 内部约定）。"""
    p = float(np.sqrt(((m * m - (ma + mb) ** 2) * (m * m - (ma - mb) ** 2))
                      / (4 * m * m)))
    e1, e2 = float(np.sqrt(ma * ma + p * p)), float(np.sqrt(mb * mb + p * p))
    ct = rng.uniform(-1, 1, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - ct ** 2)
    d = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)
    return (np.concatenate([e1 * np.ones((n, 1)), p * d], axis=1),
            np.concatenate([e2 * np.ones((n, 1)), -p * d], axis=1))


def _boost(p4, beta):
    """p4 是 (E,x,y,z)；按 3-vector beta boost。"""
    b = np.asarray(beta, dtype=float)
    if b.ndim == 1:
        b = np.broadcast_to(b, (p4.shape[0], 3))
    b2 = (b * b).sum(-1)
    g = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-15))
    g2 = (g - 1.0) / np.where(b2 > 1e-12, b2, 1.0)
    e = p4[:, 0]
    bp = (p4[:, 1:] * b).sum(-1)
    v = p4[:, 1:] + (g2 * bp + g * e)[:, None] * b
    return np.concatenate([(g * (e + bp))[:, None], v], axis=1)


def synthetic_momenta(n=48, seed=11):
    """真实的 psi(2S) -> rho f2 -> (pipi)(pipi) 四动量，布局 (E,x,y,z)。"""
    rng = np.random.default_rng(seed)
    rho, f2 = _two_body(3.686, 0.775, 1.275, n, rng)
    p1, p2 = _two_body(0.775, 0.13957039, 0.13957039, n, rng)
    p3, p4 = _two_body(1.275, 0.13957039, 0.13957039, n, rng)
    p1 = _boost(p1, -rho[:, 1:] / rho[:, 0:1])
    p2 = _boost(p2, -rho[:, 1:] / rho[:, 0:1])
    p3 = _boost(p3, -f2[:, 1:] / f2[:, 0:1])
    p4 = _boost(p4, -f2[:, 1:] / f2[:, 0:1])
    return np.stack([p1, p2, p3, p4], axis=1)


def test_generic_chain_loads_and_enumerates():
    s = specmod.load_spec(SPEC)
    assert s["top"] == "psi2s"
    assert [n for n in s["input_keys"]] == ["Pip", "Pim", "Pip2", "Pim2"]
    ls = specmod.allowed_ls(s)
    assert ls == [(0, 1), (2, 1), (2, 2), (2, 3), (4, 3)], ls
    assert not s["channels"], "这份 spec 应当完全靠自动枚举"


def test_generic_chain_omega_end_to_end():
    s = specmod.load_spec(SPEC)
    chain = specmod.to_chain(s)
    mom = synthetic_momenta()
    ls = specmod.allowed_ls(s)
    Om = RT.omega_from_momenta(mom, chain)
    K = RT.node_factors(mom, chain, ls)
    M = RT.cg_matrix(ls, 1, 1, 2, RT.LAM_SLOTS[1], RT.LAM_SLOTS[2])
    # psi2s (J=1) -> rho (J=1) + f2 (J=2)
    assert Om.shape == (mom.shape[0], 3, 3, 5), Om.shape
    assert K.shape == (len(ls), mom.shape[0]), K.shape
    assert M.shape == (len(ls), 3, 5), M.shape
    assert np.isfinite(Om).all(), "Omega 有非有限值"
    assert np.isfinite(K).all(), "K 有非有限值"
    assert np.isfinite(M).all(), "M 有非有限值"
    # 装配出的 W 也要有限
    W = RT.assemble(M, Om, np.ones(len(ls)), K)
    assert W.shape == (len(ls), mom.shape[0], 3), W.shape
    assert np.isfinite(W).all()


def test_generic_chain_emit_omega_script(tmp_path=None):
    """Omega 路径能 emit 出自足脚本（协变路径是 KK 专用，这里不涉及）。"""
    import subprocess
    import tempfile

    from agent.tensor import emitter

    s = specmod.load_spec(SPEC)
    s["_spec_path"] = SPEC
    s["bases"] = [b for b in s["bases"] if b["name"] == "omega"]
    assert s["bases"], "spec 里应当有 omega 表象"
    with tempfile.TemporaryDirectory() as tmp:
        r = emitter.emit(s, out_dir=tmp)
        assert r["scripts"], r
        script = r["scripts"][0]["script"]
        assert os.path.exists(script)
        # 合成一份 npz 给脚本跑（键名是 Pip/Pim/Pip2/Pim2，布局 x,y,z,E）
        mom = synthetic_momenta(n=16)
        npz = os.path.join(tmp, "mom.npz")
        np.savez(npz,
                 Pip=np.stack([mom[:, 0, 1], mom[:, 0, 2], mom[:, 0, 3],
                               mom[:, 0, 0]], axis=1),
                 Pim=np.stack([mom[:, 1, 1], mom[:, 1, 2], mom[:, 1, 3],
                               mom[:, 1, 0]], axis=1),
                 Pip2=np.stack([mom[:, 2, 1], mom[:, 2, 2], mom[:, 2, 3],
                                mom[:, 2, 0]], axis=1),
                 Pim2=np.stack([mom[:, 3, 1], mom[:, 3, 2], mom[:, 3, 3],
                                mom[:, 3, 0]], axis=1))
        out = os.path.join(tmp, "data")
        rc = subprocess.call([sys.executable, script, "--momenta", npz,
                              "--out", out, "--n", "16"])
        assert rc == 0, "生成脚本退出码 %d" % rc
        for fn, shape in (("Omega.npy", (16, 3, 3, 5)), ("K.npy", (5, 16)),
                          ("M.npy", (5, 3, 5)), ("g.npy", (5,)),
                          ("norm.npy", (5,))):
            p = os.path.join(out, fn)
            assert os.path.exists(p), fn
            assert np.load(p).shape == shape, (fn, np.load(p).shape)


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
