#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`agent.tensor.emitter`：同一 TOML 两次 emit 字节一致 + 产物自足可跑。

    python agent/tensor/tests/test_emitter.py
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)

from agent.tensor import emitter, spec              # noqa: E402

_TOML = '''
[chain]
name = "phif2"
top = "psi2s"
l_max = 4
suffix = ""
# 这个测试只关心 emit 的确定性与自足性，用一条道跑得快；
# 显式关掉「道表 vs 自动枚举」的交叉检查（见 test_spec.py 里对应的那条）。
check_channels = false

[input]
keys = ["Kp", "Km", "Pip", "Pim"]
layout = "x,y,z,E"

[particles.psi2s]
J = 1
mass = 3.686
parity = -1
daughters = ["phi", "f2"]

[particles.phi]
J = 1
mass = 1.019
parity = -1
daughters = ["Kp", "Km"]
l = 1

[particles.f2]
J = 2
mass = 1.275
parity = 1
daughters = ["Pip", "Pim"]
l = 2

[particles.Kp]
J = 0
mass = 0.493677
[particles.Km]
J = 0
mass = 0.493677
[particles.Pip]
J = 0
mass = 0.13957039
[particles.Pim]
J = 0
mass = 0.13957039

[[channel]]
name = "phif223"
l = 2
s = 3
'''


def _spec(tmp):
    p = os.path.join(tmp, "s.toml")
    with open(p, "w") as fh:
        fh.write(_TOML)
    s = spec.load_spec(p)
    s["_spec_path"] = p
    s["bases"] = [
        {"name": "cartesian", "kernel": "covariant", "files": [], "env": {},
         "suffix_map": {}},
        {"name": "helicity", "kernel": "helicity", "files": [], "env": {},
         "suffix_map": {}},
    ]
    return s


def _hashes(d):
    out = {}
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if os.path.isfile(p):
            out[fn] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


def test_emit_is_deterministic():
    """同一 TOML 两次 emit 必须字节一致（可复现性是框架的硬要求）。"""
    with tempfile.TemporaryDirectory() as tmp:
        s = _spec(tmp)
        a = os.path.join(tmp, "a")
        b = os.path.join(tmp, "b")
        emitter.emit(s, out_dir=a)
        emitter.emit(s, out_dir=b)
        ha, hb = _hashes(a), _hashes(b)
        assert set(ha) == set(hb), (sorted(ha), sorted(hb))
        for k in ha:
            assert ha[k] == hb[k], "产物 %s 两次不一致" % k


def test_one_toml_two_bases():
    """一份 TOML 同时产出协变张量与螺度两种 tensor 的生成代码。"""
    with tempfile.TemporaryDirectory() as tmp:
        s = _spec(tmp)
        r = emitter.emit(s, out_dir=os.path.join(tmp, "g"))
        kernels = sorted(e["kernel"] for e in r["scripts"])
        assert kernels == ["covariant", "helicity"], kernels
        for e in r["scripts"]:
            assert os.path.exists(e["script"])
            body = open(e["script"]).read()
            assert "def phif223(" in body, e["script"]
            assert "from covariant import" in body, e["script"]


def test_generated_script_runs_and_is_selfcontained():
    """产物自足：不 import agent.tensor，直接跑得动，且写出约定文件名。"""
    with tempfile.TemporaryDirectory() as tmp:
        s = _spec(tmp)
        gdir = os.path.join(tmp, "g")
        r = emitter.emit(s, out_dir=gdir)
        for e in r["scripts"]:
            body = open(e["script"]).read()
            # 产物必须是自足的：只 import 同目录的 covariant 副本与 numpy。
            # 只看真正的 import 语句（文档里提到框架名是允许的）。
            imports = [ln.strip() for ln in body.split("\n")
                       if ln.strip().startswith(("import ", "from "))]
            for ln in imports:
                assert "agent" not in ln, "产物 import 了框架本身：%s" % ln

        mom = _momenta(os.path.join(tmp, "mom.npz"))
        out = os.path.join(tmp, "out")
        for e in r["scripts"]:
            rc = subprocess.call([sys.executable, e["script"],
                                  "--momenta", mom, "--out", out, "--batch", "8"])
            assert rc == 0, e["script"]
        assert os.path.exists(os.path.join(out, "phif2_kk.npy"))
        assert os.path.exists(os.path.join(out, "phif2_hel.npy"))
        kk = np.load(os.path.join(out, "phif2_kk.npy"))
        hl = np.load(os.path.join(out, "phif2_hel.npy"))
        assert kk.shape == (1, 16, 3), kk.shape
        assert hl.shape == (1, 16, 3) and hl.dtype == np.complex128
        assert np.isfinite(kk).all() and np.isfinite(hl).all()
        # 同一份 T 的两个表象：W = EPS_A . T
        E = np.array([[1 / np.sqrt(2), -1j / np.sqrt(2), 0],
                      [0, 0, 1],
                      [-1 / np.sqrt(2), -1j / np.sqrt(2), 0]], dtype=np.complex128)
        W = np.einsum("lm,enm->enl", E, kk)
        assert np.abs(W - hl).max() < 1e-12, np.abs(W - hl).max()


def _jax_python():
    """找带 jax 的解释器（分析 venv 优先）；没有就返回 None。"""
    cands = [os.path.join(_ROOT, "analyses", "kk_dis", ".venv", "bin", "python"),
             sys.executable]
    for p in cands:
        if not os.path.exists(p):
            continue
        rc = subprocess.call([p, "-c", "import jax"],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        if rc == 0:
            return p
    return None


def test_generated_scripts_run_under_jax():
    """产物在 **jax 后端**下跑得动，且与 numpy 后端逐位一致。

    这条是"统一到 jax"的端到端判据：内核在 jit 里必须没有 tracer 违规
    （逐项赋值 / numpy 调用 / bool 转换），否则这里会直接非零退出。
    """
    jax_py = _jax_python()
    if jax_py is None:
        print("    跳过：环境里没有 jax")
        return
    with tempfile.TemporaryDirectory() as tmp:
        s = _spec(tmp)
        gdir = os.path.join(tmp, "g")
        r = emitter.emit(s, out_dir=gdir)
        mom = _momenta(os.path.join(tmp, "mom.npz"))
        outs = {}
        for backend in ("numpy", "jax"):
            out = os.path.join(tmp, "out_" + backend)
            env = dict(os.environ, TENSOR_BACKEND=backend,
                       JAX_PLATFORMS="cpu",
                       XLA_PYTHON_CLIENT_PREALLOCATE="false")
            for e in r["scripts"]:
                rc = subprocess.call([jax_py, e["script"], "--momenta", mom,
                                      "--out", out, "--batch", "8"], env=env)
                assert rc == 0, "%s（backend=%s）" % (e["script"], backend)
            outs[backend] = out

        # 逐文件比对两个后端：covariant/helicity 的产物 + Omega/K/M/g/norm
        names = sorted(os.listdir(outs["numpy"]))
        assert names, "没有产物"
        for fn in names:
            a = np.load(os.path.join(outs["numpy"], fn))
            b = np.load(os.path.join(outs["jax"], fn))
            assert a.shape == b.shape, (fn, a.shape, b.shape)
            rel = np.abs(a - b).max() / max(float(np.abs(a).max()), 1e-300)
            assert rel < 1e-12, "%s 两后端不一致 rel=%.3e" % (fn, rel)


def _momenta(path, n=16, seed=5):
    """写一份真实的 psi(2S) -> phi f2 -> (K+K-)(pi+pi-) 四动量 npz。"""
    rng = np.random.default_rng(seed)

    def two_body(m, ma, mb):
        p = float(np.sqrt(((m * m - (ma + mb) ** 2) * (m * m - (ma - mb) ** 2))
                          / (4 * m * m)))
        e1, e2 = float(np.sqrt(ma * ma + p * p)), float(np.sqrt(mb * mb + p * p))
        ct = rng.uniform(-1, 1, n)
        ph = rng.uniform(0, 2 * np.pi, n)
        st = np.sqrt(1 - ct ** 2)
        d = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)
        return (np.concatenate([p * d, e1 * np.ones((n, 1))], axis=1),
                np.concatenate([-p * d, e2 * np.ones((n, 1))], axis=1))

    def boost(p4, beta):
        b = np.asarray(beta, dtype=float)
        if b.ndim == 1:
            b = np.broadcast_to(b, (n, 3))
        b2 = (b * b).sum(-1)
        g = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-15))
        g2 = (g - 1.0) / np.where(b2 > 1e-12, b2, 1.0)
        e = p4[:, 3]
        bp = (p4[:, :3] * b).sum(-1)
        return np.concatenate(
            [p4[:, :3] + (g2 * bp + g * e)[:, None] * b, (g * (e + bp))[:, None]],
            axis=1)

    phi, f2 = two_body(3.686, 1.019, 1.275)
    Kp, Km = two_body(1.019, 0.493677, 0.493677)
    Pip, Pim = two_body(1.275, 0.13957039, 0.13957039)
    Kp, Km = boost(Kp, -phi[:, :3] / phi[:, 3:]), boost(Km, -phi[:, :3] / phi[:, 3:])
    Pip = boost(Pip, -f2[:, :3] / f2[:, 3:])
    Pim = boost(Pim, -f2[:, :3] / f2[:, 3:])
    np.savez(path, Kp=Kp, Km=Km, Pip=Pip, Pim=Pim)
    return path


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
