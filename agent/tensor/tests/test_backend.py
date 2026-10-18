#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""后端一致性：**同一份内核在 jax 与 numpy 下必须给出同一个数**。

这是"统一到 jax"之后最重要的一条回归 —— 移植 jax 时踩的坑（数组不可变、
tracer 上不能用 numpy、einsum 路径格式不同、常量表逐项赋值）都只会表现为
**数值不一致或直接报错**，不会静默通过。

判据：逐道、逐对象，`max|jax - numpy| / max|numpy| < 1e-12`。

没有 jax 时**跳过**（打印原因），不算失败：生成器环境（`emit`/`check`）本来
就不需要 jax。

    python agent/tensor/tests/test_backend.py
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as onp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)

TOL = 1e-12
_CHANNELS = ["phif001", "phif021", "phif201", "phif221", "phif222", "phif223",
             "phif243", "u_rho_1", "u_rho_2", "u_SS_1", "u_SS_2", "u_SD_1",
             "u_SD_2", "u_DS_1", "u_DS_2", "u_DD_1", "u_DD_2"]
_LS = [(0, 1), (2, 1), (2, 2), (2, 3), (4, 3)]

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

_N = 6
_NPZ = os.path.join(_ROOT, "analyses", "kk_dis", "data", "mc_truth",
                    "Momentum_kk.npz")


def _have_jax(python=None):
    """在**子进程**里探测 jax：本进程可能已被 numpy 后端锁定。"""
    py = python or sys.executable
    rc = subprocess.call([py, "-c", "import jax"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return rc == 0


def _momenta():
    d = onp.load(_NPZ)
    return {k: onp.asarray(d[k])[: _N] for k in ("Kp", "Km", "Pip", "Pim")}


# --------------------------------------------------------------- 子进程驱动
# 后端必须在 import 内核**之前**选好（`xp` 一旦解析就固定了），所以每个后端
# 各起一个子进程算一遍、把结果写成 npz，主进程再比对。
_DRIVER = r'''
import os, sys
import numpy as onp
sys.path.insert(0, %(root)r)
from agent.tensor.kernels import covariant as C, omega as RT

mom = {k: onp.asarray(onp.load(%(npz)r)[k])[:%(n)d] for k in ("Kp","Km","Pip","Pim")}
out = {}
for nm in %(channels)r:
    out[nm] = onp.asarray(C.compute_event(nm, mom))
chain = %(chain)r
m4 = RT.momenta_from_npz(%(npz)r, ["Kp","Km","Pip","Pim"], 0, %(n)d)
out["__Omega"] = onp.asarray(RT.omega_from_momenta(m4, chain))
out["__K"] = onp.asarray(RT.node_factors(m4, chain, %(ls)r))
onp.savez(%(dest)r, **out)
'''


def _run_backend(backend, dest, python=None):
    py = python or sys.executable
    env = dict(os.environ)
    env["TENSOR_BACKEND"] = backend
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    src = _DRIVER % {"root": _ROOT, "npz": _NPZ, "n": _N,
                     "channels": _CHANNELS, "chain": _CHAIN, "ls": _LS,
                     "dest": dest}
    rc = subprocess.call([py, "-c", src], env=env)
    if rc != 0:
        raise RuntimeError("后端 %s 子进程失败（rc=%d）" % (backend, rc))


def _rel(a, b):
    return float(onp.abs(a - b).max() / max(float(onp.abs(b).max()), 1e-300))


def test_jax_matches_numpy():
    """jax 与 numpy 两条后端逐道、逐对象一致。"""
    import tempfile
    if not os.path.exists(_NPZ):
        print("    跳过：找不到 %s" % _NPZ)
        return
    # 优先用带 jax 的解释器（分析 venv）；没有就跳过。
    cand = [os.path.join(_ROOT, "analyses", "kk_dis", ".venv", "bin", "python"),
            sys.executable]
    jax_py = next((p for p in cand if os.path.exists(p) and _have_jax(p)), None)
    if jax_py is None:
        print("    跳过：环境里没有 jax")
        return

    with tempfile.TemporaryDirectory() as tmp:
        a = os.path.join(tmp, "np.npz")
        b = os.path.join(tmp, "jx.npz")
        _run_backend("numpy", a, python=jax_py)
        _run_backend("jax", b, python=jax_py)
        A, B = onp.load(a), onp.load(b)
        assert set(A.files) == set(B.files), (sorted(A.files), sorted(B.files))
        worst, worst_key = 0.0, None
        for k in A.files:
            r = _rel(B[k], A[k])
            if r > worst:
                worst, worst_key = r, k
        assert worst < TOL, "最差 %s rel=%.3e（阈值 %.0e）" % (worst_key, worst, TOL)
        print("    17 道 + Omega/K：最差 %s rel=%.3e" % (worst_key, worst))


def test_backend_default_is_auto():
    """默认后端是 auto；显式 numpy 一定解析成 numpy。"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.join(tmp, "n.npz")
        _run_backend("numpy", dest)
        assert os.path.exists(dest)
    # auto 在无 jax 的解释器上必须落到 numpy，且不报错
    src = ("import sys; sys.path.insert(0, %r);"
           "from agent.tensor.kernels import _backend as B;"
           "print(B.configure(backend='auto'))" % _ROOT)
    out = subprocess.check_output([sys.executable, "-c", src]).decode().strip()
    assert out in ("numpy", "jax"), out


def test_x64_enabled_on_jax():
    """jax 后端必须打开 float64：float32 会把相对误差抬到 1e-7，够不着 1e-12。"""
    cand = [os.path.join(_ROOT, "analyses", "kk_dis", ".venv", "bin", "python"),
            sys.executable]
    jax_py = next((p for p in cand if os.path.exists(p) and _have_jax(p)), None)
    if jax_py is None:
        print("    跳过：环境里没有 jax")
        return
    src = ("import sys; sys.path.insert(0, %r);"
           "import os; os.environ['JAX_PLATFORMS']='cpu';"
           "from agent.tensor.kernels import _backend as B;"
           "import jax; assert B.configure(backend='jax')=='jax';"
           "assert jax.config.jax_enable_x64;"
           "import jax.numpy as jnp;"
           "assert jnp.zeros(2).dtype.itemsize==8, jnp.zeros(2).dtype;"
           "print('ok')" % _ROOT)
    out = subprocess.check_output([jax_py, "-c", src]).decode().strip()
    assert out == "ok", out


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
