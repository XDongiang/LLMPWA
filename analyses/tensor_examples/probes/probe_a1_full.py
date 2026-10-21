#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A1 全面复验：**全部 14 份 spec / 48 条道**，协变 vs 级联（经 EPS_A）。

判据 A1 的定义（`analyses/tensor_examples/README.md`「如何验证」）：

    W_i^{casc}(e,lam) = c_i · W_i^{cov}(e,lam)       逐事例成立，c_i 与事例无关

`c_i` 是逐道的整体复常数（耦合 + 归一化 + 相位，由拟合吸收），所以**不能**
要求 `c_i = 1`；能要求的是 `c_i` 与事例无关。本探针报三样：

  * `rel`   —— 用**一个全局** c 拟合后的最大相对残差（阈值 1e-12）
  * `c`     —— 那个复常数（**看它的相位**：`-i`/`±1` 是承重约定）
  * `spread`—— 逐事例 c(e) 的离散度；`rel` 小但 `spread` 大 = 事件依赖因子

用法：
    python analyses/tensor_examples/probes/probe_a1_full.py
    python analyses/tensor_examples/probes/probe_a1_full.py --json out.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "agent", "tensor", "tests"))
os.chdir(_ROOT)

import test_ex3_channels as T                                       # noqa: E402
import test_ex5_channels as T5                                      # noqa: E402
from agent.tensor import spec as S                                  # noqa: E402
from agent.tensor.kernels import covariant as COV                   # noqa: E402
from agent.tensor.kernels import helicity_cascade as HC             # noqa: E402
from agent.tensor.kernels.helicity import EPS_A                     # noqa: E402

SPECS = sorted(glob.glob("analyses/tensor_examples/specs/*.toml")) + \
    sorted(glob.glob("agent/tensor/specs/*.toml"))


# --------------------------------------------------------------- 运动学
#
# 每个 KIN 返回 `(momd, keys)`：`momd` 是 {末态名: (N,4)(x,y,z,E)}，
# `keys` 是该 spec 的 `[input].keys` 顺序。**不假设是 4 体** ——
# 例 1/2 是 3 体（Pip1/Pim2/Pi0），签名与 4 体不同。
def mom_ex3(n=32, seed=11):
    d = T.physical(n=n, seed=seed)
    return d, ["Kp", "Km", "Pip", "Pim"]


def mom_ex5(n=32, seed=7):
    d = T5.physical(n=n, seed=seed)
    return d, ["Kp", "Km", "Pip", "Pim"]


def _load_make_threebody():
    """按路径载入 `data/make_threebody.py`（那个目录不是包）。"""
    import importlib.util
    p = os.path.join(_ROOT, "analyses", "tensor_examples", "data",
                     "make_threebody.py")
    spec = importlib.util.spec_from_file_location("_mk3", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def mom_ex12(case, n=32, seed=5):
    """例 1/2 的三体相空间（与 `data/make_threebody.py` 同构）。"""
    mk = _load_make_threebody()
    d = mk.make(case, n=n, seed=seed)
    return d, list(d)


def mom_omega5(n=32, seed=5):
    """例 4 的五体相空间：`data/make_omega5.py`（虚拟 m_ω = 1.5）。"""
    import importlib.util
    p = os.path.join(_ROOT, "analyses", "tensor_examples", "data",
                     "make_omega5.py")
    spec = importlib.util.spec_from_file_location("_mo5", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    d = m.make(n=n, seed=seed)
    return d, list(d)


def mom_ex6(n=32, seed=5):
    """例 6 的六体相空间：`data/make_phi4pi.py`（虚拟 m_f₂ = 1.9）。"""
    import importlib.util
    p = os.path.join(_ROOT, "analyses", "tensor_examples", "data",
                     "make_phi4pi.py")
    spec = importlib.util.spec_from_file_location("_mp4", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    d = m.make(n=n, seed=seed)
    return d, list(d)


def mom_rhof2(n=32, seed=7):
    """ψ→ρf₂→(ππ)(ππ)：与 `probe_missing_covariant.py` 的 B 段同一构造。"""
    pair, boost = T._pair, T._boost
    MPI, MPSI = 0.13957039, 3.686
    rng = np.random.default_rng(seed)
    rho, f2 = pair(MPSI, 0.775, 1.275, rng, n)
    a0, b0 = pair(0.775, MPI, MPI, rng, n)
    c0, d0 = pair(1.275, MPI, MPI, rng, n)
    br = rho[:, :3] / rho[:, 3:4]
    bf = f2[:, :3] / f2[:, 3:4]
    d = {"Pip": boost(a0, br), "Pim": boost(b0, br),
         "Pip2": boost(c0, bf), "Pim2": boost(d0, bf)}
    return d, list(d)


KIN = {
    "ex3": mom_ex3,
    "ex5": mom_ex5,
    "ex1": lambda: mom_ex12("ex1"),
    "ex2": lambda: mom_ex12("ex2"),
    "rhof2": mom_rhof2,
    "ex4": mom_omega5,
    "ex6": mom_ex6,
}


# ------------------------------------------------- 通用协变道调用（不假设 4 体）
def _node_mom(momd, name, tree, particles):
    """把一个参数名解析成 (N,4)：末态直取，中间节点 = 叶子之和（递归）。"""
    if name in momd:
        return momd[name]
    ds = (tree or {}).get(name)
    if ds is None:
        p = particles.get(name) or {}
        ds = p.get("daughters")
    if not ds:
        raise KeyError(name)
    total = None
    for d in ds:
        part = _node_mom(momd, d, tree, particles)
        total = part if total is None else total + part
    return total


def cov_event(sp, cc, momd, psi):
    """通用协变道：按 `inspect.signature` 绑参数名（与 emitter 同一套规则）。

    4 体 legacy 签名 `(phi, f2, Kp, Km, Pip, Pim)` 里的 `phi`/`f2` **不是**
    `[particles]` 里的名字（`phif0` 链里 `f2` 其实指 `f0`），所以名字解析
    失败时整体回退到 `emitter._LEGACY_SIG` 的固定绑定。
    """
    import inspect
    from agent.tensor.kernels.covariant import channel_fn
    fn = channel_fn(cc["name"])
    names = [p for p in inspect.signature(fn).parameters if p != "psi"]
    parts = {k: dict(v) for k, v in sp["particles"].items()}
    try:
        args = [psi] + [_node_mom(momd, p, cc["tree"], parts) for p in names]
    except KeyError:
        if tuple(names) != ("phi", "f2", "Kp", "Km", "Pip", "Pim"):
            raise
        args = [psi, momd["Kp"] + momd["Km"], momd["Pip"] + momd["Pim"],
                momd["Kp"], momd["Km"], momd["Pip"], momd["Pim"]]
    return fn(*args)


def pick(path):
    b = os.path.basename(path)
    if "rhof2" in b:
        return "rhof2"
    if b.startswith("ex1"):
        return "ex1"
    if b.startswith("ex2"):
        return "ex2"
    if b.startswith("ex4"):
        return "ex4"
    if b.startswith("ex6"):
        return "ex6"
    return "ex3"


# --------------------------------------------------------------- 判据
def fit_c(w, ref):
    """逐事例 c(e) 与全局 c。"""
    a = w.reshape(w.shape[0], -1)
    b = ref.reshape(ref.shape[0], -1)
    num = np.einsum("ei,ei->e", np.conj(a), b)
    den = np.einsum("ei,ei->e", np.conj(a), a)
    ce = num / den
    c = np.vdot(w.reshape(-1), ref.reshape(-1)) / np.vdot(w.reshape(-1),
                                                           w.reshape(-1))
    return complex(c), ce


def rel(a, b):
    c, _ = fit_c(a, b)
    return float(np.abs(b - c * a).max() / max(np.abs(b).max(), 1e-300))


def one(spec_path, momd, keys):
    sp = S.load_spec(spec_path)
    base = S.to_chain(sp)
    # 级联内核要 (E,x,y,z)
    mom = np.stack([momd[k] for k in keys], axis=1)[:, :, [3, 0, 1, 2]]
    psi = np.broadcast_to(np.array([0.0, 0.0, 0.0, 3.686]),
                          (momd[keys[0]].shape[0], 4))
    out = []
    for cc in S.cascade_channels(sp):
        ch = HC.with_tree(base, cc["tree"] or {})
        A = HC.chain_amplitude(mom, ch, cc["ls"], ang=HC.node_angles(mom, ch))
        T = cov_event(sp, cc, momd, psi)
        w = np.einsum("lm,em->el", EPS_A, np.asarray(T)[..., :3])
        c, ce = fit_c(w, A)
        r = float(np.abs(A - c * w).max() / max(np.abs(A).max(), 1e-300))
        c0 = np.mean(ce)
        spread = float(np.abs(ce / c0 - 1).max()) if abs(c0) > 0 else np.inf
        out.append({"name": cc["name"], "rel": r, "c": c,
                    "abs_c": abs(c), "phase": float(np.angle(c)),
                    "spread": spread, "max": float(np.abs(A).max())})
    return sp, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()

    cache = {}
    ok = True
    rows = []
    for p in SPECS:
        kind = pick(p)
        if kind not in cache:
            cache[kind] = KIN[kind]()
        momd, keys = cache[kind]
        try:
            sp, res = one(p, momd, keys)
        except Exception as e:                                  # noqa: BLE001
            print("!! %-26s %s: %s" % (os.path.basename(p),
                                       type(e).__name__, e))
            ok = False
            continue
        worst = max(r["rel"] for r in res)
        print("=" * 96)
        print("%-28s depth=%d  道=%d  最大 rel=%.3e  %s"
              % (os.path.basename(p), sp["depth"], len(res), worst,
                 "OK" if worst < 1e-12 else "**FAIL**"))
        for r in res:
            good = r["rel"] < 1e-12
            ok &= good
            print("   %-14s rel=%.3e  c=%+.6f%+.6fj  |c|=%.6f  phase=%+.1f°"
                  "  spread=%.1e  %s"
                  % (r["name"], r["rel"], r["c"].real, r["c"].imag,
                     r["abs_c"], np.degrees(r["phase"]), r["spread"],
                     "OK" if good else "**FAIL**"))
            rows.append(dict(spec=os.path.basename(p), **{
                k: ([v.real, v.imag] if isinstance(v, complex) else v)
                for k, v in r.items()}))
    print()
    print("=> %s（%d 条道）" % ("全部通过" if ok else "**有失败**", len(rows)))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=1)
        print("   写到 %s" % args.json)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
