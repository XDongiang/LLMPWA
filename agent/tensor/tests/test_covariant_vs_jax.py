#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 numpy 版协变张量内核与 **jax 原版**（Tensor/{BASE,FUNC}.py）逐式对拍。

这是 A1 方案（移植而非重新推导）的**唯一可信判据**：移植只允许改变
"用哪个数组库算"，不允许改变任何一个数字。

原版是**单事例**函数（`FUNCTION.Modifiers()` 用 vmap 批量），本内核是
`(N,4)` 批量接口，所以对拍时用 `N=1`。

    # 需要 Tensor/ 源码在本地（它不在 git 里）
    TENSOR_ROOT=<...>/1_运行代码/tensor_example python -m pytest agent/tensor/tests/test_covariant_vs_jax.py
    # 或直接跑
    TENSOR_ROOT=... python agent/tensor/tests/test_covariant_vs_jax.py

没有 jax 或找不到 Tensor/ 时**跳过**（打印原因），不算失败 —— 生成器环境
只依赖 openai/toml，不该被 jax 绑住。
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_KERNELS = os.path.abspath(os.path.join(_HERE, "..", "kernels"))
if _KERNELS not in sys.path:
    sys.path.insert(0, _KERNELS)

import covariant as COV  # noqa: E402

TOL = 1e-12


def _load_jax_original(tensor_root):
    """导入原版 BASE / FUNC。返回 (JBASE, JFUNC, jax) 或 None。"""
    if not tensor_root or not os.path.isdir(tensor_root):
        print("跳过：未给出 TENSOR_ROOT（Tensor/ 源码目录）")
        return None
    tensor_pkg = os.path.join(tensor_root, "Tensor")
    if not os.path.isdir(tensor_pkg):
        print("跳过：%s 下没有 Tensor/ 包" % tensor_root)
        return None
    try:
        import jax
    except ModuleNotFoundError:
        print("跳过：环境里没有 jax")
        return None

    jax.config.update("jax_enable_x64", True)
    # 原代码仍用 `from jax.config import config`（旧 API）
    if "jax.config" not in sys.modules:
        m = types.ModuleType("jax.config")
        m.config = jax.config
        m.update = jax.config.update
        sys.modules["jax.config"] = m

    for p in (tensor_root, tensor_pkg):
        if p not in sys.path:
            sys.path.insert(0, p)
    import BASE as JBASE          # noqa: E402
    from Tensor import FUNC as JFUNC  # noqa: E402
    return JBASE, JFUNC, jax


M_PSI, M_PHI, M_F2 = 3.686, 1.019, 1.275
M_K, M_PI = 0.493677, 0.13957039


def _two_body(m_parent, m1, m2, rng, n):
    """母粒子静止系下的两体衰变：返回 (p1, p2) 各 (N,4)，序 (x,y,z,E)。"""
    lam = ((m_parent ** 2 - (m1 + m2) ** 2)
           * (m_parent ** 2 - (m1 - m2) ** 2))
    p = np.full(n, np.sqrt(max(lam, 0.0)) / (2.0 * m_parent))
    ct = rng.uniform(-1.0, 1.0, n)
    st = np.sqrt(np.maximum(1.0 - ct ** 2, 0.0))
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    d = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)
    E1 = np.sqrt(m1 ** 2 + p ** 2)
    E2 = np.sqrt(m2 ** 2 + p ** 2)
    p1 = np.concatenate([p[:, None] * d, E1[:, None]], axis=1)
    p2 = np.concatenate([-p[:, None] * d, E2[:, None]], axis=1)
    return p1, p2


def _boost(p, beta):
    """把四动量 p 按 3-速度 beta boost（分量序 (x,y,z,E)）。"""
    b2 = (beta ** 2).sum(-1)
    g = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-300))
    bp = (beta * p[:, :3]).sum(-1)
    g2 = np.where(b2 > 1e-12, (g - 1.0) / np.where(b2 > 1e-12, b2, 1.0), 0.0)
    E = g * (p[:, 3] + bp)
    v = p[:, :3] + (g2 * bp + g * p[:, 3])[:, None] * beta
    return np.concatenate([v, E[:, None]], axis=1)


def _momenta(seed=7, n=1):
    """生成**物理**的 psi(2S) -> phi f2 -> (K+K-)(pi+pi-) 事例。

    随机四动量不行：`Qabc` 会出现负的 q^2，`B1/B3` 的 sqrt 变 NaN，
    原版与本内核**都**会 NaN（那不是移植错误）。所以必须用真实衰变学。
    """
    rng = np.random.default_rng(seed)
    # psi 静止系：psi -> phi + f2
    pphi, pf2 = _two_body(M_PSI, M_PHI, M_F2, rng, n)
    # phi -> K+ K-（在 phi 静止系），f2 -> pi+ pi-
    kp0, km0 = _two_body(M_PHI, M_K, M_K, rng, n)
    pip0, pim0 = _two_body(M_F2, M_PI, M_PI, rng, n)
    # boost 到 psi 静止系
    bphi = pphi[:, :3] / pphi[:, 3:4]
    bf2 = pf2[:, :3] / pf2[:, 3:4]
    return {"Kp": _boost(kp0, bphi), "Km": _boost(km0, bphi),
            "Pip": _boost(pip0, bf2), "Pim": _boost(pim0, bf2)}


def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def check_base(jax, JBASE, verbose=True):
    """T1/T2/T3/T4/P2/P3/P4 逐式对拍。

    原版 BASE 的这些函数是**严格单事例**的（`dot` 用 `"i,j,ij->"`），
    所以这里喂 1-D 的 `(4,)`；本内核是 `(N,4)`，取 `[0]` 对齐。
    """
    mom = _momenta()
    psi = COV.PSI2S_DEFAULT
    phi = (mom["Kp"] + mom["Km"])[0]
    f2 = (mom["Pip"] + mom["Pim"])[0]
    J = lambda a: jax.numpy.array(a)  # noqa: E731

    cases = [
        ("T1_cov", JBASE.T1_cov, COV.T1_cov, (psi, phi, f2)),
        ("T2_cov", JBASE.T2_cov, COV.T2_cov, (psi, phi, f2)),
        ("T3_cov", JBASE.T3_cov, COV.T3_cov, (psi, phi, f2)),
        ("T4_cov", JBASE.T4_cov, COV.T4_cov, (psi, phi, f2)),
        ("P2", JBASE.P2, COV.P2, (psi,)),
        ("P3", JBASE.P3, COV.P3, (psi,)),
        ("P4", JBASE.P4, COV.P4, (psi,)),
    ]
    bad = []
    for name, jf, cf, args in cases:
        jv = np.asarray(jf(*[J(np.asarray(a, dtype=float)) for a in args]))
        cv = np.asarray(cf(*[np.asarray(a, dtype=float) for a in args]))
        # 原版是单事例 -> 去掉本内核的前导事例轴
        cv = cv[0]
        rel = _rel(cv, jv)
        ok = rel < TOL
        if verbose:
            print("  %-7s %-24s rel = %.3e  %s"
                  % (name, str(jv.shape), rel, "OK" if ok else "**MISMATCH**"))
        if not ok:
            bad.append(name)
    return bad


def check_channels(jax, JFUNC, verbose=True):
    """七条道逐道对拍（原版经 Modifiers() 已 vmap）。"""
    mom = _momenta()
    psi = COV.PSI2S_DEFAULT
    phi = mom["Kp"] + mom["Km"]
    f2 = mom["Pip"] + mom["Pim"]
    J = lambda a: jax.numpy.array(a)  # noqa: E731

    fn = JFUNC.FUNCTION()
    fn.Modifiers()
    n = phi.shape[0]
    # 原版把 psi 存在 self.psi2s 里（不传参），所以对拍必须用同一个 psi 口径
    psi_b = np.broadcast_to(psi, (n, 4))
    # 原版各道的**形参个数不同**：phif001 只吃 (phi,Kp,Km)，
    # phif021 吃 (phi,f0,Kp,Km)，其余吃 (phi,f2,Kp,Km,Pip,Pim)。
    args_of = {
        "phif001": lambda: (phi, mom["Kp"], mom["Km"]),
        "phif021": lambda: (phi, f2, mom["Kp"], mom["Km"]),
    }
    default_args = lambda: (phi, f2, mom["Kp"], mom["Km"], mom["Pip"], mom["Pim"])  # noqa: E731
    bad = []
    for name in COV.F0_NAMES + COV.F2_NAMES:
        jv = np.asarray(getattr(fn, name)(
            *[J(a) for a in args_of.get(name, default_args)()]))[0]
        cv = np.asarray(COV.compute_event(name, mom, psi=psi_b))[0]
        rel = _rel(cv, jv)
        ok = rel < TOL
        if verbose:
            print("  %-8s %-24s rel = %.3e  %s"
                  % (name, str(jv.shape), rel, "OK" if ok else "**MISMATCH**"))
        if not ok:
            bad.append(name)
    return bad


def main():
    tensor_root = os.environ.get("TENSOR_ROOT", "")
    loaded = _load_jax_original(tensor_root)
    if loaded is None:
        return 0
    JBASE, JFUNC, jax = loaded

    print("=== BASE 张量原语：numpy 移植 vs jax 原版 ===")
    bad = check_base(jax, JBASE)
    print("=== 七条道：numpy 移植 vs jax 原版 ===")
    bad += check_channels(jax, JFUNC)

    if bad:
        print("\nFAIL：以下对象与原版不一致 -> %s" % ", ".join(bad))
        return 1
    print("\nPASS：全部 %d 个对象与原版一致（rel < %g）"
          % (7 + len(COV.F0_NAMES + COV.F2_NAMES), TOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
