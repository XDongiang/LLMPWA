#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 5 移植对拍：`chains/kstpi/` 的 10 条道 vs jax 参考实现。

这是 A1 方案（移植而非重新推导）的唯一可信判据：移植只允许改变"用哪个
数组库算"，不允许改变任何一个数字。

需要 `Tensor/{BASE,FUNC}.py`（不在 git 里）与 jax：

    analyses/kk_dis/.venv/bin/python \
        analyses/tensor_examples/probes/verify_ex5_port.py

⚠️ **必须在导入 FUNC 之前** `jax.config.update("jax_enable_x64", True)`。
否则 jax 全程 float32（ε≈1.2e-7），对拍只能到 1e-7 —— 看起来像"移植错了"，
其实只是精度。这个坑第一次就踩了。
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

TENSOR = os.path.join(ROOT, "analyses", "kk_dis", "task",
                      "20260929_螺旋度耦合振幅推导", "1_运行代码",
                      "tensor_example")
EX5 = ["u_kst0_l", "u_kst0_r", "u_PP_1", "u_PP_2", "u_PF_1", "u_PF_2",
       "u_FP_1", "u_FP_2", "u_FF_1", "u_FF_2"]

# `(道名, 用哪个文件对拍)`。
#
# ⚠️ 2026-10-21 起 `u_PP_*`/`u_FP_*` 的**生产文件**不再是逐式移植，而是
# 修正版（内层无迹化改到 $K^*_2$ 系，见 `chains/kstpi/CORRECTION.md`）。
# jax 参考实现是**历史形式**，所以对拍要拿 `*_paper` 文件 —— 那是保留的
# 逐式移植，A5 的"逐位一致"保证落在它身上。
#
# 生产文件与 jax 的差**是本次修正的内容**，不是回归：脚本最后会把差值
# 一并打出来，作为"修正确实改了东西"的阳性对照。
EX5_PAIRS = [(n, n + "_paper" if n in ("u_PP_1", "u_PP_2", "u_FP_1", "u_FP_2")
              else n) for n in EX5]

# 生产文件与历史形式**有意不同**的道 —— 单独量差值，不计入失败。
CORRECTED = ["u_PP_1", "u_PP_2", "u_FP_1", "u_FP_2"]


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


def physical(n=32, seed=11, m_kst=1.43):
    r"""$\psi\to K^*\pi\to(\phi\pi)\pi\to(K^+K^-)(\pi\pi)$ 的物理事例。

    构造顺序保证每个 `Qabc` 的 `a -> b + c` 都是真两体衰变（$q^2>0$，
    $B_l$ 不会 NaN）。Kp/Km 要**两层** boost（φ 系 → K* 系 → ψ 系）——
    少一层会让四动量不守恒、$q^2<0$。
    """
    M_PSI, M_PHI, M_K, M_PI = 3.686, 1.019, 0.493677, 0.13957039
    rng = np.random.default_rng(seed)
    kst, spec_pi = _pair(M_PSI, m_kst, M_PI, rng, n)
    phi, pi_in = _pair(m_kst, M_PHI, M_PI, rng, n)
    kp0, km0 = _pair(M_PHI, M_K, M_K, rng, n)
    bk = kst[:, :3] / kst[:, 3:4]
    bph = phi[:, :3] / phi[:, 3:4]
    kp_kst, km_kst = _boost(kp0, bph), _boost(km0, bph)
    return {"Kp": _boost(kp_kst, bk), "Km": _boost(km_kst, bk),
            "Pip": _boost(pi_in, bk), "Pim": spec_pi,
            "phi": _boost(phi, bk)}


def load_jax():
    t = os.path.join(TENSOR, "Tensor")
    if not os.path.isdir(t):
        print("跳过：找不到 %s" % t)
        return None
    if t not in sys.path:
        sys.path.insert(0, t)
    try:
        import jax
    except ModuleNotFoundError:
        print("跳过：环境里没有 jax")
        return None
    try:
        import FUNC
    except Exception as e:                                       # noqa: BLE001
        print("跳过：导入 FUNC 失败（%s: %s）" % (type(e).__name__, e))
        return None
    jax.config.update("jax_enable_x64", True)
    f = FUNC.FUNCTION()
    f.Modifiers()
    return f


def main():
    f = load_jax()
    if f is None:
        return 1
    from agent.tensor.kernels import covariant as COV

    d = physical(n=32, seed=11)
    phi = d["phi"]
    Kp, Km, Pip, Pim = d["Kp"], d["Km"], d["Pip"], d["Pim"]
    # 参考实现的 phi 参数用 Kp+Km；用 d["phi"] 也能验，但为了与它对拍一致，
    # 显式传 phi = Kp + Km（两者数值上只差浮点误差）。
    phi_arg = Kp + Km
    mom = {"Kp": Kp, "Km": Km, "Pip": Pip, "Pim": Pim}

    print("%-10s %-10s %-14s %-14s %-12s %s"
          % ("道", "对拍文件", "|jax|max", "|numpy|max", "rel", "状态"))
    print("-" * 80)
    bad = []
    for nm, src in EX5_PAIRS:
        J = np.asarray(getattr(f, nm)(phi_arg, Kp, Km, Pip, Pim))
        N = COV.compute_event(src, mom)
        r = float(np.abs(N - J).max() / max(np.abs(J).max(), 1e-300))
        ok = r < 1e-11
        if not ok:
            bad.append((src, r))
        print("%-10s %-10s %-14.5g %-14.5g %-12.3e %s"
              % (nm, src, float(np.abs(J).max()), float(np.abs(N).max()), r,
                 "OK" if ok else "**FAIL**"))
    print("-" * 80)
    print("%s：失败 %d / %d"
          % ("PASS" if not bad else "**FAIL**", len(bad), len(EX5_PAIRS)))
    for nm, r in bad:
        print("  %s  rel = %.3e" % (nm, r))

    # 阳性对照：生产文件（修正版）与历史形式的差 —— **应该**非零
    print()
    print("修正的阳性对照（生产文件 vs 历史形式，应该非零）：")
    for nm in CORRECTED:
        J = np.asarray(getattr(f, nm)(phi_arg, Kp, Km, Pip, Pim))
        N = COV.compute_event(nm, mom)
        r = float(np.abs(N - J).max() / max(np.abs(J).max(), 1e-300))
        print("  %-10s rel = %.3e  %s"
              % (nm, r, "✅ 确实改了" if r > 1e-6 else "⚠️ 没改？"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
