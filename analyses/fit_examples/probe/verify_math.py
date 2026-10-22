#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐条核对 `math.md` 里的公式 —— 文档里的每个等式都在这里跑一遍。

为什么需要这个文件
------------------
`math.md` 写的是一组**等式**（振幅、强度、似然、梯度闭式、平坦方向）。
等式最容易在重构后悄悄失效：代码还能跑、数字还在动，但已经不是文档写的那件事。
所以这里不用 `jax.grad` 去"验证 `jax.grad`"，而是**独立重算**每个中间量：

  * 强度：用 numpy 直接从 cache 算 `I(e)`，与 `likelihood.py` 的 jax 版对比
  * 似然：用 numpy 按两项分解重算，与 jax 版对比
  * 梯度：**手推的闭式**（math.md §7）与 `jax.grad` 对比
  * 平坦方向：`g -> s·g`（s ∈ ℂ）下 L 的不变性，与 `x0ᵀHx0 = 0` 对比

任何一条对不上就非零退出。用法：

    analyses/kk_dis/.venv/bin/python analyses/fit_examples/probe/verify_math.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as onp

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(_FIT))
sys.path.insert(0, _FIT)

import jax                                                    # noqa: E402
from jax import grad, jit                                     # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "fit_likelihood", os.path.join(_FIT, "common", "likelihood.py"))
LK = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LK)
onp.set_printoptions(precision=4, suppress=True)

CASES = os.path.join(_FIT, "cases")
FAIL = []


def check(name, ok, detail=""):
    print("  %-4s %-42s %s" % ("OK" if ok else "**NO**", name, detail))
    if not ok:
        FAIL.append(name)


def rel(a, b):
    a, b = onp.asarray(a), onp.asarray(b)
    return float(onp.abs(a - b).max() / max(onp.abs(b).max(), 1e-300))


def slots(c):
    return tuple(LK.HEL_TRANSVERSE) if c.is_complex else tuple(LK.CART_TRANSVERSE)


def numpy_amplitude(c, x, prefix):
    """A(e,k) = Σ_R Σ_i g_i W_i(e,k)，纯 numpy，不经 jax。"""
    cpls = [onp.asarray(g) for g in c.unpack(x)]
    A = None
    for fam, g in zip(c.families, cpls):
        part = onp.einsum("iek,i->ek", onp.asarray(fam[prefix]), g)
        A = part if A is None else A + part
    return A


def numpy_intensity(c, x, prefix):
    A = numpy_amplitude(c, x, prefix)
    return onp.sum(onp.abs(A[:, list(slots(c))]) ** 2, axis=-1)


def numpy_likelihood(c, x):
    """math.md §5 的两项，逐项算出。"""
    I_d = numpy_intensity(c, x, "t_data")
    I_m = numpy_intensity(c, x, "t_mc")
    return -onp.log(I_d).sum() + c.n_data * onp.log(I_m.mean()), I_d, I_m


def analytic_grad(c, x):
    """math.md §7 的闭式。`g_i = c_i + i·t_i`。

        u^d_{e,i} = Σ_{k∈K} conj(A_{e,k}) W_i(e,k)
        v_i       = Σ_e (-1/I^d_e)·u^d_{e,i}
                    + (N_data/(N_mc·mean I^mc))·Σ_e u^mc_{e,i}
        ∂L/∂c_i = 2 Re v_i
        ∂L/∂t_i = −2 Im v_i
    """
    K = list(slots(c))
    cpls = [onp.asarray(g) for g in c.unpack(x)]
    A_d = numpy_amplitude(c, x, "t_data")[:, K]
    A_m = numpy_amplitude(c, x, "t_mc")[:, K]
    I_d = onp.sum(onp.abs(A_d) ** 2, axis=-1)
    mean_m = onp.sum(onp.abs(A_m) ** 2, axis=-1).mean()

    g = onp.zeros(x.size)
    for fam, cpl in zip(c.families, cpls):
        T_d = onp.asarray(fam["t_data"])[:, :, K]
        T_m = onp.asarray(fam["t_mc"])[:, :, K]
        u_d = onp.einsum("ek,iek->ei", onp.conj(A_d), T_d)
        u_m = onp.einsum("ek,iek->ei", onp.conj(A_m), T_m)
        v = (onp.einsum("e,ei->i", -1.0 / I_d, u_d)
             + onp.einsum("e,ei->i", onp.full(c.n_mc, c.n_data / mean_m), u_m)
             / c.n_mc)
        sl = fam["slice"]
        g[sl.start:sl.stop:2] = 2.0 * v.real
        g[sl.start + 1:sl.stop:2] = -2.0 * v.imag
    return g


def scale_phase(x, s, phi):
    """x -> (整体乘 s·e^{iφ}) 之后对应的实参向量。"""
    out = x.copy()
    for i in range(0, x.size, 2):
        re, im = x[i] * s, x[i + 1] * s
        out[i] = re * onp.cos(phi) - im * onp.sin(phi)
        out[i + 1] = re * onp.sin(phi) + im * onp.cos(phi)
    return out


# ---------------------------------------------------------------- 各项检查
def check_case(case, basis, n_data, n_mc):
    tag = "%s/%s" % (case, basis)
    print("\n== %s  (N_data=%d, N_mc=%d)" % (tag, n_data, n_mc))
    c = LK.Case(os.path.join(CASES, case), basis, n_data, n_mc)
    inten, inten_mc, lik = LK.make_functions(c)
    x0 = onp.asarray(c.initial(), dtype=float)

    # §3 强度
    check("§3 I_data  (numpy == jax)", rel(numpy_intensity(c, x0, "t_data"),
                                          onp.asarray(inten(x0))) < 1e-10,
          "rel=%.1e" % rel(numpy_intensity(c, x0, "t_data"),
                           onp.asarray(inten(x0))))
    check("§3 I_mc    (numpy == jax)", rel(numpy_intensity(c, x0, "t_mc"),
                                           onp.asarray(inten_mc(x0))) < 1e-10)

    # §5 似然
    L_np, I_d, I_m = numpy_likelihood(c, x0)
    check("§5 L        (numpy == jax)", abs(L_np - float(lik(x0))) < 1e-8,
          "%.6f vs %.6f" % (L_np, float(lik(x0))))

    # §7 闭式梯度
    g_an = analytic_grad(c, x0)
    g_jax = onp.asarray(jit(grad(lik))(x0))
    r = rel(g_an, g_jax)
    check("§7 grad 闭式 == jax.grad", r < 1e-10, "rel=%.1e" % r)

    # §6 整体复缩放不变性
    base = float(lik(x0))
    worst = 0.0
    for s, phi in ((1e-4, 0.0), (1e3, 0.0), (1.0, 0.7), (0.3, 2.5), (7.0, 3.14159)):
        worst = max(worst, abs(float(lik(scale_phase(x0, s, phi))) - base))
    check("§6 L 在 g->s·g 下不变", worst < 1e-8, "|ΔL|max=%.1e" % worst)

    # §6 两个精确零方向：g 与 g·x0、g·Jx0，以及 x0ᵀHx0
    Jx = x0.copy()
    for i in range(0, x0.size, 2):
        Jx[i], Jx[i + 1] = -x0[i + 1], x0[i]
    check("§6 grad·x0   = 0", abs(float(g_jax @ x0)) < 1e-6,
          "%.1e" % float(g_jax @ x0))
    check("§6 grad·Jx0  = 0", abs(float(g_jax @ Jx)) < 1e-6,
          "%.1e" % float(g_jax @ Jx))
    H = onp.asarray(jax.jacfwd(jit(grad(lik)))(x0))
    Hs = 0.5 * (H + H.T)
    check("§6 x0ᵀ·H·x0  = 0", abs(float(x0 @ Hs @ x0)) < 1e-4,
          "%.1e（|H|max=%.1e）" % (float(x0 @ Hs @ x0), onp.abs(Hs).max()))

    # §5.1 data == mc 时退化为 N·log(AM/GM)
    if c.n_data == c.n_mc and onp.array_equal(
            onp.asarray(c.families[0]["t_data"]),
            onp.asarray(c.families[0]["t_mc"])):
        am, gm = I_d.mean(), onp.exp(onp.log(I_d).mean())
        check("§5.1 L == N·log(AM/GM)", abs(float(lik(x0)) - c.n_data * onp.log(am / gm)) < 1e-8,
              "L/N=%.6f" % (float(lik(x0)) / c.n_data))
        check("§5.1 AM/GM >= 1  =>  L >= 0", am / gm >= 1.0, "AM/GM=%.6f" % (am / gm))
    return c, x0, lik


# ---------------------------------------------------------------- 跨表象 §4
def check_basis_equivalence(case):
    print("\n== %s  跨表象 (math.md §4)" % case)
    ca = LK.Case(os.path.join(CASES, case), "cartesian")
    ch = LK.Case(os.path.join(CASES, case), "helicity_cascade")
    x0 = onp.asarray(ca.initial(), dtype=float)
    ga, gh = ca.unpack(x0), ch.unpack(x0)
    for fa, fh in zip(ca.families, ch.families):
        i, j = ca.families.index(fa), ch.families.index(fh)
        Aa = onp.einsum("iek,i->ek", onp.asarray(fa["t_data"]), onp.asarray(ga[i]))
        Ah = onp.einsum("iek,i->ek", onp.asarray(fh["t_data"]), onp.asarray(gh[j]))
        Ia = onp.sum(onp.abs(Aa[:, list(LK.CART_TRANSVERSE)]) ** 2, axis=-1)
        Ih = onp.sum(onp.abs(Ah[:, list(LK.HEL_TRANSVERSE)]) ** 2, axis=-1)
        r = rel(Ia, Ih)
        mx = (fa["calib"] or {}).get("max_rel")
        comparable = mx is not None and mx <= 1e-6
        if comparable:
            # 两份 cache 是同一振幅（只差逐道常数）-> 必须逐位相等
            check("%s  I_cart == I_casc" % fa["name"], r < 1e-6,
                  "rel=%.1e  标定残差=%.1e" % (r, mx))
        else:
            # 形状不同（已知边界）：不判等，只**记录**差异确实远超阈值。
            # 写成 OK 会自相矛盾（"相等"却 rel=5e4），所以单独一行说清。
            print("  SKIP %-42s 两份 cache 形状不同（标定残差 %.1e），"
                  "不可比；实测比值 rel=%.1e" % (fa["name"], mx, r))


def main():
    print("jax %s  x64=%s  devices=%s"
          % (jax.__version__, jax.config.jax_enable_x64, jax.devices()))
    # 默认设置：n_data = n_mc = N（此时 §5.1 的退化恒等式成立）
    for case in ("psi_omega_kk", "psi_pipipi0"):
        check_case(case, "cartesian", 0, 0)
    # 不重叠切片（§5.1 不成立，归一化项真的在动）
    check_case("psi_kkpipi", "cartesian", 4000, 3000)
    check_case("psi_kkpipi", "helicity_cascade", 4000, 3000)
    check_case("psi_phi4pi", "helicity_cascade", 2000, 2000)

    check_basis_equivalence("psi_kkpipi")
    check_basis_equivalence("psi_pipipi0")

    print("\n%s" % ("=> 全部通过" if not FAIL else "=> 失败：%s" % FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
