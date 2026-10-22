#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""原型：缓存张量 -> JAX likelihood -> value/grad/hvp，**只验"能算、能微分、能快"**。

这不是最终拟合代码，是**可行性探针**。它回答四个问题：

1. 同一份 spec 出的 cartesian 与 helicity 两份 cache，在**同一组耦合**下
   给出的 likelihood 是否逐位相同？
   —— 这是最硬的判据：`I_cart = |T_x|^2+|T_y|^2` 与
   `I_hel = sum_{λ=±1} |W_λ|^2` 由 `W = EPS_A T` 保证**严格相等**，
   所以两个数字必须落到机器精度。任一侧的装订（k 分量、λ 槽位、行序）错了，
   这里立刻炸。
2. `jax.grad` 与有限差分是否一致（抓"能跑但梯度错"）。
3. `jvp(grad)` 的 HVP 与"梯度再做有限差分"是否一致（HVP 是 Newton-CG 的燃料）。
4. 单次 value+grad / HVP 的耗时（CPU 本地；GPU 数字见 README）。

用法
----
    python analyses/fit_examples/probe/proto_likelihood.py \
        --cache probe/cache/ex3_b1pi --n-data 2000 --n-mc 5000 --iters 20
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as onp

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as np
from jax import jit, grad, jvp

jax.config.update("jax_enable_x64", True)

# 与 kk_dis/run/base_functions.py 逐位相同的 EPS_A（行序 λ=+1,0,-1）
EPS_A = onp.array(
    [[1.0 / onp.sqrt(2.0), -1.0j / onp.sqrt(2.0), 0.0],
     [0.0, 0.0, 1.0],
     [-1.0 / onp.sqrt(2.0), -1.0j / onp.sqrt(2.0), 0.0]],
    dtype=onp.complex128,
)
# 初始态 e+e- 只给 λ=±1；k=2 的旧 cache 只有 (x,y)，槽位不同
TRANSVERSE = {2: (0, 1), 3: (0, 2)}


# ------------------------------------------------------------------ 数据装载
def load_cache(cache_dir, n_data, n_mc):
    """读 `cartesian/*_kk.npy` 与 `helicity/*_hel.npy`，返回两套 (n_ch,N,k)。"""
    cart = sorted(glob.glob(os.path.join(cache_dir, "cartesian", "*.npy"))
                  or glob.glob(os.path.join(cache_dir, "covariant", "*.npy")))
    hel = sorted(glob.glob(os.path.join(cache_dir, "helicity", "*.npy")))
    if not cart or not hel:
        raise SystemExit("%s 下缺 cartesian/ 或 helicity/ 的 .npy" % cache_dir)
    Tc = onp.load(cart[0])
    Th = onp.load(hel[0])
    if Tc.shape[:2] != Th.shape[:2]:
        raise SystemExit("两份 cache 的 (n_ch, N) 不一致：%s vs %s"
                         % (Tc.shape, Th.shape))
    n = Tc.shape[1]
    n_data = min(n_data, n)
    n_mc = min(n_mc, n)
    return {
        "cart_data": np.asarray(Tc[:, :n_data, :], dtype=np.float64),
        "cart_mc": np.asarray(Tc[:, n - n_mc:, :], dtype=np.float64),
        "hel_data": np.asarray(Th[:, :n_data, :], dtype=np.complex128),
        "hel_mc": np.asarray(Th[:, n - n_mc:, :], dtype=np.complex128),
        "n_ch": Tc.shape[0],
        "k_cart": Tc.shape[-1],
        "k_hel": Th.shape[-1],
    }


# ------------------------------------------------------------------ 振幅/强度
# 笛卡尔 cache 的强度**永远**取 (x, y) —— 存了 z 也不参与。
# 这不是笔误：`basis.cartesian` 的钩子就是 `CARTESIAN_TRANSVERSE = (0, 1)`，
# 与 k 无关。写成 TRANSVERSE[k] 会把 (x,z) 当成横向量，得到"能跑但错"的数。
CART_TRANSVERSE = (0, 1)


def _intensity_cart(tensor, cpl):
    """A(e,mu) = sum_i c_i T_i(e,mu)；I = |A_x|^2 + |A_y|^2（k=2 的旧口径）。"""
    amp = np.einsum("iek,i->ek", tensor, cpl)
    return np.sum(np.abs(amp[:, CART_TRANSVERSE]) ** 2, axis=-1)


def _intensity_hel(tensor, cpl):
    """A(e,lam) = sum_i c_i W_i(e,lam)；I = sum_{lam=±1} |A|^2。"""
    amp = np.einsum("iek,i->ek", tensor, cpl)
    k = tensor.shape[-1]
    return np.sum(np.abs(amp[:, TRANSVERSE[k]]) ** 2, axis=-1)


def _combined(inten_data, inten_mc, n_data):
    """kk_dis 的口径：L = -sum log I_data + n_data * log(mean I_mc)。"""
    tiny = np.finfo(np.float64).tiny
    return -np.sum(np.log(np.maximum(inten_data, tiny))) \
        + n_data * np.log(np.maximum(np.mean(inten_mc), tiny))


def make_likelihood(data, basis):
    if basis == "cartesian":
        inten = _intensity_cart
        td, tm = data["cart_data"], data["cart_mc"]
    else:
        inten = _intensity_hel
        td, tm = data["hel_data"], data["hel_mc"]
    n_data = td.shape[1]

    def likelihood(cpl):
        return _combined(inten(td, cpl), inten(tm, cpl), n_data)

    return likelihood


# ------------------------------------------------------------------ 导数自检
def _fd_grad(f, x, eps=1e-6):
    """复输入、实输出的**实参**有限差分，返回与 `jax.grad` 同约定的复数。

    `jax.grad` 对实值函数的复输入返回 `g = df/dRe - i df/dIm`（共轭梯度，
    实测 `grad(|z|^2) = 2 z*`、`grad(Re z^2) = 2z*`）。符号**必须**是减号：
    写成加号会得到 1.6 的假失配。也不能把整条复数数组当成一个实变量扰动 ——
    那样只测到 `df/dRe`。两个坑都踩过。
    """
    x = onp.asarray(x)
    g = onp.zeros_like(x, dtype=onp.complex128)
    for i in range(x.size):
        e = onp.zeros_like(x)
        e[i] = eps
        g[i] = (float(f(x + e)) - float(f(x - e))) / (2 * eps)
        e[i] = 1j * eps
        g[i] -= 1j * (float(f(x + e)) - float(f(x - e))) / (2 * eps)
    return g


def _rel(a, b):
    a, b = onp.asarray(a), onp.asarray(b)
    return float(onp.abs(a - b).max() / max(onp.abs(b).max(), 1e-300))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n-data", type=int, default=2000)
    ap.add_argument("--n-mc", type=int, default=5000)
    ap.add_argument("--iters", type=int, default=20, help="计时/拟合迭代数")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", default="", help="把结果写到这个 json")
    args = ap.parse_args()

    data = load_cache(args.cache, args.n_data, args.n_mc)
    n_ch = data["n_ch"]
    rng = onp.random.default_rng(args.seed)
    cpl0 = np.asarray(
        (rng.normal(size=n_ch) + 1j * rng.normal(size=n_ch)) * 0.1,
        dtype=np.complex128)

    out = {"cache": args.cache, "n_ch": n_ch, "k_cart": data["k_cart"],
           "k_hel": data["k_hel"],
           "n_data": int(data["cart_data"].shape[1]),
           "n_mc": int(data["cart_mc"].shape[1])}

    lik = {b: make_likelihood(data, b) for b in ("cartesian", "helicity")}
    val = {b: float(lik[b](cpl0)) for b in lik}
    out["value"] = val
    out["cart_vs_hel_rel"] = abs(val["cartesian"] - val["helicity"]) / \
        max(abs(val["helicity"]), 1e-300)

    # 两个表象的梯度也必须一致（更强：它管着拟合往哪走）
    gfun = {b: jit(grad(lik[b])) for b in lik}
    gr = {b: onp.asarray(gfun[b](cpl0)) for b in lik}
    out["grad_rel_cart_vs_hel"] = _rel(gr["cartesian"], gr["helicity"])

    # grad vs 有限差分（一阶导的独立复算）
    out["grad_vs_fd_rel"] = {
        b: _rel(gr[b], _fd_grad(lik[b], cpl0)) for b in lik}

    # HVP vs "梯度的有限差分"
    vec = np.asarray(rng.normal(size=n_ch) + 1j * rng.normal(size=n_ch),
                     dtype=np.complex128)
    hessp = {b: jit(lambda x, v: jvp(gfun[b], (x,), (v,))[1]) for b in lik}
    hv = {b: onp.asarray(hessp[b](cpl0, vec)) for b in lik}
    fd_hv = {}
    eps = 1e-5
    for b in lik:
        gp = onp.asarray(gfun[b](cpl0 + eps * vec))
        gm = onp.asarray(gfun[b](cpl0 - eps * vec))
        fd_hv[b] = (gp - gm) / (2 * eps)
    out["hvp_vs_fd_rel"] = {b: _rel(hv[b], fd_hv[b]) for b in lik}

    # 计时：编译一次，然后只量执行
    timing = {}
    for b in lik:
        f = jit(lambda x: (lik[b](x), gfun[b](x)))
        f(cpl0)[0].block_until_ready()
        t0 = time.time()
        for _ in range(args.iters):
            v, g = f(cpl0)
            float(v), onp.asarray(g)[:1]
            v.block_until_ready()
        timing[b] = (time.time() - t0) / args.iters
        h = jit(lambda x: hessp[b](x, vec))
        h(cpl0)[0].block_until_ready()
        t0 = time.time()
        for _ in range(args.iters):
            r = h(cpl0)
            r[0].block_until_ready()
        timing[b + "_hvp"] = (time.time() - t0) / args.iters
    out["seconds_per_eval"] = timing

    # 短跑一次 Newton-CG：只证明"能进优化器"，不追求收敛
    from scipy.optimize import minimize
    b = "helicity"
    x0 = onp.stack([cpl0.real, cpl0.imag], axis=1).reshape(-1)

    def flat_to_cpl(x):
        z = x[:n_ch] + 1j * x[n_ch:]
        return z

    def f(x):
        return float(lik[b](flat_to_cpl(x)))

    def g(x):
        return onp.asarray(gfun[b](flat_to_cpl(x)))

    t0 = time.time()
    res = minimize(f, x0, jac=lambda x: onp.concatenate(
        [g(x).real, g(x).imag]), method="L-BFGS-B",
        options={"maxiter": args.iters})
    out["fit"] = {"basis": b, "success": bool(res.success),
                  "nfev": int(res.nfev), "nit": int(res.nit),
                  "f0": float(res.fun), "seconds": time.time() - t0,
                  "note": "L-BFGS-B 短跑，只为证明优化器能接上"}

    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
