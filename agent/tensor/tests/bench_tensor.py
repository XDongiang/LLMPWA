#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""协变张量计算性能基准：numpy(CPU) vs jax(CPU) vs jax(GPU)。

对拍的是**同一个物理量、同一种生产负载**：
  * numpy 侧 = agent/tensor/kernels/covariant.py（A1 移植版）
  * jax   侧 = Tensor/{BASE,FUNC}.py 原版（jit + vmap，生产在用的实现）

⚠️ 必须**分批**（生产 `gen_tensor3.py --batch 128` 就是为此）：
`P4` 是 4^8 张量，单事例 float64 就 4^8*8 = 512 KB；
5865 事例一次性做 = 3 GB，CPU 上会慢到看不出真实性能。

同时给出**逐道**耗时，因为 `phif223`(P3) / `phif243`(P4) 是绝对主导项，
而 f0 族（phif001/phif021）几乎免费 —— 只看总时间会误判。

    python bench_tensor.py --backend numpy  --npz tmp/bundle/real_r5865.npz --batch 128
    python bench_tensor.py --backend jaxcpu --npz tmp/bundle/real_r5865.npz --batch 128
    python bench_tensor.py --backend jaxgpu --npz tmp/bundle/real_r5865.npz --batch 128
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["numpy", "jaxcpu", "jaxgpu"])
    ap.add_argument("--npz", default="tmp/bundle/real_r5865.npz")
    ap.add_argument("--tensor-dir", default=None)
    ap.add_argument("--channels",
                    default="phif001,phif021,phif201,phif221,phif222,phif223,phif243")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--per-channel", action="store_true", default=True)
    return ap.parse_args()


ARGS = parse_args()

# JAX_PLATFORMS 必须在 import jax 之前设置
if ARGS.backend == "jaxgpu":
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
else:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

CHANNELS = ARGS.channels.split(",")
NP = ARGS.dtype
BATCH = ARGS.batch


def find_tensor_dir():
    if ARGS.tensor_dir:
        return ARGS.tensor_dir
    for cand in ("task/20260929_螺旋度耦合振幅推导/1_运行代码/tensor_example",
                 "4_三条路线/1_运行代码/tensor_example"):
        if os.path.isdir(os.path.join(cand, "Tensor")):
            return cand
    raise SystemExit("找不到 Tensor/ 目录，请用 --tensor-dir 指定")


def load_momenta():
    d = np.load(ARGS.npz)
    return {k: np.asarray(d[k], dtype=NP) for k in ("Kp", "Km", "Pip", "Pim")}


def slices(n, batch):
    for beg in range(0, n, batch):
        yield beg, min(beg + batch, n)


# ------------------------------------------------------------------ numpy 侧
def bench_numpy(mom, cov, n):
    psi = np.broadcast_to(cov.PSI2S_DEFAULT.astype(NP), (n, 4))

    def one(nm):
        acc = 0.0
        for beg, end in slices(n, BATCH):
            sub = {k: v[beg:end] for k, v in mom.items()}
            acc += float(np.abs(cov.compute_event(nm, sub, psi=psi[beg:end])).sum())
        return acc

    def allch():
        return {nm: one(nm) for nm in CHANNELS}

    return allch, one


# -------------------------------------------------------------------- jax 侧
def build_jax(mom, tensor_dir, n):
    import jax
    if NP == "float64":
        jax.config.update("jax_enable_x64", True)
    import types
    if "jax.config" not in sys.modules:
        m = types.ModuleType("jax.config")
        m.config = jax.config
        m.update = jax.config.update
        sys.modules["jax.config"] = m
    for p in (tensor_dir, os.path.join(tensor_dir, "Tensor")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from Tensor import FUNC as JFUNC

    fn = JFUNC.FUNCTION()
    fn.Modifiers()
    J = lambda a: jax.numpy.asarray(a)  # noqa: E731

    # 各道形参个数不同（见 FUNC.py）：phif001 不吃 f2，phif021 吃 f0
    def make(nm, beg, end):
        Kp, Km = mom["Kp"][beg:end], mom["Km"][beg:end]
        Pip, Pim = mom["Pip"][beg:end], mom["Pim"][beg:end]
        phi, f2 = Kp + Km, Pip + Pim
        if nm == "phif001":
            args = (phi, Kp, Km)
        elif nm == "phif021":
            args = (phi, f2, Kp, Km)
        else:
            args = (phi, f2, Kp, Km, Pip, Pim)
        f = getattr(fn, nm)
        return jax.jit(lambda: f(*[J(x) for x in args]))

    compiled = {(nm, beg): make(nm, beg, end) for nm in CHANNELS
                for beg, end in slices(n, BATCH)}

    def one(nm):
        acc = 0.0
        for beg, end in slices(n, BATCH):
            acc += float(np.abs(np.asarray(jax.device_get(compiled[(nm, beg)]()))).sum())
        return acc

    def allch():
        return {nm: one(nm) for nm in CHANNELS}

    return jax, allch, one


def timeit(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), float(np.median(ts))


def main():
    mom = load_momenta()
    n = mom["Kp"].shape[0]
    print("=" * 78)
    print("backend=%-7s dtype=%-8s N=%d batch=%d channels=%d"
          % (ARGS.backend, NP, n, BATCH, len(CHANNELS)))

    if ARGS.backend == "numpy":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "kernels"))
        import covariant as cov
        allch, one = bench_numpy(mom, cov, n)
        extra = ""
    else:
        jax, allch, one = build_jax(mom, find_tensor_dir(), n)
        devs = jax.devices()
        extra = "  device=%s" % (devs[0],)
        print("jax %s  devices=%s" % (jax.__version__, devs))

    best, med = timeit(allch, ARGS.warmup, ARGS.repeat)
    checksum = sum(allch().values())
    print("TOTAL  best=%.4f s  median=%.4f s  throughput=%.0f ev/s  checksum=%.6e%s"
          % (best, med, n / best, checksum, extra))

    if ARGS.per_channel:
        print("-" * 78)
        print("per-channel (best of %d):" % ARGS.repeat)
        for nm in CHANNELS:
            b, m = timeit(lambda nm=nm: one(nm), ARGS.warmup, ARGS.repeat)
            print("  %-8s best=%.4f s  (%.1f%% of total)  %.0f ev/s"
                  % (nm, b, 100.0 * b / best if best else 0.0, n / b if b else 0.0))
    print("=" * 78)


if __name__ == "__main__":
    main()
