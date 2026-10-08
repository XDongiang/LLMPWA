#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split the per-iteration amplitude cost into its three stages.

Answers "what is worth precomputing?" with numbers, for the MC-scale sample
(3.6M events, the dominant term in the likelihood):

    stage 1  propagators        BW / Flatte evaluated on the per-event masses
    stage 2  couplings x propagators  (the tensor contraction)
    stage 3  total + EPS_A rotation + |.|^2

Usage:
    python run/bench_amplitude_split.py --prefix mc_ --repeats 9
"""

import argparse
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
os.chdir(_root)
for _p in (_root, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as onp  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import config, jit  # noqa: E402

config.update("jax_enable_x64", True)

import base_functions as B  # noqa: E402
import likelihood_function as L  # noqa: E402


def timeit(fn, n):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.time()
        fn()
        ts.append(time.time() - t0)
    return min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="mc_")
    ap.add_argument("--repeats", type=int, default=9)
    ap.add_argument("--mode", default="1")
    args = ap.parse_args()
    os.environ["HELICITY_LAYER"] = args.mode

    data = B.normalize_data(B.load_data())
    a0, _, _ = B.make_initial_args()
    params = L.extract_parameters(jnp.asarray(a0))

    phi_s = jnp.asarray(data[args.prefix + "phi_kk"])
    f_s = jnp.asarray(data[args.prefix + "f_kk"])
    f0_t = jnp.asarray(data[args.prefix + "phif0_kk"])
    f2_t = jnp.asarray(data[args.prefix + "phif2_kk"])
    print("prefix=%s  events=%d  f0 %s  f2 %s  HELICITY_LAYER=%s"
          % (args.prefix, phi_s.shape[0], f0_t.shape, f2_t.shape, args.mode))

    p = params

    def props(ps):
        pm, pw = 1.02, 0.004
        q = ps["f0_980"]
        fp = B.flatte980(q["mass"], q["g_pipi"], q["rg"], f_s)[:, None, :]
        a = L._propagator_product(pm, pw, phi_s, fp)
        q = ps["f0_bw"]
        fp = jnp.stack([B.BW(q["mass"][i], q["width"][i], f_s) for i in range(2)], axis=1)
        b = L._propagator_product(pm, pw, phi_s, fp)
        q = ps["f2_1270"]
        fp = B.flatte1270(q["mass"], q["width"], f_s)[:, None, :]
        c = L._propagator_product(pm, pw, phi_s, fp)
        q = ps["f2_bw"]
        fp = jnp.stack([B.BW(q["mass"][i], q["width"][i], f_s) for i in range(3)], axis=1)
        d = L._propagator_product(pm, pw, phi_s, fp)
        return a, b, c, d

    def coupled(ps):
        pr = props(ps)
        q = ps["f0_980"]
        r0 = L._coupled_components(f0_t, q["const"][None, :], q["theta"][None, :], pr[0])
        q = ps["f0_bw"]
        r1 = L._coupled_components(f0_t, q["const"], q["theta"], pr[1])
        q = ps["f2_1270"]
        r2 = L._coupled_components(f2_t, q["const"][None, :], q["theta"][None, :], pr[2])
        q = ps["f2_bw"]
        r3 = L._coupled_components(f2_t, q["const"], q["theta"], pr[3])
        return r0, r1, r2, r3

    def intensity(ps):
        return B.intensity_from_amplitude(B.apply_helicity(L._total_amplitude(coupled(ps))))

    jp = jit(props)
    jc = jit(coupled)
    ji = jit(intensity)

    n = args.repeats
    t_all = timeit(lambda: ji(params).block_until_ready(), n)
    pl = jp(params)
    # props only: return a dummy scalar so the tree is not the output
    jp_only = jit(lambda ps: sum(x[0].sum() for x in props(ps)))
    t_p = timeit(lambda: jp_only(params).block_until_ready(), n)
    jc_only = jit(lambda ps: sum(x[0].sum() + x[1].sum() for x in coupled(ps)))
    t_c = timeit(lambda: jc_only(params).block_until_ready(), n)

    # rotation alone, on the total of the (already computed) coupled blocks
    tot = L._total_amplitude(coupled(params))
    jr = jit(lambda x: B.to_helicity(x)[0].sum())
    t_r = timeit(lambda: jr(tot).block_until_ready(), n)

    print("\n%-34s %8.4f s" % ("full intensity (all stages)", t_all))
    print("%-34s %8.4f s  (%4.1f%%)" % ("stage 1  propagators", t_p, 100 * t_p / t_all))
    print("%-34s %8.4f s  (%4.1f%%)" % ("stage 2  contraction (incl. 1)", t_c, 100 * t_c / t_all))
    print("%-34s %8.4f s  (%4.1f%%)" % ("stage 2 only (contraction)", t_c - t_p,
                                        100 * (t_c - t_p) / t_all))
    print("%-34s %8.4f s  (%4.1f%%)" % ("stage 3  rotation+abs2", t_all - t_c,
                                        100 * (t_all - t_c) / t_all))
    print("%-34s %8.4f s  (%4.1f%%)" % ("  of which EPS_A rotation", t_r,
                                        100 * t_r / t_all))
    print("\nper-event costs: propagators %.3f ns, contraction %.3f ns"
          % (1e9 * t_p / phi_s.shape[0], 1e9 * (t_c - t_p) / phi_s.shape[0]))


if __name__ == "__main__":
    main()
