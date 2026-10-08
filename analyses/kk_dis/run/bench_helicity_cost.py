#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure what the EPS_A rotation actually costs inside the fit loop.

Compares, on identical full-scale data and parameters, the stable per-call time
of jit(NLL), jit(grad) and jit(hvp) for

    HELICITY_LAYER=0   Cartesian only
    HELICITY_LAYER=1   Cartesian + to_helicity() applied per coupled block

so the difference is the price of the rotation.  Repeat `--repeats` times and
report the minimum (the most stable statistic for GPU work).

Usage:
    python run/bench_helicity_cost.py --repeats 7
"""

import argparse
import contextlib
import json
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
from jax import config, device_put, grad, jit, jvp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

config.update("jax_enable_x64", True)

from base_functions import load_data, make_initial_args, normalize_data, shard_data_distributed  # noqa: E402
from likelihood_function import make_distributed_likelihood  # noqa: E402


def _mesh_context(mesh):
    if hasattr(jax, "set_mesh"):
        return jax.set_mesh(mesh)
    return contextlib.nullcontext()


def timeit(fn, n):
    fn()  # warm
    ts = []
    for _ in range(n):
        t0 = time.time()
        fn()
        ts.append(time.time() - t0)
    return min(ts), sum(ts) / len(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--out", default="output/bench_helicity_cost.json")
    args = ap.parse_args()

    print("devices: %s" % jax.devices())
    data = normalize_data(load_data())
    data_size = int(data["data_phi_kk"].shape[0])
    mesh = Mesh(jax.devices(), axis_names=("event",))
    args0, _, _ = make_initial_args()
    replicated = NamedSharding(mesh, P())

    out = {}
    with _mesh_context(mesh):
        jax_data = shard_data_distributed(data, mesh)
        rep = lambda v: device_put(jnp.asarray(v), replicated)
        x0 = rep(args0)
        v0 = rep(onp.ones_like(args0))

        for mode in ("0", "1"):
            os.environ["HELICITY_LAYER"] = mode
            jax.clear_caches()
            nll = make_distributed_likelihood(data_size)
            gn = grad(nll, argnums=0)

            def hvp(a, vector, d):
                z = jax.tree_util.tree_map(jnp.zeros_like, d)
                return jvp(lambda x, dd: gn(x, dd), (a, d), (vector, z))[1]

            jn, jg, jh = jit(nll), jit(gn), jit(hvp)
            t0 = time.time()
            val = float(jn(x0, jax_data))
            compile_s = time.time() - t0
            mn, an = timeit(lambda: jn(x0, jax_data).block_until_ready(), args.repeats)
            mg, ag = timeit(lambda: jg(x0, jax_data).block_until_ready(), args.repeats)
            mh, ah = timeit(lambda: jh(x0, v0, jax_data).block_until_ready(), args.repeats)
            out[mode] = {"nll_min": mn, "nll_avg": an, "grad_min": mg, "grad_avg": ag,
                         "hvp_min": mh, "hvp_avg": ah, "nll_value": val,
                         "compile_s": compile_s}
            print("HELICITY_LAYER=%s  nll=%.12g  compile %.1fs" % (mode, val, compile_s))
            print("   NLL  min %.3fs avg %.3fs" % (mn, an))
            print("   grad min %.3fs avg %.3fs" % (mg, ag))
            print("   hvp  min %.3fs avg %.3fs" % (mh, ah))

    a, b = out["0"], out["1"]
    print("\n--- cost of the rotation (helicity - cartesian, min) ---")
    for k in ("nll", "grad", "hvp"):
        d = b[k + "_min"] - a[k + "_min"]
        print("   %-5s %+.3fs  (%+.1f%%)" % (k, d, 100 * d / max(a[k + "_min"], 1e-9)))
    print("   NLL values equal: %s" % (a["nll_value"] == b["nll_value"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("written %s" % args.out)


if __name__ == "__main__":
    main()
