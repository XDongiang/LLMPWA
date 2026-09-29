#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decisive test: start from EXPLICITLY NON-PHYSICAL initial parameters and see
whether the fit (a) stays at / converges to a non-physical solution, or
(b) self-corrects back to physical values.

Runs the same non-physical start under:
  * RANGE-ON  (default likelihood with gaussian/fraction range penalties)
  * FREE      (RUN_FREE_RANGE=1, penalties disabled)

Useful to determine whether the range penalty genuinely guards against
non-physical solutions, or whether the data likelihood alone is enough.
"""
import os
import sys

import numpy as onp

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
os.chdir(_root)
for p in (_root, _here):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax import config, device_put, grad, jit, jvp
from scipy.optimize import minimize

from base_functions import load_data, make_initial_args, normalize_data, shard_data_distributed

# params to force non-physical (arg_index -> non-physical value)
# f0_1710_width=arg6, f2_1270_mass=arg11, f2_1270_width=arg12,
# f2_2150_width=arg36, f2_2340_mass=arg47
NONPHYS = {
    6: -0.05,        # negative width (strongly non-physical)
    11: 0.80,        # f2(1270) mass way too low
    12: 0.001,       # near-zero width
    36: -0.02,       # negative f2(2150) width
    47: 3.50,        # f2(2340) mass way too high
}


def make_unphys_base(base):
    b = onp.array(base, dtype=float).copy()
    for i, v in NONPHYS.items():
        b[i] = v
    return b


def build_likelihood(free):
    if free:
        from likelihood_function import make_distributed_likelihood
    else:
        from likelihood_function import make_distributed_likelihood
    return make_distributed_likelihood


def run(x0, data, data_size, mesh, replicated, free, maxiter):
    from likelihood_function import make_distributed_likelihood
    jax_data = shard_data_distributed(data, mesh)
    nll = make_distributed_likelihood(data_size)
    grad_nll = grad(nll, argnums=0)

    def hvp_nll(args, vector, data_arg):
        zero = jax.tree_util.tree_map(jnp.zeros_like, data_arg)
        return jvp(lambda a, d: grad_nll(a, d), (args, data_arg), (vector, zero))[1]

    jit_nll = jit(nll)
    jit_grad = jit(grad_nll)
    jit_hvp = jit(hvp_nll)

    def rep(v):
        return device_put(jnp.asarray(v), replicated)

    x0r = rep(x0)
    direction0 = rep(onp.zeros_like(x0))

    def objective(x):
        return float(jit_nll(rep(x), jax_data))

    def gradient_value(x):
        return onp.asarray(jit_grad(rep(x), jax_data), dtype=onp.float64)

    def hessian_vector(x, vector):
        return onp.asarray(jit_hvp(rep(x), rep(vector), jax_data), dtype=onp.float64)

    res = minimize(objective, onp.asarray(x0, dtype=float),
                   jac=gradient_value, hessp=hessian_vector,
                   method="Newton-CG",
                   options={"disp": False, "xtol": 1e-8, "maxiter": maxiter})
    return res.x, res.fun


def widths_of(a):
    idx = {'f0_1710_width': 6, 'f2_1270_width': 12, 'f2_1525_width': 24,
           'f2_2150_width': 36, 'f2_2340_width': 48}
    return {l: float(a[i]) for l, i in idx.items()}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxiter", type=int, default=60)
    args = ap.parse_args()

    config.update("jax_enable_x64", True)
    data = normalize_data(load_data())
    data_size = int(data["data_phi_kk"].shape[0])
    base, _, _ = make_initial_args()
    x0 = make_unphys_base(base)
    mesh = jax.sharding.Mesh(jax.devices(), axis_names=("event",))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    print("NON-PHYSICAL INITIAL PARAMS:")
    for i, v in NONPHYS.items():
        print("   arg%-3d initial = %+.4f" % (i, v))

    free = os.environ.get("RUN_FREE_RANGE") == "1"
    tag = "FREE" if free else "RANGE-ON"
    xfin, nllj = run(x0, data, data_size, mesh, replicated, free, args.maxiter)
    w = widths_of(xfin)
    print("\n==== %s ==== final nll=%.4f" % (tag, nllj))
    for i, v in NONPHYS.items():
        print("   arg%-3d  init=%+.4f  final=%+.4f" % (i, v, xfin[i]))
    print("   widths:", {l: round(v, 4) for l, v in w.items()})
    bad = [l for l, v in w.items() if v <= 0]
    print("   non-positive widths final:", bad if bad else "none")


if __name__ == "__main__":
    main()
