#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment: test whether REMOVING the range (gaussian/fraction) penalties lets
the fit find non-physical solutions.

We run several perturbed starts from the baseline free_params.toml, each with a
bounded number of Newton-CG iterations, and compare the final resonance
mass/width of:
  * the STANDARD likelihood (range penalties ON), and
  * the FREE-RANGE likelihood (likelihood_function_free.py: penalties OFF).

The physical-resonance windows (from free_params.toml `range`) define what we
check: if the free fit drives a mass/width outside a sensible window (or to an
absurd value like a negative/zero-width or a mass that wanders far), that is a
non-physical solution the range constraint was meant to prevent.

Runs on CPU (subset) for speed; each start limited to --maxiter iterations.
"""
import argparse
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

# Broad "physically sensible" bands used to detect clearly non-physical values.
# A mass/width deep inside these is acceptable; outside (e.g. non-positive width,
# mass far from any resonance) means the unconstrained fit found a non-physical
# solution the range penalty was meant to suppress.
RESONANCE_WINDOWS = {
    "f0_980_mass": (0.90, 1.30),
    "f0_1710_mass": (1.40, 2.20),
    "f0_1710_width": (0.01, 1.50),
    "f2_1270_mass": (1.10, 1.60),
    "f2_1270_width": (0.01, 1.50),
    "f2_1525_mass": (1.30, 1.90),
    "f2_1525_width": (0.01, 1.50),
    "f2_2150_mass": (1.90, 2.60),
    "f2_2150_width": (0.01, 1.50),
    "f2_2340_mass": (2.10, 2.90),
    "f2_2340_width": (0.01, 1.50),
}
# width must be strictly positive for a Breit-Wigner to be physical
WIDTH_POSITIVE = ["f0_1710_width", "f2_1270_width", "f2_1525_width", "f2_2150_width", "f2_2340_width"]


def perturb(base, seed, disturb=50.0):
    rng = onp.random.RandomState(seed)
    return base * ((rng.rand(base.shape[0]) - 0.5) / disturb + 1.0)


def make_likelihood(data, data_size, free):
    if free:
        from likelihood_function_free import make_distributed_likelihood
    else:
        from likelihood_function import make_distributed_likelihood
    return make_distributed_likelihood(data_size)


def run_start(base, seed, data, data_size, mesh, replicated, free, maxiter):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    jax_data = shard_data_distributed(data, mesh)
    nll = make_likelihood(data, data_size, free)
    grad_nll = grad(nll, argnums=0)

    def hvp_nll(args, vector, data_arg):
        zero = jax.tree_util.tree_map(jnp.zeros_like, data_arg)
        return jvp(lambda a, d: grad_nll(a, d), (args, data_arg), (vector, zero))[1]

    jit_nll = jit(nll)
    jit_grad = jit(grad_nll)
    jit_hvp = jit(hvp_nll)

    def rep(v):
        return device_put(jnp.asarray(v), replicated)

    x0 = perturb(base, seed)
    x0r = rep(x0)
    direction0 = rep(onp.zeros_like(base))

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
    return res.x, res.fun, x0


def param_report(args):
    """Return {label: (value, window) } for the resonance masses/widths."""
    m = {}
    m["f0_980_mass"] = args[0]
    m["f0_1710_mass"] = args[5]
    m["f0_1710_width"] = args[6]
    m["f2_1270_mass"] = args[11]
    m["f2_1270_width"] = args[12]
    m["f2_1525_mass"] = args[23]
    m["f2_1525_width"] = args[24]
    m["f2_2150_mass"] = args[35]
    m["f2_2150_width"] = args[36]
    m["f2_2340_mass"] = args[47]
    m["f2_2340_width"] = args[48]
    return m


def check_violations(args):
    rep = param_report(args)
    viol = []
    for label, (lo, hi) in RESONANCE_WINDOWS.items():
        v = rep[label]
        if not (lo <= v <= hi):
            reason = "outside physical band"
            if label in WIDTH_POSITIVE and v <= 0.0:
                reason = "NON-POSITIVE width (non-physical)"
            viol.append("%s=%.4f (band %.3f-%.3f, %s)" % (label, v, lo, hi, reason))
    return rep, viol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    config.update("jax_enable_x64", True)
    data = normalize_data(load_data())
    data_size = int(data["data_phi_kk"].shape[0])
    base, _, _ = make_initial_args()
    mesh = jax.sharding.Mesh(jax.devices(), axis_names=("event",))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    for free in (False, True):
        tag = "FREE" if free else "RANGE-ON"
        print("\n########## %s ##########" % tag)
        outs = []
        for s in range(args.starts):
            seed = args.seed + s
            xfin, nll, x0 = run_start(base, seed, data, data_size, mesh, replicated, free, args.maxiter)
            rep, viol = check_violations(xfin)
            flag = "  <-- NON-PHYSICAL" if viol else "  ok"
            print("  start %d seed %d nll=%.4f%s" % (s, seed, nll, flag))
            for lbl, (lo, hi) in RESONANCE_WINDOWS.items():
                v = rep[lbl]
                mark = " *" if not (lo <= v <= hi) else ""
                print("        %-14s = %8.4f   [%.3f, %.3f]%s" % (lbl, v, lo, hi, mark))
            if viol:
                outs.append((tag, s, viol))
        if not outs:
            print("  => no non-physical solutions found in %d starts" % args.starts)
        else:
            print("  => %d non-physical solution(s)" % len(outs))


if __name__ == "__main__":
    main()
