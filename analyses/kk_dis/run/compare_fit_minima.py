#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Are the two amplitude layers the SAME function?

The full Newton-CG fits of `run/fit_layer.py` stop at slightly different points
(139 vs 168 iterations, NLL differs by 3e-4).  This script decides whether that
is a *function* difference or a *stopping-point* difference, by evaluating the
2x2 cross table

                    x_cartesian        x_helicity
    NLL_cartesian   A                  B
    NLL_helicity    C                  D

If A == C and B == D (to float64 round-off) the two layers are the same function
and the fitted-point difference is pure optimiser noise.  The Hessian is
ill-conditioned (~1e9) and the Newton-CG tolerance acts on the step, so a
round-off-level difference in the gradient is enough to stop elsewhere on a flat
ridge.

Usage:
    python run/compare_fit_minima.py \
        --cartesian output/fit_layer_0 --helicity output/fit_layer_1
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

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
from jax import config, device_put, grad, jit  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

config.update("jax_enable_x64", True)

from base_functions import load_data, normalize_data, shard_data_distributed  # noqa: E402
from likelihood_function import make_distributed_likelihood  # noqa: E402


def _mesh_context(mesh):
    if hasattr(jax, "set_mesh"):
        return jax.set_mesh(mesh)
    return contextlib.nullcontext()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cartesian", default="output/fit_layer_0")
    ap.add_argument("--helicity", default="output/fit_layer_1")
    ap.add_argument("--out", default="output/fit_minima_cross.json")
    args = ap.parse_args()

    x_cart = onp.load(os.path.join(args.cartesian, "fit_result_values.npy"))
    x_hel = onp.load(os.path.join(args.helicity, "fit_result_values.npy"))
    e_cart = onp.load(os.path.join(args.cartesian, "fit_result_errors.npy"))
    e_hel = onp.load(os.path.join(args.helicity, "fit_result_errors.npy"))

    data = normalize_data(load_data())
    data_size = int(data["data_phi_kk"].shape[0])
    mesh = Mesh(jax.devices(), axis_names=("event",))
    replicated = NamedSharding(mesh, P())

    table = {}
    with _mesh_context(mesh):
        jax_data = shard_data_distributed(data, mesh)
        rep = lambda v: device_put(jnp.asarray(v), replicated)
        for mode in ("0", "1"):
            os.environ["HELICITY_LAYER"] = mode
            jax.clear_caches()
            nll = make_distributed_likelihood(data_size)
            jit_nll = jit(nll)
            jit_grad = jit(grad(nll, argnums=0))
            for tag, x in (("x_cartesian", x_cart), ("x_helicity", x_hel)):
                table["NLL_%s(%s)" % ({"0": "cartesian", "1": "helicity"}[mode], tag)] = \
                    float(jit_nll(rep(x), jax_data))
                g = onp.asarray(jit_grad(rep(x), jax_data), dtype=onp.float64)
                table["|grad|inf_%s(%s)" % ({"0": "cartesian", "1": "helicity"}[mode], tag)] = \
                    float(onp.abs(g).max())

    print("=" * 74)
    print("NLL cross table (rows = amplitude layer, cols = fitted point)")
    print("=" * 74)
    hdr = "%-12s %-24s %-24s" % ("", "x_cartesian", "x_helicity")
    print(hdr)
    for mode in ("cartesian", "helicity"):
        print("%-12s %-24.12g %-24.12g"
              % (mode, table["NLL_%s(x_cartesian)" % mode],
                 table["NLL_%s(x_helicity)" % mode]))

    same_fn = (abs(table["NLL_cartesian(x_cartesian)"] - table["NLL_helicity(x_cartesian)"])
               < 1e-9 and
               abs(table["NLL_cartesian(x_helicity)"] - table["NLL_helicity(x_helicity)"])
               < 1e-9)
    gap = abs(table["NLL_cartesian(x_cartesian)"] - table["NLL_helicity(x_helicity)"])
    print("\nsame function (column-wise agreement < 1e-9): %s" % same_fn)
    print("NLL gap between the two fitted points      : %.6e" % gap)

    sig = onp.maximum(e_hel, 1e-300)
    dv = onp.abs(x_hel - x_cart)
    print("\nfitted points")
    print("  max|dx|            = %.6e" % dv.max())
    print("  max |dx| / sigma   = %.6e" % onp.max(dv / sig))
    print("  median |dx|/sigma  = %.6e" % onp.median(dv / sig))
    print("  max |dsigma|/sigma = %.6e" % onp.max(onp.abs(e_hel - e_cart) / sig))

    summary = dict(table=table, same_function=bool(same_fn), nll_gap=float(gap),
                   max_dx_over_sigma=float(onp.max(dv / sig)),
                   median_dx_over_sigma=float(onp.median(dv / sig)),
                   max_dsigma_over_sigma=float(onp.max(onp.abs(e_hel - e_cart) / sig)))
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nwritten %s" % args.out)


if __name__ == "__main__":
    main()
