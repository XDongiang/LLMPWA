#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Demonstrate the lambda = 0 path on a 3-component (P2-1) tensor cache.

With a k=2 cache the helicity layer can only reach lambda = +-1: the lambda=0
slot produced by `to_helicity` is exactly zero.  With a k=3 cache the *same*
code fills that slot in.  This script runs the real pipeline blocks (couplings +
propagators, with the real parameters) on each dataset and reports how the
intensity splits over the three helicity amplitudes.

Usage:
    KK_DATA_ROOT=data/tensor3 N_REPEAT=1 python run/check_lambda0_cache.py
"""

import os
import sys

import numpy as onp

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
os.chdir(_root)
for _p in (_root, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from jax import config  # noqa: E402

config.update("jax_enable_x64", True)

import base_functions as B  # noqa: E402
import likelihood_function as L  # noqa: E402


def main():
    root = os.environ.get("KK_DATA_ROOT", "data")
    print("KK_DATA_ROOT = %s" % root)
    data = B.normalize_data(B.load_data())
    args, _, _ = B.make_initial_args()
    os.environ["HELICITY_LAYER"] = "1"

    print("data_phif0_kk %s   data_phif2_kk %s"
          % (onp.asarray(data["data_phif0_kk"]).shape,
             onp.asarray(data["data_phif2_kk"]).shape))
    k = onp.asarray(data["data_phif0_kk"]).shape[-1]
    print("stored components k = %d  ->  lambda=0 %s\n"
          % (k, "AVAILABLE" if k >= 3 else "absent (slot is exactly 0)"))

    comps = L._components_for_prefix(args, data, "data_")
    print("pipeline blocks: %d   block shape %s"
          % (len(comps), onp.asarray(comps[0]).shape))
    print("%-9s %-16s %-12s %-12s %-12s" % ("block", "sum|a|^2", "share +1", "share 0", "share -1"))
    tot = onp.zeros(3)
    per_block = []
    for ci, c in enumerate(comps):
        # `_components_for_prefix` returns the cache basis; the EPS_A rotation
        # is applied downstream (report 8.6), so do it explicitly here.
        c = B.apply_helicity(c)
        re, im = onp.asarray(c[0]), onp.asarray(c[1])
        a2 = re ** 2 + im ** 2                       # (l, e, 3)
        s = a2.sum(axis=tuple(range(a2.ndim - 1)))
        tot += s
        per_block.append(s)
        share = s / max(s.sum(), 1e-300)
        print("%-9d %-16.6e %-12.6f %-12.6f %-12.6f"
              % (ci, s.sum(), share[0], share[1], share[2]))
    share = tot / max(tot.sum(), 1e-300)
    print("%-9s %-16.6e %-12.6f %-12.6f %-12.6f"
          % ("TOTAL", tot.sum(), share[0], share[1], share[2]))

    # also per-channel, using unit couplings, to separate the 7 tensor channels
    print("\nper-channel (unit coupling, unit propagator):")
    print("%-9s %-12s %-12s %-12s" % ("channel", "|a+1|^2", "|a0|^2", "|a-1|^2"))
    names0 = ["phif001", "phif021"]
    names2 = ["phif201", "phif221", "phif222", "phif223", "phif243"]
    for nm, cat, idx in ([(n, "data_phif0_kk", i) for i, n in enumerate(names0)]
                         + [(n, "data_phif2_kk", i) for i, n in enumerate(names2)]):
        t = onp.asarray(data[cat])[idx][None, :, :]          # (i=1, e, k)
        # dplex_dconstruct(r, theta) stacks (Re, Im): (1, 0) is a unit coupling
        const = onp.ones((1, 1))
        theta = onp.zeros((1, 1))
        # propagators must be a dplex pair (Re, Im), shape (2, l, e)
        prop = onp.stack([onp.ones((1, t.shape[1])), onp.zeros((1, t.shape[1]))])
        c = L._coupled_components(t, const, theta, prop)
        c = B.apply_helicity(c)
        re, im = onp.asarray(c[0]), onp.asarray(c[1])
        s = (re ** 2 + im ** 2).sum(axis=(0, 1))
        share = s / max(s.sum(), 1e-300)
        print("%-9s %-12.6f %-12.6f %-12.6f" % (nm, share[0], share[1], share[2]))

    ie = onp.asarray(L.event_intensity(args, data, "data_"), dtype=onp.float64)
    print("\nper-event intensity (lambda=+-1 only): n=%d  min=%.6g  max=%.6g"
          % (ie.size, ie.min(), ie.max()))
    print("NOTE: lambda=0 is deliberately NOT added to the intensity -- an")
    print("      unpolarised transverse psi populates only lambda=+-1.")


if __name__ == "__main__":
    main()
