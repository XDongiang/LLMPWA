#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full-scale regression for the hybrid helicity amplitude layer.

Runs the REAL pipeline (full `load_data()` + `normalize_data()`, real 10k events
tiled x3, 1.2M MC tiled x3, 450k truth) on the actual device mesh, and compares

    HELICITY_LAYER=0  (Cartesian)   vs   HELICITY_LAYER=1  (helicity)

for: per-event intensity, combined NLL, its gradient, one HVP, plus a few
Newton-CG iterations from the initial parameters.

This is the experiment report section 5 item 3 asks for; the NLL is never
expected to be bit identical, only equal to float64 round-off.

Usage (inside the jax-fit container or a jax env with cwd = analysis root):
    python run/test_helicity_layer_full.py
Env knobs:
    N_REPEAT   tile factor for data/mc (default 3, same as production)
    FIT_ITERS  number of Newton-CG iterations per mode (default 3, 0 disables)
    OUT_JSON   where to drop the machine-readable summary
"""

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
from scipy.optimize import minimize  # noqa: E402

config.update("jax_enable_x64", True)

from base_functions import (  # noqa: E402
    load_data,
    make_initial_args,
    normalize_data,
    shard_data_distributed,
)
from likelihood_function import make_distributed_likelihood  # noqa: E402


def _mesh_context(mesh):
    if hasattr(jax, "set_mesh"):
        return jax.set_mesh(mesh)
    return contextlib.nullcontext()


def evaluate(mode, mesh, data, args, fit_iters):
    """Compile and evaluate one amplitude layer end to end."""
    os.environ["HELICITY_LAYER"] = mode
    jax.clear_caches()

    data_size = int(data["data_phi_kk"].shape[0])
    replicated = NamedSharding(mesh, P())
    out = {"mode": mode}

    with _mesh_context(mesh):
        t0 = time.time()
        jax_data = shard_data_distributed(data, mesh)
        out["shard_secs"] = time.time() - t0

        nll = make_distributed_likelihood(data_size)
        grad_nll = grad(nll, argnums=0)

        def hvp_nll(a, vector, data_arg):
            zero = jax.tree_util.tree_map(jnp.zeros_like, data_arg)
            return jvp(lambda x, d: grad_nll(x, d), (a, data_arg), (vector, zero))[1]

        jit_nll = jit(nll)
        jit_grad = jit(grad_nll)
        jit_hvp = jit(hvp_nll)
        rep = lambda v: device_put(jnp.asarray(v), replicated)

        x0 = rep(args)
        v0 = rep(onp.ones_like(args))

        t0 = time.time()
        value = float(jit_nll(x0, jax_data))
        out["nll"] = value
        out["nll_secs"] = time.time() - t0

        t0 = time.time()
        g = onp.asarray(jit_grad(x0, jax_data), dtype=onp.float64)
        out["grad"] = g.tolist()
        out["grad_secs"] = time.time() - t0

        t0 = time.time()
        hv = onp.asarray(jit_hvp(x0, v0, jax_data), dtype=onp.float64)
        out["hvp"] = hv.tolist()
        out["hvp_secs"] = time.time() - t0

        t0 = time.time()
        intensity = onp.asarray(_event_intensity(jax_data, x0), dtype=onp.float64)
        out["intensity_sum"] = float(intensity.sum())
        out["intensity_max"] = float(intensity.max())
        out["intensity_secs"] = time.time() - t0

        if fit_iters > 0:
            def objective(x):
                return float(jit_nll(rep(x), jax_data))

            def gradient_value(x):
                return onp.asarray(jit_grad(rep(x), jax_data), dtype=onp.float64)

            def hessian_vector(x, vector):
                return onp.asarray(jit_hvp(rep(x), rep(vector), jax_data),
                                   dtype=onp.float64)

            t0 = time.time()
            res = minimize(
                objective,
                x0=onp.asarray(args, dtype=onp.float64),
                jac=gradient_value,
                hessp=hessian_vector,
                method="Newton-CG",
                options={"disp": False, "maxiter": fit_iters, "xtol": 1e-10},
            )
            out["fit_secs"] = time.time() - t0
            out["fit_x"] = onp.asarray(res.x, dtype=onp.float64).tolist()
            out["fit_nll"] = float(res.fun)
            out["fit_nit"] = int(res.nit)
            out["fit_success"] = bool(res.success)

    return out


def _event_intensity(jax_data, args):
    from likelihood_function import event_intensity
    return event_intensity(args, jax_data, "data_")


def cmp_block(name, a, b):
    a = onp.asarray(a, dtype=onp.float64)
    b = onp.asarray(b, dtype=onp.float64)
    scale = max(float(onp.abs(b).max()), 1e-300)
    ad = float(onp.abs(a - b).max())
    return {"name": name, "max_abs": ad, "rel": ad / scale,
            "ok": bool(ad / scale < 1e-13)}


def main():
    n_repeat = int(os.environ.get("N_REPEAT", "3"))
    fit_iters = int(os.environ.get("FIT_ITERS", "3"))
    out_json = os.environ.get("OUT_JSON", "output/helicity_layer_full.json")

    print("=" * 78)
    print("[full-scale regression] devices=%s  N_REPEAT=%d  FIT_ITERS=%d"
          % (jax.devices(), n_repeat, fit_iters))
    print("=" * 78)

    t0 = time.time()
    data = normalize_data(load_data())
    print("loaded+normalised data in %.1fs: n_data=%d n_mc=%d n_truth=%d"
          % (time.time() - t0,
             data["data_phi_kk"].shape[0],
             data["mc_phi_kk"].shape[0],
             data["truth_phi_kk"].shape[0]))
    for k in ("phif0_kk", "phif2_kk"):
        print("   data_%-10s %s   mc_%-10s %s"
              % (k, data["data_" + k].shape, k, data["mc_" + k].shape))

    mesh = Mesh(jax.devices(), axis_names=("event",))
    args, _, _ = make_initial_args()
    print("n_args = %d" % args.size)

    cart = evaluate("0", mesh, data, args, fit_iters)
    hel = evaluate("1", mesh, data, args, fit_iters)

    print("\n--- timings (measured after jit compile) ---")
    for tag, r in (("cartesian", cart), ("helicity", hel)):
        print("  %-10s nll %.2fs  grad %.2fs  hvp %.2fs  intensity %.2fs%s"
              % (tag, r["nll_secs"], r["grad_secs"], r["hvp_secs"],
                 r["intensity_secs"],
                 "  fit %.1fs" % r["fit_secs"] if "fit_secs" in r else ""))

    print("\n--- cartesian vs helicity ---")
    checks = [
        cmp_block("gradient", cart["grad"], hel["grad"]),
        cmp_block("hvp(J 1)", cart["hvp"], hel["hvp"]),
    ]
    nll_rel = abs(cart["nll"] - hel["nll"]) / max(abs(hel["nll"]), 1e-300)
    print("  %-22s %.12g vs %.12g   rel = %.3e   %s"
          % ("combined NLL", cart["nll"], hel["nll"], nll_rel,
             "OK" if nll_rel < 1e-13 else "**MISMATCH**"))
    print("  %-22s sum %.10g vs %.10g   rel = %.3e"
          % ("intensity sum", cart["intensity_sum"], hel["intensity_sum"],
             abs(cart["intensity_sum"] - hel["intensity_sum"])
             / max(abs(hel["intensity_sum"]), 1e-300)))
    for c in checks:
        print("  %-22s max|d| = %.3e   rel = %.3e   %s"
              % (c["name"], c["max_abs"], c["rel"], "OK" if c["ok"] else "**MISMATCH**"))

    ok = nll_rel < 1e-13 and all(c["ok"] for c in checks)

    if fit_iters > 0:
        dx = cmp_block("fit x", cart["fit_x"], hel["fit_x"])
        dnll = abs(cart["fit_nll"] - hel["fit_nll"]) / max(abs(hel["fit_nll"]), 1e-300)
        print("\n--- %d Newton-CG iterations from the same start ---" % fit_iters)
        print("  nit            %d vs %d" % (cart["fit_nit"], hel["fit_nit"]))
        print("  nll            %.12g vs %.12g   rel = %.3e"
              % (cart["fit_nll"], hel["fit_nll"], dnll))
        print("  max|dx|        %.3e   (|x| ~ %.3g)   rel = %.3e   %s"
              % (dx["max_abs"], float(onp.abs(hel["fit_x"]).max()), dx["rel"],
                 "OK" if dx["ok"] else "**MISMATCH**"))
        ok &= dnll < 1e-12

    print("\n=== %s ===" % ("PASS" if ok else "FAIL"))

    summary = {
        "n_repeat": n_repeat, "fit_iters": fit_iters,
        "devices": [str(d) for d in jax.devices()],
        "n_data": int(data["data_phi_kk"].shape[0]),
        "n_mc": int(data["mc_phi_kk"].shape[0]),
        "n_truth": int(data["truth_phi_kk"].shape[0]),
        "cartesian": {k: v for k, v in cart.items() if k not in ("grad", "hvp", "fit_x")},
        "helicity": {k: v for k, v in hel.items() if k not in ("grad", "hvp", "fit_x")},
        "checks": checks + [{"name": "combined NLL", "rel": nll_rel, "ok": bool(nll_rel < 1e-13)}],
        "pass": bool(ok),
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("summary written to %s" % out_json)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
