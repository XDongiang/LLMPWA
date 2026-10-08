#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production Newton-CG fit with a selectable amplitude layer.

Same computation as run/fit_script.py (full data, full 65-parameter Newton-CG,
Hessian by columns, covariance, errors), but the amplitude layer and the output
directory come from the environment so the two layers can be run back to back
and compared without touching each other's results:

    HELICITY_LAYER=0|1        amplitude layer
    FIT_OUT_DIR=<dir>         where values/errors/toml are written (default
                              output/fit_layer_<mode>)
    FIT_ITERS=<n>             optional cap on Newton-CG iterations (0 = no cap)

Usage:
    HELICITY_LAYER=0 FIT_OUT_DIR=output/fit_layer_cartesian \
        python run/fit_layer.py
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
    load_data, make_initial_args, normalize_data, save_result, setup_logging,
    shard_data_distributed,
)
from likelihood_function import make_distributed_likelihood  # noqa: E402


def _mesh_context(mesh):
    if hasattr(jax, "set_mesh"):
        return jax.set_mesh(mesh)
    return contextlib.nullcontext()


def main():
    mode = os.environ.get("HELICITY_LAYER", "0")
    out_dir = os.environ.get("FIT_OUT_DIR", "output/fit_layer_%s" % mode)
    fit_iters = int(os.environ.get("FIT_ITERS", "0"))
    os.environ["HELICITY_LAYER"] = mode
    jax.clear_caches()

    config.update("jax_enable_x64", True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logging()
    logger.info("fit_layer start: HELICITY_LAYER=%s out=%s fit_iters=%d",
                mode, out_dir, fit_iters)

    data = normalize_data(load_data())
    data_size = int(data["data_phi_kk"].shape[0])
    mesh = Mesh(jax.devices(), axis_names=("event",))
    initial_args, _, _ = make_initial_args()
    replicated = NamedSharding(mesh, P())

    with _mesh_context(mesh):
        jax_data = shard_data_distributed(data, mesh)
        nll = make_distributed_likelihood(data_size)
        grad_nll = grad(nll, argnums=0)

        def hvp_nll(a, vector, data_arg):
            zero = jax.tree_util.tree_map(jnp.zeros_like, data_arg)
            return jvp(lambda x, d: grad_nll(x, d), (a, data_arg), (vector, zero))[1]

        jit_nll, jit_grad, jit_hvp = jit(nll), jit(grad_nll), jit(hvp_nll)
        rep = lambda v: device_put(jnp.asarray(v), replicated)

        x0 = rep(initial_args)
        logger.info("likelihood smoke: %s", float(jit_nll(x0, jax_data)))

        def objective(x):
            return float(jit_nll(rep(x), jax_data))

        def gradient_value(x):
            return onp.asarray(jit_grad(rep(x), jax_data), dtype=onp.float64)

        def hessian_vector(x, vector):
            return onp.asarray(jit_hvp(rep(x), rep(vector), jax_data), dtype=onp.float64)

        counter = [0]
        t0 = time.time()

        def callback(xk):
            counter[0] += 1
            logger.info("iteration %d: nll=%.12g  (%.1fs)",
                        counter[0], objective(xk), time.time() - t0)

        opts = {"disp": False, "xtol": 1e-8}
        if fit_iters > 0:
            opts["maxiter"] = fit_iters
        result = minimize(
            objective, x0=onp.asarray(initial_args, dtype=onp.float64),
            jac=gradient_value, hessp=hessian_vector,
            method="Newton-CG", callback=callback, options=opts,
        )
        fit_secs = time.time() - t0

        n_parameters = result.x.size
        logger.info("fit done: nit=%d nll=%.12g success=%s (%.1fs); Hessian next",
                    result.nit, result.fun, result.success, fit_secs)
        hessian = onp.column_stack(
            [hessian_vector(result.x, onp.eye(n_parameters)[:, i])
             for i in range(n_parameters)]
        )
        hessian = 0.5 * (hessian + hessian.T)
        try:
            covariance = onp.linalg.inv(hessian)
        except onp.linalg.LinAlgError:
            covariance = onp.linalg.pinv(hessian)
        ferror = onp.sqrt(onp.maximum(onp.real(onp.diag(covariance)), 0.0))

        onp.save(os.path.join(out_dir, "fit_result_values.npy"),
                 onp.asarray(result.x, dtype=onp.float64))
        onp.save(os.path.join(out_dir, "fit_result_errors.npy"), ferror)
        save_result(result.x, ferror, os.path.join(out_dir, "free_params_fitted.toml"))

        summary = {
            "mode": mode, "out_dir": out_dir, "nit": int(result.nit),
            "nll": float(result.fun), "success": bool(result.success),
            "message": str(result.message), "fit_secs": fit_secs,
            "n_eval": counter[0],
            "hessian_cond": float(onp.linalg.cond(hessian)),
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("written %s : %s", out_dir, summary)
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
