import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
os.chdir(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)
if _here not in sys.path:
    sys.path.insert(0, _here)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as np
import numpy as onp
from jax import config, device_put, grad, jit, jvp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from scipy.optimize import minimize

from base_functions import (
    load_data,
    make_initial_args,
    normalize_data,
    save_result,
    setup_logging,
    shard_data_distributed,
)
from likelihood_function import make_distributed_likelihood


def init_distributed():
    coordinator = os.environ.get("JAX_COORDINATOR_ADDRESS", "")
    num_processes = int(os.environ.get("JAX_NUM_PROCESSES", 1))
    process_id = int(os.environ.get("JAX_PROCESS_ID", 0))
    if num_processes > 1:
        jax.distributed.initialize(
            coordinator_address=coordinator,
            num_processes=num_processes,
            process_id=process_id,
        )
    return num_processes, process_id


def build_mesh():
    return Mesh(onp.asarray(jax.devices()), axis_names=("event",))


def main():
    num_processes, process_id = init_distributed()
    is_chief = process_id == 0
    config.update("jax_enable_x64", True)
    logger = setup_logging()

    if is_chief:
        logger.info(
            "Starting distributed fit with %d process(es), %d global device(s)",
            num_processes,
            jax.device_count(),
        )

    data = normalize_data(load_data())
    data_size = len(data["data_phi_kk"])
    mesh = build_mesh()
    mesh_context = jax.set_mesh(mesh) if hasattr(jax, "set_mesh") else mesh

    with mesh_context:
        jax_data = shard_data_distributed(data, mesh)
        nll = make_distributed_likelihood(data_size)
        grad_fn = grad(nll, argnums=0)

        def hvp_fn(args, vector, data_arg):
            return jvp(
                lambda x: grad_fn(x, data_arg),
                (args,),
                (vector,),
            )[1]

        jit_likelihood = jit(nll)
        jit_grad = jit(grad_fn)
        jit_hvp = jit(hvp_fn)

        args_list, _, _ = make_initial_args()
        parameter_sharding = NamedSharding(mesh, P())

        def parameters(value):
            return device_put(np.asarray(value), parameter_sharding)

        initial_device_args = parameters(args_list)
        smoke_value = float(jit_likelihood(initial_device_args, jax_data))
        smoke_hvp = jit_hvp(
            initial_device_args,
            parameters(onp.ones_like(args_list)),
            jax_data,
        )
        smoke_hvp.block_until_ready()
        if is_chief:
            logger.info(
                "Compiled likelihood/HVP smoke: nll=%.12g, hvp_shape=%s, data_size=%d",
                smoke_value,
                smoke_hvp.shape,
                data_size,
            )

        iteration = [0]
        start_time = time.time()

        def objective(x):
            return float(jit_likelihood(parameters(x), jax_data))

        def gradient(x):
            return onp.asarray(jit_grad(parameters(x), jax_data))

        def hessian_product(x, vector):
            return onp.asarray(
                jit_hvp(parameters(x), parameters(vector), jax_data)
            )

        def callback(x):
            value = objective(x)
            iteration[0] += 1
            if is_chief:
                logger.info(
                    "Iteration %d: nll=%.12g elapsed=%.1fs",
                    iteration[0],
                    value,
                    time.time() - start_time,
                )

        result = minimize(
            fun=objective,
            x0=onp.asarray(args_list),
            jac=gradient,
            hessp=hessian_product,
            method="Newton-CG",
            callback=callback,
            options={"disp": False, "xtol": 1e-8},
        )

        fitted_args = onp.asarray(result.x)
        columns = []
        for index in range(fitted_args.size):
            basis = onp.zeros_like(fitted_args)
            basis[index] = 1.0
            columns.append(hessian_product(fitted_args, basis))
        hessian = onp.column_stack(columns)
        hessian = 0.5 * (hessian + hessian.T)
        covariance = onp.linalg.inv(hessian)
        ferror = onp.sqrt(onp.diag(covariance))

    if is_chief:
        output_dir = os.path.join("output", "fit")
        os.makedirs(output_dir, exist_ok=True)
        onp.save(os.path.join(output_dir, "fit_result_values.npy"), fitted_args)
        onp.save(os.path.join(output_dir, "fit_result_errors.npy"), ferror)
        save_result(
            fitted_args,
            ferror,
            os.path.join(output_dir, "free_params_fitted.toml"),
        )
        logger.info(
            "Fit finished: success=%s status=%d nll=%.12g message=%s",
            result.success,
            result.status,
            result.fun,
            result.message,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
