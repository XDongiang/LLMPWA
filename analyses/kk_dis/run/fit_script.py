import contextlib
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
os.chdir(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)
if _here not in sys.path:
    sys.path.insert(0, _here)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
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
    return Mesh(jax.devices(), axis_names=("event",))


def _mesh_context(mesh):
    if hasattr(jax, "set_mesh"):
        return jax.set_mesh(mesh)
    return contextlib.nullcontext()


def main():
    num_processes, process_id = init_distributed()
    is_chief = process_id == 0
    config.update("jax_enable_x64", True)
    os.makedirs("logs", exist_ok=True)  # setup_logging 写 logs/fit.log 需要目录存在
    logger = setup_logging()

    if is_chief:
        logger.info("starting distributed fit: processes=%d devices=%d", num_processes, len(jax.devices()))

    data = normalize_data(load_data())
    data_size = int(data["data_phi_kk"].shape[0])
    mesh = build_mesh()
    initial_args, _, _ = make_initial_args()
    replicated = NamedSharding(mesh, P())

    with _mesh_context(mesh):
        jax_data = shard_data_distributed(data, mesh)
        nll = make_distributed_likelihood(data_size)
        grad_nll = grad(nll, argnums=0)

        def hvp_nll(args, vector, data_arg):
            zero_data = jax.tree_util.tree_map(jnp.zeros_like, data_arg)
            return jvp(
                lambda a, d: grad_nll(a, d),
                (args, data_arg),
                (vector, zero_data),
            )[1]

        jit_nll = jit(nll)
        jit_grad = jit(grad_nll)
        jit_hvp = jit(hvp_nll)

        def replicated_array(value):
            return device_put(jnp.asarray(value), replicated)

        x0 = replicated_array(initial_args)
        direction0 = replicated_array(onp.zeros_like(initial_args))
        
        smoke_value = jit_nll(x0, jax_data)
        smoke_hvp = jit_hvp(x0, direction0, jax_data)
        if is_chief:
            logger.info("likelihood smoke: value=%s, hvp_shape=%s", float(smoke_value), tuple(smoke_hvp.shape))

        def objective(x):
            return float(jit_nll(replicated_array(x), jax_data))

        def gradient_value(x):
            return onp.asarray(jit_grad(replicated_array(x), jax_data), dtype=onp.float64)

        def hessian_vector(x, vector):
            return onp.asarray(
                jit_hvp(replicated_array(x), replicated_array(vector), jax_data),
                dtype=onp.float64,
            )

        iteration = [0]

        def callback(xk):
            value = objective(xk)
            iteration[0] += 1
            if is_chief:
                logger.info("iteration %d: nll=%.12g", iteration[0], value)

        result = minimize(
            objective,
            x0=onp.asarray(initial_args, dtype=onp.float64),
            jac=gradient_value,
            hessp=hessian_vector,
            method="Newton-CG",
            callback=callback,
            options={"disp": False, "xtol": 1e-8},
        )

        n_parameters = result.x.size
        hessian = onp.column_stack(
            [hessian_vector(result.x, onp.eye(n_parameters, dtype=onp.float64)[:, i])
             for i in range(n_parameters)]
        )
        hessian = 0.5 * (hessian + hessian.T)
        try:
            covariance = onp.linalg.inv(hessian)
        except onp.linalg.LinAlgError:
            covariance = onp.linalg.pinv(hessian)
        diagonal = onp.real(onp.diag(covariance))
        ferror = onp.sqrt(onp.maximum(diagonal, 0.0))

        if is_chief:
            os.makedirs("output/fit", exist_ok=True)
            onp.save("output/fit/fit_result_values.npy", onp.asarray(result.x))
            onp.save("output/fit/fit_result_errors.npy", ferror)
            save_result(result.x, ferror, "output/fit/free_params_fitted.toml")
            logger.info("fit complete: success=%s, nll=%.12g, message=%s", result.success, result.fun, result.message)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
