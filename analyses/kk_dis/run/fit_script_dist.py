"""
kk_dis 分布式拟合主入口(只跑在一个进程的 optimizer 方案)。

核心设计(规避 SciPy Newton-CG 在分布式下的 CG 路径分叉死锁):
    - 只在 chief (process 0) 上跑 scipy.optimize.minimize
    - 其它进程 (workers) 不跑 SciPy, 而是进入 serve_optimizer_requests():
      循环用 multihost_utils.broadcast_one_to_all 接收 chief 广播的
      "optimizer 请求" (objective / gradient / hessp / stop), 然后执行对应
      的 JAX 计算(collectives 与 chief 严格一一对应)。
    - 每次 optimizer 需要 JAX 计算时, chief 先 broadcast 请求, 再在本端执行,
      从而保证两端 collectives 完全同步。

因此即使 Newton-CG 的 CG 子迭代次数随机变化, 两端也始终同步, 不会死锁。

本地单进程 (JAX_NUM_PROCESSES=1) 时退化为普通 SciPy 拟合, 行为与原 fit_script 一致。

用法:
    docker run ... jax-fit:latest /opt/env/bin/python run/fit_script_dist.py
"""
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
import numpy as onp
from jax.experimental import multihost_utils
from jax import config, device_put, grad, jit, jvp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from scipy.optimize import minimize

from base_functions import (
    load_data,
    make_initial_args,
    normalize_data,
    np,
    save_result,
    setup_logging,
    shard_data_distributed,
)
from likelihood_function import make_distributed_likelihood


OP_STOP = 0
OP_OBJECTIVE = 1
OP_GRADIENT = 2
OP_HESSIAN_PRODUCT = 3


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


def broadcast_optimizer_request(op, x, vector):
    packet = {
        "op": onp.asarray(op, dtype=onp.int32),
        "x": onp.asarray(x, dtype=onp.float64),
        "vector": onp.asarray(vector, dtype=onp.float64),
    }
    return multihost_utils.broadcast_one_to_all(packet)


def main():
    num_processes, process_id = init_distributed()
    is_chief = process_id == 0
    config.update("jax_enable_x64", True)
    logger = setup_logging()

    if is_chief:
        logger.info(
            "Starting distributed fit with %d process(es), %d device(s)",
            num_processes,
            len(jax.devices()),
        )

    data = normalize_data(load_data())
    data_size = len(data["data_phi_kk"])
    mesh = build_mesh()

    with jax.set_mesh(mesh):
        jax_data = shard_data_distributed(data, mesh)
        nll = make_distributed_likelihood(data_size)
        grad_fn = grad(nll, argnums=0)

        def hvp_fn(args, vector, data_arg):
            return jvp(
                lambda varied_args: grad_fn(varied_args, data_arg),
                (args,),
                (vector,),
            )[1]

        jit_likelihood = jit(nll)
        jit_grad = jit(grad_fn)
        jit_hvp = jit(hvp_fn)

        args_list, _, _ = make_initial_args()

        # Optional overrides for multi-start comparison:
        #   KKDIS_X0_NPY   -> path to a .npy of the initial parameter vector
        #   KKDIS_OUTDIR   -> subdirectory under output/ for this run's artifacts
        _x0_env = os.environ.get("KKDIS_X0_NPY", "")
        if _x0_env:
            args_list = onp.asarray(
                onp.load(_x0_env), dtype=onp.float64
            )
            if is_chief:
                logger.info("Overriding initial args from %s (n=%d)", _x0_env, args_list.size)
        param_sharding = NamedSharding(mesh, P())

        def put_param(value):
            return device_put(np.asarray(value), param_sharding)

        zero_vector = onp.zeros_like(args_list)

        initial_args = put_param(args_list)
        smoke_value = float(jit_likelihood(initial_args, jax_data))
        smoke_hvp = onp.asarray(
            jit_hvp(initial_args, put_param(onp.ones_like(args_list)), jax_data)
        )
        if is_chief:
            logger.info(
                "JIT smoke complete: likelihood=%.12g, HVP shape=%s",
                smoke_value,
                smoke_hvp.shape,
            )

        if num_processes > 1:
            multihost_utils.sync_global_devices("fit-smoke-complete")

        iteration = 0
        start_time = time.time()

        def raw_objective(x):
            return float(jit_likelihood(put_param(x), jax_data))

        def raw_objective_grad(x):
            return onp.asarray(jit_grad(put_param(x), jax_data))

        def raw_objective_hvp(x, vector):
            return onp.asarray(
                jit_hvp(put_param(x), put_param(vector), jax_data)
            )

        def serve_optimizer_requests():
            while True:
                packet = broadcast_optimizer_request(
                    OP_STOP,
                    zero_vector,
                    zero_vector,
                )
                op = int(onp.asarray(packet["op"]))
                if op == OP_STOP:
                    break
                x = onp.asarray(packet["x"])
                vector = onp.asarray(packet["vector"])
                if op == OP_OBJECTIVE:
                    raw_objective(x)
                elif op == OP_GRADIENT:
                    raw_objective_grad(x)
                elif op == OP_HESSIAN_PRODUCT:
                    raw_objective_hvp(x, vector)
                else:
                    raise ValueError("Unknown optimizer request op: {}".format(op))

        if num_processes > 1 and not is_chief:
            serve_optimizer_requests()
            return 0

        def dispatch(op, x, vector=None):
            if num_processes > 1:
                broadcast_optimizer_request(
                    op,
                    x,
                    zero_vector if vector is None else vector,
                )

        def objective(x):
            dispatch(OP_OBJECTIVE, x)
            return raw_objective(x)

        def objective_grad(x):
            dispatch(OP_GRADIENT, x)
            return raw_objective_grad(x)

        def objective_hvp(x, vector):
            dispatch(OP_HESSIAN_PRODUCT, x, vector)
            return raw_objective_hvp(x, vector)

        def callback(x):
            nonlocal iteration
            iteration += 1
            value = objective(x)
            if is_chief:
                logger.info(
                    "Iteration %d: likelihood=%.12g elapsed=%.1fs",
                    iteration,
                    value,
                    time.time() - start_time,
                )

        try:
            _maxiter = int(os.environ.get("KKDIS_MAXITER", "0"))
            _opts = {"disp": False, "xtol": 1.0e-8}
            if _maxiter > 0:
                _opts["maxiter"] = _maxiter
            result = minimize(
                fun=objective,
                x0=onp.asarray(args_list),
                jac=objective_grad,
                hessp=objective_hvp,
                method="Newton-CG",
                callback=callback,
                options=_opts,
            )

            fitted_args = onp.asarray(result.x)
            basis = onp.eye(fitted_args.size)
            hessian = onp.column_stack(
                [
                    objective_hvp(fitted_args, basis[:, i])
                    for i in range(fitted_args.size)
                ]
            )
            hessian = 0.5 * (hessian + hessian.T)
            covariance = onp.linalg.inv(hessian)
            ferror = onp.sqrt(onp.diag(covariance))
        finally:
            if num_processes > 1:
                broadcast_optimizer_request(OP_STOP, zero_vector, zero_vector)

        if is_chief:
            output_dir = os.path.join(
                "output", os.environ.get("KKDIS_OUTDIR", "fit")
            )
            os.makedirs(output_dir, exist_ok=True)
            onp.save(os.path.join(output_dir, "fit_result_values.npy"), fitted_args)
            onp.save(os.path.join(output_dir, "fit_result_errors.npy"), ferror)
            save_result(
                fitted_args,
                ferror,
                os.path.join(output_dir, "free_params_fitted.toml"),
            )
            logger.info(
                "Fit finished: success=%s status=%d likelihood=%.12g iterations=%s",
                result.success,
                result.status,
                float(result.fun),
                getattr(result, "nit", "unknown"),
            )
            logger.info("Results written to %s", output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
