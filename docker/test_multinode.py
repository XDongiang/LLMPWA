"""
多节点 JAX 通信测试脚本

使用方式见 run_mulit_node.md，将 your_script.py 替换为 test_multinode.py
"""
import os
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh, NamedSharding


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


def test_basic_info(num_processes, process_id):
    print(f"[Process {process_id}/{num_processes}] "
          f"local devices: {jax.local_devices()}")
    print(f"[Process {process_id}/{num_processes}] "
          f"global devices: {jax.devices()}")
    print(f"[Process {process_id}/{num_processes}] "
          f"local device count: {jax.local_device_count()}, "
          f"global device count: {jax.device_count()}")


def test_allreduce(process_id, num_processes):
    """每个进程贡献一个值，psum 后所有进程应得到相同的全局和"""
    devices = jax.devices()
    mesh = Mesh(devices, ("devices",))

    # 每个 process 贡献本地设备数量个元素，值为 process_id + 1
    num_local = jax.local_device_count()
    local_data = jnp.full((num_local,), float(process_id + 1))

    sharding = NamedSharding(mesh, P("devices",))
    out_sharding = NamedSharding(mesh, P())  # 输出 replicated（全局求和）

    # 用 make_array_from_process_local_data 从各 process 的本地数据构建全局分布式数组
    global_arr = jax.make_array_from_process_local_data(sharding, local_data)

    # jnp.sum 在分片输入 + replicated 输出下，XLA 会自动插入跨设备 all-reduce
    result = jax.jit(
        jnp.sum,
        in_shardings=sharding,
        out_shardings=out_sharding,
    )(global_arr)

    # process p 的每个本地设备贡献 (p+1)，总和 = sum_p (p+1) * num_local
    expected = float(sum(i + 1 for i in range(num_processes)) * num_local)
    actual = float(result)
    ok = abs(actual - expected) < 1e-5
    print(f"[Process {process_id}] allreduce: local={float(process_id + 1):.1f}, "
          f"global_sum={actual:.1f}, expected={expected:.1f}, "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_sharded_matmul(process_id):
    """在所有设备上做分片矩阵乘法，与单设备结果对比"""
    devices = jax.devices()
    n = len(devices)
    mesh = Mesh(devices, ("devices",))

    key = jax.random.PRNGKey(42)
    A = jax.random.normal(key, (n * 8, 64))
    B = jax.random.normal(key, (64, 32))
    ref = A @ B  # 单设备参考结果

    # A 按行分片到各设备，B 复制
    sharding_A = NamedSharding(mesh, P("devices", None))
    sharding_B = NamedSharding(mesh, P(None, None))
    sharding_out = NamedSharding(mesh, P("devices", None))

    with jax.set_mesh(mesh):
        A_s = jax.device_put(A, sharding_A)
        B_s = jax.device_put(B, sharding_B)
        result = jax.jit(
            jnp.dot,
            in_shardings=(sharding_A, sharding_B),
            out_shardings=sharding_out,
        )(A_s, B_s)

    err = float(jnp.max(jnp.abs(result - ref)))
    ok = err < 1e-3
    print(f"[Process {process_id}] sharded matmul: max_abs_err={err:.2e}, "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    num_processes, process_id = init_distributed()

    print(f"\n{'='*50}")
    print(f" JAX Multi-Node Test  (JAX {jax.__version__})")
    print(f"{'='*50}\n")

    test_basic_info(num_processes, process_id)
    print()

    results = []
    results.append(test_allreduce(process_id, num_processes))
    results.append(test_sharded_matmul(process_id))

    print()
    overall = all(results)
    print(f"[Process {process_id}] Overall: {'ALL PASS' if overall else 'SOME TESTS FAILED'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
