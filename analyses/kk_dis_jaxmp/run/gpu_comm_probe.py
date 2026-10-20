"""多卡通信探针：验证 JAX 在单节点多 GPU 上的集合通信是否真的走 GPU 间链路。

跑法（在 sbatch 里）:
    python3 run/gpu_comm_probe.py            # 单进程，看所有可见 GPU
    python3 run/gpu_comm_probe.py --mp 4     # 4 进程，每进程一卡，走分布式初始化

设计要点：**测试的是 fit_object_kk.py 真正用的那条路径** —— 对分片在事例轴上的
数组做普通 ``jnp.sum``，由 GSPMD 自动插入跨设备 all-reduce（代码里没有显式 psum）。
"""
import argparse
import os
import sys
import time


def banner(msg):
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70, flush=True)


def _sum_bench(jnp, jax, sharded, reps=10):
    """对分片数组在分片轴上求和（= 真实代码的归约路径），返回 (ms, 校验值)。"""
    f = jax.jit(lambda t: jnp.sum(t, axis=0))
    out = f(sharded)
    out.block_until_ready()  # 编译 + warmup
    t0 = time.time()
    for _ in range(reps):
        out = f(sharded)
        out.block_until_ready()
    dt = (time.time() - t0) / reps
    return dt, float(jax.device_get(out)[0])


def probe_single_process():
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    banner("1. device discovery (single process)")
    devs = jax.devices()
    print("jax version :", jax.__version__)
    print("backend     :", jax.default_backend())
    print("num devices :", len(devs))
    for d in devs:
        print("   ", d, "| platform:", d.platform, "| id:", d.id)

    if len(devs) < 2:
        print("\n[!] 只有一个设备可见 —— 无法做卡间通信测试。")
        print("    （单卡作业里这是正常的；多卡测试要 --gres=gpu:4）")
        return 0

    n = len(devs)
    mesh = Mesh(devs, axis_names=("event",))
    sh = NamedSharding(mesh, P("event"))

    banner("2. cross-device reduce over the sharded axis (the real code path)")
    # 每张卡放自己的 rank 值，沿分片轴求和 => 必须跨设备通信才能得到 0+1+2+3=6。
    x = jax.device_put(jnp.arange(n, dtype=jnp.float32), sh)
    print("input       :", jax.device_get(x))
    print("shards      :", [(s.device.id, tuple(s.data.shape)) for s in x.addressable_shards])
    got = jax.device_get(jax.jit(lambda t: jnp.sum(t))(x))
    want = float(sum(range(n)))
    print(f"reduce sum  : {got}  (期望 {want:.0f})")
    print("reduce sum  :", "PASS" if abs(float(got) - want) < 1e-6 else "FAIL")

    banner("3. mesh sharding: each device holds only its own slice")
    a = jax.device_put(jnp.ones((n * 8, 16), dtype=jnp.float32), sh)
    print("array sharding:", a.sharding)
    print("is fully addressable:", a.is_fully_addressable)
    print("shards per device   :", len(a.addressable_shards))
    for s in a.addressable_shards:
        print("    device", s.device.id, "holds slice", tuple(s.data.shape))
    ssum = jax.device_get(jax.jit(lambda t: jnp.sum(t, axis=0))(a))
    print("sum over sharded axis:", ssum[:4], "... (期望 32) ->",
          "PASS" if abs(float(ssum[0]) - 32.0) < 1e-6 else "FAIL")

    banner("4. measured cross-device bandwidth (sharded-axis reduce)")
    for mb in (8, 64, 256):
        nelem = mb * 1024 * 1024 // 4
        big = jax.device_put(jnp.ones((nelem, n), dtype=jnp.float32), sh)
        dt, chk = _sum_bench(jnp, jax, big)
        gbs = (mb / 1024.0) * 2 * (n - 1) / n / dt
        print(f"   {mb:4d} MiB: {dt*1000:8.2f} ms/iter  ~{gbs:7.1f} GB/s (有效)  check={chk:.0f}")

    banner("5. XLA / CUDA build info")
    try:
        from jax._src import xla_bridge
        print("xla platform_version:", xla_bridge.get_backend().platform_version)
    except Exception as e:
        print("(platform_version 读取失败:", e, ")")
    return 0


def probe_multi_process(num_procs):
    """每进程一卡；由启动器设置 JAX_* 环境变量。"""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    rank = int(os.environ.get("JAX_PROCESS_ID", "0"))
    coord = os.environ.get("JAX_COORDINATOR_ADDRESS", "")
    nproc = int(os.environ.get("JAX_NUM_PROCESSES", str(num_procs)))

    banner(f"multi-process init: rank={rank}/{nproc} coord={coord}")
    if not coord:
        print("[!] JAX_COORDINATOR_ADDRESS 未设置")
        return 1

    t0 = time.time()
    jax.distributed.initialize(coord, nproc, rank)
    print(f"distributed.initialize ok in {time.time() - t0:.1f}s")
    devs = jax.devices()
    print("global devices:", len(devs), devs)
    print("process_index :", jax.process_index(), "of", jax.process_count())
    print("local devices :", jax.local_devices())

    mesh = Mesh(devs, axis_names=("x",))
    sh = NamedSharding(mesh, P("x"))

    # 关键：全局数组在所有进程上必须**完全一致**（JAX 会校验），
    # 每个进程只持有自己那一片，归约正确 => 跨进程通信真的发生了。
    global_arr = jnp.arange(nproc, dtype=jnp.float32)
    sharded = jax.device_put(global_arr, sh)
    # 不能对跨进程的数组直接 device_get（它不是本进程完全可寻址的），
    # 这本身就是"真分片"的证据。用 addressable_shards / allgather 来看。
    print("fully addressable here:", sharded.is_fully_addressable, "(跨进程时应为 False)")
    print("this rank holds:", [(s.device.id, tuple(s.data.shape)) for s in sharded.addressable_shards])
    from jax.experimental import multihost_utils
    gathered = multihost_utils.process_allgather(sharded, tiled=True)
    print("allgather ->", jax.device_get(gathered))

    total = jax.device_get(jax.jit(lambda t: jnp.sum(t))(sharded))
    want = float(sum(range(nproc)))
    ok = abs(float(total) - want) < 1e-6
    print(f"cross-process reduce: got={total} want={want} -> {'PASS' if ok else 'FAIL'}")

    banner("cross-process bandwidth (sharded-axis reduce)")
    nelem = 64 * 1024 * 1024 // 4
    big = jax.device_put(jnp.ones((nelem, nproc), dtype=jnp.float32), sh)
    dt, chk = _sum_bench(jnp, jax, big, reps=5)
    gbs = (64 / 1024.0) * 2 * (nproc - 1) / nproc / dt
    print(f"   64 MiB: {dt*1000:.2f} ms/iter ~{gbs:.1f} GB/s (有效)  check={chk:.0f}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp", type=int, default=0,
                    help="多进程模式：总进程数（每进程一卡）")
    args = ap.parse_args()
    if args.mp and args.mp > 1:
        return probe_multi_process(args.mp)
    return probe_single_process()


if __name__ == "__main__":
    sys.exit(main())
