"""真正的卡间带宽基准。

方法论要点（前面几版踩过的坑，都在这里修掉了）:
  * 必须 jit 后再计时，否则测到的是 trace/compile 开销（表现为与 size 无关的常数时间）。
  * 分片数组必须作为 **jit 的显式参数**传入，不能闭包捕获 —— 闭包里的 jax 数组会被
    编成 jaxpr 常量，在分布式下直接报 "spans non-addressable devices"。
    （这正是本分析 README 里那条"唯一必要的源码改动"的同一个坑。）
  * `multihost_utils.process_allgather` 是跨**进程**原语；设备级 all-gather 用
    `jax.lax.all_gather`。
  * 对分片轴做 sum 只会 all-reduce 一个极小的 (nproc,) 向量，测不到带宽。

测三个真的搬数据的原语，给出 algbw（按用户数据量）与 busbw（按链路实际流量）:
  1. all-gather : 每卡把分片发给所有卡，最终每卡持有全集。
  2. all-reduce : ring 算法，每卡流量 2(n-1)/n。
  3. p2p ring   : 每卡把整块发给下一个 rank，纯链路带宽。
"""
import argparse
import os
import sys
import time


def banner(m):
    print("\n" + "=" * 70)
    print(m)
    print("=" * 70, flush=True)


def _timeit(fn, arg, reps=20, warmup=5):
    for _ in range(warmup):
        fn(arg).block_until_ready()
    t0 = time.time()
    for _ in range(reps):
        fn(arg).block_until_ready()
    return (time.time() - t0) / reps


def bench(jax, jnp, mesh, sh, n, sizes_mib, label):
    # v0.11: shard_map 在 jax.experimental；参数名是 check_rep（老版本叫 check_vma）。
    try:
        from jax.experimental.shard_map import shard_map
    except ImportError:  # pragma: no cover
        from jax.shard_map import shard_map
    import inspect
    _params = set(inspect.signature(shard_map).parameters)
    _rep_kw = "check_rep" if "check_rep" in _params else "check_vma"
    from jax.sharding import PartitionSpec as P

    banner(f"real cross-device bandwidth [{label}]  n_devices={n}")

    gather = jax.jit(shard_map(
        lambda x: jax.lax.all_gather(x, "x", axis=0, tiled=True),
        mesh=mesh, in_specs=P("x"), out_specs=P(), **{_rep_kw: False}))
    reduce_ = jax.jit(shard_map(
        lambda x: jax.lax.psum(x, "x"),
        mesh=mesh, in_specs=P("x"), out_specs=P("x")))
    ring = jax.jit(shard_map(
        lambda x: jax.lax.ppermute(x, "x", [(i, (i + 1) % n) for i in range(n)]),
        mesh=mesh, in_specs=P("x"), out_specs=P("x")))

    print(f"{'size':>7} | {'all-gather':>23} | {'all-reduce':>23} | {'p2p ring':>23}")
    print(f"{'MiB':>7} | {'ms':>7} {'algbw':>6} {'busbw':>6} | {'ms':>7} {'algbw':>6} {'busbw':>6} | {'ms':>7} {'algbw':>6} {'busbw':>6}")
    print("-" * 94)

    for mib in sizes_mib:
        nelem = mib * 1024 * 1024 // 4
        a = jax.device_put(jnp.ones((nelem,), dtype=jnp.float32), sh)

        dt_g = _timeit(gather, a)
        dt_r = _timeit(reduce_, a)
        dt_p = _timeit(ring, a)

        g_alg = (mib / 1024) / dt_g
        g_bus = g_alg * (n - 1) / n
        r_alg = (mib / 1024) / dt_r
        r_bus = r_alg * 2 * (n - 1) / n
        p_alg = (mib / 1024) / dt_p
        p_bus = p_alg

        print(f"{mib:7d} | {dt_g*1000:7.3f} {g_alg:6.1f} {g_bus:6.1f} | "
              f"{dt_r*1000:7.3f} {r_alg:6.1f} {r_bus:6.1f} | "
              f"{dt_p*1000:7.3f} {p_alg:6.1f} {p_bus:6.1f}")

    # 正确性：只看**本进程可寻址**的 shard，避免 device_get 跨进程失败。
    # a 的全局形状是 nelem，被分片到 n 个设备 -> 每卡 nelem/n。
    # all-gather 后每卡持有全集 nelem（复制）；all-reduce 后每卡仍是 nelem/n。
    mib = 64
    nelem = mib * 1024 * 1024 // 4
    a = jax.device_put(jnp.ones((nelem,), dtype=jnp.float32), sh)
    g = gather(a)
    gsh = [tuple(s.data.shape) for s in g.addressable_shards]
    r = reduce_(a)
    rsh = [tuple(s.data.shape) for s in r.addressable_shards]
    print(f"\ncorrectness (64 MiB of ones, global nelem={nelem}, n={n}):")
    print(f"  all-gather  shard shapes = {gsh[:2]}  (期望每卡全集 {nelem})")
    print(f"  all-reduce  shard shapes = {rsh[:2]}  (期望每卡 {nelem // n})")
    ok_g = all(s[0] == nelem for s in gsh)
    ok_r = all(s[0] == nelem // n for s in rsh)
    print(f"  -> all-gather {'OK' if ok_g else 'BAD'} | all-reduce {'OK' if ok_r else 'BAD'}")
    print("单位 GB/s = 1e9 B/s。busbw = 链路实际流量口径（NCCL 常用）。")
    return 0 if (ok_g and ok_r) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp", type=int, default=0)
    ap.add_argument("--sizes", type=str, default="8,64,256")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    sizes = [int(s) for s in args.sizes.split(",")]

    if args.mp and args.mp > 1:
        rank = int(os.environ.get("JAX_PROCESS_ID", "0"))
        coord = os.environ.get("JAX_COORDINATOR_ADDRESS", "")
        nproc = int(os.environ.get("JAX_NUM_PROCESSES", str(args.mp)))
        banner(f"distributed init: rank={rank}/{nproc} coord={coord}")
        jax.distributed.initialize(coord, nproc, rank)
        n = len(jax.devices())
        mesh = Mesh(jax.devices(), axis_names=("x",))
        sh = NamedSharding(mesh, P("x"))
        print("process", jax.process_index(), "local devices", jax.local_devices(), "global", n)
        return bench(jax, jnp, mesh, sh, n, sizes, f"distributed rank {rank}")

    n = len(jax.devices())
    if n < 2:
        print(f"[!] 只有 {n} 个设备，需要 >=2 才能测卡间通信。")
        return 0
    mesh = Mesh(jax.devices(), axis_names=("x",))
    sh = NamedSharding(mesh, P("x"))
    return bench(jax, jnp, mesh, sh, n, sizes, "single process")


if __name__ == "__main__":
    sys.exit(main())
