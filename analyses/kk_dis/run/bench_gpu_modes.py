#!/usr/bin/env python
"""A/B 对比：单卡并行(2 work 各 1 GPU) vs 原生双 GPU(single fit 用 2 卡) 的拟合耗时。

用法（远程 HEP1，~/LLMPWA/kk_dis 下执行）:
    python run/bench_gpu_modes.py --starts 4 --tag bench [--mode dual]
    # mode:
    #   dual  : 每起点原生双 GPU（Mesh 用全部 2 卡），串行跑 N 个起点
    #   single: 每起点单卡，2 起点并行（模拟 run100 的并行方式）
    # 输出 output/multistart/<tag>_summary.toml + 每起点墙钟

注意: dual 模式每起点用 2 卡全部显存/算力；single 模式 2 起点并行各占 1 卡。
两种模式的"单次拟合"是同一份数据（18000 事件），只差 GPU 并行策略。
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as onp
import toml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

def load_free_params():
    with open(os.path.join(_ROOT, "run", "free_params.toml"), "r") as f:
        data = toml.load(f)["data"]
    data = sorted(data, key=lambda d: d["arg_index"])
    return onp.array([d["value"] for d in data], dtype=onp.float64), data

def find_const_theta_indices(paths):
    const_idx = [i for i, p in enumerate(paths) if ".Amplitude.const" in p]
    theta_idx = [i for i, p in enumerate(paths) if ".Amplitude.theta" in p]
    return onp.array(const_idx, dtype=int), onp.array(theta_idx, dtype=int)

def perturb_args(base_args, const_idx, theta_idx, seed, disturb=50.0):
    rng = onp.random.RandomState(seed)
    args = onp.array(base_args, dtype=float).copy()
    if const_idx.size > 0:
        theta = 2 * onp.pi * rng.rand(theta_idx.shape[0])
        args[theta_idx] = 0.1 * onp.cos(theta)
        args[const_idx] = 0.1 * onp.sin(theta)
    args = args * ((rng.rand(args.shape[0]) - 0.5) / disturb + 1.0)
    return args

def build_work_dir(tag, start_idx, values, data, os_root):
    wd = os.path.join("work", f"{tag}_start_{start_idx}")
    os.makedirs(wd, exist_ok=True)
    os.makedirs(os.path.join(wd, "logs"), exist_ok=True)
    os.makedirs(os.path.join(wd, "output", "fit"), exist_ok=True)
    if not os.path.lexists(os.path.join(wd, "data")):
        os.symlink(os.path.join(os_root, "data"), os.path.join(wd, "data"))
    for d in ("config", "gen", "document"):
        src, dst = os.path.join(os_root, d), os.path.join(wd, d)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)
    run_dst = os.path.join(wd, "run")
    os.makedirs(run_dst, exist_ok=True)
    for fn in ("base_functions.py", "likelihood_function.py", "fit_script.py", "free_params.toml"):
        src = os.path.join(os_root, "run", fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(run_dst, fn))
    new_data = [dict(d) for d in data]
    for i, d in enumerate(new_data):
        d["value"] = float(values[i])
    with open(os.path.join(wd, "run", "free_params.toml"), "w") as f:
        toml.dump({"data": new_data}, f)
    return wd

def run_one_fit(wd, gpu_env, timeout_min):
    """在 wd 里跑一次 fit_script；返回 (wall_seconds, returncode, success, nll, iterations)"""
    env = dict(os.environ)
    env["JAX_NUM_PROCESSES"] = "1"
    if gpu_env is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_env
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "run/fit_script.py"],
            cwd=wd, env=env, capture_output=True, text=True,
            timeout=timeout_min * 60.0,
        )
        wall = time.time() - t0
        m = re.findall(r"fit complete: success=(\S+), nll=([-\d.]+)", proc.stdout)
        # 迭代数 = stdout 中最大 iteration N + 1
        iters = 0
        for line in (proc.stdout or "").splitlines():
            im = re.search(r"iteration (\d+):", line)
            if im:
                iters = max(iters, int(im.group(1)))
        if m and m[-1][0] == "True":
            return wall, proc.returncode, True, float(m[-1][1]), iters
        return wall, proc.returncode, False, None, iters
    except subprocess.TimeoutExpired:
        return time.time() - t0, 124, False, None, -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--mode", choices=["dual", "single"], required=True,
                    help="dual=原生双GPU每起点; single=2起点并行各1GPU")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout-min", type=float, default=720)
    args = ap.parse_args()

    os.chdir(_ROOT)
    values, data = load_free_params()
    paths = [d["path"] for d in data]
    const_idx, theta_idx = find_const_theta_indices(paths)
    os.makedirs("output/multistart", exist_ok=True)
    summary_path = os.path.join("output", "multistart", f"{args.tag}_{args.mode}_summary.toml")

    results = []

    def do_one(start_idx, gpu_env):
        if start_idx == 0:
            x0 = values
        else:
            x0 = perturb_args(values, const_idx, theta_idx, args.seed + start_idx * 1000)
        wd = build_work_dir(args.tag, start_idx, x0, data, _ROOT)
        wall, rc, succ, nll, iters = run_one_fit(wd, gpu_env, args.timeout_min)
        r = {"start": start_idx, "mode": args.mode, "gpu_env": gpu_env,
             "wall_sec": round(wall, 1), "returncode": rc, "success": succ, "nll": nll,
             "iterations": iters}
        print(f"start {start_idx} {args.mode} gpu={gpu_env}: {wall:.1f}s rc={rc} succ={succ} nll={nll} iters={iters}", flush=True)
        return r

    if args.mode == "dual":
        for s in range(args.starts):
            results.append(do_one(s, None))   # 不设 CUDA_VISIBLE_DEVICES → 用全部设备
    else:  # single: 2 起点并行，交替 GPU
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = {}
            for slot in range(2):
                if slot < args.starts:
                    futures[ex.submit(do_one, slot, str(slot % 2))] = (slot, slot % 2)
            completed = 0
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    s0, gpu = futures.pop(fut)
                    results.append(fut.result())
                    completed += 1
                    nxt = completed + 1  # 下一序号
                    if nxt + 1 <= args.starts:
                        pass
                # 补任务
                while len(futures) < 2 and len(results) + len(futures) < args.starts:
                    nxt = len(results) + len(futures)
                    gpu = nxt % 2
                    futures[ex.submit(do_one, nxt, str(gpu))] = (nxt, gpu)

    # 汇总
    ok = [r for r in results if r.get("success")]
    import statistics
    print("\n=== %s mode ===" % args.mode, flush=True)
    if ok:
        ws = [r["wall_sec"] for r in ok]
        its = [r.get("iterations") or 0 for r in ok]
        print(f"single-fit wall: min={min(ws):.1f}s max={max(ws):.1f}s mean={statistics.mean(ws):.1f}s median={statistics.median(ws):.1f}s", flush=True)
        print(f"iterations: min={min(its)} max={max(its)} mean={statistics.mean(its):.0f}", flush=True)
        print(f"wall/iter: {statistics.mean(ws)/max(statistics.mean(its),1)*1000:.0f} ms/iter", flush=True)
        print(f"total wall for {len(results)} starts (parallel where applicable): ~{sum(ws):.0f}s raw / actual wall varies", flush=True)
    print(f"success {len(ok)}/{len(results)}", flush=True)
    with open(summary_path, "w") as f:
        toml.dump({"tag": args.tag, "mode": args.mode, "results": results}, f)
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
