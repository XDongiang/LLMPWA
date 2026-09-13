#!/usr/bin/env python
"""kk_dis 多起点随机初值拟合 runner v3（并行 + 隔离 work dir + 断点续跑）。

用法（远程 HEP1，~/LLMPWA/kk_dis 下执行）:
    python run/multi_start_fit.py --starts 100 --seed 2024 --tag run100 \
        [--parallel 2] [--disturb 50] [--timeout-min 720] [--resume]

行为:
    1. 读取 run/free_params.toml 的 value 作为"基线初值"
    2. 对每个起点 s=0..N-1: 扰动算法见 document/random_initial_perturbation.md
       - const/theta 耦合对（Amplitude.constN/thetaN）：重新采样到半径 0.1 的圆上
       - 全部参数：乘性抖动 (rand()-0.5)/disturb + 1.0（默认 disturb=50 → ±1%）
       - 起点 0 用基线（不扰动）
    3. 每个起点在独立 work 目录 work_{tag}/start_{N}/ 下跑 fit_script.py（子进程）
       - work 目录内软链 data/，复制小文件（config/run/gen/document/...），
         完全隔离：并发起点各自的 free_params.toml / output/fit / logs 互不干扰
    4. 并发控制：--parallel N（同跑 N 个起点）。单机双 GPU 建议 2：
       - 但注意 fit_script 默认用全部设备（Mesh(jax.devices())），并发 2 起点时
         需用 CUDA_VISIBLE_DEVICES=0 / =1 各占一卡
    5. 每个起点结果写回 output/multistart/<tag>/start_{N}/ + 总表 summary.toml
    6. --resume：跳过已有成功结果（断点续跑，白跑不重来）

设计要点:
    - subprocess 调 fit_script.py，每个起点 cwd=独立 work 目录
    - 并发用 ThreadPoolExecutor + CUDA_VISIBLE_DEVICES 映射（worker i -> GPU i % 2）
    - success/nll 从 fit_script stdout 解析；timeout 捕获为 returncode=124
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as onp
import toml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
os.chdir(_ROOT)

COPY_DIRS = ["config", "gen", "document"]
COPY_FILES = ["llm_config_fit.toml", "resonances_config.toml", "node_config.toml"]

def load_free_params():
    with open("run/free_params.toml", "r") as f:
        data = toml.load(f)["data"]
    data = sorted(data, key=lambda d: d["arg_index"])
    values = onp.array([d["value"] for d in data], dtype=onp.float64)
    return values, data


def find_const_theta_indices(paths):
    """从 free_params.toml 的 path 字段中找出所有 Amplitude.constN / Amplitude.thetaN 的 arg_index"""
    const_idx = [i for i, p in enumerate(paths) if ".Amplitude.const" in p]
    theta_idx = [i for i, p in enumerate(paths) if ".Amplitude.theta" in p]
    return onp.array(const_idx, dtype=int), onp.array(theta_idx, dtype=int)


def perturb_args(base_args, const_idx, theta_idx, seed, disturb=50.0):
    """见 document/random_initial_perturbation.md"""
    rng = onp.random.RandomState(seed)
    args = onp.array(base_args, dtype=float).copy()
    if const_idx.size > 0:
        theta = 2 * onp.pi * rng.rand(theta_idx.shape[0])
        args[theta_idx] = 0.1 * onp.cos(theta)
        args[const_idx] = 0.1 * onp.sin(theta)
    args = args * ((rng.rand(args.shape[0]) - 0.5) / disturb + 1.0)
    return args


def build_work_dir(tag, start_idx, values, data):
    """创建并返回 start_N 的独立 work 目录（软链 data，复制小文件，写入 free_params.toml）"""
    wd = os.path.join("work", f"{tag}_start_{start_idx}")
    if not os.path.exists(wd):
        os.makedirs(wd)
    # fit_script 的 setup_logging 写 logs/fit.log，需要 logs/ 存在
    os.makedirs(os.path.join(wd, "logs"), exist_ok=True)
    # output/ 目录
    os.makedirs(os.path.join(wd, "output", "fit"), exist_ok=True)
    # data/ 软链（811M 不复制）
    if not os.path.lexists(os.path.join(wd, "data")):
        os.symlink(os.path.join(_ROOT, "data"), os.path.join(wd, "data"))
    # run/ 目录：复制代码文件（不复制多起点自己的产物）；覆盖陈旧副本
    run_dst = os.path.join(wd, "run")
    os.makedirs(run_dst, exist_ok=True)
    for fn in ("base_functions.py", "likelihood_function.py", "fit_script.py", "free_params.toml"):
        src = os.path.join(_ROOT, "run", fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(run_dst, fn))
    # 其他小目录/文件
    for d in COPY_DIRS:
        src = os.path.join(_ROOT, d)
        dst = os.path.join(wd, d)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)
    for fn in COPY_FILES:
        src = os.path.join(_ROOT, fn)
        if os.path.exists(src) and not os.path.exists(os.path.join(wd, fn)):
            shutil.copy2(src, os.path.join(wd, fn))
    # 写入本起点 free_params.toml（扰动后的初值）
    new_data = [dict(d) for d in data]
    for i, d in enumerate(new_data):
        d["value"] = float(values[i])
    with open(os.path.join(wd, "run", "free_params.toml"), "w") as f:
        toml.dump({"data": new_data}, f)
    return wd


def run_one_start(tag, start_idx, seed, disturb, timeout_min, gpu_id):
    """在独立 work 目录跑一次拟合；返回结果 dict。"""
    values, data = load_free_params()
    paths = [d["path"] for d in data]
    const_idx, theta_idx = find_const_theta_indices(paths)
    if start_idx == 0:
        x0 = values
    else:
        x0 = perturb_args(values, const_idx, theta_idx, seed, disturb=disturb)
    wd = build_work_dir(tag, start_idx, x0, data)

    env = dict(os.environ)
    env["JAX_NUM_PROCESSES"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    result = {"start": start_idx, "seed": seed, "gpu": gpu_id, "workdir": wd}
    try:
        proc = subprocess.run(
            [sys.executable, "run/fit_script.py"],
            cwd=wd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_min * 60.0,
        )
        result["returncode"] = proc.returncode
        if proc.returncode == 0:
            m = re.findall(r"fit complete: success=(\S+), nll=([-\d.]+)", proc.stdout)
            if m:
                succ, nll = m[-1]
                result["success"] = succ == "True"
                result["nll"] = float(nll)
            else:
                result["success"], result["nll"] = False, None
                result["note"] = "no fit-complete line in stdout"
        else:
            result["success"], result["nll"] = False, None
            result["stderr_tail"] = (proc.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        result["returncode"] = 124  # timeout convention
        result["success"], result["nll"] = False, None
        result["note"] = f"timeout after {timeout_min}min"
    # 收集产物（fit_script 在 wd/output/fit/ 下写）
    out_dst = os.path.join("output", "multistart", tag, f"start_{start_idx}")
    os.makedirs(out_dst, exist_ok=True)
    for fn in ("fit_result_values.npy", "fit_result_errors.npy", "free_params_fitted.toml"):
        s = os.path.join(wd, "output", "fit", fn)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(out_dst, fn))
    return result


def worker(start_idx, slot, args, existing):
    """在固定 slot（=GPU) 上跑一次拟合；slot 决定 CUDA_VISIBLE_DEVICES，与起点序号无关。"""
    seed = args.seed + start_idx * 1000
    gpu = slot % args.gpus
    if start_idx in existing.get("by_start", {}):
        r = existing["by_start"][start_idx]
        print(f"=== start {start_idx}/{args.starts} (seed {seed}) RESUMED success nll={r.get('nll')} ===", flush=True)
        return r
    print(f"=== start {start_idx}/{args.starts} (seed {seed}, gpu {gpu}) ===", flush=True)
    r = run_one_start(args.tag, start_idx, seed, args.disturb, args.timeout_min, gpu)
    print(f"    returncode={r.get('returncode')} success={r.get('success')} nll={r.get('nll')}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--tag", default="run100")
    ap.add_argument("--disturb", type=float, default=50.0)
    ap.add_argument("--timeout-min", type=float, default=720)
    ap.add_argument("--parallel", type=int, default=2,
                    help="同时运行的起点数（即 worker 槽位数；单机双卡建议 2）")
    ap.add_argument("--gpus", type=int, default=2,
                    help="GPU 数；slot % gpus 决定每 worker 绑定的卡（默认 2）")
    ap.add_argument("--resume", action="store_true", help="跳过已成功起点（断点续跑）")
    args = ap.parse_args()

    os.makedirs(os.path.join("output", "multistart"), exist_ok=True)
    summary_path = os.path.join("output", "multistart", f"{args.tag}_summary.toml")

    # 断点续跑：读取已有成功结果（按 start 索引）
    existing = {"by_start": {}}
    if args.resume and os.path.exists(summary_path):
        try:
            prev = toml.load(summary_path)
            for r in prev.get("results", []):
                if r.get("success"):
                    existing["by_start"][r["start"]] = r
        except Exception as e:
            print(f"WARN: resume summary unreadable ({e}); starting fresh", flush=True)

    results = []
    best = None

    def record(r):
        nonlocal best
        results.append(r)
        if r.get("success"):
            if best is None or r["nll"] < best["nll"]:
                best = r
        # 原子写 summary（临时文件 + rename，断点续跑可靠）
        tmp = summary_path + ".tmp"
        with open(tmp, "w") as f:
            toml.dump({
                "tag": args.tag, "seed": args.seed, "starts": args.starts,
                "results": results, "best": best,
            }, f)
        os.replace(tmp, summary_path)

    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

    order = [s for s in range(args.starts)]
    # 起点 0（基线）优先：先在 GPU0 上发车
    order.sort(key=lambda s: (0 if s == 0 else 1, s))

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        it = iter(order)
        futures = {}
        for slot in range(args.parallel):
            try:
                s = next(it)
            except StopIteration:
                break
            futures[ex.submit(worker, s, slot, args, existing)] = (s, slot)
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                s0, slot = futures.pop(fut)
                rec = fut.result()
                record(rec)
                try:
                    s = next(it)
                except StopIteration:
                    continue
                # 该 slot 空出，补下一个任务（仍用同一 slot=同一 GPU）
                futures[ex.submit(worker, s, slot, args, existing)] = (s, slot)

    # 统计
    ok = [r for r in results if r.get("success")]
    n_ok = len(ok)
    n_all = len(results)
    nlls = [r["nll"] for r in ok]
    print("\n========== SUMMARY ==========", flush=True)
    print(f"starts: {n_all} | success: {n_ok} ({100.0 * n_ok / max(n_all, 1):.1f}%)", flush=True)
    if nlls:
        import statistics
        print(f"nll: min={min(nlls):.6f} max={max(nlls):.6f} mean={statistics.mean(nlls):.6f} "
              f"median={statistics.median(nlls):.6f}", flush=True)
        print(f"BEST start={best['start']} nll={best['nll']} (gpu {best.get('gpu')})", flush=True)
    else:
        print("NO successful start", flush=True)
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
