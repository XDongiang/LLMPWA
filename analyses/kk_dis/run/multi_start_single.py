#!/usr/bin/env python
"""kk_dis single-node multi-start driver.

Generates `--starts` perturbed initial parameter vectors (reusing the same
RandomState-derived perturbation as multi_start_fit.py) and runs fit_script_dist.py
in single-node mode (JAX_NUM_PROCESSES=1) for each, sequentially or in parallel on
the 2 local GPUs, reporting success rate and elapsed time per start.

Usage (on HEP1, run via dockerized python if needed, or directly if jax env present):
    python run/multi_start_single.py --starts 10 --seed 2024 --tag ms_single \
        [--parallel 2] [--timeout-min 40]

Output:
    output/ms_single/summary.toml  +  per-start fit artifacts under output/ms_single/start_N/
"""
import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as onp
import toml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
os.chdir(_ROOT)


def load_free_params():
    with open("run/free_params.toml", "r") as f:
        data = toml.load(f)["data"]
    data = sorted(data, key=lambda d: d["arg_index"])
    values = onp.array([d["value"] for d in data], dtype=onp.float64)
    return values


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


def run_one(tag, start_idx, seed, timeout_min, gpu_id, disturb):
    values = load_free_params()
    paths = [p["path"] for p in sorted(
        toml.load("run/free_params.toml")["data"], key=lambda d: d["arg_index"])]
    const_idx, theta_idx = find_const_theta_indices(paths)
    x0 = values if start_idx == 0 else perturb_args(values, const_idx, theta_idx, seed, disturb)

    x0_path = os.path.join("output", tag, f"start_{start_idx}", "x0.npy")
    os.makedirs(os.path.dirname(x0_path), exist_ok=True)
    onp.save(x0_path, x0)

    env = dict(os.environ)
    env["JAX_NUM_PROCESSES"] = "1"
    env["JAX_PROCESS_ID"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["KKDIS_X0_NPY"] = x0_path
    env["KKDIS_OUTDIR"] = os.path.join(tag, f"start_{start_idx}")

    t0 = time.time()
    result = {"start": start_idx, "seed": seed, "gpu": gpu_id}
    try:
        proc = subprocess.run(
            [sys.executable, "run/fit_script_dist.py"],
            cwd=_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_min * 60.0,
        )
        result["returncode"] = proc.returncode
        m = re.search(r"Fit finished: success=(\S+) status=(-?\d+) likelihood=([-\d.eE+]+) iterations=(\S+)",
                      proc.stdout + proc.stderr)
        if m:
            result["success"] = m.group(1) == "True"
            result["nll"] = float(m.group(3))
            result["iterations"] = m.group(4)
            result["status"] = int(m.group(2))
        else:
            result["success"], result["nll"] = False, None
            result["note"] = "no Fit-finished line"
            result["stdout_tail"] = (proc.stdout or "")[-1500:]
            result["stderr_tail"] = (proc.stderr or "")[-1500:]
    except subprocess.TimeoutExpired:
        result["returncode"] = 124
        result["success"], result["nll"] = False, None
        result["note"] = f"timeout after {timeout_min}min"
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--tag", default="ms_single")
    ap.add_argument("--disturb", type=float, default=50.0)
    ap.add_argument("--timeout-min", type=float, default=40)
    ap.add_argument("--parallel", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(os.path.join("output", args.tag), exist_ok=True)
    summary_path = os.path.join("output", args.tag, "summary.toml")
    results = []

    def worker(start_idx, slot):
        seed = args.seed + start_idx * 1000
        gpu = slot % 2
        print(f"=== start {start_idx}/{args.starts} (seed {seed}, gpu {gpu}) ===", flush=True)
        r = run_one(args.tag, start_idx, seed, args.timeout_min, gpu, args.disturb)
        print(f"    rc={r.get('returncode')} ok={r.get('success')} nll={r.get('nll')} "
              f"iter={r.get('iterations')} {r.get('elapsed_s')}s {r.get('note','')}", flush=True)
        return r

    order = [s for s in range(args.starts)]
    order.sort(key=lambda s: (0 if s == 0 else 1, s))

    if args.parallel > 1 and args.starts > 1:
        from concurrent.futures import wait, FIRST_COMPLETED
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            it = iter(order)
            futures = {}
            for slot in range(args.parallel):
                try:
                    s = next(it)
                except StopIteration:
                    break
                futures[ex.submit(worker, s, slot)] = s
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    s = futures.pop(fut)
                    results.append(fut.result())
                    try:
                        nxt = next(it)
                    except StopIteration:
                        continue
                    futures[ex.submit(worker, nxt, s % args.parallel)] = nxt
    else:
        for s in order:
            results.append(worker(s, 0))

    ok = [r for r in results if r.get("success")]
    nlls = [r["nll"] for r in ok]
    with open(summary_path, "w") as f:
        toml.dump({"tag": args.tag, "seed": args.seed, "starts": args.starts,
                   "results": results}, f)
    print("\n========== SUMMARY ==========", flush=True)
    print(f"starts: {len(results)} | success: {len(ok)} "
          f"({100.0*len(ok)/max(len(results),1):.1f}%)", flush=True)
    if nlls:
        import statistics
        print(f"nll: min={min(nlls):.6f} max={max(nlls):.6f} "
              f"mean={statistics.mean(nlls):.6f}", flush=True)
    else:
        print("NO successful start", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
