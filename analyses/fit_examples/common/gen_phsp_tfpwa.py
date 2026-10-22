#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在**只有 tf-pwa 的环境**里跑的相空间生成器（零依赖：tf_pwa + numpy + json）。

它故意不读 spec TOML、不 import 本仓库 —— 因为要跑在
`tf-pwa-check:latest` 这个只有 tensorflow-cpu + tf-pwa 的镜像里。

输入是一个 json（`--jobs`），每个 job 是：
    {"name": "...", "m0": 3.686, "masses": [...], "keys": [...],
     "n": 20000, "seed": 0, "layout": "x,y,z,E", "out": "/path/momenta.npz"}

输出就是 `np.savez(out, **{key: (N,4)})` 加上 `<out>.meta.json`（守恒/在壳/
两体阈值）。所有检查都写进 meta，本仓库侧只读 meta 做验收，不重复实现。

    docker run --rm -v <hostdir>:<hostdir> -v $PWD/tmp/tf-pwa:/src:ro \\
        -e PYTHONPATH=/src -e TF_CPP_MIN_LOG_LEVEL=3 tf-pwa-check:latest \\
        python /path/gen_phsp_tfpwa.py --jobs jobs.json

⚠️ tf-pwa 的 `generate_phsp` 是**通用 n 体**（`ChainGenerator` 递归两体），
输出布局是 `(E, x, y, z)`；这里统一换成 spec 声明的 `layout`（默认
`x,y,z,E`，即协变张量内核的原生约定）。换序只在一个地方做。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def reorder(p4, layout):
    order = [s.strip() for s in layout.replace(" ", "").split(",")]
    if order == ["E", "x", "y", "z"]:
        return p4
    if order == ["x", "y", "z", "E"]:
        return p4[:, [1, 2, 3, 0]]
    raise SystemExit("不认识的 layout %r" % layout)


def _mass(q, layout):
    """不变质量，**按 layout 取分量**。

    tf-pwa 原生是 (E,x,y,z)，spec 默认是 (x,y,z,E) —— 写死任何一个都会算错，
    而且错得"很像真的"：把 (E,x,y,z) 当 (x,y,z,E) 用，m 会变成 2.3 GeV 而不是
    0.4937，两个方向都自洽地错。所以这里显式分派。
    """
    if layout == "E,x,y,z":
        e, s = q[:, 0], q[:, 1] ** 2 + q[:, 2] ** 2 + q[:, 3] ** 2
    else:
        e, s = q[:, 3], q[:, 0] ** 2 + q[:, 1] ** 2 + q[:, 2] ** 2
    return np.sqrt(np.abs(e ** 2 - s))


def _split(q, layout):
    """按 layout 拆出 (E, 三动量)。**不要**用 q[:,1:4] 这种"看起来对"的切片。"""
    if layout.replace(" ", "") == "E,x,y,z":
        return q[:, 0], q[:, 1:4]
    return q[:, 3], q[:, 0:3]


def checks(mom, masses, layout="E,x,y,z"):
    keys = list(mom)
    tot = sum(mom[k] for k in keys)
    energy, psum = _split(tot, layout)
    # ⚠️ 分母不能用 |sum p| —— 质心系里它**本来就是 0**，"相对残差"会炸成
    # 1e15（第一次就是这么翻车的）。用三动量里最大的单粒子模长作尺度，
    # 这样这个数是"守恒残差 / 典型动量"，有 O(1) 的意义。
    scale = max(float(np.abs(_split(mom[k], layout)[1]).max()) for k in keys)
    out = {
        "conservation_abs": float(np.abs(psum).max()),
        "conservation_rel": float(np.abs(psum).max() / max(scale, 1e-300)),
        "energy_abs": float(np.abs(energy - energy.mean()).max()),
        "onshell_max_abs": {},
        "two_body_min": {},
    }
    for k, m in zip(keys, masses):
        out["onshell_max_abs"][k] = float(np.abs(_mass(mom[k], layout) - m).max())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            mm = _mass(mom[keys[i]] + mom[keys[j]], layout)
            thr = masses[i] + masses[j]
            out["two_body_min"]["%s+%s" % (keys[i], keys[j])] = {
                "min": float(mm.min()), "max": float(mm.max()),
                "thr": float(thr), "margin": float(mm.min() - thr)}
    return out


def run_job(job):
    import tensorflow as tf
    from tf_pwa.phasespace import generate_phsp

    keys, masses = list(job["keys"]), list(job["masses"])
    tf.random.set_seed(int(job.get("seed", 0)))
    t0 = time.time()
    parts = generate_phsp(float(job["m0"]), masses, N=int(job["n"]))
    dt = time.time() - t0
    arrs = [np.asarray(x, dtype=np.float64) for x in parts]
    if len(arrs) != len(keys):
        raise SystemExit("%s: tf-pwa 给了 %d 个末态，要 %d 个"
                         % (job["name"], len(arrs), len(keys)))
    mom = dict(zip(keys, arrs))
    layout = job.get("layout", "x,y,z,E")
    # 先换算到目标布局，**之后所有检查都在目标布局里做**
    out = {k: reorder(mom[k], layout) for k in keys}
    info = checks(out, masses, layout)
    path = job["out"]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez(path, **out)
    meta = {"name": job["name"], "n": int(job["n"]), "seed": int(job.get("seed", 0)),
            "m0": float(job["m0"]), "keys": keys, "masses": masses,
            "layout": layout, "generator": "tf_pwa.phasespace.generate_phsp",
            "seconds": round(dt, 3), "events_per_s": round(job["n"] / max(dt, 1e-9), 1),
            **info}
    with open(os.path.splitext(path)[0] + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    worst = min(v["margin"] for v in info["two_body_min"].values())
    print("%-22s N=%-6d %5.1fs %8.0f ev/s  consv=%.1e  dM=%.1e  margin=%.2e"
          % (job["name"], job["n"], dt, job["n"] / max(dt, 1e-9),
             info["conservation_rel"],
             max(info["onshell_max_abs"].values()), worst))
    return {"name": job["name"], "ok": bool(info["conservation_rel"] < 1e-10
                                            and worst > 1e-6),
            "meta": meta, "path": path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--only", default="", help="只跑名字匹配的 job（子串）")
    args = ap.parse_args()
    with open(args.jobs, encoding="utf-8") as fh:
        jobs = json.load(fh)
    results = []
    for job in jobs:
        if args.only and args.only not in job["name"]:
            continue
        results.append(run_job(job))
    bad = [r["name"] for r in results if not r["ok"]]
    print("\n%d jobs, %d bad %s" % (len(results), len(bad), bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
