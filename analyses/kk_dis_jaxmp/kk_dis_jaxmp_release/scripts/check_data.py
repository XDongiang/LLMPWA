#!/usr/bin/env python
"""检查数据集是否满足 fit_object_kk.py 的布局要求（不加载进内存）。

用法:
    python tools/check_data.py                 # 检查默认 data/
    python tools/check_data.py --root data/draw_mc
    python tools/check_data.py --scan          # 扫描候选目录并给出可用组合

旧模型需要 10 个逻辑张量，其中 4 个（u_rho_r/l、u_b_r/l）只有部分数据集才有。
本脚本会指出：缺哪些文件、各组事例数是否自洽、权重长度是否与实验数据匹配，
并直接打印出可用的环境变量组合。
"""
import argparse
import os
import sys

import numpy as onp

TENSORS = ("phif0_kk", "phif2_kk", "u_rho_r_kk", "u_rho_l_kk", "u_b_r_kk", "u_b_l_kk")
SCALARS = ("phi_kk", "f_kk", "b124_kk", "b123_kk")
ALL = TENSORS + SCALARS


def event_count(path):
    a = onp.load(path, mmap_mode="r")
    return a.shape[0] if a.ndim == 1 else a.shape[1]


def probe(real, mc, truth, weight):
    """返回 (是否可用, 报告行, 环境变量建议)。"""
    report = []
    ok = True

    def group(label, d):
        nonlocal ok
        present, missing, sizes = [], [], {}
        for n in ALL:
            p = os.path.join(d, n + ".npy")
            if os.path.exists(p):
                present.append(n)
                sizes[n] = event_count(p)
            else:
                missing.append(n)
        if missing:
            ok = False
            report.append("  %-10s %s  缺: %s" % (label, d, ", ".join(missing)))
        else:
            report.append("  %-10s %s  齐全" % (label, d))
        # 似然把 22 条道逐事例相加后再乘权重，所以**同一组内所有张量**的事例数
        # 必须完全一致（旧版逐文件 array_split，长度不一致同样会炸）。
        if sizes and len(set(sizes.values())) > 1:
            ok = False
            by_n = {}
            for k, v in sorted(sizes.items()):
                by_n.setdefault(v, []).append(k)
            report.append("      !! 事例数不一致: %s"
                          % "; ".join("%d -> %s" % (n, ",".join(ks))
                                      for n, ks in sorted(by_n.items())))
        return sizes

    rs = group("实验数据", real)
    ms = group("MC 积分", mc)
    ts = group("truth-MC", truth)

    # 权重
    if os.path.exists(weight):
        wn = onp.load(weight, mmap_mode="r").shape[0]
        dn = rs.get("phi_kk")
        if dn is not None and wn != dn:
            ok = False
            report.append("  %-10s %s  长度 %d != 实验事例数 %d  ->  不匹配"
                          % ("权重", weight, wn, dn))
        else:
            report.append("  %-10s %s  长度 %d  匹配" % ("权重", weight, wn))
    else:
        report.append("  %-10s %s  缺失（会退化为不加权）" % ("权重", weight))

    env = ("KK_REAL_DIR=%s KK_MC_DIR=%s KK_TRUTH_DIR=%s KK_WEIGHT_FILE=%s"
           % (real, mc, truth, weight))
    return ok, report, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--real", default=None)
    ap.add_argument("--mc", default=None)
    ap.add_argument("--truth", default=None)
    ap.add_argument("--weight", default=None)
    ap.add_argument("--scan", action="store_true",
                    help="扫描所有含 real_data 的目录并给出可用组合")
    args = ap.parse_args()

    if args.scan:
        cands = []
        for dirpath, dirnames, _ in os.walk("data"):
            for d in dirnames:
                if d in ("real_data", "mc_truth", "mc_int", "weight"):
                    cands.append(os.path.join(dirpath, d))
        reals = sorted(d for d in cands if d.endswith("real_data"))
        mcs = sorted(d for d in cands if d.endswith(("mc_truth", "mc_int", "draw_mc")))
        truths = sorted(d for d in cands if d.endswith("mc_truth"))
        weights = sorted(d for d in cands if d.endswith("weight"))
        print("候选实验数据目录:", reals)
        print("候选 MC 目录     :", mcs)
        print("候选 truth 目录  :", truths)
        print("候选权重目录     :", weights)
        print()
        found = False
        for r in reals:
            for m in set([r] + mcs):
                for t in truths:
                    for w in weights:
                        wp = os.path.join(w, "weight_kk.npy")
                        ok, rep, env = probe(r, m, t, wp)
                        if ok:
                            found = True
                            print("=== 可用组合 ===")
                            print("\n".join(rep))
                            print("  " + env)
                            print()
        if not found:
            print("没有找到完全可用的组合：现有数据集里 u_rho_*/u_b_* 与主数组事例数不一致。")
            print("旧版是按事例轴逐文件 array_split 的，所以长度不一致也能跑；")
            print("新版要求每个张量内部自洽。需要重新导出一份一致的数据集。")
        return 0

    real = args.real or os.path.join(args.root, "real_data")
    mc = args.mc or os.path.join(args.root, "mc_int")
    if not os.path.isdir(mc):
        mc = os.path.join(args.root, "mc_truth")
    truth = args.truth or os.path.join(args.root, "mc_truth")
    weight = args.weight or os.path.join(args.root, "weight", "weight_kk.npy")

    ok, rep, env = probe(real, mc, truth, weight)
    print("\n".join(rep))
    print()
    if ok:
        print("可用。环境变量:")
        print("  " + env)
        return 0
    print("不可用（见上面 !! 与 缺: 标记）。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
