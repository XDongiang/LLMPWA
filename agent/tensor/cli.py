#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`agent.tensor` 的命令行入口。

    python -m agent.tensor show   <spec.toml>    # 链/道表/表象（不写文件）
    python -m agent.tensor check  <spec.toml>    # 校验拓扑、量子数、道表、表象
    python -m agent.tensor bases                 # 列出 4 个表象钩子
    python -m agent.tensor chains                # 列出 chain.<name>/chain.<channel>.py
    python -m agent.tensor emit   <spec.toml>    # 按 [[basis]] 产出自足脚本
    python -m agent.tensor run    <spec.toml> --momenta <npz> --out <dir>
                                                 # emit + 直接跑一遍
                                                 # （若同时出了 cartesian 与 omega，
                                                 #   自动跑跨算法一致性检验）
    python -m agent.tensor verify <spec.toml> --cartesian <dir> --omega <dir>
                                                 # 只做跨算法一致性检验

与 `specgen` 的关系：`agent.tensor` 是**框架级**版本（跨分析复用、一份 TOML
出多种表象），`specgen` 是它前身的一份分析内副本。产物约定完全兼容。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from . import basis as basismod
from . import chains as chainsmod
from . import crosscheck as ccmod
from . import emitter
from . import spec as specmod


def _load(path):
    spec = specmod.load_spec(path)
    spec["_spec_path"] = os.path.abspath(path)
    return spec


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m agent.tensor", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, help_ in (("show", "打印链/道表/表象"), ("check", "校验 TOML"),
                        ("emit", "按 [[basis]] 产出自足脚本"),
                        ("run", "emit 并直接跑一遍"),
                        ("verify", "跨算法一致性检验（已有产物）")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("spec", nargs="?" if name in ("bases", "chains") else None)
        p.add_argument("--out-dir", default="", help="覆盖 [chain].output")
        if name in ("run",):
            p.add_argument("--momenta", "--npz", dest="momenta", required=True)
            p.add_argument("--n", type=int, default=0)
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--out", required=True, help="数据产物目录")
            p.add_argument("--no-verify", action="store_true",
                           help="跳过 run 之后的跨算法一致性检验")
            p.add_argument("--tol", type=float, default=ccmod.DEFAULT_TOL,
                           help="跨算法检验的相对残差阈值")
        if name == "verify":
            p.add_argument("--cartesian", required=True,
                           help="cartesian cache 所在目录（<chain>_kk.npy）")
            p.add_argument("--omega", required=True,
                           help="Omega/K/M/g 所在目录")
            p.add_argument("--tol", type=float, default=ccmod.DEFAULT_TOL,
                           help="相对残差阈值")

    sub.add_parser("bases", help="列出 4 个表象钩子")
    sub.add_parser("chains", help="列出 chain.<name>/chain.<channel>.py")

    args = ap.parse_args(argv)

    if args.cmd == "bases":
        print(basismod.describe_all())
        print("\n一致性自检：", basismod.check_consistency())
        return 0

    if args.cmd == "chains":
        for c in chainsmod.list_chains():
            chs = chainsmod.list_channels(c)
            print("%-8s %d 条道：%s" % (c, len(chs), ", ".join(chs)))
        return 0

    spec = _load(args.spec)

    if args.cmd == "check":
        print(specmod.describe(spec))
        bases = spec["bases"] or [{"name": "cartesian"}]
        for b in bases:
            basismod.get(b["name"])           # 未知表象立刻报错
        print("\n表象（%d）：%s" % (len(bases), ", ".join(b["name"] for b in bases)))
        print("\nOK: 拓扑 / 量子数 / 道表 / 表象校验通过")
        return 0

    if args.cmd == "show":
        print(specmod.describe(spec))
        print("\n自动枚举的全部 (l,s)（宇称+三角）：", specmod.allowed_ls(spec))
        print("表象：", [b["name"] for b in spec["bases"]] or ["(默认 cartesian)"])
        return 0

    if args.cmd == "verify":
        return _verify(spec, args.cartesian, args.omega, args.tol)

    res = emitter.emit(spec, out_dir=args.out_dir or None)
    print("产物目录 %s" % res["out_dir"])
    for f in res["scripts"]:
        print("  生成 %s" % f)
    print("  runtime 副本 %s" % res["runtime"])
    print("  %d 条道：%s" % (len(res["channels"]),
                             ", ".join("%s(l%d,s%d)" % (n, l, s)
                                       for (n, l, s, _f) in res["channels"])))

    if args.cmd == "emit":
        return 0

    rc = 0
    for entry in res["scripts"]:
        cmd = [sys.executable, entry["script"], "--momenta", args.momenta,
               "--out", args.out, "--n", str(args.n), "--start", str(args.start)]
        print("\n$ " + " ".join(cmd))
        rc |= subprocess.call(cmd)

    # 跨算法一致性检验：只有同时出了 cartesian（或 helicity）与 omega 才有意义
    kernels = {e["kernel"] for e in res["scripts"]}
    if rc == 0 and not args.no_verify and "omega" in kernels and (
            kernels & {"covariant", "helicity"}):
        print()
        rc |= _verify(spec, args.out, args.out, args.tol)
    return rc


def _verify(spec, cart_dir, omega_dir, tol):
    """跨算法一致性检验。返回 0/1（供 `run` 汇总退出码）。"""
    names = [n for (n, _l, _s, _f) in specmod.resolve_channels(spec)]
    suffix = spec["raw"].get("chain", {}).get("suffix", "")
    try:
        res = ccmod.check_dirs(cart_dir, omega_dir, chain=spec["name"],
                               suffix=suffix, names=names, tol=tol)
    except ccmod.CrossCheckError as e:
        print("跨算法一致性检验**无法进行**：%s" % e)
        return 1
    print(ccmod.report(res))
    if res["ok"]:
        return 0
    print("\n提示：两条路线的 psi(2S) 口径必须一致 —— covariant 侧用 "
          "`[chain].psi_mass` 的静态近似，omega 侧用末态四动量之和。\n"
          "真实样本要写 psi_mass = 3.685782956（见 psi_mass_residual.md §6b），"
          "用默认的 3.686 会让残差停在 ~1e-4。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
