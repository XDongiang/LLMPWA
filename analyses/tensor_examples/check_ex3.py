#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""例 3（rho'/b1）端到端回归：合成四动量 -> 生成脚本 -> 产物自检。

三层验证，从弱到强：

1. **生成物 vs 框架内核**：产物脚本把 `chains/<chain>/<channel>.py` 的函数体
   内联后算出来的 cache，必须与直接调 `agent.tensor.kernels.covariant.
   compute_event` 逐位一致。这条抓的是 emitter 的组装 bug（漏道、错序、
   函数体截断）。
2. **cartesian vs helicity**：同一份 TOML 出的两个表象必须满足
   `W = EPS_A . T`（`agent.tensor.crosscheck`）。这条抓的是 store 体写错。
3. **数值有限性**：没有 NaN / Inf。相空间抽样保证 `Qabc > 0`，
   所以这里出现 NaN 就是真 bug（而不是"抽到了非物理四动量"）。

用法
----
    python check_ex3.py                    # 用 data/ex3_phipipi.npz（缺则现生成）
    python check_ex3.py --n 500            # 只跑前 500 事例（快）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # LLMPWA 仓库根
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.tensor import crosscheck as cc              # noqa: E402
from agent.tensor.kernels import covariant as COV      # noqa: E402

# 例 3 的两棵树各自的道表（顺序 = spec 里 [[channel]] 的顺序 = cache 行序）
#
# `cascade` 给出螺旋度级联表象的产物文件名（**与 helicity 逐位同格式**）；
# `a1` 决定要不要跑判据 A1（级联 vs 协变逐道对拍）。rho' 的协变道
# **不是**逐节点级联 —— 它的 Eq.(46) 是两个 epsilon 张起来的、没有中间态
# 投影算符，所以 A1 对它不成立（见 README「已知边界」）。
TREES = {
    "ex3_rhopi": {"chain": "rhopi",
                  "names": ["u_rho_1", "u_rho_2"],
                  "out": "rhopi_kk.npy", "hel": "rhopi_hel.npy",
                  "cascade": "rhopi_casc.npy", "a1": False},
    "ex3_b1pi": {"chain": "b1pi",
                 "names": ["u_SS_1", "u_SS_2", "u_SD_1", "u_SD_2",
                           "u_DS_1", "u_DS_2", "u_DD_1", "u_DD_2"],
                 "out": "b1pi_kk.npy", "hel": "b1pi_hel.npy",
                 "cascade": "b1pi_casc.npy", "a1": True},
}
PSI_MASS = 3.686


def load_momenta(npz, n=None, start=0):
    d = np.load(npz)
    sl = slice(start, None if not n else start + n)
    return {k: np.asarray(d[k][sl], dtype=float)
            for k in ("Kp", "Km", "Pip", "Pim")}


def reference(mom, names):
    """框架内核直算：{道名: (N,4)} -> (n_ch, N, 4)。"""
    n = mom["Kp"].shape[0]
    psi = np.broadcast_to(np.array([0.0, 0.0, 0.0, PSI_MASS]), (n, 4))
    return np.stack([COV.compute_event(nm, mom, psi=psi) for nm in names], axis=0)


def run_script(script, npz, out, n, start=0):
    cmd = [sys.executable, script, "--momenta", npz, "--out", out,
           "--n", str(n), "--start", str(start)]
    print("  $ " + " ".join(os.path.relpath(c, ROOT) if c.startswith(ROOT) else c
                            for c in cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:], file=sys.stderr)
        raise SystemExit("生成脚本失败：%s" % script)
    return r.stdout.strip().splitlines()[-1]


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def check_tree(tree, npz, n, start, gen_root, out_root, keep):
    info = TREES[tree]
    spec = os.path.join(HERE, "specs", "%s.toml" % tree)
    gdir = os.path.join(gen_root, tree)
    odir = os.path.join(out_root, tree)
    os.makedirs(odir, exist_ok=True)
    ok = True

    print("\n=== %s（%d 条道，chain=%s）===" % (tree, len(info["names"]),
                                               info["chain"]))

    # 1) 生成（幂等：每次重跑，产物目录自足）
    scripts = ["gen_tensor_%s_%s.py" % (tree, k)
               for k in ("covariant", "helicity", "helicity_cascade")]
    if not all(os.path.exists(os.path.join(gdir, s)) for s in scripts):
        r = subprocess.run([sys.executable, "-m", "agent.tensor", "emit", spec],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout, r.stderr, file=sys.stderr)
            raise SystemExit("emit 失败：%s" % spec)
    for s in scripts:
        script = os.path.join(gdir, s)
        if not os.path.exists(script):
            continue
        print("  跑 %s" % os.path.relpath(script, ROOT))
        run_script(script, npz, odir, n, start)

    # 2) 生成物 vs 框架内核
    mom = load_momenta(npz, n, start)
    ref = reference(mom, info["names"])
    got = np.load(os.path.join(odir, info["out"]))
    if got.shape != ref[..., :got.shape[-1]].shape:
        print("  **FAIL** 形状：产物 %s vs 参考 %s"
              % (got.shape, ref[..., :got.shape[-1]].shape))
        return False
    r1 = rel(got, ref[..., :got.shape[-1]])
    good1 = r1 < 1e-12
    print("  生成物 vs 框架内核（前 %d 个洛伦兹分量） rel = %.3e  %s"
          % (got.shape[-1], r1, "OK" if good1 else "**FAIL**"))
    ok &= good1

    # 3) 数值有限性
    finite = bool(np.isfinite(got).all())
    mx = float(np.abs(got).max())
    print("  有限性 max|T| = %.6g  %s" % (mx, "OK" if finite else "**有 NaN/Inf**"))
    ok &= finite and mx > 0

    # 4) cartesian vs helicity：W = EPS_A . T
    cart = np.load(os.path.join(odir, info["out"]))
    hel = np.load(os.path.join(odir, info["hel"]))
    T = cart.astype(np.complex128)
    W = np.einsum("lm,iem->iel", cc.EPS_A.astype(np.complex128), T)
    r2 = rel(hel, W)
    good2 = r2 < 1e-12
    print("  helicity vs EPS_A . cartesian rel = %.3e  %s"
          % (r2, "OK" if good2 else "**FAIL**"))
    ok &= good2

    # 5) 判据 A7：螺旋度级联的 cache 形状/dtype 与 helicity **精确**一致。
    #    §4.5 的承诺就是"拟合侧零改动"，所以这一条必须是精确相等而不是近似。
    casc_path = os.path.join(odir, info["cascade"])
    if os.path.exists(casc_path) and info["cascade"] != info["hel"]:
        casc = np.load(casc_path)
        same = (casc.shape == hel.shape and casc.dtype == hel.dtype)
        print("  A7 级联 cache 形状/dtype 与 helicity 一致：%s vs %s  %s"
              % (casc.shape, hel.shape, "OK" if same else "**FAIL**"))
        ok &= bool(same)
        finite_c = bool(np.isfinite(casc).all())
        print("  A7 级联有限性 max|W| = %.6g  %s"
              % (float(np.abs(casc).max()), "OK" if finite_c else "**有 NaN/Inf**"))
        ok &= finite_c and float(np.abs(casc).max()) > 0

    # 6) 判据 A1：级联 vs 协变逐道对拍（经 EPS_A）。
    #    这是**最强**的一条：covariant.py 是手写公式、与 omega/级联完全独立。
    if info.get("a1"):
        print("  A1 级联 vs 协变逐道对拍（经 EPS_A）：")
        casc = np.load(os.path.join(odir, info["cascade"]))
        worst = 0.0
        for i, nm in enumerate(info["names"]):
            s = np.vdot(hel[i], casc[i]) / np.vdot(casc[i], casc[i])
            r = float(np.abs(hel[i] - s * casc[i]).max()
                      / max(np.abs(hel[i]).max(), 1e-300))
            worst = max(worst, r)
            print("    %-8s rel = %.3e  %s" % (nm, r, "OK" if r < 1e-12 else "**FAIL**"))
        ok &= worst < 1e-12

        # 7) 判据 A3：b1 的 8 条道两两不重合（omega 时代 12/28 对逐位重合）
        print("  A3 b1 8 条道两两不重合：")
        n = len(info["names"])
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rs = []
        for i, j in pairs:
            a, b = casc[i], casc[j]
            s = np.vdot(b, a) / np.vdot(a, a)
            rs.append(float(np.abs(b - s * a).max() / max(np.abs(b).max(), 1e-300)))
        mn = min(rs)
        print("    min over %d pairs = %.4e  %s"
              % (len(pairs), mn, "OK" if mn > 1e-6 else "**FAIL**"))
        ok &= mn > 1e-6

    if keep:
        print("  产物留在 %s" % os.path.relpath(odir, ROOT))
    return ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=os.path.join(HERE, "data", "ex3_phipipi.npz"))
    ap.add_argument("--n", type=int, default=0, help="0 = 全部")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--gen", default=os.path.join(HERE, "generated"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--keep", action="store_true", help="保留产物目录")
    ap.add_argument("--tree", default="", help="只跑一棵树（ex3_rhopi / ex3_b1pi）")
    args = ap.parse_args()

    if not os.path.exists(args.npz):
        print("没有 %s，先生成" % args.npz)
        r = subprocess.run([sys.executable,
                            os.path.join(HERE, "data", "make_momenta.py"),
                            "--out", args.npz, "--n", "2000", "--seed", "7"])
        if r.returncode != 0:
            raise SystemExit("合成数据失败")

    trees = [args.tree] if args.tree else list(TREES)
    ok = True
    for t in trees:
        ok &= check_tree(t, args.npz, args.n, args.start, args.gen, args.out,
                         args.keep)
    print("\n%s" % ("PASS：例 3 的 rho'/b1 道全部通过"
                    if ok else "FAIL：见上面的 **FAIL** 行"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
