#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端复验：**全部 14 份 spec** —— emit → 跑产物 → L1/L2/L3 + A1/A7。

比 `check_ex3.py` 覆盖更广（那个只管例 3 的两棵树）。逐 spec 做：

  * **L1**：`emit` 出的产物脚本算的 cache == 直接调内核（rel < 1e-12）
  * **L2**：`helicity == EPS_A · cartesian`（rel < 1e-12）
  * **L3**：有限、非恒零
  * **A1**：级联产物 vs 协变产物，逐道一个**全局**复常数（rel < 1e-12）
  * **A7**：级联 cache 形状/dtype 与 helicity 精确一致

用法：
    python analyses/tensor_examples/probes/verify_all_specs.py
    python analyses/tensor_examples/probes/verify_all_specs.py --spec ex1_pipipi0_rho
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from agent.tensor import emitter, spec as S                        # noqa: E402
from agent.tensor.kernels.helicity import EPS_A                    # noqa: E402

sys.path.insert(0, _HERE)
from probe_a1_full import KIN, cov_event, pick                      # noqa: E402

SPECS = sorted(glob.glob("analyses/tensor_examples/specs/*.toml")) + \
    sorted(glob.glob("agent/tensor/specs/*.toml"))

# 已知的 A1 失配 —— **不是**缺道，是公式结构与逐节点级联不同（有据可查）：
#   ex3_rhopi : Eq.(46) 中间没有投影算符（边界 3，见 chains/notes/rhopi.notes.md）
# 这些道 L1/L2/L3/A7 仍然必须过，只是 A1 单独记账。
#
# ⚠️ `ex5_kst2` 的 `u_PP_*`/`u_FP_*` **2026-10-21 已修**（内层无迹化改到
#    $K^*_2$ 系，见 `agent/tensor/chains/kstpi/CORRECTION.md`），所以不再
#    豁免。历史形式保留在 `*_paper.py`，由对照 spec `ex5_kst2_paper.toml`
#    与 `probe_ex5_paper_form.py` 负责展示。
A1_KNOWN = {
    ("ex3_rhopi", "u_rho_1"), ("ex3_rhopi", "u_rho_2"),
}

# 故意的对照 spec：4 条 `*_paper` 就是历史形式（缺陷本身），A1 必然失败。
A1_SKIP_SPECS = {"ex5_kst2_paper"}

KIN_CACHE = {}


def kin(kind):
    if kind not in KIN_CACHE:
        KIN_CACHE[kind] = KIN[kind]()
    return KIN_CACHE[kind]


def rel(a, b):
    c = np.vdot(a.reshape(-1), b.reshape(-1)) / np.vdot(a.reshape(-1),
                                                         a.reshape(-1))
    return float(np.abs(b - c * a).max() / max(np.abs(b).max(), 1e-300))


def run(script, npz, out):
    r = subprocess.run([sys.executable, script, "--momenta", npz, "--out", out,
                        "--batch", "64"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("%s\n%s" % (r.stdout[-1500:], r.stderr[-1500:]))


def check(spec_path, tmp):
    name = os.path.splitext(os.path.basename(spec_path))[0]
    kind = pick(spec_path)
    momd, keys = kin(kind)
    sp = S.load_spec(spec_path)
    base = S.to_chain(sp)
    cc = S.cascade_channels(sp)
    n = momd[keys[0]].shape[0]
    psi = np.broadcast_to(np.array([0.0, 0.0, 0.0, 3.686]), (n, 4))

    gdir = os.path.join(tmp, name)
    res = emitter.emit(sp, out_dir=gdir)
    kernels = sorted(e["kernel"] for e in res["scripts"])

    npz = os.path.join(gdir, "mom.npz")
    np.savez(npz, **{k: momd[k] for k in keys})

    got = {}
    for e in res["scripts"]:
        if e["kernel"] == "omega":
            continue
        out = os.path.join(gdir, "out_" + e["kernel"])
        run(e["script"], npz, out)
        for fn in os.listdir(out):
            got[(e["kernel"], fn)] = np.load(os.path.join(out, fn))

    # 参考：内核直算
    ref = np.stack([np.asarray(cov_event(sp, c, momd, psi)) for c in cc], axis=0)
    ok = True
    detail = {}

    # ---- L1：产物 == 内核（只比前 3 个洛伦兹分量）
    for (kern, fn), arr in got.items():
        if kern not in ("covariant", "helicity"):
            continue
        a = arr.astype(np.complex128)
        if kern == "helicity":
            a = np.einsum("lm,ecm->ecl", np.linalg.inv(EPS_A), a)
        r = rel(a[..., :3], ref[..., :3])
        ok &= r < 1e-12
        detail["L1_" + kern] = r

    # ---- L2：helicity == EPS_A . cartesian
    ck = [v for (k, _f), v in got.items() if k == "covariant"]
    hl = [v for (k, _f), v in got.items() if k == "helicity"]
    if ck and hl:
        T = np.asarray(ck[0]).astype(np.complex128)
        W = np.asarray(hl[0]).astype(np.complex128)
        Wexp = np.einsum("lm,icm->icl", EPS_A.astype(np.complex128), T)
        r = rel(Wexp, W)
        ok &= r < 1e-12
        detail["L2"] = r

    # ---- L3
    for key, arr in got.items():
        if key[0] == "omega":
            continue
        fin = bool(np.isfinite(arr).all())
        mx = float(np.abs(arr).max())
        ok &= fin and mx > 0
        detail["L3_" + key[0]] = mx

    # ---- A7 + A1：级联 vs helicity
    casc = [v for (k, _f), v in got.items() if k == "helicity_cascade"]
    if casc and hl:
        c0 = np.asarray(casc[0])
        h0 = np.asarray(hl[0])
        same = c0.shape == h0.shape and c0.dtype == h0.dtype
        ok &= bool(same)
        detail["A7_shape"] = 1.0 if same else 0.0
        worst = 0.0
        for i, ch in enumerate(cc):
            r = rel(c0[i], h0[i])
            if (name, ch["name"]) in A1_KNOWN:
                detail["A1_known_" + ch["name"]] = r
            else:
                worst = max(worst, r)
        ok &= worst < 1e-12
        detail["A1"] = worst

    return name, kernels, ok, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="")
    args = ap.parse_args()

    specs = SPECS
    if args.spec:
        specs = [p for p in SPECS if args.spec in p]

    allok = True
    tmp = tempfile.mkdtemp(prefix="verify_all_")
    skipped = []
    try:
        for p in specs:
            nm = os.path.basename(p)[:-5]
            if nm in A1_SKIP_SPECS:
                # 故意的对照 spec（含历史形式），A1 必然失败 —— 整份跳过。
                skipped.append(nm)
                continue
            try:
                name, kernels, ok, d = check(p, tmp)
            except Exception as e:                              # noqa: BLE001
                print("!! %-26s %s: %s" % (os.path.basename(p),
                                           type(e).__name__, e))
                allok = False
                continue
            allok &= ok
            print("%-26s %-46s %s" % (
                name, ",".join(kernels), "OK" if ok else "**FAIL**"))
            for k in sorted(d):
                print("      %-16s %.3e" % (k, d[k]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for nm in skipped:
        print("%-26s %-46s SKIP（对照 spec，含历史形式）" % (nm, "-"))
    print("\n=> %s" % ("全部通过" if allok else "**有失败**"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
