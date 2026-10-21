#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""手写协变道**完整性**回归（2026-10-21）。

背景：`chains/<chain>/<channel>.py` 是逐道的手工搬运，`emitter` 按
`def <name>` 取体内联。缺一个道，`emit` 到那一步会 `LookupError`；
而"能算"（级联路线）与"能 emit"（协变路线）是两件事。

2026-10-21 之前有 **18 个道、7 个 spec** 缺手写函数（只有 `helicity_cascade`
能出产物）。本文件把"14 个 spec / 48 条道**全部**有手写协变道、且三条路线
都出得来"钉成断言 —— 若哪天删了一个文件，这里会失败。

覆盖：

1. **完整性**：每个 spec 的每条道都能在 `chains/` 里找到（`all_channels()`）；
2. **发现性**：`chains.load_channel` 能导入并调用，返回 (N,4) 有限；
3. **emit 三路线**：`cartesian` / `helicity` / `helicity_cascade` 都能 emit
   （缺手写道的 spec 会在这里 `LookupError`）；
4. **A1 一致性**：级联产物 vs 协变产物，逐道一个全局复常数（rel < 1e-12）
   —— 只对"级联可表示"的道成立，所以例 3 ρ′ 与例 5 的 4 条按已知机制豁免
   （见 `chains/notes/`）。

    python agent/tensor/tests/test_handwritten_complete.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from agent.tensor import chains as chainsmod                    # noqa: E402
from agent.tensor import emitter, spec as specmod               # noqa: E402
from agent.tensor.kernels.helicity import EPS_A                 # noqa: E402

SPECS = sorted(glob.glob(os.path.join(_ROOT, "analyses", "tensor_examples",
                                      "specs", "*.toml"))) + \
    sorted(glob.glob(os.path.join(_ROOT, "agent", "tensor", "specs", "*.toml")))

# 已知的 A1 失配（不是缺道，是公式结构问题 —— 见 chains/notes/）
#
#   ex3_rhopi : Eq.(46) 中间没有投影算符，与逐节点级联是两种振幅（边界 3）
#   ex5_kst2  : u_PP_*/u_FP_* 的内层无迹化用错参考系（**已修**，见
#               chains/kstpi/CORRECTION.md —— 生产文件已改用修正版）
A1_EXEMPT = {
    ("ex3_rhopi.toml", "u_rho_1"), ("ex3_rhopi.toml", "u_rho_2"),
}

# 这份 spec 是**故意的对照实验**：4 条 `*_paper` 就是历史形式（未修正），
# A1 必然失败 —— 它存在的意义就是展示"论文形式 vs 修正形式"的差。
# 所以整份 spec 跳过 A1，只验 L1/L2/L3（emit 与产物自洽）。
A1_SKIP_SPECS = {"ex5_kst2_paper.toml"}


def test_every_spec_channel_has_a_handwritten_file():
    """14 个 spec 的每一条道都能在 `chains/` 里找到。"""
    allc = chainsmod.all_channels()
    missing = []
    for p in SPECS:
        sp = specmod.load_spec(p)
        for c in specmod.cascade_channels(sp):
            if c["name"] not in allc:
                missing.append((os.path.basename(p), c["name"]))
    assert not missing, "缺手写协变道：%s" % missing
    # 道名全局唯一
    assert len(set(allc)) == len(allc)


def test_every_handwritten_file_is_importable_and_finite():
    """每个手写道文件都能导入、调用，对物理四动量返回 (N,4) 且有限。"""
    sys.path.insert(0, os.path.join(_ROOT, "analyses", "tensor_examples",
                                    "probes"))
    from probe_a1_full import KIN, cov_event, pick                # noqa: E402
    kin = {}
    for p in SPECS:
        sp = specmod.load_spec(p)
        cc = specmod.cascade_channels(sp)
        kind = pick(p)
        if kind not in kin:
            kin[kind] = KIN[kind]()
        momd, keys = kin[kind]
        n = momd[keys[0]].shape[0]
        psi = np.broadcast_to(np.array([0.0, 0.0, 0.0, 3.686]), (n, 4))
        for c in cc:
            a = np.asarray(cov_event(sp, c, momd, psi))
            assert a.shape == (n, 4), (p, c["name"], a.shape)
            assert np.all(np.isfinite(a)), (p, c["name"])
            assert np.abs(a).max() > 0, (p, c["name"], "恒零")


def test_emit_produces_three_routes_for_every_spec():
    """每个 spec 都能 emit 出 cartesian / helicity（+ 它自己声明的表象）。

    `emit` 只写生成器脚本（不跑），所以这条不需要四动量、很快。
    """
    import tempfile
    for p in SPECS:
        sp = specmod.load_spec(p)
        with tempfile.TemporaryDirectory() as td:
            res = emitter.emit(sp, out_dir=td)
            kernels = sorted(e["kernel"] for e in res["scripts"])
            want = sorted({b.get("kernel", b["name"]) for b in sp["bases"]})
            assert kernels == want, (os.path.basename(p), kernels, want)
            # 没有 fallback（回退到内核里的 legacy 副本）
            fb = [e.get("fallback") or [] for e in res["scripts"]]
            assert not any(fb), (os.path.basename(p), fb)


def _rel(a, b):
    c = np.vdot(a.reshape(-1), b.reshape(-1)) / np.vdot(a.reshape(-1),
                                                         a.reshape(-1))
    return float(np.abs(b - c * a).max() / max(np.abs(b).max(), 1e-300))


def test_a1_cascade_vs_handwritten_covariant():
    """级联 vs 手写协变，逐道一个全局复常数（rel < 1e-12）。

    只豁免 `A1_EXEMPT` 里那几条 —— 且豁免是**有据可查**的（见文件头）。
    """
    sys.path.insert(0, os.path.join(_ROOT, "analyses", "tensor_examples",
                                    "probes"))
    from probe_a1_full import KIN, cov_event, pick                # noqa: E402
    from agent.tensor.kernels import helicity_cascade as HC       # noqa: E402

    kin = {}
    bad, exempted = [], []
    for p in SPECS:
        name = os.path.basename(p)
        if name in A1_SKIP_SPECS:
            continue                       # 故意的对照 spec（见文件头）
        sp = specmod.load_spec(p)
        base = specmod.to_chain(sp)
        kind = pick(p)
        if kind not in kin:
            kin[kind] = KIN[kind]()
        momd, keys = kin[kind]
        n = momd[keys[0]].shape[0]
        mom = np.stack([momd[k] for k in keys], axis=1)[:, :, [3, 0, 1, 2]]
        psi = np.broadcast_to(np.array([0.0, 0.0, 0.0, 3.686]), (n, 4))
        for c in specmod.cascade_channels(sp):
            ch = HC.with_tree(base, c["tree"] or {})
            A = HC.chain_amplitude(mom, ch, c["ls"], ang=HC.node_angles(mom, ch))
            w = np.einsum("lm,em->el", EPS_A,
                          np.asarray(cov_event(sp, c, momd, psi))[..., :3])
            r = _rel(w, A)
            if (name, c["name"]) in A1_EXEMPT:
                exempted.append((name, c["name"], r))
            elif r >= 1e-12:
                bad.append((name, c["name"], r))
    assert not bad, "A1 失配（未豁免）：%s" % bad
    # 豁免的必须**确实**失配 —— 否则说明它被修好了，该从表里删掉
    for name, cname, r in exempted:
        assert r > 1e-6, (
            "%s/%s 的 A1 残差 %.3e —— 已经对了，请把它从 A1_EXEMPT 删掉"
            % (name, cname, r))


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    ok = 0
    for n, f in tests:
        try:
            f()
            print("  OK   %s" % n)
            ok += 1
        except AssertionError as e:
            print("  FAIL %s: %s" % (n, e))
        except Exception as e:                                   # noqa: BLE001
            print("  ERR  %s: %s: %s" % (n, type(e).__name__, e))
    print("%s: %d/%d" % ("PASS" if ok == len(tests) else "**FAIL**",
                         ok, len(tests)))
    return 0 if ok == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
