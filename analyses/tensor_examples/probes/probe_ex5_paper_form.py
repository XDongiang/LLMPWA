#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""例 5 `u_PP_*` / `u_FP_*`：**历史形式** vs **修正版** 的逐道对拍。

这两个形式现在都在 `agent/tensor/chains/kstpi/` 里：

| 形式 | 文件 | 用途 |
|---|---|---|
| 历史形式（= `Tensor/FUNC.py` 逐式移植） | `u_PP_1_paper.py` … | 与 jax 对拍（A5）；对照基线 |
| **修正版（生产）** | `u_PP_1.py` … | 实际生成 cache 用 |

本探针同时量三件事：

1. **历史形式**对级联的 A1 残差 —— 应该**大**（这是被修的缺陷）；
2. **修正版**对级联的 A1 残差 —— 应该到机器精度；
3. 两者之间的相对差 —— 应该**非零**（否则等于没改）。

判据 A1 = 经 `EPS_A` 投到螺旋度后，与该道的级联振幅相差一个**全局复常数**
（常数本身不必是 1）。

机制与公式见 [`../../../agent/tensor/chains/kstpi/CORRECTION.md`](
../../../agent/tensor/chains/kstpi/CORRECTION.md)。

    python analyses/tensor_examples/probes/probe_ex5_paper_form.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "agent", "tensor", "tests"))
os.chdir(_ROOT)

import test_ex5_channels as T5                                   # noqa: E402
from agent.tensor import chains as C                             # noqa: E402
from agent.tensor.kernels import helicity_cascade as HC          # noqa: E402
from agent.tensor.kernels.helicity import EPS_A                  # noqa: E402

M_PSI = 3.686

# 级联侧：与 spec `ex5_kst2.toml` 的 `[[channel]].ls` / `.tree` 一致
BASE = {"psi2s": {"J": 1, "mass": M_PSI, "daughters": ["Kstar", "Pim"]},
        "Kstar": {"J": 2, "mass": 1.43, "daughters": ["phi", "Pip"]},
        "phi": {"J": 1, "mass": 1.019, "daughters": ["Kp", "Km"]},
        "Kp": {"J": 0, "mass": 0.493677}, "Km": {"J": 0, "mass": 0.493677},
        "Pip": {"J": 0, "mass": 0.13957039},
        "Pim": {"J": 0, "mass": 0.13957039}}
TREES = {"l": {"psi2s": ["Kstar", "Pip"], "Kstar": ["phi", "Pim"]},
         "r": {"psi2s": ["Kstar", "Pim"], "Kstar": ["phi", "Pip"]}}
INDEX = {"Kp": 0, "Km": 1, "Pip": 2, "Pim": 3}

# 道 -> (顶层 (l,s), 内层 (l,s), 树)
CHANNELS = {
    "u_PP_1": ((1, 2), (1, 1), "l"), "u_PP_2": ((1, 2), (1, 1), "r"),
    "u_FP_1": ((3, 2), (1, 1), "l"), "u_FP_2": ((3, 2), (1, 1), "r"),
}


def cascade(name, d):
    """该道在级联内核下的振幅 (N,3)。"""
    top_ls, in_ls, side = CHANNELS[name]
    mom = np.stack([d[k][:, [3, 0, 1, 2]]
                    for k in ("Kp", "Km", "Pip", "Pim")], axis=1)
    ch = HC.with_tree({"top": "psi2s", "index": INDEX,
                       "nodes": {k: dict(v) for k, v in BASE.items()}},
                      TREES[side])
    return HC.chain_amplitude(mom, ch,
                              {"psi2s": [top_ls], "Kstar": [in_ls],
                               "phi": [(1, 0)]},
                              ang=HC.node_angles(mom, ch))


def fit(a, b):
    """a -> b 的最佳全局复常数，与相对残差。"""
    c = np.vdot(a.reshape(-1), b.reshape(-1)) / np.vdot(a.reshape(-1),
                                                         a.reshape(-1))
    return complex(c), float(np.abs(b - c * a).max() / max(np.abs(b).max(),
                                                          1e-300))


def w_of(T):
    """协变 3-矢量 -> 螺旋度振幅。"""
    return np.einsum("lm,em->el", EPS_A, np.asarray(T)[..., :3])


def main():
    print("=" * 100)
    print("历史形式 vs 修正版（判据 A1：与级联相差一个全局复常数）")
    print("=" * 100)
    print()
    print("  %-9s %-6s %-13s %-13s %-13s"
          % ("道", "seed", "历史形式 rel", "修正版 rel", "两者之差"))
    print("  " + "-" * 74)

    worst_paper, worst_fixed = 0.0, 0.0
    min_delta = 1.0
    for seed in (11, 23, 7):
        d = T5.physical(n=40, seed=seed)
        psi = np.broadcast_to(np.array([0.0, 0.0, 0.0, M_PSI]),
                              (d["Kp"].shape[0], 4))
        args = (psi, d["Kp"] + d["Km"], None, d["Kp"], d["Km"],
                d["Pip"], d["Pim"])
        for nm in CHANNELS:
            A = cascade(nm, d)
            T_paper = C.load_channel("kstpi", nm + "_paper")(*args)
            T_fixed = C.load_channel("kstpi", nm)(*args)
            w_p, w_f = w_of(T_paper), w_of(T_fixed)
            _, r_p = fit(w_p, A)
            _, r_f = fit(w_f, A)
            delta = float(np.abs(w_f - w_p).max() / max(np.abs(w_p).max(),
                                                        1e-300))
            worst_paper = max(worst_paper, r_p)
            worst_fixed = max(worst_fixed, r_f)
            min_delta = min(min_delta, delta)
            print("  %-9s %-6d %-13.3e %-13.3e %-13.3e"
                  % (nm, seed, r_p, r_f, delta))

    print()
    print("  历史形式最差 rel = %.3e  （应该**大**：这是被修的缺陷）" % worst_paper)
    print("  修正版最差 rel   = %.3e  （应该到机器精度）" % worst_fixed)
    print("  两者最小差       = %.3e  （应该非零：否则等于没改）" % min_delta)
    print()
    ok = (worst_paper > 1e-3 and worst_fixed < 1e-12 and min_delta > 1e-3)
    print("  => %s" % ("✅ 修正有效：历史形式不满足 A1，修正版满足，且两者确实不同"
                       if ok else "**异常**：三者关系不符合预期"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
