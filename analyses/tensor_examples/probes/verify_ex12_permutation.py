#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""例 1/2 的「置换求和」验收。

文档（`document/ψ衰变到介子的分波分析协变张量形式.md` 200–304 行）把振幅
写成**置换求和**：

    U^μ = 第一项 + (1↔3) + (2↔3)

框架的表达方式是「一条道 = 一棵树 = 一种配对」，所以三条道就是三项。
本脚本分两级验证：

**第 1 级：逐道恒等式**（只对 ρ 家族成立）
    每条道 = `s_i·c·ε_{μνλσ}p₁^νp₂^λp₃^σ · B_l(top)·B_l(pair)`，
    `c` 是全局复常数。这一级给"能对上文档显式公式"的信心。

**第 2 级：置换协变性**（对所有 J 成立，这才是真正的判据）
    把末态粒子按 (1↔3) 或 (2↔3) 重标号后，`_12` 道的振幅必须等于
    对应的 `_13` / `_23` 道（相差一个置换符号）。这条不依赖"能不能
    化成单个 ε 张量"，所以 J=3 也能验。

第 2 级能抓住树子节点顺序导致的符号错误 —— 那个坑很隐蔽：每条道单看
都自洽，只有做置换才暴露。

    python analyses/tensor_examples/probes/verify_ex12_permutation.py
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(_TE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.tensor import spec as specmod                       # noqa: E402
from agent.tensor.kernels import helicity_cascade as HC        # noqa: E402
from agent.tensor.kernels import omega as RT                   # noqa: E402

# spec, 相空间, 道名前缀, J, l
CASES = [
    ("ex1_pipipi0_rho.toml", "ex1_pipipi0.npz", "u_rho", 1, 1),
    ("ex1_pipipi0_rho3.toml", "ex1_pipipi0_rho3.npz", "u_rho3", 3, 3),
    ("ex2_kkpi0_rho.toml", "ex2_kkpi0.npz", "u_rho_K", 1, 1),
    ("ex2_kkpi0_kstar.toml", "ex2_kkpi0.npz", "u_kstar", 1, 1),
]

# 例 2 的两族的置换结构不同：
#   ρ' 族只有 (12) 一项（K+K- 对；换 1<->3 会变成 Kπ，不是同一个 ρ'）
#   K* 族有 (13)/(23) 两项（Kπ 对；换 1<->2 不产生新配对，K 同种粒子）
TWO_CHANNEL = {"u_kstar"}

# 三条道的名字 vs 它「把哪两个粒子配成共振」
PAIRING = {
    "u_rho_12": (0, 1), "u_rho_13": (0, 2), "u_rho_23": (1, 2),
    "u_rho3_12": (0, 1), "u_rho3_13": (0, 2), "u_rho3_23": (1, 2),
}


def _load(spec_name, npz_name):
    sp = specmod.load_spec(os.path.join(_TE, "specs", spec_name))
    d = np.load(os.path.join(_TE, "data", npz_name))
    keys = sp["input_keys"]
    # npz 布局 = spec 的 layout（x,y,z,E），内核要 (E,x,y,z)
    mom = np.stack([d[k] for k in keys], axis=1)[:, :, [3, 0, 1, 2]]
    return sp, mom, keys


def _fit(pred, obs):
    """`obs ≈ c·pred` 的 c 与最大相对残差。"""
    p = np.asarray(pred).reshape(-1)
    o = np.asarray(obs).reshape(-1)
    c = np.vdot(p, o) / np.vdot(p, p)
    return complex(c), float(np.abs(o - c * p).max() / np.abs(o).max())


def _amp(sp, base, chans, name, mom):
    c = chans[name]
    chain = HC.with_tree(base, c["tree"])
    return HC.chain_amplitude(mom, chain, c["ls"], covariant=True)


# --------------------------------------------------- 第 2 级：置换协变性
def check_permutation(sp, base, chans, mom, prefix, verbose=True):
    """把某条道在「重标号后的动量」上算，必须等于对应的另一条道（差一个符号）。

    注意这不是"任意两条道相等"，而是**同一棵树 + 重标号动量**等于
    **另一棵树 + 原动量** —— 这才是"置换求和"正确性的来源。

    置换对按各族实际有几条道取：

    * 例 1 的 ρ/ρ₃ 族三条 `{12,13,23}` 齐全 -> 以 `_12` 为起点
    * 例 2 的 ρ′ 族只有 `_12`（K⁺K⁻ 对）-> 没有置换伙伴，跳过
    * 例 2 的 K* 族只有 `_13/_23`（Kπ 对）-> 以 `_13` 为起点试 (0<->1)
    """
    assert mom.shape[1:] == (len(sp["input_keys"]), 4), mom.shape

    if prefix + "_12" in chans:
        start, pairs = prefix + "_12", []
        if prefix + "_13" in chans:
            pairs.append((prefix + "_13", (1, 2)))
        if prefix + "_23" in chans:
            pairs.append((prefix + "_23", (0, 2)))
    elif prefix + "_13" in chans and prefix + "_23" in chans:
        # K* 族：K*(13) 把 K+ 与 K- 的角色互换 -> K*(23)
        start, pairs = prefix + "_13", [(prefix + "_23", (0, 1))]
    else:
        if verbose:
            print("    跳过：该族只有一条道，没有置换伙伴")
        return 0.0

    if not pairs:
        if verbose:
            print("    跳过：该族只有一条道，没有置换伙伴")
        return 0.0

    worst = 0.0
    for other, (i, j) in pairs:
        perm = list(range(mom.shape[1]))
        perm[i], perm[j] = perm[j], perm[i]
        moved = _amp(sp, base, chans, start, mom[:, perm, :])
        c, r = _fit(moved, _amp(sp, base, chans, other, mom))
        worst = max(worst, r)
        if verbose:
            print("    %s(交换 %d<->%d 的动量) vs %-12s 残差=%.2e  符号=%s"
                  % (start, i, j, other, r, "+" if c.real > 0 else "-"))
        assert r < 1e-12, "%s 置换协变性残差 %.3e" % (other, r)
        assert abs(abs(c) - 1.0) < 1e-9, "|c| 应为 1，得到 %s" % abs(c)
    return worst


# --------------------------------------------------- 第 1 级：逐道恒等式
def check_eps_identity(sp, base, chans, mom, prefix, l, verbose=True):
    """ρ 家族：每道 = s·c·ε·p₁p₂p₃·B_l(top)·B_l(pair)，c 全局。"""
    from agent.tensor.kernels.covariant import leci_four
    from agent.tensor.kernels.helicity import EPS_A

    e4 = np.einsum("mnls,en,el,es->em", leci_four(),
                   mom[:, 0], mom[:, 1], mom[:, 2])
    eps = np.einsum("al,el->ea", EPS_A, e4[:, 1:4])

    names = [prefix + s for s in ("_12", "_13", "_23")]
    names = [n for n in names if n in chans]
    cs, rows = [], []
    for nm in names:
        chain = HC.with_tree(base, chans[nm]["tree"])
        A = HC.chain_amplitude(mom, chain, chans[nm]["ls"], covariant=True)
        # 文档写 B_l(Q)；框架的 node_barrier 是 q^l B_l(q^2)，
        # q^l 已在张量里，所以这里只用 Bl。
        B = np.ones(mom.shape[0])
        for n, v in chain["nodes"].items():
            ds = v.get("daughters")
            if not ds:
                continue
            pa = RT._node_momentum(mom, chain, n)
            pb = RT._node_momentum(mom, chain, ds[0])
            pc = RT._node_momentum(mom, chain, ds[1])
            B = B * np.asarray(RT.Bl(int(l), RT.q2_relative(pa, pb, pc)))
        c, r = _fit(eps * B[:, None], A)
        cs.append(c)
        rows.append((nm, r, B, A))

    if verbose:
        print("    %-12s %-10s %-14s %s" % ("道", "符号", "|c|", "残差"))
        for (nm, r, _B, _A), c in zip(rows, cs):
            print("    %-12s %-10s %-14.9f %.2e"
                  % (nm, "+1" if c.imag > 0 else "-1", abs(c), r))
    for nm, r, _B, _A in rows:
        assert r < 1e-12, "%s: ε·B 恒等式残差 %.3e" % (nm, r)
    spread = np.abs(np.abs(cs) / abs(cs[0]) - 1).max()
    assert spread < 1e-9, "各道的 |c| 不一致：%.3e" % spread
    return cs


def main():
    bad = 0
    for spec_name, npz_name, prefix, J, l in CASES:
        print("=== %s（J=%d）===" % (spec_name, J))
        sp, mom, keys = _load(spec_name, npz_name)
        base = specmod.to_chain(sp)
        chans = {c["name"]: c for c in specmod.cascade_channels(sp)}

        print("  [置换协变性]")
        try:
            check_permutation(sp, base, chans, mom, prefix)
            print("    OK")
        except AssertionError as e:
            bad += 1
            print("    FAIL: %s" % e)

        print("  [逐道 eps·B 恒等式]")
        if J == 1:
            try:
                cs = check_eps_identity(sp, base, chans, mom, prefix, l)
                print("    OK（符号 %s）"
                      % ["+" if c.imag > 0 else "-" for c in cs])
            except AssertionError as e:
                bad += 1
                print("    FAIL: %s" % e)
        else:
            # J>=3 时文档的 T~^(3) 是秩 2 张量，ε·p_ψ 缩并后**不能**化成一个
            # ε_{μνλσ}p1p2p3；所以这一级只对 J=1 有意义。这不是缺陷。
            print("    跳过：J=%d 的 ε·p_ψ·T~·t~ 不能化简为单个 ε 张量" % J)
        print()

    print("%s: %d 项失败" % ("PASS" if not bad else "**FAIL**", bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
