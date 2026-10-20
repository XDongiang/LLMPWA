#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""例 1（`ψ→π⁺π⁻π⁰`）与例 2（`ψ→K⁺K⁻π⁰`）的「置换求和」回归。

文档（`document/ψ衰变到介子的分波分析协变张量形式.md` 200–304 行）把三体
振幅写成**置换求和**。框架的表达方式是「一条道 = 一棵树 = 一种配对」，
所以 N 项就对应 N 条道。本文件验两件**结构**上的事：

**1. 置换协变性**（所有 J 都成立，这是主判据）

    把某条道在「末态粒子重标号后的动量」上算，必须等于对应的另一条道
    （相差一个模为 1 的符号）：

        A_12(交换 slot i<->j 的动量) == c · A_other(原动量)，|c| = 1

    这条不依赖"振幅能否化成单个 ε 张量"，所以 J=3 也能验。它能抓住
    **树子节点顺序**导致的符号错误 —— 那个坑很隐蔽：每条道单看都自洽，
    只有做置换才暴露（例 1 一开始就是 `B12 − B13 + B23` 而不是全加）。

**2. 逐道 `ε·B` 恒等式**（只对 J=1 成立）

    每条道 = `s_i · c · ε_{μνλσ}p₁^νp₂^λp₃^σ · B_l(top)·B_l(pair)`，
    `c` 是对全部道、全部事例都相同的复常数。

    J≥3 时文档的 `T̃⁽³⁾` 是秩 2 张量，`ε·p_ψ` 缩并后**不能**化简成单个
    `ε_{μνλσ}p₁p₂p₃`（那是 J=1 的特殊性），所以这一级对 ρ₃/K*₃ 不适用 ——
    不是缺陷。

相空间在文件内自己合成（三体：ψ→R+spec，R→pair），不依赖 `data/*.npz`
（那些是生成物，不入库）。

    python agent/tensor/tests/test_ex12_permutation.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.tensor import spec as specmod                       # noqa: E402
from agent.tensor.kernels import helicity_cascade as HC        # noqa: E402
from agent.tensor.kernels import omega as RT                   # noqa: E402

M_PSI = 3.686
M_PI, M_PI0, M_K = 0.13957039, 0.1349768, 0.493677
SPECS = os.path.join(_ROOT, "analyses", "tensor_examples", "specs")

# spec 文件, 道名前缀, J, l, 末态质量, 造相空间用的中间态质量
CASES = {
    "ex1_rho": ("ex1_pipipi0_rho.toml", "u_rho", 1, 1,
                (M_PI, M_PI, M_PI0), 1.5),
    "ex1_rho3": ("ex1_pipipi0_rho3.toml", "u_rho3", 3, 3,
                 (M_PI, M_PI, M_PI0), 2.2),
    "ex2_rho": ("ex2_kkpi0_rho.toml", "u_rho_K", 1, 1,
                (M_K, M_K, M_PI0), 1.9),
    "ex2_kstar": ("ex2_kkpi0_kstar.toml", "u_kstar", 1, 1,
                  (M_K, M_K, M_PI0), 1.9),
}


# ------------------------------------------------------------- 三体相空间
def _pair(m0, m1, m2, rng, n):
    """母粒子静止系两体衰变，(x,y,z,E)。"""
    lam = (m0**2 - (m1 + m2) ** 2) * (m0**2 - (m1 - m2) ** 2)
    p = np.sqrt(max(lam, 0.0)) / (2.0 * m0)
    ct = rng.uniform(-1.0, 1.0, n)
    st = np.sqrt(np.maximum(1.0 - ct**2, 0.0))
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    d = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)
    return (np.concatenate([p * d, np.full((n, 1), np.sqrt(m1**2 + p**2))], axis=1),
            np.concatenate([-p * d, np.full((n, 1), np.sqrt(m2**2 + p**2))], axis=1))


def _boost(p, beta):
    e, v = p[:, 3], p[:, :3]
    b2 = (beta**2).sum(-1)
    g = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-300))
    bp = (beta * v).sum(-1)
    g2 = np.where(b2 > 1e-12, (g - 1.0) / np.where(b2 > 1e-12, b2, 1.0), 0.0)
    return np.concatenate(
        [v + (g2 * bp + g * e)[:, None] * beta, (g * (e + bp))[:, None]], axis=1)


def threebody(mass, m_res, n=64, seed=5):
    """ψ → R(→1+2) + 3 的合法三体相空间，返回 `(N, 3, 4)` 的 (E,x,y,z)。"""
    rng = np.random.default_rng(seed)
    res, p3 = _pair(M_PSI, m_res, mass[2], rng, n)
    p1, p2 = _pair(m_res, mass[0], mass[1], rng, n)
    b = res[:, :3] / res[:, 3:4]
    mom = np.stack([_boost(p1, b), _boost(p2, b), p3], axis=1)   # (x,y,z,E)
    return mom[:, :, [3, 0, 1, 2]]                               # -> (E,x,y,z)


def _fit(pred, obs):
    """`obs ≈ c·pred` 的 c 与最大相对残差。"""
    p = np.asarray(pred).reshape(-1)
    o = np.asarray(obs).reshape(-1)
    c = np.vdot(p, o) / np.vdot(p, p)
    return complex(c), float(np.abs(o - c * p).max() / np.abs(o).max())


def _load(key):
    spec_name, prefix, J, l, mass, m_res = CASES[key]
    sp = specmod.load_spec(os.path.join(SPECS, spec_name))
    mom = threebody(mass, m_res)
    base = specmod.to_chain(sp)
    chans = {c["name"]: c for c in specmod.cascade_channels(sp)}
    return sp, base, chans, mom, prefix, J, l


def _amp(base, chans, name, mom):
    c = chans[name]
    return HC.chain_amplitude(mom, HC.with_tree(base, c["tree"]),
                              c["ls"], covariant=True)


# --------------------------------------------------------- 1) 置换协变性
def test_permutation_covariance():
    """重标号动量后，一条道必须等于对应的另一条道（|c| = 1）。"""
    for key in CASES:
        sp, base, chans, mom, prefix, J, _l = _load(key)
        # 起点与配对：按该族实际有几条道取
        if prefix + "_12" in chans:
            start, pairs = prefix + "_12", []
            if prefix + "_13" in chans:
                pairs.append((prefix + "_13", (1, 2)))
            if prefix + "_23" in chans:
                pairs.append((prefix + "_23", (0, 2)))
        elif prefix + "_13" in chans and prefix + "_23" in chans:
            # K* 族：把 K+ 与 K- 的角色互换 -> 另一种 Kπ 配对
            start, pairs = prefix + "_13", [(prefix + "_23", (0, 1))]
        else:
            # 例 2 的 ρ′ 族只有 KK̄ 一对，没有置换伙伴
            continue
        for other, (i, j) in pairs:
            perm = list(range(mom.shape[1]))
            perm[i], perm[j] = perm[j], perm[i]
            moved = _amp(base, chans, start, mom[:, perm, :])
            c, r = _fit(moved, _amp(base, chans, other, mom))
            assert r < 1e-12, "%s: %s 置换残差 %.3e" % (key, other, r)
            assert abs(abs(c) - 1.0) < 1e-9, (
                "%s: %s 的 |c| = %.9f，应为 1" % (key, other, abs(c)))


# ------------------------------------------------------- 2) ε·B 恒等式
def test_eps_barrier_identity_j1():
    """J=1 的每条道 = s·c·ε·p₁p₂p₃·B_l(top)·B_l(pair)，c 为全局常数。"""
    from agent.tensor.kernels.covariant import leci_four
    from agent.tensor.kernels.helicity import EPS_A

    for key in CASES:
        sp, base, chans, mom, prefix, J, l = _load(key)
        if J != 1:
            continue        # J>=3 的 T~ 不能化简成单个 ε 张量
        e4 = np.einsum("mnls,en,el,es->em", leci_four(),
                       mom[:, 0], mom[:, 1], mom[:, 2])
        eps = np.einsum("al,el->ea", EPS_A, e4[:, 1:4])

        cs = []
        for nm in sorted(chans):
            chain = HC.with_tree(base, chans[nm]["tree"])
            A = HC.chain_amplitude(mom, chain, chans[nm]["ls"], covariant=True)
            # 文档写 B_l(Q)；框架 node_barrier 是 q^l B_l(q^2)，q^l 已在张量里
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
            assert r < 1e-12, "%s/%s: ε·B 恒等式残差 %.3e" % (key, nm, r)
            cs.append(c)
        spread = np.abs(np.abs(cs) / abs(cs[0]) - 1).max()
        assert spread < 1e-9, "%s: 各道 |c| 不一致 %.3e" % (key, spread)


# ------------------------------------------------------------ 3) spec 自检
def test_specs_pass_check():
    """四份 spec 的拓扑 / 量子数 / 道表 / 表象校验全过。

    直接调 `specmod.load_spec`（它内部就 `validate`），不调 CLI ——
    CLI 会往 stdout 打整份拓扑树，把这个测试套件的输出淹掉。
    """
    for key, (spec_name, *_rest) in CASES.items():
        path = os.path.join(SPECS, spec_name)
        sp = specmod.load_spec(path)           # 内部 validate() 失败会抛异常
        assert sp["bases"], key
        assert sp["leaves"], key


# -------------------------------------------------- 4) 三体末态是真的三体
def test_phase_space_is_genuine_threebody():
    """相空间满足四动量守恒，且三个末态粒子的不变质量都高于阈值。"""
    for key in CASES:
        sp, base, chans, mom, prefix, J, l = _load(key)
        mass = CASES[key][4]
        e = mom[:, :, 0].sum(1)
        px = mom[:, :, 1].sum(1)
        assert np.allclose(e, M_PSI, atol=1e-10), e[:3]
        assert np.allclose(px, 0.0, atol=1e-10), px[:3]
        # 两两不变质量必须 >= 两粒子质量之和（否则不是物理的三体相空间）
        for i, j in ((0, 1), (0, 2), (1, 2)):
            mij = np.sqrt(np.maximum(
                (mom[:, i, 0] + mom[:, j, 0]) ** 2
                - ((mom[:, i, 1:] + mom[:, j, 1:]) ** 2).sum(-1), 0.0))
            assert mij.min() >= mass[i] + mass[j] - 1e-9, (
                "%s: m(%d,%d) 低于阈值" % (key, i, j))


# ------------------------------------------- 5) emit 端到端（产物 == 内核）
def test_emit_end_to_end():
    """`emit` 出的 cascade 产物，与直接调内核**逐位相同**。

    两件事一起验：

    * 本套四个 spec 都**没有手写道函数**，所以 `emit` 应当只出
      `helicity_cascade`（`cartesian`/`helicity` 会 `LookupError`）；
    * 产物脚本跑出来的 `.npy` 与直接调内核逐位一致（生成器内联的代码
      与内核是同一份，不该有任何差别）。

    注意 `emitter.emit` 只写**生成器脚本**，`.npy` 要跑脚本才产出。
    """
    import subprocess
    import tempfile

    from agent.tensor import emitter

    for key, (spec_name, prefix, J, l, mass, m_res) in CASES.items():
        sp = specmod.load_spec(os.path.join(SPECS, spec_name))
        base = specmod.to_chain(sp)
        chans = specmod.cascade_channels(sp)
        mom = threebody(mass, m_res)

        with tempfile.TemporaryDirectory() as td:
            res = emitter.emit(sp, out_dir=td)
            kernels = [e["kernel"] for e in res["scripts"]]
            assert kernels == ["helicity_cascade"], (key, kernels)

            src = open(res["scripts"][0]["script"], encoding="utf-8").read()
            assert "import agent" not in src, key     # 产物自足

            # 写回 spec 的 layout（x,y,z,E）再跑产物脚本
            keys = sp["input_keys"]
            npz = os.path.join(td, "mom.npz")
            np.savez(npz, **{keys[i]: mom[:, i][:, [1, 2, 3, 0]]
                             for i in range(len(keys))})
            out = os.path.join(td, "out")
            r = subprocess.run([sys.executable, res["scripts"][0]["script"],
                                "--momenta", npz, "--out", out],
                               capture_output=True, text=True)
            assert r.returncode == 0, "%s: 产物脚本失败\n%s%s" % (
                key, r.stdout, r.stderr)

            # 产物名 = <chain_name>_hel.npy（`emitter.py:567`；suffix 只用于
            # 生成器脚本内部的参数，不进文件名）
            chain_name = (sp["raw"].get("chain", {}).get("chain_name")
                          or sp["name"])
            want = np.load(os.path.join(out, chain_name + "_hel.npy"))

            ours = np.stack([
                HC.chain_amplitude(mom, HC.with_tree(base, c["tree"]),
                                   c["ls"], covariant=True)
                for c in chans], axis=0)
            assert ours.shape == want.shape, (key, ours.shape, want.shape)
            d = np.abs(ours - want).max() / np.abs(want).max()
            assert d < 1e-12, "%s: emit 产物与内核差 %.3e" % (key, d)


# ---------- 6) omega 对"置换族"是静默错的（回归护栏）
def test_omega_is_silently_wrong_for_permutation_family():
    """守住"置换族不能用 omega"这个结论，并留下它**静默错**的证据。

    `omega.omega_from_momenta(mom, chain)` 只吃 **base chain**
    （`to_chain(spec)`），返回**一份逐道共用**的 `Omega`。三条置换道的树
    不同（配对、旁观者都不同）却共用它，所以只有与 base 树一致的那条道
    对得上 —— 而且**不会报错**。

    实测（`ex1_pipipi0_rho`）：`u_rho_12` 残差 2e-16（对），
    `u_rho_13`/`u_rho_23` 残差 0.27–0.30（错）。这条测试把这个不对称钉住：
    如果哪天 omega 修好了，这里会失败，提醒把本测试反过来写。

    ⚠️ 判据不是"三体"，也不是"顶层两子节点都衰变" —— `ex2_kkpi0_rho`
    也是三体、顶层也是"一个衰变一个末态"，但它**只有一条道**、树与 base
    相同，omega 实测残差 3.3e-16（**正确**）。正确判据是
    "**是否存在道，其 tree 与 base 树不同**"。
    """
    import subprocess
    import tempfile

    from agent.tensor import emitter

    spec_name, prefix, _J, _l, mass, m_res = CASES["ex1_rho"]
    sp = specmod.load_spec(os.path.join(SPECS, spec_name))
    base = specmod.to_chain(sp)
    chans = specmod.cascade_channels(sp)
    mom = threebody(mass, m_res)

    sp2 = dict(sp)
    sp2["bases"] = [{"name": "omega", "kernel": "omega"}]
    with tempfile.TemporaryDirectory() as td:
        res = emitter.emit(sp2, out_dir=td)
        keys = sp["input_keys"]
        npz = os.path.join(td, "m.npz")
        np.savez(npz, **{keys[i]: mom[:, i][:, [1, 2, 3, 0]]
                         for i in range(len(keys))})
        out = os.path.join(td, "o")
        r = subprocess.run([sys.executable, res["scripts"][0]["script"],
                            "--momenta", npz, "--out", out],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

        omega = np.load(os.path.join(out, "Omega_pipipi0.npy"))
        K = np.load(os.path.join(out, "K_pipipi0.npy"))
        M = np.load(os.path.join(out, "M_pipipi0.npy"))
        g = np.load(os.path.join(out, "g_pipipi0.npy"))
        W = (np.einsum("iab,enab->ien", M, omega)
             * K[:, :, None] * g[:, None, None])

        ours = np.stack([
            HC.chain_amplitude(mom, HC.with_tree(base, c["tree"]),
                               c["ls"], covariant=True)
            for c in chans], axis=0)

        rels = [_fit(W[i], ours[i])[1] for i in range(len(chans))]

        # 第一条道是对的 —— 说明 omega 不是"整体坏掉"，而是树结构错配
        assert rels[0] < 1e-12, (
            "omega 对 %s 应当正确（残差 %.3e）" % (chans[0]["name"], rels[0]))
        # 另外两条静默错：残差必须是 O(0.1)，不是数值噪声
        for i in (1, 2):
            assert rels[i] > 1e-3, (
                "omega 对 %s 居然对了（残差 %.3e）—— 如果 omega 修好了，"
                "请把这条测试反过来写" % (chans[i]["name"], rels[i]))


def test_omega_guard_rejects_permutation_family():
    """`spec.py` 必须**拒绝**"置换族 + omega"，但放行单道三体。

    这条守住 `validate` 里那条新加的拦截，防止有人以为"三体就该禁 omega"
    而写成过严的判据（那会误伤 `ex2_kkpi0_rho`）。
    """
    # 1) 置换族 + omega -> 必须拒绝，且错误信息要点出"树与 base chain 不同"
    for key in ("ex1_rho", "ex1_rho3", "ex2_kstar"):
        spec_name = CASES[key][0]
        sp = specmod.load_spec(os.path.join(SPECS, spec_name))
        sp2 = dict(sp)
        sp2["bases"] = [{"name": "omega", "kernel": "omega"}]
        try:
            specmod.validate(sp2)
        except specmod.SpecError as e:
            msg = str(e)
            assert "树" in msg and "omega" in msg, (key, msg)
        else:
            raise AssertionError("%s + omega 应该被拒绝" % key)

    # 2) 单道三体 + omega -> 必须放行（实测它是对的，残差 3.3e-16）
    spec_name = CASES["ex2_rho"][0]
    sp = specmod.load_spec(os.path.join(SPECS, spec_name))
    sp2 = dict(sp)
    sp2["bases"] = [{"name": "omega", "kernel": "omega"}]
    specmod.validate(sp2)          # 不抛异常即通过


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print("  OK   %s" % fn.__name__)
        except Exception as e:                                  # noqa: BLE001
            bad += 1
            print("  FAIL %s: %s" % (fn.__name__, e))
    print("%s: %d/%d" % ("PASS" if not bad else "**FAIL**",
                         len(fns) - bad, len(fns)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
