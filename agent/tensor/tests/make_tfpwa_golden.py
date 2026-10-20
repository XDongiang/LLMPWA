#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重新生成 `test_helicity_vs_tfpwa.py` 用的 tf-pwa 金标准（fixture）。

**只有在需要更新金标准时才跑这个脚本** —— 它需要 TensorFlow + tf-pwa，
本仓库的运行环境两者都没有。日常跑测试只用 fixture，不碰这个脚本。

    # 在装了 tf-pwa 的环境里（例如远程 tf-pwa-check docker）
    TFPWA=<tf-pwa 源码目录> python make_tfpwa_golden.py > tfpwa_b1pi_golden.json

本机没有 TensorFlow 也没有 tf-pwa，所以这一步只能在有 tf-pwa 的机器上做。
实测可用的做法（HEP1 上的 docker，tf-pwa 源码挂在 /w）：

    docker run --rm -v $PWD:/s -v /path/to/tf-pwa:/w -w /s \
        -e TFPWA=/w -e PYTHONPATH=/w tf-pwa-check:latest \
        /opt/tfpwa/bin/python /s/make_tfpwa_golden.py > out.txt

那个镜像的 stdout 前面会带一段 CUDA 横幅，所以要从第一个 `{` 开始截：

    python -c "s=open('out.txt').read(); open('tfpwa_b1pi_golden.json','w') \
        .write(s[s.index('{\"seed\"'):])"

生成后把 `tfpwa_b1pi_golden.json` 放到本目录，跑
`python agent/tensor/tests/test_helicity_vs_tfpwa.py` 应 4/4 通过。

--------------------------------------------------------------------------
金标准内容（例 3 的 b1pi 链，8 条道，seed=11，N=8）
--------------------------------------------------------------------------

1. `angles` —— tf-pwa `cal_angle_from_momentum` 的逐节点 `(alpha, beta)`；
2. `cg`     —— tf-pwa `HelicityDecay.get_cg_matrix()` 的 CG 矩阵；
3. `channels` —— tf-pwa 的**链振幅** `get_amp3`，flags 见下。

**为什么位垒要关掉**：tf-pwa 的 Blatt-Weisskopf 用 `d = 3`（GeV⁻¹）参数化

    B'_l(q²) = sqrt( P_l(q₀²d²) / P_l(q²d²) )

而本仓库（以及项目自己的 jax 参考实现 `Tensor/BASE.py`）用 `Q0 = 0.197321`

    B_l(q²) = sqrt( P_l(Q0²) / P_l(q²) )，q^l B_l

两者是**不同的约定**，不是同一个函数（实测形状都不同，不只是差常数）。
另外 tf-pwa 在 `add_covariant_term=True` 时还会往位垒里塞一个逐事例的
`mass` 因子。所以"螺度计算对不对"这一层必须在位垒因子**之外**比：
fixture 里存的是 `has_barrier_factor=False` 的振幅，测试端把本仓库的
位垒乘子除掉再比。

**`add_covariant_term=True`**：对应本仓库 `chain_amplitude(..., covariant=True)`
的 Chung Eq.52 项（`f^J_{|lam|}(gamma)`）。这一项是螺度形式的一部分，要打开。

本脚本每次运行都重建 Particle 对象并重算 angle data（tf-pwa 的 `Particle`
会累积 `.decay`，复用对象会串味）。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("TFPWA", "/w"))

from tf_pwa.amp.core import DecayGroup, HelicityDecay, Particle   # noqa: E402
from tf_pwa.cal_angle import cal_angle_from_momentum              # noqa: E402

M_PSI, M_PHI, M_K, M_PI = 3.686, 1.019, 0.493677, 0.13957039
M_B1, M_F2 = 1.23, 1.275
SEED, NEV = 11, 8


# --- 与 agent/tensor/tests/test_ex3_channels.py::physical 逐字一致的合成 ---
def _pair(m_parent, m1, m2, rng, n):
    """母粒子静止系两体衰变，(x,y,z,E)。"""
    lam = ((m_parent ** 2 - (m1 + m2) ** 2)
           * (m_parent ** 2 - (m1 - m2) ** 2))
    p = np.sqrt(max(lam, 0.0)) / (2.0 * m_parent)
    ct = rng.uniform(-1.0, 1.0, n)
    st = np.sqrt(np.maximum(1.0 - ct ** 2, 0.0))
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    d = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)
    e1 = np.sqrt(m1 ** 2 + p ** 2)
    e2 = np.sqrt(m2 ** 2 + p ** 2)
    return (np.concatenate([p * d, np.full((n, 1), e1)], axis=1),
            np.concatenate([-p * d, np.full((n, 1), e2)], axis=1))


def _boost(p, beta):
    b2 = (beta ** 2).sum(-1)
    g = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-300))
    bp = (beta * p[:, :3]).sum(-1)
    g2 = np.where(b2 > 1e-12, (g - 1.0) / np.where(b2 > 1e-12, b2, 1.0), 0.0)
    return np.concatenate(
        [p[:, :3] + (g2 * bp + g * p[:, 3])[:, None] * beta,
         (g * (p[:, 3] + bp))[:, None]], axis=1)


def physical(n=NEV, seed=SEED):
    """psi -> phi f2 -> (K+K-)(pi+pi-) 的物理事例，(x,y,z,E) 布局。"""
    rng = np.random.default_rng(seed)
    phi, f2 = _pair(M_PSI, M_PHI, M_F2, rng, n)
    kp0, km0 = _pair(M_PHI, M_K, M_K, rng, n)
    pip0, pim0 = _pair(M_F2, M_PI, M_PI, rng, n)
    bphi = phi[:, :3] / phi[:, 3:4]
    bf2 = f2[:, :3] / f2[:, 3:4]
    return {"Kp": _boost(kp0, bphi), "Km": _boost(km0, bphi),
            "Pip": _boost(pip0, bf2), "Pim": _boost(pim0, bf2)}


MOMD = physical()
# tf-pwa 的四动量布局是 (T,X,Y,Z)
TFPWA_MOM = {k: MOMD[k][:, [3, 0, 1, 2]] for k in MOMD}

# 道名 -> (顶层节点 (l,s), b1 节点 (l,s))；phi 节点恒为 (1,0)
LS = {"SS": ((0, 1), (0, 1)), "SD": ((0, 1), (2, 1)),
      "DS": ((2, 1), (0, 1)), "DD": ((2, 1), (2, 1))}


def build(tag, top_ls, inner_ls):
    """建一条 b1pi 链。`tag`='1' -> 旁观者 Pim；'2' -> 旁观者 Pip。

    每次都新建 Particle 对象（tf-pwa 会往对象上挂 `.decay`）。
    """
    spec = "Pim" if tag == "1" else "Pip"
    other = "Pip" if tag == "1" else "Pim"
    psi = Particle("psi", J=1, P=-1, mass=M_PSI)
    b1 = Particle("b1", J=1, P=1, mass=M_B1)
    phi = Particle("phi", J=1, P=-1, mass=M_PHI)
    sp = Particle(spec, J=0, P=-1, mass=M_PI)
    ot = Particle(other, J=0, P=-1, mass=M_PI)
    kp = Particle("Kp", J=0, P=-1, mass=M_K)
    km = Particle("Km", J=0, P=-1, mass=M_K)
    # 位垒关掉：与我们的 Q0=0.197321 约定不同，见模块 docstring
    kw = dict(add_covariant_term=True, has_barrier_factor=False)
    HelicityDecay(psi, [b1, sp], ls_list=[list(top_ls)], **kw)
    HelicityDecay(b1, [phi, ot], ls_list=[list(inner_ls)], **kw)
    HelicityDecay(phi, [kp, km], ls_list=[[1, 0]], **kw)
    dg = DecayGroup(psi.chain_decay())
    dg.init_params()
    p = {sp: TFPWA_MOM[spec], ot: TFPWA_MOM[other],
         kp: TFPWA_MOM["Kp"], km: TFPWA_MOM["Km"]}
    data = cal_angle_from_momentum(p, dg, center_mass=False)
    return dg, data


def main():
    out = {
        "seed": SEED,
        "n_ev": NEV,
        "convention": {
            "generator": "agent/tensor/tests/make_tfpwa_golden.py",
            "tfpwa_commit": "e503951 (v0.2.2-209)",
            "add_covariant_term": True,
            "has_barrier_factor": False,
            "note": "tf-pwa 位垒用 d=3 参数化，本仓库用 Q0=0.197321，两者不同；"
                    "故 fixture 存无位垒振幅，测试端除掉本仓库位垒再比",
        },
        "channels": {},
        "angles": {},
        "cg": {},
        "momenta": {k: v.tolist() for k, v in TFPWA_MOM.items()},
    }

    # ---- 1) 逐节点角（任意一条链的树都是同一组节点） ---------------------
    dg1, data1 = build("1", (0, 1), (0, 1))
    for chain_key, dec_map in data1["decay"].items():
        # dec_map: {衰变节点: {子粒子: {"ang": {"alpha","beta","gamma"}, ...}}}
        for dec_i, pmap in dec_map.items():
            for pobj, entry in pmap.items():
                if not isinstance(entry, dict) or "ang" not in entry:
                    continue
                ang = entry["ang"]
                out["angles"].setdefault(str(dec_i), {})[str(pobj)] = {
                    "alpha": np.asarray(ang["alpha"]).tolist(),
                    "beta": np.asarray(ang["beta"]).tolist(),
                }

    # ---- 2) CG 矩阵 ------------------------------------------------------
    # 顶层 psi(J=1) -> b1(J=1) + pi(J=0)：ls_list 同时给 (0,1) 与 (2,1)
    d_top = HelicityDecay(Particle("A", J=1, P=-1),
                          [Particle("B", J=1, P=1), Particle("C", J=0, P=-1)],
                          ls_list=[[0, 1], [2, 1]])
    out["cg"]["top_J1_1_0"] = np.asarray(d_top.get_cg_matrix()).real.tolist()
    # 内层 b1(J=1) -> phi(J=1) + pi(J=0)
    d_inn = HelicityDecay(Particle("A", J=1, P=-1),
                          [Particle("B", J=1, P=-1), Particle("C", J=0, P=-1)],
                          ls_list=[[0, 1], [2, 1]])
    out["cg"]["inner_J1_1_0"] = np.asarray(d_inn.get_cg_matrix()).real.tolist()

    # ---- 3) 无位垒链振幅 -------------------------------------------------
    for tag in ("1", "2"):
        for nm, (tl, il) in LS.items():
            dg, data = build(tag, tl, il)
            # (N, 3, 1, 1, 1, 1) -> 取 (lam_b, lam_c) = (0, 0) 槽位 -> (N, 3)
            amp = np.asarray(dg.get_amp3(data))[..., 0, 0, 0, 0]
            key = "u_%s_%s" % (nm, tag)
            # JSON 不能存复数：拆成 re / im
            out["channels"][key] = {"re": amp.real.tolist(),
                                    "im": amp.imag.tolist()}
            print("  %-9s |amp|max=%.6e" % (key, np.abs(amp).max()),
                  file=sys.stderr)

    print(json.dumps(out))


if __name__ == "__main__":
    main()
