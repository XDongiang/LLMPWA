#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 J=5 / l=5 的 tf-pwa 金标准（外部验证协变张量路线的 J=5）。

背景
----
2026-10-20 给 `kernels/covariant.py` 补了论文 Eq.(258) 的通用 `t_l`，协变
张量路线第一次能表达 J=5（ρ₅ / K*₅）。本仓库内部已经验过：

  * `t_l(1..4)` 与写死的 `T1_cov..T4_cov` 逐位一致（rel ~ 1e-16）；
  * ρ₅ 的 cartesian/helicity/cascade 三条路线互相一致（rel ~ 4e-15）。

但那些都是**内部自洽**。这里用 tf-pwa 做**外部**对拍。

⚠️ 为什么必须把配对质量放在壳上
-------------------------------
tf-pwa 的 `has_barrier_factor=False` 关掉的是 **B_l**，但它仍然乘
`q^l`（`has_ql`，见 `tf_pwa/amp/core.py::get_barrier_factor2`）；而论文的
`t~^(l)` 里是 `Q_abc^l B_l(Q_abc)`，本仓库的 `t_l` 只有 `B_l`，`q^l` 那
一半由拟合参数 `g_i` 吸收（与 BW 同处，见 `tensor_examples_feasibility.md`）。

也就是说**位垒口径本来就不一样**，这是既有约定差异，不是 bug。既有
fixture（`make_tfpwa_golden.py`）的做法是"在位垒之外比"。对 J=5 我们采用
更干净的做法：**让配对质量恰好等于共振质量 m_ρ₅**。

    造 psi -> rho5(2.5) + spec 的两体衰变，再让 rho5 -> a + b 各向同性。

这样 `q_in` 与 `q_top` 都只由质量决定、**逐事例为常数**，于是任何
`q^l / B_l` 因子都退化成整体常数，被"逐道拟合一个 c"吸收。剩下的
角分布 / 全部 λ 分量 / 置换结构必须逐点对上 —— 这才是张量公式的判据。

实测（在壳数据）：三条道 `tf/ours` 都是**逐事例标量**且 λ 方向一致，
overlap = 1.000000，残差 ~2e-15。这就是"J=5 张量公式与 tf-pwa 一致"
的外部证据。

跑法（远程 docker，tf-pwa 源码挂在 /w）：

    docker run --rm -v $PWD:/s -v /path/to/tf-pwa:/w -w /s \
        -e TFPWA=/w -e PYTHONPATH=/w tf-pwa-check:latest \
        /opt/tfpwa/bin/python /s/make_tfpwa_j5_golden.py > out.txt

然后从第一个 `{` 截出 JSON。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("TFPWA", "/w"))

from tf_pwa.amp.core import DecayGroup, HelicityDecay, Particle   # noqa: E402
from tf_pwa.cal_angle import cal_angle_from_momentum              # noqa: E402

M_PSI, M_RHO5, M_PI = 3.686, 2.5, 0.13957039
SEED, NEV = 33, 8

# 道名 -> (内层第一子, 内层第二子, 顶层旁观者)。内层顺序 = 共振配对顺序。
ROLE = {"12": ("Pip", "Pim", "Pi0"),
        "13": ("Pi0", "Pip", "Pim"),
        "23": ("Pi0", "Pim", "Pip")}


def _pair(m_parent, m1, m2, rng, n):
    lam = ((m_parent ** 2 - (m1 + m2) ** 2)
           * (m_parent ** 2 - (m1 - m2) ** 2))
    p = np.sqrt(max(lam, 0.0)) / (2.0 * m_parent)
    ct = rng.uniform(-1.0, 1.0, n)
    st = np.sqrt(np.maximum(1.0 - ct ** 2, 0.0))
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    d = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)
    return (np.concatenate([p * d, np.full((n, 1),
                                           np.sqrt(m1 ** 2 + p ** 2))], axis=1),
            np.concatenate([-p * d, np.full((n, 1),
                                            np.sqrt(m2 ** 2 + p ** 2))], axis=1))


def _boost(p, beta):
    b2 = (beta ** 2).sum(-1)
    g = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-300))
    bp = (beta * p[:, :3]).sum(-1)
    g2 = np.where(b2 > 1e-12, (g - 1.0) / np.where(b2 > 1e-12, b2, 1.0), 0.0)
    return np.concatenate(
        [p[:, :3] + (g2 * bp + g * p[:, 3])[:, None] * beta,
         (g * (p[:, 3] + bp))[:, None]], axis=1)


def physical(tag, n=NEV, seed=SEED):
    """造一组"共振在壳"的动量：psi -> rho5(2.5) + spec，rho5 -> a + b。

    `a`/`b` 的质量和恒为 m_rho5，`spec` 是旁观者 —— 三个**角色**互不相同。

    然后按 `tag` 把这三个角色指派到 (Pip, Pim, Pi0) 三个**名字**上，
    使 `ROLE[tag]` 指定的那一对正好是 (a, b)：

        _12: rho=(Pip,Pim)  -> Pip=a, Pim=b, Pi0=spec
        _13: rho=(Pi0,Pip)  -> Pi0=a, Pip=b, Pim=spec
        _23: rho=(Pi0,Pim)  -> Pi0=a, Pim=b, Pip=spec

    ⚠️ 这一步是关键：若三条道共用同一组动量（只改角色命名），
    则只有 _12 的配对在壳，_13/_23 的配对质量会逐事例变化，于是
    tf-pwa 的 `q^l`（它即使 `has_barrier_factor=False` 也仍乘 q^l）
    不再退化成常数，对拍就会失败。见模块 docstring。
    """
    rng = np.random.default_rng(seed)
    res, spec = _pair(M_PSI, M_RHO5, M_PI, rng, n)      # psi -> rho5 + spec
    a0, b0 = _pair(M_RHO5, M_PI, M_PI, rng, n)          # rho5 -> a + b
    beta = res[:, :3] / res[:, 3:4]
    a, b = _boost(a0, beta), _boost(b0, beta)

    def tf(p):
        return p[:, [3, 0, 1, 2]]

    # 角色 -> 名字
    assign = {"12": {"Pip": a, "Pim": b, "Pi0": spec},
              "13": {"Pi0": a, "Pip": b, "Pim": spec},
              "23": {"Pi0": a, "Pim": b, "Pip": spec}}[tag]
    return {k: tf(v) for k, v in assign.items()}


def amp(tag, mom):
    """按 `tag` 的树建链并算 tf-pwa 振幅，返回 (N, 3)（顶层 psi 的 3 个螺度）。"""
    d1, d2, spec = ROLE[tag]
    psi = Particle("psi", J=1, P=-1, mass=M_PSI)
    rho = Particle("rho5", J=5, P=-1, mass=M_RHO5)
    p1 = Particle(d1, J=0, P=-1, mass=M_PI)
    p2 = Particle(d2, J=0, P=-1, mass=M_PI)
    sp = Particle(spec, J=0, P=-1, mass=M_PI)
    # 位垒关掉：与我们的 Q0=0.197321 / q^l 口径不同，见模块 docstring
    kw = dict(add_covariant_term=True, has_barrier_factor=False)
    HelicityDecay(psi, [rho, sp], ls_list=[[5, 5]], **kw)
    HelicityDecay(rho, [p1, p2], ls_list=[[5, 0]], **kw)
    dg = DecayGroup(psi.chain_decay())
    dg.init_params()
    data = cal_angle_from_momentum({p1: mom[d1], p2: mom[d2], sp: mom[spec]},
                                   dg, center_mass=False)
    return np.asarray(dg.get_amp3(data))[..., 0, 0, 0]


def main():
    out = {
        "seed": SEED,
        "n_ev": NEV,
        "convention": {
            "generator": "agent/tensor/tests/make_tfpwa_j5_golden.py",
            "tfpwa_commit": "e503951 (v0.2.2-209)",
            "add_covariant_term": True,
            "has_barrier_factor": False,
            "on_shell_pair": True,
            "note": "配对质量恒 = m_rho5^2，使 q^l/B_l 退化成常数；"
                    "这样比的是角分布与全部 lambda 分量（张量结构）",
        },
        "momenta": {},
        "channels": {},
    }
    for tag in ("12", "13", "23"):
        mom = physical(tag)
        out["momenta"][tag] = {k: v.tolist() for k, v in mom.items()}
        a = amp(tag, mom)
        out["channels"][tag] = {"re": a.real.tolist(), "im": a.imag.tolist()}
        print("  _%s shape=%s |amp|max=%.6e" % (tag, a.shape, np.abs(a).max()),
              file=sys.stderr)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
