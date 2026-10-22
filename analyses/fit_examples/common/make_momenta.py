#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 **tf-pwa 自己的相空间生成器**造一批四动量（`tf_pwa.phasespace.generate_phsp`）。

为什么用 tf-pwa 而不是自己写
---------------------------
`analyses/tensor_examples/data/make_*.py` 里已经有手写的两体/三体生成器，它们
**只覆盖那几条链**。这里要覆盖论文里全部 15 份 spec（3/4/5/6 体），手写会变成
六份近似重复的代码，而且每份都要自己证守恒。tf-pwa 的 `generate_phsp` 是
**通用的 n 体**（按 `ChainGenerator` 递归做两体），已经内部检查过守恒，还顺便
给了 `phsp_volume()` 做归一化参考。所以：

    数据用 tf-pwa 造 —— 这是**独立实现**，同时也是与 tf-pwa 对拍的入口。

⚠️ 它必须跑在**有 tf-pwa 的环境**里。本机没有（`python -c "import tf_pwa"` 失败），
远程 HEP1 也没有装进任何 conda env，但有现成的 docker 镜像
`tf-pwa-check:latest`（tensorflow-cpu 2.21）与源码 `tmp/tf-pwa`：

    docker run --rm -v $PWD/tmp/tf-pwa:/src:ro \\
        -e PYTHONPATH=/src -e TF_CPP_MIN_LOG_LEVEL=3 tf-pwa-check:latest \\
        python fit_examples/common/make_momenta.py --spec <spec.toml> --n 20000 ...

⚠️ **不需要 GPU**：相空间抽样是纯 numpy/TF 的随机数，CPU 更快（实测 3 体
~1.4e5 ev/s、6 体 ~7e2 ev/s）。

产物
----
`--out data/momenta.npz`，键 = spec 的 `[input].keys`（= 四动量轴顺序），
每个值 `(N, 4)`，布局与 spec 的 `[input].layout` 一致（默认 `x,y,z,E`，
与协变张量内核的原生约定相同）。**tf-pwa 原生是 (E,x,y,z)**，这里在写盘前
统一换序 —— 换序只做一次、只在一个地方做。

同时写 `momenta.meta.json`：N、seed、各末态质量、守恒残差、在壳残差、
每个两体子系统的质量区间。**这些数字是验收线**：守恒 >1e-10 或某个两体
子系统撞到阈值（`Qabc≈0`）就说明这批数据会让 `B_l` 退化，必须重抽。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:
    import tomllib as _toml
except ModuleNotFoundError:                              # python < 3.11
    import tomli as _toml


def _load_spec(path):
    """**不 import agent.tensor** —— 自己读 TOML。

    理由：这个脚本要跑在**只有 tf-pwa 的 docker 镜像**里（见上面的用法），
    那个环境里没有本仓库。所以它只能依赖 tomllib + numpy + tf_pwa。
    需要读的字段很少（`[input]` 与 `[particles.*]` 的 mass），自己读比把整个
    `agent/tensor` 塞进容器省事得多，也不会引入版本漂移。
    """
    with open(path, "rb") as fh:
        raw = _toml.load(fh)
    keys = list(raw["input"]["keys"])
    layout = raw["input"].get("layout", "x,y,z,E")
    # top = 唯一一个不出现为别人 daughter 的粒子
    daughters = set()
    for node in raw.get("particles", {}).values():
        daughters.update(node.get("daughters") or [])
    tops = [n for n in raw.get("particles", {}) if n not in daughters]
    if len(tops) != 1:
        raise SystemExit("%s：顶层母粒子不唯一 %s" % (path, tops))
    return raw, {"keys": keys, "layout": layout, "top": tops[0],
                 "nodes": raw.get("particles", {})}


def _final_masses(chain, keys):
    """末态质量，按 `[input].keys` 的顺序。缺质量直接报错（不猜）。"""
    out = []
    for k in keys:
        node = chain["nodes"].get(k)
        if node is None:
            raise SystemExit("spec 里没有粒子 %r（[input].keys 与 [particles.*] 不符）" % k)
        m = node.get("mass")
        if m is None:
            raise SystemExit("粒子 %r 缺 mass；相空间生成与张量内核都要求显式质量" % k)
        out.append(float(m))
    return out


def generate(keys, masses, m0, n, seed=0):
    """调用 tf-pwa 的通用 n 体生成器；返回 {key: (N,4)} 布局 (E,x,y,z)。"""
    import tensorflow as tf
    from tf_pwa.phasespace import generate_phsp

    tf.random.set_seed(seed)
    parts = generate_phsp(m0, masses, N=n)
    arrs = [np.asarray(x, dtype=np.float64) for x in parts]
    if len(arrs) != len(keys):
        raise SystemExit("tf-pwa 返回 %d 个末态，spec 要 %d 个" % (len(arrs), len(keys)))
    return dict(zip(keys, arrs)), m0


def reorder_to_layout(p4, layout):
    """(E,x,y,z) -> `layout` 声明的顺序。目前只支持 x,y,z,E 与 E,x,y,z。"""
    order = [s.strip() for s in layout.replace(" ", "").split(",")]
    if order == ["E", "x", "y", "z"]:
        return p4
    if order == ["x", "y", "z", "E"]:
        return p4[:, [1, 2, 3, 0]]
    raise SystemExit(
        "[input].layout = %r 不认识；生成器只保证 E,x,y,z 与 x,y,z,E 两种" % layout)


def invariant_mass(q, layout="x,y,z,E"):
    """不变质量，**按 layout 取分量**。

    写死任何一个约定都会算错，而且错得"很像真的"（把 (E,x,y,z) 当
    (x,y,z,E) 用，0.4937 会算成 2.3）。生成器原生是 (E,x,y,z)，
    写盘是 `layout` —— 检查必须显式选择，不能靠"大家都知道"。
    """
    if layout.replace(" ", "") == "E,x,y,z":
        e, s = q[:, 0], q[:, 1] ** 2 + q[:, 2] ** 2 + q[:, 3] ** 2
    else:
        e, s = q[:, 3], q[:, 0] ** 2 + q[:, 1] ** 2 + q[:, 2] ** 2
    return np.sqrt(np.abs(e ** 2 - s))


def _split(q, layout):
    """按 layout 拆 (E, 三动量)。**不要**用 q[:,1:4] —— 那只有 (E,x,y,z) 对。"""
    if layout.replace(" ", "") == "E,x,y,z":
        return q[:, 0], q[:, 1:4]
    return q[:, 3], q[:, 0:3]


def checks(mom, masses, m0, layout="E,x,y,z"):
    """守恒 / 在壳 / 两体子系统阈值。返回一个可写进 meta 的 dict。"""
    keys = list(mom)
    tot = sum(mom[k] for k in keys)
    energy, psum = _split(tot, layout)
    # ⚠️ 分母不能用 |sum p|：质心系里它本来就是 0，相对残差会炸成 1e15。
    # 用"最大的单粒子三动量模长"作尺度，这个数才有 O(1) 的含义。
    scale = max(float(np.abs(_split(mom[k], layout)[1]).max()) for k in keys)
    cons = float(np.abs(psum).max() / max(scale, 1e-300))
    e_cons = float(np.abs(energy - m0).max() / m0)
    offshell = {k: float(np.abs(invariant_mass(mom[k], layout) - m).max())
                for k, m in zip(keys, masses)}
    return {"conservation_abs": float(np.abs(psum).max()),
            "conservation_rel": cons, "energy_rel": e_cons,
            "onshell_max_abs": offshell,
            "two_body_min": _pair_ranges(mom, masses, layout)}


def _pair_ranges(mom, masses, layout="E,x,y,z"):
    """所有两体组合的不变质量最小值，以及它与阈值之差。

    `Qabc < 0` 会让 `B_l` 的 `sqrt` 出 NaN（jax 原版与 numpy 版都一样）。
    两体子系统的质量**理论上**不可能低于阈值，但浮点 + 生成器的边界抽样
    可能刚好贴到阈值上，那时 `Qabc≈0`、`B_l` 很大。所以把
    `min m_ij - (m_i+m_j)` 记下来；<1e-6 GeV 就该重抽。
    """
    keys = list(mom)
    out = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            m = invariant_mass(mom[keys[i]] + mom[keys[j]], layout)
            thr = masses[i] + masses[j]
            out["%s+%s" % (keys[i], keys[j])] = {
                "min": float(m.min()), "max": float(m.max()),
                "thr": float(thr), "margin": float(m.min() - thr)}
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="tensor_examples/specs/*.toml")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="momenta.npz 的路径")
    ap.add_argument("--m0", type=float, default=None,
                    help="母粒子质量，默认取 spec 里 top 节点的 mass")
    ap.add_argument("--from-npz", default="",
                    help="**不走 tf-pwa**：从这个 npz 取前 n 个事例（本地没有 "
                         "tf-pwa 时的退路）。会写进 meta 的 generator 字段，"
                         "下游据此知道这批数据不是 tf-pwa 造的。")
    args = ap.parse_args()

    sp, chain = _load_spec(args.spec)
    keys = chain["keys"]
    masses = _final_masses(chain, keys)
    m0 = args.m0 if args.m0 is not None else float(
        chain["nodes"][chain["top"]]["mass"])
    if m0 <= sum(masses):
        raise SystemExit("m0=%.4f 低于阈值 %.4f" % (m0, sum(masses)))

    print("[make_momenta] %s: %d 体，m0=%.4f，N=%d"
          % (os.path.basename(args.spec), len(keys), m0, args.n))
    if args.from_npz:
        src = np.load(args.from_npz)
        mom = {k: np.asarray(src[k][:args.n], dtype=np.float64) for k in keys}
        n_got = {v.shape[0] for v in mom.values()}
        if len(n_got) != 1:
            raise SystemExit("%s 里各末态的事例数不一致：%s" % (args.from_npz, n_got))
        generator = "npz:%s" % os.path.abspath(args.from_npz)
    else:
        mom, _ = generate(keys, masses, m0, args.n, args.seed)
        generator = "tf_pwa.phasespace.generate_phsp"
    # `generate` 返回 (E,x,y,z)；--from-npz 的输入按 layout 读
    native = "E,x,y,z" if not args.from_npz else chain["layout"]
    info = checks(mom, masses, m0, native)
    layout = chain["layout"]
    out = {k: reorder_to_layout(mom[k], layout) for k in keys}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **out)
    meta = {"spec": os.path.abspath(args.spec), "n": args.n, "seed": args.seed,
            "m0": m0, "keys": keys, "masses": masses, "layout": layout,
            "generator": generator, **info}
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print("  守恒 rel=%.2e  能量 rel=%.2e  在壳 max=%.2e"
          % (info["conservation_rel"], info["energy_rel"],
             max(info["onshell_max_abs"].values())))
    worst = min(v["margin"] for v in info["two_body_min"].values())
    print("  两体子系统最小阈值余量 %.3e GeV  %s"
          % (worst, "OK" if worst > 1e-6 else "**太贴阈值，重抽**"))
    print("  saved %s  %s" % (args.out, {k: out[k].shape for k in keys}))
    return 0 if (info["conservation_rel"] < 1e-10 and worst > 1e-6) else 1


if __name__ == "__main__":
    sys.exit(main())
