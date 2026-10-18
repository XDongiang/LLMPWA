#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把一份链描述（TOML）编译成一个**自足**的预计算脚本。

产物目录（默认 spec 里的 `[chain].output`）：

    <out>/gen_tensor_<name>.py   生成的预计算脚本（jax 优先，无 tf-pwa）
    <out>/runtime.py             运行时副本（kernels/omega.py 原样拷贝）
    <out>/_backend.py            后端选择层（jax / numpy，原样拷贝）
    <out>/covariant.py           协变内核副本（kernel=covariant/helicity 时）
    <out>/chain_spec.toml        冻结的输入描述
    <out>/README.md              这次生成了什么、怎么跑

生成的脚本按 batch 做 `jit`（jax 后端；numpy 后端下 jit 是恒等装饰器），
用哪个后端由环境变量 `TENSOR_BACKEND` 决定（auto / jax / numpy，默认 auto）。

生成脚本做的事（与 method4_omega/gen_tensor.py 相同的产物约定）：

    Omega<suf>.npy   (N, nA, nb, nc) complex128
    K<suf>.npy       (n_ch, N)      float64
    M<suf>.npy       (n_ch, nA, nb) float64
    g<suf>.npy       (n_ch,)        float64   （默认全 1）
    norm<suf>.npy    (n_ch,)        float64

其中 suf 由 `[chain].suffix` 给出（f2 族为空串，f0 族为 "0"），
这样两族可以并存在同一个数据目录里，与 base_functions._assemble_omega_cache 一致。
"""
from __future__ import annotations

import os
import shutil
import textwrap

from . import basis as basismod
from . import spec as specmod

_HEADER = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""{title}

**本文件由 agent.tensor 从 TOML 自动生成，不要手改**；改参数请改
`{spec_name}` 后重跑 `python -m agent.tensor emit {spec_name}`。

链：{chain_one_line}
道：
{channel_lines}

产物（写进 --out，后缀 `{suffix}`）：

    Omega{suffix}.npy  (N, {nA}, {nb}, {nc})  complex128   与 (l,s) 无关的角向对象
    K{suffix}.npy      ({n_ch}, N)        float64      逐道节点标量 K_i(e)
    M{suffix}.npy      ({n_ch}, {nb}, {nc})    float64      常量 CG 矩阵
    g{suffix}.npy      ({n_ch},)          float64      逐道归一化常数（默认 1）
    norm{suffix}.npy   ({n_ch},)          float64      1/mean|W_i|

装配（与 tf-pwa 的 H.D 同构）：

    W_i[e, lam_A] = g_i * K_i(e) * sum_{{a,b}} M_i[lam_A, a, b] * Omega[e, lam_A, a, b]

`g_i` **默认为 1**：本生成器不假设每个道都有对应的协变张量表达式，
所以不做标定。若确实有参照 cache，可用 `--calib <phif2_kk.npy>` 做一次最小二乘标定。

用法：
    python {script_name} --momenta <Momentum_kk.npz> --n 20000 --out <dir>
    python {script_name} --momenta <Momentum_kk.npz> --out <dir> --calib <ref.npy>
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as onp

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _backend as _B   # noqa: E402   （同目录的后端选择：jax 优先，回退 numpy）
import runtime as RT    # noqa: E402   （同目录的运行时副本，无 tf-pwa 依赖）

# 注意这里**不需要** xp：Omega/K 的计算全在 runtime 里（它自己取 xp），
# 本脚本只负责分批搬运与写盘，用的都是 host numpy。
jit = _B.jit
to_numpy = _B.to_numpy

# ------------------------------------------------------------------ 冻结的链描述
CHAIN = {chain!r}

# 道表：(名字, l, s)  —— (l,s) 来自 TOML 的显式 [[channel]] 或自动枚举
CHANNELS = {channels!r}
LS_LIST = [(l, s) for (_n, l, s) in CHANNELS]
NAMES = [_n for (_n, _l, _s) in CHANNELS]

# 输入
INPUT_KEYS = {keys!r}
INPUT_LAYOUT = {layout!r}

# 顶层节点的量子数（Omega 的轴由它决定）
TOP = CHAIN["top"]
_TOPNODE = CHAIN["nodes"][TOP]
_LAM_A = RT.LAM_SLOTS[_TOPNODE["J"]]
_LAM_B = RT.LAM_SLOTS[CHAIN["nodes"][_TOPNODE["daughters"][0]]["J"]]
_LAM_C = RT.LAM_SLOTS[CHAIN["nodes"][_TOPNODE["daughters"][1]]["J"]]
_JTOP = _TOPNODE["J"]
_JB = CHAIN["nodes"][_TOPNODE["daughters"][0]]["J"]
_JC = CHAIN["nodes"][_TOPNODE["daughters"][1]]["J"]


def cg_matrices():
    """逐道常量 CG 矩阵 M_i[lam_A, lam_b, lam_c]。"""
    return RT.cg_matrix(LS_LIST, _JTOP, _JB, _JC, _LAM_B, _LAM_C)


def omega(mom, covariant=True):
    return RT.omega_from_momenta(mom, CHAIN, covariant=covariant)


def node_factors(mom):
    return RT.node_factors(mom, CHAIN, LS_LIST)


# 分批 + jit：Omega 是 (N, nA, nb, nc) complex128，1.2M 事例一次算会占掉
# ~0.9 GB 显存；分批后显存只与 --batch 有关。numpy 后端下 jit 是恒等装饰器。
#
# `covariant` 必须声明为 static：jit 默认把 bool 当**追踪值**，函数体里的
# `if covariant:` 会直接抛 TracerBoolConversionError。
_OMEGA = jit(omega, static_argnames=("covariant",))
_NODE_FACTORS = jit(node_factors)


def assemble(M, Om, g, K):
    return RT.assemble(M, Om, g, K)


def _calibrate(M, Om, K, calib_path, n_ev):
    """可选：用参照的笛卡尔张量 cache 做一次最小二乘标定 g_i。

    参照 = EPS_A . T（路线 3 的 cache），EPS_A 是 psi(2S) 横向极化投影。
    只有当**每个道**都有对应的协变张量时才可用；默认不做这一步。
    """
    T = onp.asarray(onp.load(calib_path), dtype=onp.float64)
    if T.shape[0] != len(NAMES):
        raise SystemExit("--calib 的通道数 %d 与道表 %d 不一致"
                         % (T.shape[0], len(NAMES)))
    if T.shape[1] < n_ev:
        raise SystemExit("--calib 只有 %d 个事例，少于请求的 %d"
                         % (T.shape[1], n_ev))
    T = T[:, :n_ev]
    if T.shape[-1] < 3:
        # ⚠️ 交付的旧 cache 只有 (x,y)（CacheTensor.py:220 丢掉了 z）。
        # 补零后 EPS_A 的 lambda=0 行必然对不上，标定会失败——这里**直接报错**，
        # 而不是静默补零给出一个看着像样的错数。
        raise SystemExit(
            "--calib 只有 %d 个洛伦兹分量，缺 z；EPS_A 的 lambda=0 行无法复现。\\n"
            "请用 run/gen_tensor3.py --keep 3 重新生成 3 分量 cache。"
            % T.shape[-1])
    EPS_A = onp.array([[1 / onp.sqrt(2), -1j / onp.sqrt(2), 0],
                       [0, 0, 1],
                       [-1 / onp.sqrt(2), -1j / onp.sqrt(2), 0]], dtype=onp.complex128)
    W_ref = onp.einsum("lm,iem->iel", EPS_A, T[..., :3])       # (n_ch, N, nA)
    P = onp.einsum("iab,enab->ien", M, Om)
    g = onp.zeros(len(NAMES))
    for i in range(len(NAMES)):
        p, w, k = P[i], W_ref[i], K[i]
        g[i] = (onp.sum(onp.sum(onp.conj(p) * w, axis=-1) * k)
                / onp.sum(onp.sum(onp.abs(p) ** 2, axis=-1) * k ** 2)).real
    res = onp.abs(assemble(M, Om, g, K) - W_ref).max() / onp.abs(W_ref).max()
    return g, W_ref, res


def main():
    ap = argparse.ArgumentParser(description="自动生成的 Omega 预计算（{name}）")
    ap.add_argument("--momenta", "--npz", dest="momenta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = 全部")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128,
                    help="逐批计算的批大小（控制显存占用）")
    ap.add_argument("--calib", default="",
                    help="可选：参照协变张量 cache（EPS_A.T），用来标定 g_i；"
                         "不给则 g_i = 1")
    ap.add_argument("--g-file", default="", help="读已有的 g.npy，跳过标定")
    ap.add_argument("--no-covariant", action="store_true",
                    help="关掉 Chung Eq.52 的协变项（只用于对照，不要用于生产）")
    ap.add_argument("--suffix", default={suffix!r}, help="产物文件名后缀")
    args = ap.parse_args()

    t0 = time.time()
    mom = RT.momenta_from_npz(args.momenta, INPUT_KEYS, args.start, args.n or None,
                              order=INPUT_LAYOUT)
    n = mom.shape[0]
    print("[%s] %d events from %s  backend=%s"
          % ({name!r}, n, args.momenta, _B.backend_name()))

    # 逐 batch 算 Omega / K 再拼回 host，避免把整份数据留在显存里。
    batch = args.batch
    Om = K = None
    for beg in range(0, n, batch):
        end = min(beg + batch, n)
        om_blk = to_numpy(_OMEGA(mom[beg:end], covariant=not args.no_covariant))
        k_blk = to_numpy(_NODE_FACTORS(mom[beg:end]))
        if Om is None:
            Om = onp.zeros((n,) + om_blk.shape[1:], dtype=om_blk.dtype)
            K = onp.zeros((k_blk.shape[0], n), dtype=k_blk.dtype)
        Om[beg:end] = om_blk
        K[:, beg:end] = k_blk
    M = cg_matrices()
    print("  Omega %s   K %s   M %s" % (Om.shape, K.shape, M.shape))

    if args.g_file and os.path.exists(args.g_file):
        g = onp.load(args.g_file)
        print("  g 读自 %s" % args.g_file)
    elif args.calib:
        g, W_ref, res = _calibrate(M, Om, K, args.calib, mom.shape[0])
        print("  g 标定自 %s  自检 max|dW|/max|W| = %.3e  %s"
              % (args.calib, res, "OK" if res < 1e-10 else "**MISMATCH**"))
    else:
        g = onp.ones(len(NAMES))
        print("  g 默认全 1（未提供 --calib）")

    W = assemble(M, Om, g, K)
    norm = 1.0 / onp.sqrt((onp.abs(W) ** 2).sum(-1)).mean(axis=1)

    os.makedirs(args.out, exist_ok=True)
    suf = args.suffix
    for fn, arr in (("Omega%s.npy" % suf, Om), ("K%s.npy" % suf, K),
                    ("M%s.npy" % suf, M), ("g%s.npy" % suf, g),
                    ("norm%s.npy" % suf, norm)):
        onp.save(os.path.join(args.out, fn), arr)
    print("  saved -> %s  Omega/K/M/g/norm 后缀 %r  (%.1fs)"
          % (args.out, suf, time.time() - t0))


if __name__ == "__main__":
    main()
'''


def emit(spec, out_dir=None, runtime_src=None):
    """编译 spec -> 产物目录，按 `[[basis]]` 各出一份自足脚本。

    返回 dict：

        {"out_dir", "scripts": [ {"basis", "kernel", "script", "suffix"} ],
         "runtime", "channels", "script"}     # script = 第一份，兼容旧调用

    省略 `[[basis]]` 时只出 Omega 一份。
    """
    ch = specmod.resolve_channels(spec)
    chain = specmod.to_chain(spec)
    name = spec["name"]
    out_dir = out_dir or spec["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    suffix = spec["raw"].get("chain", {}).get("suffix", "")
    bases = spec["bases"] or [{"name": "omega", "kernel": "omega",
                               "files": [], "env": {}, "suffix_map": {}}]

    # 运行时副本：产出目录自足，不依赖 agent.tensor
    kdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
    rt_dst = os.path.join(out_dir, "runtime.py")
    rt_src = runtime_src or os.path.join(kdir, "omega.py")
    shutil.copyfile(rt_src, rt_dst)

    # 后端选择层：产物脚本 import 它来决定用 jax 还是 numpy。
    shutil.copyfile(os.path.join(kdir, "_backend.py"),
                    os.path.join(out_dir, "_backend.py"))

    # 冻结描述
    if spec.get("_spec_path"):
        shutil.copyfile(spec["_spec_path"],
                        os.path.join(out_dir, "chain_spec.toml"))

    scripts = []
    for b in bases:
        kern = b.get("kernel", b["name"])
        # `check` 会校验 basis.name，`emit` 也必须校验，否则名字写错时
        # 产物里的 README 会记下一个不存在的表象名，而拟合侧按
        # TENSOR_BASIS 去取时才发现 —— 那时已经在远程跑了一半。
        try:
            basismod.get(b["name"])
        except KeyError as e:
            raise ValueError(
                "[[basis]] name=%r 不是已知表象：%s\n"
                "（注意 name 与 kernel 是两件事：name 是拟合侧的表象名，"
                "kernel 是生成用哪个内核；如 cartesian 配 kernel=\"covariant\"）"
                % (b["name"], e))
        if kern == "omega":
            scripts.append(_emit_omega(spec, ch, chain, name, out_dir,
                                       suffix, b))
        elif kern in ("covariant", "helicity"):
            scripts.append(_emit_covariant(spec, ch, chain, name, out_dir,
                                           suffix, b))
        else:
            raise ValueError("未知 kernel %r（可用：omega / covariant / helicity）"
                             % kern)

    readme = _readme(spec, ch, scripts, suffix)
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(readme)

    # `fallback` = 在 `chains/` 里找不到、退回 `kernels/covariant.py` 的道名。
    # 非空说明这份 spec 点到了内核里那 7 条 legacy 道（正常），但若点到的名字
    # 两边都没有，`_extract_channel_body` 早就 FileNotFoundError 了。
    fallback = sorted({nm for e in scripts for nm in e.get("fallback") or []})
    from_chain = {}
    for e in scripts:
        from_chain.update(e.get("from_chain") or {})
    return {"out_dir": out_dir, "scripts": scripts, "runtime": rt_dst,
            "channels": ch, "script": scripts[0]["script"],
            "fallback": fallback, "from_chain": from_chain}


# --------------------------------------------------------------- Omega 侧
def _emit_omega(spec, ch, chain, name, out_dir, suffix, basis):
    """把 spec 编译成 method4_omega 约定的 Omega 预计算脚本。"""
    top = chain["nodes"][chain["top"]]
    jb = chain["nodes"][top["daughters"][0]]["J"]
    jc = chain["nodes"][top["daughters"][1]]["J"]
    nA, nb, nc = 2 * top["J"] + 1, 2 * jb + 1, 2 * jc + 1
    n_ch = len(ch)

    script_name = "gen_tensor_%s.py" % name
    channel_lines = "\n".join(
        "    %-18s (l,s)=(%d,%d)" % (n, l, s) for (n, l, s, _f) in ch)
    chain_one_line = "%s (J=%d) -> %s + %s" % (
        chain["top"], top["J"], top["daughters"][0], top["daughters"][1])

    body = _HEADER.format(
        title="%s —— 自动生成的 Omega 预计算" % name,
        spec_name="chain_spec.toml",
        chain_one_line=chain_one_line,
        channel_lines=channel_lines,
        suffix=suffix,
        nA=nA, nb=nb, nc=nc,
        n_ch=n_ch,
        script_name=script_name,
        name=name,
        chain=chain,
        channels=[(n, l, s) for (n, l, s, _f) in ch],
        keys=spec["input_keys"],
        layout=spec["input_layout"],
    )

    script_path = os.path.join(out_dir, script_name)
    with open(script_path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return {"basis": basis["name"], "kernel": "omega", "script": script_path,
            "suffix": suffix, "env": dict(basis.get("env") or {})}


# ----------------------------------------------------------- 协变/螺度侧
_COV_HEADER = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""{title}

**本文件由 agent.tensor 从 TOML 自动生成，不要手改**；改参数请改
`{spec_name}` 后重跑 `python -m agent.tensor emit {spec_name}`。

链：{chain_one_line}
道（{n_ch} 条）：
{channel_lines}

产物（写进 --out）：

    {out_names}

`kernel = "{kernel}"`：{kernel_doc}
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as onp

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _backend as _B     # noqa: E402  后端选择（jax 优先，无 jax 回退 numpy）
import covariant as CV    # noqa: E402  同目录的协变张量内核副本
from covariant import (  # noqa: E402,F401  道函数体直接调用这些原语
    B1, B2, B3, B4, Gt_con, Gt_cov, P2, P3, P4, Qabc, T1_cov, T2_cov, T3_cov,
    T4_cov, dot, invm, leci_four, metric,
    _einsum, _leci_con, _metric_axis, _raise_last, _s1_vector,
)

xp = _B.xp
jit = _B.jit
to_numpy = _B.to_numpy

CHAIN = {chain!r}
CHANNELS = {channels!r}
NAMES = [_n for (_n, _l, _s) in CHANNELS]
LS_LIST = [(l, s) for (_n, l, s) in CHANNELS]

INPUT_KEYS = {keys!r}
INPUT_LAYOUT = {layout!r}
PSI_MASS = {psi_mass!r}
OUT_NAME = {out_name!r}

# 道名 -> 该道的 (l,s)；生成时从 chain.<name>/chain.<channel>.py 组装
# （本脚本是组装后的自足产物，运行时不需要框架包）。
{channel_imports}


def _assemble_funcs():
    """把内联进来的道函数收成 {{道名: callable}}。"""
    return {{nm: globals()[nm] for nm in NAMES}}


_FUNCS = _assemble_funcs()


def _psi(mom, n):
    """psi(2S) 四动量：静态近似 (0,0,0,m)，布局 (x,y,z,E)。"""
    return onp.broadcast_to(onp.array([0.0, 0.0, 0.0, PSI_MASS]), (n, 4))


def raw_tensors(mom):
    """逐道算原始协变张量 T_i(e, mu)，返回 {{道名: (N, 4)}}。

    道函数的签名固定是 `(psi, phi, f2, Kp, Km, Pip, Pim)`（见
    `agent/tensor/chains/README` 的约定）。`phi` / `f2` 只对
    `<phi f_J|LS>` 那些道有意义，rho'/b1 族忽略它们 —— 但签名保持统一，
    这样道表可以混装两个族的道。
    """
    Kp, Km = mom[:, CHAIN["index"]["Kp"]], mom[:, CHAIN["index"]["Km"]]
    Pip, Pim = mom[:, CHAIN["index"]["Pip"]], mom[:, CHAIN["index"]["Pim"]]
    phi, f2 = Kp + Km, Pip + Pim
    psi = _psi(mom, Kp.shape[0])
    args = (psi, phi, f2, Kp, Km, Pip, Pim)
    return {{nm: _FUNCS[nm](*args) for nm in NAMES}}


def stored(mom):
    """按 kernel 把原始张量变成要存盘的对象。"""
    raw = raw_tensors(mom)
    {store_body}


# 每个 batch 走一次 jit：XLA 把 psi(2S) 这类编译期常量折掉，并在 GPU 上并行。
# numpy 后端下 `jit` 是恒等装饰器，行为不变。
_STORED = jit(stored)


def main():
    ap = argparse.ArgumentParser(description="自动生成的协变/螺度张量预计算（{name}）")
    ap.add_argument("--momenta", "--npz", dest="momenta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = 全部")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    t0 = time.time()
    mom = _momenta_from_npz(args.momenta, INPUT_KEYS, args.start,
                            args.n or None, order=INPUT_LAYOUT)
    n = mom.shape[0]
    print("[{name}] %d events from %s  backend=%s"
          % (n, args.momenta, _B.backend_name()))

    out = None
    for beg in range(0, n, args.batch):
        end = min(beg + args.batch, n)
        # 搬回 host 再堆叠：产物要写 .npy，且这样显存不随事例数增长。
        blk = _B.tree_to_numpy(_STORED(mom[beg:end]))
        # 每道 (N, k) -> 堆成 (n_ch, N, k)，与 phif0_kk.npy / phif2_kk.npy 的
        # 既有约定一致（道轴在前）。
        arr = onp.stack([blk[nm] for nm in NAMES], axis=0)
        if out is None:
            out = onp.zeros(arr.shape[:1] + (n,) + arr.shape[2:], dtype=arr.dtype)
        out[:, beg:end] = arr
        if beg % (args.batch * 20) == 0 or end == n:
            print("  %7d/%d  %.1f ev/s  elapsed %.1fs"
                  % (end, n, end / max(time.time() - t0, 1e-9),
                     time.time() - t0), flush=True)

    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, OUT_NAME)
    onp.save(dest, out)
    print("  saved %-22s %s %s" % (OUT_NAME, out.shape, out.dtype))
    print("done in %.1fs" % (time.time() - t0))


def _momenta_from_npz(path, keys, start=0, n=None, order="x,y,z,E"):
    """读 npz，**保持 `order` 声明的分量顺序不变**。

    协变张量内核（`covariant.py`）的原生约定是 `(x, y, z, E)` —— 与
    `Tensor/BASE.py` 一致，也与 `[input].layout` 的默认值一致。所以这里
    **不做**到 `(E,x,y,z)` 的置换：那个约定属于 `kernels/omega.py`，由它自己的
    `momenta_from_npz` 负责。

    曾经在这里照抄了 omega 的置换，结果生成物与生产 cache 差 0.98（完全错位）；
    分量顺序是承重约定，不要"顺手统一"。
    """
    d = onp.load(path)
    sl = slice(start, None if n is None else start + n)
    arrs = [onp.asarray(d[k][sl], dtype=onp.float64) for k in keys]
    mom = onp.stack(arrs, axis=1)                      # (N, n_part, 4)
    declared = order.replace(" ", "").split(",")
    if declared != ["x", "y", "z", "E"]:
        raise SystemExit(
            "[input].layout = %r，但协变张量内核的原生约定是 'x,y,z,E'；"
            "请改 TOML，或在此处显式做一次置换" % order)
    return mom


if __name__ == "__main__":
    main()
'''


def _emit_covariant(spec, ch, chain, name, out_dir, suffix, basis):
    """按 chain.<name>/chain.<channel>.py 组装协变/螺度张量预计算脚本。

    组装方式：把每个道的函数体从 `agent/tensor/chains/<chain>/<channel>.py`
    里原样取出、内联进产物（产物自足，运行时不 import agent.tensor），
    并在同目录写一份 `covariant.py` 内核副本。
    """
    import re

    from . import chains as chainsmod

    kernel = basis.get("kernel", basis["name"])
    chain_name = spec["raw"].get("chain", {}).get("chain_name") or spec["name"]
    psi_mass = float(spec["raw"].get("chain", {}).get("psi_mass", 3.686))

    # 1) 取道函数体：优先 chain.<name>/chain.<channel>.py，退回 kernels.covariant
    #
    # 一份 spec 可以跨多个 chain 目录取道（例 3 同时用 rhopi 的 rho' 与
    # b1pi 的 b1 —— 两条链的树不同，不能合成一个目录）。所以先按
    # `[chain].chain_name` 指定的目录找，找不到再跨目录按道名找。
    bodies, missing, from_chain = [], [], {}
    all_ch = chainsmod.all_channels()
    for (nm, _l, _s, _f) in ch:
        body = None
        if chain_name in chainsmod.list_chains() and \
                nm in chainsmod.list_channels(chain_name):
            body = _extract_channel_body(chain_name, nm)
            from_chain[nm] = chain_name
        elif nm in all_ch:
            body = _extract_channel_body(all_ch[nm], nm)
            from_chain[nm] = all_ch[nm]
        if body is None:
            body = _extract_channel_body(None, nm)
            missing.append(nm)
        bodies.append(body)

    channel_imports = "\n\n".join(bodies)

    # 2) 内核副本
    kdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
    shutil.copyfile(os.path.join(kdir, "covariant.py"),
                    os.path.join(out_dir, "covariant.py"))

    # 3) 产物文件名 + store 体
    chain_name_out = spec["raw"].get("chain", {}).get("chain_name") or name
    if kernel == "helicity":
        out_name = "%s_hel.npy" % chain_name_out
        out_names = "%s   (n_ch, N, 3) complex128" % out_name
        kernel_doc = ("预计算 W_i(e,lambda) = EPS_A . T_i，直接存螺旋度振幅")
        store_body = '''EPS_A = onp.array([[1/onp.sqrt(2), -1j/onp.sqrt(2), 0],
                       [0, 0, 1],
                       [-1/onp.sqrt(2), -1j/onp.sqrt(2), 0]], dtype=onp.complex128)
    return {nm: xp.einsum("lm,em->el", EPS_A, raw[nm][..., :3]) for nm in NAMES}'''
    else:
        out_name = "%s_kk.npy" % chain_name_out
        out_names = "%s    (n_ch, N, 3) float64" % out_name
        kernel_doc = "预计算笛卡尔协变张量 T_i(e,mu)，mu = x,y,z"
        store_body = '''return {nm: xp.asarray(raw[nm], dtype=xp.float64)[..., :3]
            for nm in NAMES}'''

    script_name = "gen_tensor_%s_%s.py" % (name, kernel)
    top = chain["nodes"][chain["top"]]
    chain_one_line = "%s (J=%d) -> %s + %s" % (
        chain["top"], top["J"], top["daughters"][0], top["daughters"][1])
    channel_lines = "\n".join(
        "    %-18s (l,s)=(%d,%d)" % (n, l, s) for (n, l, s, _f) in ch)

    body = _COV_HEADER.format(
        title="%s —— 自动生成的%s预计算" % (
            name, "螺度" if kernel == "helicity" else "协变张量"),
        spec_name="chain_spec.toml", chain_one_line=chain_one_line,
        channel_lines=channel_lines, n_ch=len(ch), out_names=out_names,
        kernel=kernel, kernel_doc=kernel_doc, chain=chain,
        channels=[(n, l, s) for (n, l, s, _f) in ch],
        keys=spec["input_keys"], layout=spec["input_layout"],
        psi_mass=psi_mass, out_name=out_name,
        channel_imports=channel_imports,
        store_body=store_body, name=name)

    script_path = os.path.join(out_dir, script_name)
    with open(script_path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return {"basis": basis["name"], "kernel": kernel, "script": script_path,
            "suffix": suffix, "env": dict(basis.get("env") or {}),
            "from_chain_dir": chain_name, "from_chain": from_chain,
            "fallback": missing}


def _extract_channel_body(chain_name, channel):
    """从 chains/<chain>/<channel>.py 或 kernels/covariant.py 取出函数体源码。"""
    import re
    here = os.path.dirname(os.path.abspath(__file__))
    if chain_name:
        p = os.path.join(here, "chains", chain_name, channel + ".py")
    else:
        p = os.path.join(here, "kernels", "covariant.py")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    src = open(p, encoding="utf-8").read()
    pat = re.compile(r"^def %s\(.*?(?=^def |\Z)" % re.escape(channel),
                     re.M | re.S)
    m = pat.search(src)
    if not m:
        raise LookupError("在 %s 里找不到 def %s" % (p, channel))
    text = m.group(0).rstrip() + "\n"
    if not chain_name:
        # 内核里的私有名去掉下划线前缀，便于内联后调用
        text = re.sub(r"\b_(phif\d+)\b", r"\1", text)
    return text


def _readme(spec, ch, scripts, suffix):
    rows = "\n".join("| `%s` | (%d, %d) |" % (n, l, s) for (n, l, s, _f) in ch)
    tbl = "\n".join(
        "| `%s` | `%s` | `%s` | `%s` |" % (
            s["basis"], s["kernel"], os.path.basename(s["script"]),
            ", ".join("%s=%s" % kv for kv in sorted((s.get("env") or {}).items())) or "-")
        for s in scripts)
    return textwrap.dedent("""\
        # `%s` 的预计算（自动生成）

        由 `agent.tensor` 从 `chain_spec.toml` 编译而来。**不要手改生成物**；
        改参数请改 TOML 后重跑 `python -m agent.tensor emit <spec>`。

        ## 表象 / 脚本

        | 表象 | kernel | 脚本 | 环境变量 |
        |---|---|---|---|
        %s

        ## 跑

        ```bash
        python <脚本> --momenta <Momentum_kk.npz> --n 20000 --out <数据目录>
        ```

        ## 道表

        | 道 | (l, s) |
        |---|---|
        %s
        """) % (spec["name"], tbl, rows)
