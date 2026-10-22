#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用 JAX likelihood：**一份代码吃所有 case**，只换 `<case>/case.toml` 与盘上的 cache。

它是什么
--------
`kk_dis/run/likelihood_function.py` 是一个**具体的**物理似然（f₀/f₂ 六族、
flatté 传播子、高斯先验、份额罚项混在一起）。这里是它的**结构骨架**：

    A(e) = Σ_族  f_R(e) · Σ_道 g_i · W_i(e)           <- 张量缓存 + 耦合 + 传播子
    I(e) = Σ_{λ=±1} |A_λ(e)|²                          <- 横向极化（e+e- 初态）
    L    = -Σ_data log I + N_data · log(mean_MC I)     <- extended likelihood

**没有**份额罚项、没有高斯先验、没有 flatté：那些是"拟合结果"层面的事，
本分析的第一目标是证明「tensor 缓存 + JAX 高性能计算」这条链路能跑通。
结构留了扩展位（`_family_propagator`）。

参数化
------
每个道一个复耦合 `g_i = const_i + i·theta_i`，拟合向量是**实数**（`const` 与
`theta` 交替排列）。这样 `jax.grad` / `jvp` 都在实空间里，没有复梯度的
共轭约定问题 —— 与 `kk_dis` 的 `dplex_dconstruct` 同构。

两份 cache 的等价性
-------------------
同一个 spec 出两份 cache：

    cartesian         T_i(e, μ)     实数 (n_ch, N, 3)，μ = x,y,z
    helicity_cascade  W_i(e, λ)     复数 (n_ch, N, 3)，λ = +1,0,-1

`W = EPS_A · T` 是**恒等**变换，所以

    |T_x|² + |T_y|²  ==  |W_{+1}|² + |W_{-1}|²

必须**逐事例**成立（1e-15）。`--basis both` 就是跑这条判据。⚠️ 坑：
笛卡尔侧的横向分量**固定**是 (x, y)，与 k 无关；写成 `(x, z)` 会得到
"能跑但错"的数（实测 rel 0.04）。

用法
----
    python analyses/fit_examples/common/likelihood.py <case_dir> --basis both
    python analyses/fit_examples/common/likelihood.py <case_dir> --check --fit
    python analyses/fit_examples/common/likelihood.py <case_dir> --gpu
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as onp

try:
    import tomllib as _toml
except ModuleNotFoundError:
    import tomli as _toml

# XLA 的两个环境开关必须在 import jax **之前**设
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as np
from jax import jit, grad, jvp

jax.config.update("jax_enable_x64", True)

# 与 kk_dis/run/base_functions.py 逐位相同的 EPS_A（行序 λ = +1, 0, -1）
EPS_A = onp.array(
    [[1.0 / onp.sqrt(2.0), -1.0j / onp.sqrt(2.0), 0.0],
     [0.0, 0.0, 1.0],
     [-1.0 / onp.sqrt(2.0), -1.0j / onp.sqrt(2.0), 0.0]],
    dtype=onp.complex128,
)

# ⚠️ 两条路线的"横向"槽位**不是**同一件事：
#   * 螺旋度 cache：末轴是 λ 槽位，(0, 2) = (λ=+1, λ=-1)
#   * 笛卡尔 cache：末轴是洛伦兹指标，(0, 1) = (x, y)；即使存了 z 也不参与
#     （`agent/tensor/basis.py` 的 `CARTESIAN_TRANSVERSE` 就是这样定义的）
HEL_TRANSVERSE = (0, 2)
CART_TRANSVERSE = (0, 1)


# ------------------------------------------------------------------ 读 case
def load_case(case_dir):
    with open(os.path.join(case_dir, "case.toml"), "rb") as fh:
        case = _toml.load(fh)
    idx_path = os.path.join(case_dir, "tensors", "index.toml")
    index = {}
    if os.path.exists(idx_path):
        with open(idx_path, "rb") as fh:
            index = _toml.load(fh)
    return case, index


def load_params(case_dir):
    """读 `params/free_params.toml` -> (全部初值, 固定掩码, 路径列表)。"""
    path = os.path.join(case_dir, "params", "free_params.toml")
    with open(path, "rb") as fh:
        rows = _toml.load(fh)["data"]
    rows.sort(key=lambda r: r["arg_index"])
    if [r["arg_index"] for r in rows] != list(range(len(rows))):
        raise SystemExit("%s 的 arg_index 不是 0..n-1 连续" % path)
    values = onp.array([r["value"] for r in rows], dtype=onp.float64)
    fixed = onp.array([bool(r.get("fixed", False)) for r in rows])
    return values, fixed, [r["path"] for r in rows]


def _tensor_path(case_dir, index, family, basis):
    stem = _spec_stem(family)
    fname = None
    for fam in index.get("family", []):
        if fam.get("name") == family or fam.get("spec") == stem:
            fname = fam.get(basis + "_file")
            break
    if fname is None:
        cands = glob.glob(os.path.join(case_dir, "tensors", stem, basis, "*.npy"))
        if not cands:
            raise SystemExit("缺 tensors/%s/%s 的 cache（先跑 case.py build）"
                             % (stem, basis))
        fname = os.path.basename(cands[0])
    path = os.path.join(case_dir, "tensors", stem, basis, fname)
    if not os.path.exists(path):
        raise SystemExit("缺 %s" % path)
    return path


def _spec_stem(spec):
    """`[[family]].spec` 写不写 `.toml` 都接受（归一在这里，一次）。"""
    return os.path.splitext(os.path.basename(spec))[0]


def _channel_names(case_dir, spec):
    with open(os.path.join(case_dir, "specs", _spec_stem(spec) + ".toml"),
              "rb") as fh:
        raw = _toml.load(fh)
    chans = raw.get("channel") or []
    if not chans:
        raise SystemExit("%s 没有显式 [[channel]]；耦合需要显式的道名" % spec)
    return [c["name"] for c in chans]


# ------------------------------------------------------------------ 跨表象标定
def load_calibration(case_dir, index, family):
    """每个道的**复标定常数** `c_i`，使得

        W_i(e, λ)  ==  c_i · (EPS_A · T_i(e, μ))_λ        逐事例

    为什么需要它：同一个 spec 的两份 cache **不是同一个数组**，而是相差一个
    逐道复常数（`kk_dis` 报告 §5 实测 route1/route4 的 `|c_i|` = 3.4641/6.5320，
    正是当初标定的 `g_i`）。所以：

        ❌ 在**固定参数点**上跨表象比 likelihood —— 差几个百分点，看起来很糟
        ✅ 比**逐道形状**（本函数给出的 `c_i`），或比**收敛后**的最优值

    本函数返回 `{"source": "index.toml"|"computed"|"none", "c": ndarray}`。
    `index.toml` 里若有 `calibration_cart_to_casc` 就直接用（由 `case.py`
    的 `--calibrate` 阶段写入）；否则返回全 1（并在调用方报警）。
    """
    for fam in index.get("family", []):
        if fam.get("name") != family:
            continue
        c = fam.get("calibration_cart_to_casc")
        if c:
            # TOML 没有复数标量：index.toml 存的是 [[re, im], ...]
            arr = onp.asarray(c, dtype=onp.float64)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise SystemExit("calibration_cart_to_casc 要写成 [[re, im], ...]，"
                                 "收到 %s" % (c,))
            return {"source": "index.toml",
                    "c": arr[:, 0] + 1j * arr[:, 1],
                    "max_rel": float(fam.get("calibration_max_rel", 1.0))}
    return {"source": "none", "c": None}


def compute_calibration(t_cart, t_casc):
    """从两份 cache 反解逐道复常数 `c_i = <W|EPS_A T>/<W|W>`。

    返回 (c, 逐事例相对残差的最大值)。残差大说明**不是**同一个振幅
    （形状不同，而不是标定不同）—— 那就是真问题，不能靠标定掩盖。
    已知的三处（见 README"已知边界"）：
        ex3_rhopi  ~0.30   Eq.(46) 里没有投影算符
        ex6 rr/f2s ~0.5-1  f₂ 支的母系自旋投影错配
    """
    W = onp.einsum("lm,iem->iel", EPS_A, onp.asarray(t_cart, dtype=onp.complex128))
    C = onp.asarray(t_casc)
    cs, worst = [], 0.0
    for i in range(W.shape[0]):
        denom = onp.vdot(C[i], C[i])
        c = onp.vdot(C[i], W[i]) / denom if abs(denom) > 0 else 0.0
        cs.append(c)
        r = float(onp.abs(W[i] - c * C[i]).max() / max(onp.abs(W[i]).max(), 1e-300))
        worst = max(worst, r)
    return onp.asarray(cs), worst


# ------------------------------------------------------------------ 数据结构
class Case:
    """case.toml + 盘上 cache -> 一组不可变数组。**只做一次搬运**。"""

    def __init__(self, case_dir, basis, n_data=0, n_mc=0):
        self.dir = case_dir
        self.case, self.index = load_case(case_dir)
        self.basis = basis
        self.values, self.fixed, self.paths = load_params(case_dir)
        self.free_idx = onp.flatnonzero(~self.fixed)
        self.values_j = np.asarray(self.values)

        self.families = []
        pos = 0
        for fam in self.case["family"]:
            chans = _channel_names(case_dir, fam["spec"])
            n_ch = len(chans)
            sl = slice(pos, pos + 2 * n_ch)
            pos += 2 * n_ch
            # 目录按 spec 名（一份 spec = 一组道表）；族名是逻辑分组
            src = _tensor_path(case_dir, self.index, fam["spec"], basis)
            arr = onp.load(src)
            if arr.ndim != 3:
                raise SystemExit("%s 的 rank=%d，期望 3 (n_ch, N, k)"
                                 % (src, arr.ndim))
            if arr.shape[0] != n_ch:
                raise SystemExit(
                    "%s 有 %d 个道，但 spec 里 %d 个 [[channel]] —— "
                    "改了 spec 却没重跑 case.py build --only tensors"
                    % (src, arr.shape[0], n_ch))
            calib = load_calibration(case_dir, self.index, fam["name"])
            # index.toml 里的 [[family]].name 是逻辑族名（与 calibration 绑定）
            self.families.append({
                "name": fam["name"], "channels": chans, "slice": sl,
                "source": src, "tensor": arr, "n_ch": n_ch,
                "k": arr.shape[-1], "is_complex": bool(onp.iscomplexobj(arr)),
                "meta": fam, "calib": calib,
            })
        if pos != len(self.values):
            raise SystemExit(
                "自由参数 %d 个，但 case 里的道一共要 %d 个 —— "
                "改了 spec 的道表却没重跑 case.py build --only params"
                % (len(self.values), pos))

        # 所有族必须共享同一套槽位语义（复/实）与事例数
        kinds = {f["is_complex"] for f in self.families}
        if len(kinds) != 1:
            raise SystemExit("同一个 case 里混了实数与复数 cache：%s"
                             % [(f["name"], f["is_complex"]) for f in self.families])
        self.is_complex = kinds.pop()
        n = self.families[0]["tensor"].shape[1]
        self.n_data = int(min(n_data or n, n))
        self.n_mc = int(min(n_mc or n, n))
        for fam in self.families:
            t = fam["tensor"]
            if t.shape[1] != n:
                raise SystemExit("%s 的事例数 %d != %d"
                                 % (fam["source"], t.shape[1], n))
            # 跨表象标定：把逐道复常数**乘进张量**（而不是乘进耦合）。
            # 这样拟合出来的 g_i 就是"以这个表象的 cache 为单位"的耦合，
            # 两个表象之间只差一个可吸收的常数 —— 与 kk_dis §5 的结论一致。
            c = fam["calib"]["c"]
            # 只在**真的是常数关系**时才标定。判据用 index.toml 里记的残差：
            # 残差大 = 形状不同（缺投影算符 / f₂ 自旋投影错配），此时乘一个
            # 最小二乘常数只会把差异*摊平*成一个假的几倍因子 —— psi_phi4pi
            # 实测会得到 I_rel ~ 1e4 的噪声。宁可让它如实报错。
            if c is not None and fam["calib"].get("max_rel", 1.0) > 1e-6:
                c = None
            # ⚠️ 方向：`c` 的定义是 `W = c · C`（W = EPS_A·T 是协变侧，
            # C 是级联侧）。要让两侧可比，必须把**级联侧乘上 c**
            # （等价于协变侧除以 c）。乘错方向会差 c²，强度上差 c⁴ ——
            # 实测 I 最大差 ~1e6 倍，非常显眼但很容易被当成"物理不同"。
            self.calibrated = bool(c is not None) and basis == "helicity_cascade"
            if self.calibrated:
                if c.size != t.shape[0]:
                    raise SystemExit(
                        "标定常数 %d 个 != 道数 %d（%s）"
                        % (c.size, t.shape[0], fam["name"]))
                t = t * c[:, None, None].astype(t.dtype)
            fam["t_data"] = np.asarray(t[:, :self.n_data])
            fam["t_mc"] = np.asarray(t[:, n - self.n_mc:])

    # -------------------------------------------------- 参数 -> 每族复耦合
    def unpack(self, x):
        """扁平**实**参 -> [每族 (n_ch,) complex]。

        `g_i = const_i + i·theta_i`：`free_params` 里 const/theta 交替排列。
        固定项用文件里的值（`jnp.where` 而不是 `.at[].set` —— 后者在
        `jax.jit` 里对常量索引虽然能工作，但会把整个数组标成可写，
        后续所有对 `full` 的读取都变慢）。
        """
        full = self.values_j.at[self.free_idx].set(x)
        out = []
        for fam in self.families:
            sl = fam["slice"]
            # slice 的 start/stop 是 Python int，静态切片可以在 tracing 里用
            seg = full[sl]
            out.append(seg[0::2] + 1j * seg[1::2])
        return out

    def initial(self):
        return np.asarray(self.values[self.free_idx])


# ------------------------------------------------------------------ 物理
def _family_propagator(fam, s=None):
    """传播子 / 线形因子。**当前恒等 = 1**，这是刻意留的空位。

    接 BW / flatté / K-matrix 时只改这里：`fam["meta"]` 已经带了
    `propagator`/`mass`/`width`/`sbc`，调用方（`_amplitude`）不需要动。
    想按事件算就必须把 `s_e` 传进来 —— 现在没有，所以它只能是常数 1。
    """
    return 1.0


def _amplitude(families, cpls, prefix):
    """A(e, k) = Σ_族 f_R · Σ_i g_i W_i(e, k)，各族相干相加。

    所有族共用同一个末轴（`k`）：cartesian 是 μ = x,y,z，螺旋度是 λ = +1,0,-1。
    这是"同一批末态、不同共振族"能相干的前提 —— 与 `kk_dis` 的
    `_total_amplitude`（四类振幅逐 k 相加）同构。
    """
    total = None
    for fam, cpl in zip(families, cpls):
        amp = np.einsum("iek,i->ek", fam[prefix], cpl)
        amp = amp * _family_propagator(fam)
        total = amp if total is None else total + amp
    return total


def _intensity(amp, is_complex):
    """I(e) = Σ_{横向槽位} |A|²。槽位是承重约定，见文件头的 ⚠️。

    **两侧都取 |A|²**，不是 Re²：耦合 g_i 是复数，所以即使是实张量
    `T_i(e,μ)`，`A_μ = Σ_i g_i T_i,μ` 也是复数。丢掉虚部会得到一个
    "能跑但错"的强度（在纯实耦合时恰好对，掩盖 bug）。

    等价性的来源是线性：`W = EPS_A·T` 逐分量线性，所以
    `Σ_{λ=±1} |Σ_i g_i W_i,λ|² = Σ_{μ=x,y} |Σ_i g_i T_i,μ|²` 逐事例成立。
    """
    slots = HEL_TRANSVERSE if is_complex else CART_TRANSVERSE
    return np.sum(np.abs(amp[:, slots]) ** 2, axis=-1)


def make_functions(c):
    """返回 (intensity_data, intensity_mc, likelihood)，都吃扁平实参。"""

    def intensity_data(x):
        return _intensity(_amplitude(c.families, c.unpack(x), "t_data"),
                          c.is_complex)

    def intensity_mc(x):
        return _intensity(_amplitude(c.families, c.unpack(x), "t_mc"),
                          c.is_complex)

    def likelihood(x):
        tiny = np.finfo(np.float64).tiny
        return (-np.sum(np.log(np.maximum(intensity_data(x), tiny)))
                + c.n_data * np.log(np.maximum(np.mean(intensity_mc(x)), tiny)))

    return intensity_data, intensity_mc, likelihood


# ------------------------------------------------------------------ 自检
def _rel(a, b):
    a, b = onp.asarray(a), onp.asarray(b)
    return float(onp.abs(a - b).max() / max(onp.abs(b).max(), 1e-300))


def _fd_grad(f, x, eps=1e-6):
    """中心差分。参数是**实数**，所以这里没有 jax 复梯度的共轭约定问题 ——
    那正是"把复耦合拆成 const/theta 两个实参"的额外好处之一。
    """
    x = onp.asarray(x)
    g = onp.zeros(x.size)
    for i in range(x.size):
        e = onp.zeros_like(x)
        e[i] = eps
        g[i] = (float(f(x + e)) - float(f(x - e))) / (2 * eps)
    return g


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case_dir")
    ap.add_argument("--basis", default="both",
                    choices=["both", "cartesian", "helicity_cascade"])
    ap.add_argument("--n-data", type=int, default=0)
    ap.add_argument("--n-mc", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="grad/HVP 有限差分自检")
    ap.add_argument("--fit", action="store_true", help="短跑一次 L-BFGS-B")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--gpu", action="store_true",
                    help="要求 jax 看到 GPU（看不到就报错，不静默降级）")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    devs = jax.devices()
    if args.gpu and not any(d.platform == "gpu" for d in devs):
        raise SystemExit("--gpu 要求 GPU，但 jax 只看到 %s" % (devs,))

    case_dir = os.path.abspath(args.case_dir)
    bases = (["cartesian", "helicity_cascade"] if args.basis == "both"
             else [args.basis])
    out = {"case": os.path.basename(case_dir),
           "dir": case_dir,
           "devices": [str(d) for d in devs],
           "jax": jax.__version__, "x64": bool(jax.config.jax_enable_x64)}

    loaded = {}
    for b in bases:
        c = Case(case_dir, b, args.n_data, args.n_mc)
        inten, inten_mc, lik = make_functions(c)
        loaded[b] = {"case": c, "intensity": inten, "intensity_mc": inten_mc,
                     "likelihood": lik}
        out.setdefault("setup", {})[b] = {
            "families": [f["name"] for f in c.families],
            "n_ch": sum(f["n_ch"] for f in c.families),
            "n_free": int(c.free_idx.size), "n_data": c.n_data,
            "n_mc": c.n_mc, "k": c.families[0]["k"],
            "dtype": "complex128" if c.is_complex else "float64",
            "sources": {f["name"]: os.path.relpath(f["source"], case_dir)
                        for f in c.families},
        }

    x0 = loaded[bases[0]]["case"].initial()
    rng = onp.random.default_rng(0)
    x_alt = x0 * 0.37 + 0.02 * rng.normal(size=x0.size)

    # ---- 1) 两份 cache：强度与似然必须逐位相同（最强的一条判据）
    #
    # 判据要**逐族**看，不能只看总量：ex3_rhopi 是已知的形状失配（Eq.46 没有
    # 投影算符，rel 0.31），它的贡献会把其余三族的 1e-7 完全盖住。
    # 所以先给逐族残差，再给"去掉已知失配族之后"的总残差。
    if len(bases) == 2:
        ca, cc = loaded["cartesian"]["case"], loaded["helicity_cascade"]["case"]
        # 哪些族**可比**：只有当两份 cache 严格差一个逐道复常数时才有意义。
        # ex3_rhopi / ex6_rr / ex6_f2s 是形状不同（缺投影算符、f₂ 自旋投影错配），
        # 它们的逐族比值是"拿两个不同振幅相除"，量级毫无物理含义
        # （实测 ex6_f2s 能到 1e7）。所以这类族单独列名，不混进总残差。
        comparable = {}
        for fam in ca.families:
            rel = (fam["calib"] or {}).get("max_rel")
            comparable[fam["name"]] = (rel is not None and rel <= 1e-6)
        out["comparable_families"] = comparable
        # 总残差只在**可比族**上算。把不可比族整个剔掉，而不是让它们的
        # 荒谬比值（1e7）污染一个看起来"整体差很多"的数字。
        keep = [f for f in ca.families if comparable[f["name"]]]
        eq = {}
        for tag, x in (("initial", x0), ("perturbed", x_alt)):
            entry = {}
            if keep:
                # ⚠️ 两个 Case 的 families 是**不同的 dict 对象**，不能互相
                # `.index()`（会 ValueError）。一律用**位置下标**对齐：两侧的
                # 族顺序都由 case.toml 决定，必然一致。
                gc, gh = ca.unpack(x), cc.unpack(x)
                idx = [i for i, f in enumerate(ca.families)
                       if comparable[f["name"]]]
                entry["intensity_rel_on_comparable"] = _rel(
                    _intensity(_amplitude([ca.families[i] for i in idx],
                                          [gc[i] for i in idx], "t_data"),
                               ca.is_complex),
                    _intensity(_amplitude([cc.families[i] for i in idx],
                                          [gh[i] for i in idx], "t_data"),
                               cc.is_complex))
            if not all(comparable.values()):
                entry["excluded"] = [k for k, v in comparable.items() if not v]
            per = {}
            for fam_c, fam_h in zip(ca.families, cc.families):
                if not comparable[fam_c["name"]]:
                    # 不可比：给出**形状残差**（A1 判据的残差），而不是比值。
                    # 比值在这里是"两个不同振幅相除"，没有物理含义。
                    per[fam_c["name"]] = {
                        "shape_rel": (fam_c["calib"] or {}).get("max_rel"),
                        "note": "两个 cache 是不同振幅，不能相除"}
                    continue
                xc = ca.unpack(x)[ca.families.index(fam_c)]
                xh = cc.unpack(x)[cc.families.index(fam_h)]
                ac = _amplitude([fam_c], [xc], "t_data")
                ah = _amplitude([fam_h], [xh], "t_data")
                per[fam_c["name"]] = _rel(_intensity(ac, ca.is_complex),
                                          _intensity(ah, cc.is_complex))
            entry["per_family"] = per
            eq[tag] = entry
        out["cart_vs_casc"] = eq

    # ---- 2) 有限性 / 量级（L3）
    san = {}
    for b in bases:
        i = onp.asarray(loaded[b]["intensity"](x0))
        v = float(loaded[b]["likelihood"](x0))
        san[b] = {"likelihood": v, "I_min": float(i.min()),
                  "I_max": float(i.max()),
                  "finite": bool(onp.isfinite(i).all() and onp.isfinite(v))}
    out["sanity"] = san

    # ---- 3) grad / HVP 自检
    if args.check:
        chk = {}
        for b in bases:
            lik = loaded[b]["likelihood"]
            gfun = jit(grad(lik))
            g = onp.asarray(gfun(x0))
            vec = onp.asarray(rng.normal(size=x0.size))
            hv = onp.asarray(jit(lambda v: jvp(gfun, (x0,), (v,))[1])(vec))
            eps = 1e-6
            fd = (onp.asarray(gfun(x0 + eps * vec))
                  - onp.asarray(gfun(x0 - eps * vec))) / (2 * eps)
            chk[b] = {"grad_vs_fd_rel": _rel(g, _fd_grad(lik, x0)),
                      "hvp_vs_fd_rel": _rel(hv, fd),
                      "hvp_max": float(onp.abs(hv).max())}
        out["derivative_check"] = chk

    # ---- 4) 计时（先编译，再只量执行）
    #
    # ⚠️ 必须等**梯度**也 ready：异步分发下 `v.block_until_ready()` 只等前半个
    # 计算，测出来的"value+grad"其实是"value"的时间，会系统性偏小 2 倍。
    timing, throughput = {}, {}
    n_ev = loaded[bases[0]]["case"].n_data + loaded[bases[0]]["case"].n_mc
    for b in bases:
        lik = loaded[b]["likelihood"]
        f = jit(lambda x: (lik(x), grad(lik)(x)))
        jax.block_until_ready(f(x0))          # 触发编译并等待
        t0 = time.time()
        out_last = None
        for _ in range(args.iters):
            out_last = f(x0)                  # 异步分发，不等待
        jax.block_until_ready(out_last)       # 只等最后一次，前面必然已完成
        dt = (time.time() - t0) / args.iters
        timing[b] = dt
        throughput[b] = n_ev / dt
    out["seconds_per_value_grad"] = timing
    out["events_per_s_value_grad"] = throughput

    # ---- 5) 短跑优化器
    if args.fit:
        from scipy.optimize import minimize
        b = bases[-1]
        lik = loaded[b]["likelihood"]
        gfun = jit(grad(lik))
        t0 = time.time()
        res = minimize(lambda x: float(lik(x)), x0,
                       jac=lambda x: onp.asarray(gfun(x)),
                       method="L-BFGS-B", options={"maxiter": args.iters})
        out["fit"] = {"basis": b, "success": bool(res.success),
                      "nit": int(res.nit), "nfev": int(res.nfev),
                      "f0": float(res.fun), "seconds": round(time.time() - t0, 3),
                      "note": "短跑，未收敛是预期的"}
        os.makedirs(os.path.join(case_dir, "output"), exist_ok=True)
        onp.save(os.path.join(case_dir, "output", "fitted_%s.npy" % b),
                 onp.asarray(res.x))

    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
