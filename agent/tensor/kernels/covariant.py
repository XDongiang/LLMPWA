#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""协变张量内核（Chung 形式）—— 纯 numpy，无 jax / 无 tf-pwa 依赖。

来源
----
本文件是 `task/20260929_螺旋度耦合振幅推导/1_运行代码/tensor_example/Tensor/
{BASE,FUNC}.py` 的**逐式移植**：Chung, PRD 48 (1993) 1225 的协变张量形式。

移植时只做两件事，**不动任何公式**：
  1. `jax.numpy` -> `numpy`，去掉 `@jit`（jax 的 jit 只是加速，不改变数值）；
  2. 把 `FUNCTION` 类里 `self.psi2s` 这个硬编码常量提成显式参数 `psi`，
     这样 psi 四动量口径（`psi_mass_residual.md` §6b）可以由调用方控制。

约定
----
* 度规 `diag(-1,-1,-1,1)`，四矢量序 `(x, y, z, E)`（与 `Tensor/BASE.py` 一致）。
  注意这与 `specgen/runtime.py` 的 `(E,x,y,z)` 相反 —— 两个内核各自保持自己
  原生的约定，由 `agent/tensor/basis.py` 在边界处转换。
* 全部逐事例函数接受 `(N, 4)` 的批量输入，返回 `(N, k)`。

    T1_cov(a,b,c) -> (N,4)       Eq.(10)
    T2_cov(a,b,c) -> (N,4,4)     Eq.(11)
    T4_cov(a,b,c) -> (N,4,4,4,4) Eq.(23)

七条道（与 `FUNC.FUNCTION` 的 `phif001/phif021/phif201/phif221/phif222/
phif223/phif243` 一一对应）由 `channels()` 返回。
"""
from __future__ import annotations

import os

import numpy as np

# R = 1 -> Q0
Q0 = 0.197321

# 硬编码的 psi(2S) 四动量（FUNC.py:15），(x,y,z,E)
PSI2S_DEFAULT = np.array([0.0, 0.0, 0.0, 3.686])


# --------------------------------------------------------------------- 度规
def metric():
    """G_{mu,nu} = diag(-1,-1,-1,1)。"""
    G = np.eye(4)
    G[0, 0] = -1.0
    G[1, 1] = -1.0
    G[2, 2] = -1.0
    return G


# 度规是对角的，所以"用 G 升 k 个指标"就退化成乘以符号外积：
#     T^{o...} = sum_{i...} G^{oi} ... T_{i...} = g[o]...g[t] * T_{o...t}
# 因为 G 只有对角项，求和里只剩 i=o, j=p, ... 一项。这是**恒等变形**（与
# einsum 结果逐位相同，见 .dsh/bench/probe_metric.py），但 P3 那种 6 指标的
# 升指标从 3.1 s/16 事例降到 2e-6 s —— einsum 会在内部物化巨大的中间张量。
_METRIC_SIGN = np.array([-1.0, -1.0, -1.0, 1.0])


def _sign_outer(k):
    """g 的 k 重外积，形状 (4,)*k；对最后 k 个指标做对角度规升指标。"""
    out = _METRIC_SIGN
    for _ in range(k - 1):
        out = np.multiply.outer(out, _METRIC_SIGN)
    return out


# 收缩路径缓存：`optimize="optimal"` 会做一次全局搜索，对 phif222 那种 7 操作数
# 的收缩值 ~100x（0.081 s -> 0.0001 s / 128 事例）。路径只与下标串有关，与数组
# 形状无关（本例里形状固定），所以按下标串缓存一次即可。结果与默认路径**逐位相同**。
_EINSUM_PATHS = {}


def _einsum(subscripts, *operands):
    """`np.einsum` + 缓存的 optimal 收缩路径。

    只对 >=3 个操作数的收缩做路径搜索（2 个操作数时 numpy 本身已是最优）。
    """
    if len(operands) < 3:
        return np.einsum(subscripts, *operands)
    path = _EINSUM_PATHS.get(subscripts)
    if path is None:
        path, _info = np.einsum_path(subscripts, *operands, optimize="optimal")
        _EINSUM_PATHS[subscripts] = path
    return np.einsum(subscripts, *operands, optimize=path)


def _raise_last(t, k):
    """把 (...,4,...,4) 的最后 k 个指标升上去（乘常数符号外积，恒等变形）。

    对应原版的 `np.einsum("...,eim,ejn,...->...", t, Gb, Gb, ...)`。
    """
    return t * _sign_outer(k)


def _metric_axis(t, axis):
    """把 t 的第 axis 轴用对角度规缩并（等价于 `np.einsum("...ei...,eij->...ej...")`）。"""
    shape = [1] * t.ndim
    shape[axis] = 4
    return t * _METRIC_SIGN.reshape(shape)


_LECI_CON = None


def _leci_con():
    """leci^{mu,nu,rho,sigma} = G G G G · leci_{...}（对角度规 -> 逐轴乘 g）。"""
    global _LECI_CON
    if _LECI_CON is None:
        _LECI_CON = _raise_last(leci_four(), 4)
    return _LECI_CON


def leci_four():
    """epsilon_{mu,nu,rho,sigma}（全反对称，逐行照抄 BASE.py::leci_four）。"""
    leci = np.zeros((4, 4, 4, 4))
    leci[0, 1, 2, 3] = 1
    leci[0, 1, 3, 2] = -1
    leci[0, 3, 1, 2] = 1
    leci[3, 0, 1, 2] = -1
    leci[3, 0, 2, 1] = 1
    leci[0, 3, 2, 1] = -1
    leci[0, 2, 3, 1] = 1
    leci[0, 2, 1, 3] = -1
    leci[2, 0, 1, 3] = 1
    leci[2, 0, 3, 1] = -1
    leci[2, 3, 0, 1] = 1
    leci[3, 2, 0, 1] = -1
    leci[3, 2, 1, 0] = 1
    leci[2, 3, 1, 0] = -1
    leci[2, 1, 3, 0] = 1
    leci[2, 1, 0, 3] = -1
    leci[1, 2, 0, 3] = 1
    leci[1, 2, 3, 0] = -1
    leci[1, 3, 2, 0] = 1
    leci[3, 1, 2, 0] = -1
    leci[3, 1, 0, 2] = 1
    leci[1, 3, 0, 2] = -1
    leci[1, 0, 3, 2] = 1
    leci[1, 0, 2, 3] = -1
    return leci


# --------------------------------------------------------------------- 标量
def _atleast2(v):
    """把 (4,) 或 (N,4) 统一成 (N,4)；本内核内部一律批量。"""
    v = np.asarray(v, dtype=float)
    return v[None, :] if v.ndim == 1 else v


def dot(a, b):
    """a . b = a_i b_j G^{ij}（度规升指标）。接受 (4,) 或 (N,4)。

    单事例时返回标量，批量时返回 (N,) —— 与原版 BASE.py::dot 的
    "单事例返回标量" 行为一致，方便逐式对拍。
    """
    scalar_in = np.asarray(a).ndim == 1
    a, b = _atleast2(a), _atleast2(b)
    G = metric()
    out = np.einsum("ei,ej,ij->e", a, b, G)
    return out[0] if scalar_in else out


def invm(a):
    return np.sqrt(dot(a, a))


def Gt_con(a):
    """Eq.(2) 的 \tilde{G}^{mu,nu} 逆变。接受 (4,) 或 (N,4)，返回 (N,4,4)。"""
    a = _atleast2(a)
    G = metric()
    return G - np.einsum("ei,ej->eij", a, a) / dot(a, a)[:, None, None]


def Gt_cov(a):
    """\tilde{G}_{mu,nu} 协变。接受 (4,) 或 (N,4)，返回 (N,4,4)。"""
    a = _atleast2(a)
    gt_con = Gt_con(a)
    return _raise_last(gt_con, 2)


def Qabc(a, b, c):
    """Eq.(13)。接受 (4,) 或 (N,4)。"""
    a, b, c = _atleast2(a), _atleast2(b), _atleast2(c)
    Sa, Sb, Sc = dot(a, a), dot(b, b), dot(c, c)
    return (Sa + Sb - Sc) ** 2.0 / (4.0 * Sa) - Sb


def B1(a, b, c):
    """Eq.(14) Blatt-Weisskopf l=1。"""
    return np.sqrt(2.0 / (Qabc(a, b, c) + Q0 ** 2))


def B2(a, b, c):
    """Eq.(15) l=2。"""
    q = Qabc(a, b, c)
    return np.sqrt(13.0 / (q ** 2 + 3 * q * (Q0 ** 2) + 9 * (Q0 ** 4)))


def B3(a, b, c):
    """Eq.(16) l=3。"""
    q = Qabc(a, b, c)
    return np.sqrt(277.0 / (q ** 3 + 6 * (q ** 2) * (Q0 ** 2)
                            + 45 * q * (Q0 ** 4) + 225 * (Q0 ** 6)))


def B4(a, b, c):
    """Eq.(17) l=4。"""
    q = Qabc(a, b, c)
    return np.sqrt(12746.0 / (q ** 4 + 10 * (q ** 3) * (Q0 ** 2)
                              + 135 * (q ** 2) * (Q0 ** 4)
                              + 1575 * q * (Q0 ** 6) + 11025 * (Q0 ** 8)))


# --------------------------------------------------------------- 节点张量
def T1_cov(a, b, c):
    """Eq.(10)：\tilde{t}^{(1)}_{mu}，返回 (N,4)。"""
    a, b, c = _atleast2(a), _atleast2(b), _atleast2(c)
    gt_cov = Gt_cov(a)
    r = b - c
    return np.einsum("eij,ej->ei", gt_cov, r) * B1(a, b, c)[:, None]


def T2_cov(a, b, c):
    """Eq.(11)：\tilde{t}^{(2)}_{mu nu}，返回 (N,4,4)。"""
    a, b, c = _atleast2(a), _atleast2(b), _atleast2(c)
    gt_cov = Gt_cov(a)
    r = b - c
    t_r = np.einsum("eij,ej->ei", gt_cov, r)
    return (np.einsum("ei,ej->eij", t_r, t_r)
            - (1.0 / 3) * dot(t_r, t_r)[:, None, None] * gt_cov) * B2(a, b, c)[:, None, None]


def T3_cov(a, b, c):
    """Eq.(12)：\tilde{t}^{(3)}_{mu nu rho}，返回 (N,4,4,4)。"""
    a, b, c = _atleast2(a), _atleast2(b), _atleast2(c)
    gt_cov = Gt_cov(a)
    r = b - c
    t_r = np.einsum("eij,ej->ei", gt_cov, r)
    gt_r_1 = np.einsum("eij,ek->eijk", gt_cov, t_r)
    gt_r_2 = np.einsum("ejk,ei->eijk", gt_cov, t_r)
    gt_r_3 = np.einsum("eki,ej->eijk", gt_cov, t_r)
    rrr = np.einsum("ei,ej,ek->eijk", t_r, t_r, t_r)
    return (rrr - (1.0 / 5) * dot(t_r, t_r)[:, None, None, None]
            * (gt_r_1 + gt_r_2 + gt_r_3)) * B3(a, b, c)[:, None, None, None]


def T4_cov(a, b, c):
    """Eq.(23)：\tilde{t}^{(4)}，返回 (N,4,4,4,4)。"""
    a, b, c = _atleast2(a), _atleast2(b), _atleast2(c)
    r = b - c
    p4 = P4(a)
    return np.einsum("eijklmnop,em,en,eo,ep->eijkl", p4, r, r, r, r) \
        * B4(a, b, c)[:, None, None, None, None]


# ------------------------------------------------------------- 投影算子
#
# 重要：P2/P3/P4 **只依赖传入的四动量 a**，与事例无关。生产里 a 恒为 psi(2S)
# 且是常数，所以这三个张量对整份数据只需算一次。原版 Tensor/FUNC.py 之所以
# 看不出这个开销，是因为 psi2s 是 jit 的编译期常量，XLA 在 trace 阶段就把它
# 常量折叠掉了（等价于免费的缓存）；numpy 版没有这一步，必须自己缓存，
# 否则每个 batch 都会重建 4^8 的 P4（实测 7.2 s / 128 事例）。
_PROJ_CACHE = {}
_PROJ_CACHE_MAX = 8


def _proj_key(a):
    """把 a 规范化成缓存键 + 用于构建的输入。

    psi(2S) 对每个事例都相同，所以先压成**单事例**再缓存：这样 (1,4,4,...) 的
    结果可以跨 batch 广播复用，不会因为最后一批 105 vs 128 而失效。
    若 a 真的逐事例不同，则退回按整块缓存（正确，只是没有复用收益）。
    """
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[0] > 1:
        first = arr[0]
        if np.array_equal(arr, np.broadcast_to(first, arr.shape)):
            arr = arr[:1]
    return (arr.shape, arr.tobytes()), arr


def _proj_cached(name, a, build):
    """对投影算子做记忆化；a 变化（如换 psi 方向）时自动失效。"""
    key, arr = _proj_key(a)
    key = (name,) + key
    hit = _PROJ_CACHE.get(key)
    if hit is not None:
        return hit
    val = build(arr)
    if len(_PROJ_CACHE) >= _PROJ_CACHE_MAX:
        _PROJ_CACHE.clear()
    _PROJ_CACHE[key] = val
    return val


def clear_projection_cache():
    """清空 P2/P3/P4 缓存（换 psi 质量或做数值对拍时调用）。"""
    _PROJ_CACHE.clear()


def P2(a):
    """Eq.(20)：P^{(2)}，返回 (N,4,4,4,4)。"""
    return _proj_cached("P2", a, _P2_impl)


def P3(a):
    """Eq.(21)：P^{(3)}，返回 (N,4,4,4,4,4,4)。"""
    return _proj_cached("P3", a, _P3_impl)


def P4(a):
    """Eq.(22)：P^{(4)}，返回 (N,4,4,4,4,4,4,4,4)。"""
    return _proj_cached("P4", a, _P4_impl)


def _P2_impl(a):
    a = _atleast2(a)
    gt_cov = Gt_cov(a)
    g_g_1 = np.einsum("eij,ekl->eikjl", gt_cov, gt_cov)
    g_g_2 = np.einsum("eil,ekj->eikjl", gt_cov, gt_cov)
    g_g_3 = np.einsum("eik,ejl->eikjl", gt_cov, gt_cov)
    return 1.0 / 2 * (g_g_1 + g_g_2) - 1.0 / 3 * g_g_3


def _P3_impl(a):
    a = _atleast2(a)
    gt_cov = Gt_cov(a)
    g = gt_cov
    # 6 terms
    get_ggg = np.einsum("eij,ekl,emn->eikmjln", g, g, g)
    get_ggg = get_ggg + np.einsum("eij,ekn,eml->eikmjln", g, g, g)
    get_ggg = get_ggg + np.einsum("eil,ekj,emn->eikmjln", g, g, g)
    get_ggg = get_ggg + np.einsum("eil,ekn,emj->eikmjln", g, g, g)
    get_ggg = get_ggg + np.einsum("ein,ekl,emj->eikmjln", g, g, g)
    get_ggg = get_ggg + np.einsum("ein,ekj,eml->eikmjln", g, g, g)
    # 9 terms
    get_ggg_ = np.einsum("eik,ejl,emn->eikmjln", g, g, g)
    get_ggg_ = get_ggg_ + np.einsum("eik,eln,emj->eikmjln", g, g, g)
    get_ggg_ = get_ggg_ + np.einsum("eik,ejn,eml->eikmjln", g, g, g)
    get_ggg_ = get_ggg_ + np.einsum("eim,ejn,ekl->eikmjln", g, g, g)
    get_ggg_ = get_ggg_ + np.einsum("eim,ejl,ekn->eikmjln", g, g, g)
    get_ggg_ = get_ggg_ + np.einsum("eim,eln,ekj->eikmjln", g, g, g)
    get_ggg_ = get_ggg_ + np.einsum("ekm,eln,eij->eikmjln", g, g, g)
    get_ggg_ = get_ggg_ + np.einsum("ekm,ejl,ein->eikmjln", g, g, g)
    get_ggg_ = get_ggg_ + np.einsum("ekm,ejn,eil->eikmjln", g, g, g)
    return -1.0 / 6 * get_ggg + 1.0 / 15 * get_ggg_


def _P4_impl(a):
    a = _atleast2(a)
    g = Gt_cov(a)
    n = g.shape[0]

    # 24 terms: (m,n,o,p) <- (i,j,k,l) 的全排列（偶置换）
    perms24 = [
        "mi,nj,ok,pl", "mi,nj,ol,pk", "mi,nk,oj,pl", "mi,nk,ol,pj",
        "mi,nl,oj,pk", "mi,nl,ok,pj", "mj,ni,ok,pl", "mj,ni,ol,pk",
        "mj,nk,oi,pl", "mj,nk,ol,pi", "mj,nl,oi,pk", "mj,nl,ok,pi",
        "mk,ni,oj,pl", "mk,ni,ol,pj", "mk,nj,oi,pl", "mk,nj,ol,pi",
        "mk,nl,oi,pj", "mk,nl,oj,pi", "ml,ni,oj,pk", "ml,ni,ok,pj",
        "ml,nj,oi,pk", "ml,nj,ok,pi", "ml,nk,oi,pj", "ml,nk,oj,pi",
    ]
    def _sub(pat, out):
        """把单事例下标 "mi,nj,ok,pl" 变成批量 "emi,enj,eok,epl->e..."。"""
        lhs = ",".join("e" + t for t in pat.split(","))
        return lhs + "->" + out

    get_gggg = np.zeros((n, 4, 4, 4, 4, 4, 4, 4, 4))
    for p in perms24:
        get_gggg = get_gggg + np.einsum(_sub(p, "emnopijkl"), g, g, g, g)

    # 72 terms
    perms72 = [
        "ij,mn,pl,ok", "ij,mn,ol,pk", "ij,mo,pl,nk", "ij,mo,nl,pk",
        "ij,mp,ol,nk", "ij,mp,nl,ok", "ij,no,pl,mk", "ij,no,ml,pk",
        "ij,np,ol,mk", "ij,np,ml,ok", "ij,op,nl,mk", "ij,op,ml,nk",
        "ik,mn,pl,oj", "ik,mn,ol,pj", "ik,mo,pl,nj", "ik,mo,nl,pj",
        "ik,mp,ol,nj", "ik,mp,nl,oj", "ik,no,pl,mj", "ik,no,ml,pj",
        "ik,np,ol,mj", "ik,np,ml,oj", "ik,op,nl,mj", "ik,op,ml,nj",
        "il,mn,pk,oj", "il,mn,ok,pj", "il,mo,pk,nj", "il,mo,nk,pj",
        "il,mp,ok,nj", "il,mp,nk,oj", "il,no,pk,mj", "il,no,mk,pj",
        "il,np,ok,mj", "il,np,mk,oj", "il,op,nk,mj", "il,op,mk,nj",
        "jk,mn,pl,oi", "jk,mn,ol,pi", "jk,mo,pl,ni", "jk,mo,nl,pi",
        "jk,mp,ol,ni", "jk,mp,nl,oi", "jk,no,pl,mi", "jk,no,ml,pi",
        "jk,np,ol,mi", "jk,np,ml,oi", "jk,op,nl,mi", "jk,op,ml,ni",
        "jl,mn,pk,oi", "jl,mn,ok,pi", "jl,mo,pk,ni", "jl,mo,nk,pi",
        "jl,mp,ok,ni", "jl,mp,nk,oi", "jl,no,pk,mi", "jl,no,mk,pi",
        "jl,np,ok,mi", "jl,np,mk,oi", "jl,op,nk,mi", "jl,op,mk,ni",
        "kl,mn,pj,oi", "kl,mn,oj,pi", "kl,mo,pj,ni", "kl,mo,nj,pi",
        "kl,mp,oj,ni", "kl,mp,nj,oi", "kl,no,pj,mi", "kl,no,mj,pi",
        "kl,np,oj,mi", "kl,np,mj,oi", "kl,op,nj,mi", "kl,op,mj,ni",
    ]
    get_g_g_g_g = np.zeros_like(get_gggg)
    for p in perms72:
        get_g_g_g_g = get_g_g_g_g + np.einsum(_sub(p, "emnopijkl"), g, g, g, g)

    # 9 terms
    perms9 = [
        "ij,kl,mn,op", "ij,kl,mo,np", "ij,kl,mp,no",
        "ik,jl,mn,op", "ik,jl,mo,np", "ik,jl,mp,no",
        "il,jk,mn,op", "il,jk,mo,np", "il,jk,mp,no",
    ]
    get_gggg_ = np.zeros_like(get_gggg)
    for p in perms9:
        get_gggg_ = get_gggg_ + np.einsum(_sub(p, "emnopijkl"), g, g, g, g)

    return 1.0 / 24 * get_gggg - 1.0 / 84 * get_g_g_g_g + 1.0 / 105 * get_gggg_


# ------------------------------------------------------------------ 七条道
def _s1_vector(psi, phi, f2, Kp, Km, Pip, Pim):
    """Eq.(41) 修正版：纯 S=1 自旋矢量（见 FUNC.py::_s1_vector 的推导注释）。"""
    t2_cov = T2_cov(f2, Pip, Pim)
    t2_con = _raise_last(t2_cov, 2)
    t1_cov = T1_cov(phi, Kp, Km)
    t1_til = np.einsum("eij,ej->ei", Gt_con(psi), t1_cov)
    W = np.einsum("eij,ej->ei", t2_con, t1_til)
    sig = np.einsum("eij,ei,ej->e", t2_con, psi, psi)
    m2 = dot(psi, psi)
    return W - (1.0 / (3.0 * m2))[:, None] * sig[:, None] * t1_til


def _phif001(psi, phi, f0, Kp, Km, Pip, Pim):
    r"""Eq.(36)：<\phi f_0|01>。"""
    t1_cov = T1_cov(phi, Kp, Km)
    return _metric_axis(t1_cov, 1)


def _phif021(psi, phi, f0, Kp, Km, Pip, Pim):
    r"""Eq.(37)：<\phi f_0|21>。"""
    t2_cov = T2_cov(psi, phi, f0)
    t2_con = _raise_last(t2_cov, 2)
    t1_cov = T1_cov(phi, Kp, Km)
    return np.einsum("eij,ej->ei", t2_con, t1_cov)


def _phif201(psi, phi, f2, Kp, Km, Pip, Pim):
    r"""Eq.(41) 修正版：<\phi f_2|01>。"""
    return _s1_vector(psi, phi, f2, Kp, Km, Pip, Pim)


def _phif221(psi, phi, f2, Kp, Km, Pip, Pim):
    r"""Eq.(42) 修正版：<\phi f_2|21>。"""
    W = _s1_vector(psi, phi, f2, Kp, Km, Pip, Pim)
    t2_cov = T2_cov(psi, phi, f2)
    t2_con = _raise_last(t2_cov, 2)
    return np.einsum("eij,ej->ei", t2_con, W)


def _phif222(psi, phi, f2, Kp, Km, Pip, Pim):
    r"""Eq.(43)：<\phi f_2|22>。"""
    leci_cov = leci_four()
    leci_con = _leci_con()
    psi2s_cov = _metric_axis(psi, 1)
    t2_phif2 = _metric_axis(T2_cov(psi, phi, f2), 2)
    t2_34 = _metric_axis(T2_cov(f2, Pip, Pim), 1)
    t1_con = _metric_axis(T1_cov(phi, Kp, Km), 1)
    # 这两条 7 操作数收缩是整条道的全部代价：默认路径 0.10 s / 128 事例，
    # `_einsum` 的 optimal 路径 0.0001 s —— 差 1000x，结果逐位相同。
    return (_einsum("ijkl,ej,emk,lnop,enm,eo,ep->ei",
                    leci_con, psi2s_cov, t2_phif2, leci_cov, t2_34, psi, t1_con)
            + _einsum("ijkl,ej,emk,mnop,enl,eo,ep->ei",
                      leci_con, psi2s_cov, t2_phif2, leci_cov, t2_34, psi, t1_con))


def _phif223(psi, phi, f2, Kp, Km, Pip, Pim):
    r"""Eq.(44)：<\phi f_2|23>。"""
    p3_cov = P3(psi)
    p3_con = _raise_last(p3_cov, 6)
    t2_phif2 = T2_cov(psi, phi, f2)
    t2_34 = T2_cov(f2, Pip, Pim)
    t1_12 = T1_cov(phi, Kp, Km)
    return np.einsum("eijklmn,ejk,elm,en->ei", p3_con, t2_phif2, t2_34, t1_12)


def _phif243(psi, phi, f2, Kp, Km, Pip, Pim):
    r"""Eq.(45)：<\phi f_2|43>。"""
    t4_cov = T4_cov(psi, phi, f2)
    t4_con = _raise_last(t4_cov, 4)
    t2_cov = T2_cov(f2, Pip, Pim)
    t1_cov = T1_cov(phi, Kp, Km)
    return np.einsum("eijkl,ej,ekl->ei", t4_con, t1_cov, t2_cov)


# f0 族与 f2 族的道名（与 Tensor/FUNC.py 的 F0_NAMES / F2_NAMES 一致）
F0_NAMES = ["phif001", "phif021"]
F2_NAMES = ["phif201", "phif221", "phif222", "phif223", "phif243"]

_CHANNELS = {
    "phif001": _phif001,
    "phif021": _phif021,
    "phif201": _phif201,
    "phif221": _phif221,
    "phif222": _phif222,
    "phif223": _phif223,
    "phif243": _phif243,
}

# 上面 7 条道在 `chains/` 里也有同名文件（emitter 从那里取源码组装产物）。
# 它们在这里保留一份函数副本是历史原因（本内核是 A1 移植的落点）。
#
# 之后新增的道（例 3 的 rho'/b1、例 5 的 K*_0/K*_2）**只写在 `chains/` 里**，
# 由 `channel_fn` 惰性回退过去 —— 避免同一份公式存两处、日后改一处漏一处。
_LEGACY_CHAINS = {"phif001": "phif0", "phif021": "phif0",
                  "phif201": "phif2", "phif221": "phif2",
                  "phif222": "phif2", "phif223": "phif2", "phif243": "phif2"}


def _chains_module():
    """取 `agent.tensor.chains` —— 兼容三种导入方式。

    1. `agent.tensor` 内部（正常路径）：`from .. import chains`；
    2. 被当作**顶层模块**直接导入（`tests/test_covariant_vs_jax.py` 就把
       `kernels/` 加进 sys.path 后 `import covariant`）；
    3. 被 `emitter` 拷贝成产物目录里的 `covariant.py` 后导入。

    后两种情况下相对导入无父包可依，于是**向上找**含
    `agent/tensor/chains/__init__.py` 的目录。找不到就抛 ImportError ——
    产物脚本本身不调用 `channel_fn`（道函数体已内联），所以这只是
    "在产物目录里手搓调用"时的兜底。
    """
    try:
        from .. import chains as m
        return m
    except ImportError:
        pass
    import sys
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        root = os.path.dirname(d)
        if root == d:
            break
        if os.path.isfile(os.path.join(root, "agent", "tensor", "chains",
                                       "__init__.py")):
            if root not in sys.path:
                sys.path.insert(0, root)
            from agent.tensor import chains as m
            return m
        d = root
    raise ImportError(
        "找不到 agent/tensor/chains/ —— 无法按道名装配新道函数。"
        "产物脚本不依赖它（道函数体已内联），只有在产物目录里直接调用 "
        "channel_fn() 时才需要。")


def _chain_channel_map():
    """惰性扫 `chains/`，返回 {道名: 链名}（不含上面 7 条 legacy 道）。"""
    out = {}
    for c in _chains_module().list_chains():
        for nm in _chains_module().list_channels(c):
            out.setdefault(nm, c)
    return out


def channel_fn(name):
    """按道名取函数：fn(psi, phi, f2, Kp, Km, Pip, Pim) -> (N, 4)。

    先在 `_CHANNELS`（本文件的 7 条 legacy 道）里找，找不到就回退到
    `agent/tensor/chains/<chain>/<channel>.py`（新增道的唯一存放处）。
    """
    if name in _CHANNELS:
        return _CHANNELS[name]
    chainsmod = _chains_module()
    for chain in chainsmod.list_chains():
        if name in chainsmod.list_channels(chain):
            return chainsmod.load_channel(chain, name)
    raise KeyError("未知道 %r；可用：%s" % (name, sorted(available_channels())))


def available_channels():
    """本内核能算的全部道名（legacy 7 条 + `chains/` 里的全部）。"""
    out = set(_CHANNELS)
    out.update(_chain_channel_map())
    return out


def compute_event(name, momenta, psi=None):
    """单事例（或一批）计算一条道的协变张量。

    Parameters
    ----------
    name    : 道名（phif001 等）
    momenta : dict，键 Kp/Km/Pip/Pim，各为 (N,4) 的 (x,y,z,E) 数组
    psi     : (4,) 或 (N,4) 的 psi(2S) 四动量；None 用 PSI2S_DEFAULT

    Returns
    -------
    (N, 4) 的复数/实数张量（本内核为实数）
    """
    Kp, Km = momenta["Kp"], momenta["Km"]
    Pip, Pim = momenta["Pip"], momenta["Pim"]
    phi = Kp + Km
    f2 = Pip + Pim
    if psi is None:
        psi = np.broadcast_to(PSI2S_DEFAULT, Kp.shape)
    return channel_fn(name)(psi, phi, f2, Kp, Km, Pip, Pim)
