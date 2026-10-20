#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""协变张量内核（Chung 形式）—— 后端无关，无 tf-pwa 依赖。

来源
----
本文件是 `task/20260929_螺旋度耦合振幅推导/1_运行代码/tensor_example/Tensor/
{BASE,FUNC}.py` 的**逐式移植**：Chung, PRD 48 (1993) 1225 的协变张量形式。

移植时只做两件事，**不动任何公式**：
  1. `jax.numpy` -> `_backend.xp`（jax 优先，无 jax 回退 numpy）；
  2. 把 `FUNCTION` 类里 `self.psi2s` 这个硬编码常量提成显式参数 `psi`，
     这样 psi 四动量口径（`psi_mass_residual.md` §6b）可以由调用方控制。

与事例无关的常量（度规、ε 张量、CG 表）用 host numpy 建：jax 数组不可变，
且 XLA 会在编译期把它们折叠成常量。

约定
----
* 度规 `diag(-1,-1,-1,1)`，四矢量序 `(x, y, z, E)`（与 `Tensor/BASE.py` 一致）。
  注意这与 `specgen/runtime.py` 的 `(E,x,y,z)` 相反 —— 两个内核各自保持自己
  原生的约定，由 `agent/tensor/basis.py` 在边界处转换。
* 全部逐事例函数接受 `(N, 4)` 的批量输入，返回 `(N, k)`。函数在 jax 下
  可被 `jit` 追踪（不用逐项赋值、不调用 numpy）。

    T1_cov(a,b,c) -> (N,4)       Eq.(10)
    T2_cov(a,b,c) -> (N,4,4)     Eq.(11)
    T4_cov(a,b,c) -> (N,4,4,4,4) Eq.(23)

七条道（与 `FUNC.FUNCTION` 的 `phif001/phif021/phif201/phif221/phif222/
phif223/phif243` 一一对应）由 `channels()` 返回。
"""
from __future__ import annotations

import itertools
import os

import numpy as onp

try:                      # 包内（正常路径）
    from . import _backend as _B
except ImportError:       # 被 emitter 拷成产物目录里的顶层 covariant.py
    import _backend as _B

xp = _B.xp

# R = 1 -> Q0
Q0 = 0.197321

# 硬编码的 psi(2S) 四动量（FUNC.py:15），(x,y,z,E)
#
# 用 host numpy 建**常量**：这些值必须在 import 期就可用，不能因为"碰一下
# xp"就把 jax 拖进来（`emit`/`check` 走的是无 jax 路径）。jax 侧把它们当
# 常量参与运算，XLA 会做常量折叠 —— 与原版 `Tensor/FUNC.py` 里 jit 编译期
# 常量折叠是同一件事。
PSI2S_DEFAULT = onp.array([0.0, 0.0, 0.0, 3.686])


# --------------------------------------------------------------------- 度规
def metric():
    """G_{mu,nu} = diag(-1,-1,-1,1)。

    返回 host 常量：jax 数组不支持 `G[i, j] = ...` 逐项赋值（在 jit 里更是
    直接报错），而度规是**与事例无关的常数**，没有理由做成设备数组。
    """
    return onp.diag(onp.array([-1.0, -1.0, -1.0, 1.0]))


# 度规是对角的，所以"用 G 升 k 个指标"就退化成乘以符号外积：
#     T^{o...} = sum_{i...} G^{oi} ... T_{i...} = g[o]...g[t] * T_{o...t}
# 因为 G 只有对角项，求和里只剩 i=o, j=p, ... 一项。这是**恒等变形**（与
# einsum 结果逐位相同，见 .dsh/bench/probe_metric.py），但 P3 那种 6 指标的
# 升指标从 3.1 s/16 事例降到 2e-6 s —— einsum 会在内部物化巨大的中间张量。
_METRIC_SIGN = onp.array([-1.0, -1.0, -1.0, 1.0])


def _sign_outer(k):
    """g 的 k 重外积，形状 (4,)*k；对最后 k 个指标做对角度规升指标。

    用 host numpy 预先算好常数外积（jax 下 `xp.multiply.outer` 会把它变成
    设备数组，且 `multiply.outer` 这种逐次外积在 XLA 里不如直接乘常数省事）。
    """
    out = _METRIC_SIGN
    for _ in range(k - 1):
        out = onp.multiply.outer(out, _METRIC_SIGN)
    return out


def _einsum(subscripts, *operands):
    """后端感知的收缩；numpy 侧带 optimal 路径缓存。

    收敛到 `_backend.einsum` —— 那边知道 jax 不接受 numpy `einsum_path` 返回
    的路径列表（元素含字符串，会 TypeError），所以 jax 侧直接用 XLA 自己的
    收缩规划。
    """
    return _B.einsum(subscripts, *operands)


def _raise_last(t, k):
    """把 (...,4,...,4) 的最后 k 个指标升上去（乘常数符号外积，恒等变形）。

    对应原版的 `xp.einsum("...,eim,ejn,...->...", t, Gb, Gb, ...)`。
    """
    return t * _sign_outer(k)


def _metric_axis(t, axis):
    """把 t 的第 axis 轴用对角度规缩并（等价于 `xp.einsum("...ei...,eij->...ej...")`）。"""
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
    """epsilon_{mu,nu,rho,sigma}（全反对称，逐行照抄 BASE.py::leci_four）。

    返回 **host numpy 常量**：jax 数组不支持逐项赋值，而这是与事例无关的常数。
    原版 `Tensor/BASE.py::leci_four` 也是用 `onp.zeros` 建的。
    """
    leci = onp.zeros((4, 4, 4, 4))
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
    v = xp.asarray(v, dtype=float)
    return v[None, :] if v.ndim == 1 else v


def dot(a, b):
    """a . b = a_i b_j G^{ij}（度规升指标）。接受 (4,) 或 (N,4)。

    单事例时返回标量，批量时返回 (N,) —— 与原版 BASE.py::dot 的
    "单事例返回标量" 行为一致，方便逐式对拍。
    """
    scalar_in = xp.asarray(a).ndim == 1
    a, b = _atleast2(a), _atleast2(b)
    G = metric()
    out = xp.einsum("ei,ej,ij->e", a, b, G)
    return out[0] if scalar_in else out


def invm(a):
    return xp.sqrt(dot(a, a))


def Gt_con(a):
    """Eq.(2) 的 \tilde{G}^{mu,nu} 逆变。接受 (4,) 或 (N,4)，返回 (N,4,4)。"""
    a = _atleast2(a)
    G = metric()
    return G - xp.einsum("ei,ej->eij", a, a) / dot(a, a)[:, None, None]


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
    return xp.sqrt(2.0 / (Qabc(a, b, c) + Q0 ** 2))


def B2(a, b, c):
    """Eq.(15) l=2。"""
    q = Qabc(a, b, c)
    return xp.sqrt(13.0 / (q ** 2 + 3 * q * (Q0 ** 2) + 9 * (Q0 ** 4)))


def B3(a, b, c):
    """Eq.(16) l=3。"""
    q = Qabc(a, b, c)
    return xp.sqrt(277.0 / (q ** 3 + 6 * (q ** 2) * (Q0 ** 2)
                            + 45 * q * (Q0 ** 4) + 225 * (Q0 ** 6)))


def B4(a, b, c):
    """Eq.(17) l=4。"""
    q = Qabc(a, b, c)
    return xp.sqrt(12746.0 / (q ** 4 + 10 * (q ** 3) * (Q0 ** 2)
                              + 135 * (q ** 2) * (Q0 ** 4)
                              + 1575 * q * (Q0 ** 6) + 11025 * (Q0 ** 8)))


def B5(a, b, c):
    r"""l=5 —— 论文 Eq.(17) 之后**被注释掉**的那一行（本仓库补上）。

    论文原文：
        %B_5(Q_{abc})\!&=&\! \\sqrt{998881\\over
        %Q_{abc}^{10}+15Q_{abc}^8Q_0^2+
        %315Q_{abc}^6Q^4_0+6300Q_{abc}^4Q^6_0+99225Q_{abc}^2Q^8_0+893025Q_0^{10}}

    系数 `[1, 15, 315, 6300, 99225, 893025]` 与 tf-pwa
    `breit_wigner.py::Bprime_polynomial` 的 `L=5` 行逐项一致（互为独立来源）。
    """
    q = Qabc(a, b, c)
    return xp.sqrt(998881.0 / (q ** 5 + 15 * (q ** 4) * (Q0 ** 2)
                               + 315 * (q ** 3) * (Q0 ** 4)
                               + 6300 * (q ** 2) * (Q0 ** 6)
                               + 99225 * q * (Q0 ** 8) + 893025 * (Q0 ** 10)))


# l=6 的系数由反向 Bessel 多项式现算（与 `kernels/omega.py::_bprime_coeff`
# 同源，都是 tf-pwa `get_bprime_coeff` 的口径）。写死是为了避免在这个内核里
# 再引入一遍生成逻辑 —— 两处的一致性由 `tests/test_covariant_general.py` 钉住。
_BPRIME_COEFF_6 = [1.0, 21.0, 630.0, 18900.0, 496125.0, 9823275.0, 108056025.0]


def B6(a, b, c):
    r"""l=6（`_BPRIME_COEFF_6` 的 num = P_6(1) = 118394977）。"""
    q = Qabc(a, b, c)
    den = 0.0
    for i, cf in enumerate(_BPRIME_COEFF_6):
        den = den + cf * q ** (6 - i) * (Q0 ** 2) ** i
    return xp.sqrt(sum(_BPRIME_COEFF_6) / den)


# l -> B_l。`t_l` 与 `T_l` 都用它按 l 取。
_BL = {0: lambda a, b, c: xp.ones(_atleast2(a).shape[0]),
       1: B1, 2: B2, 3: B3, 4: B4, 5: B5, 6: B6}


# --------------------------------------------------------------- 节点张量
def T1_cov(a, b, c):
    """Eq.(10)：\tilde{t}^{(1)}_{mu}，返回 (N,4)。"""
    a, b, c = _atleast2(a), _atleast2(b), _atleast2(c)
    gt_cov = Gt_cov(a)
    r = b - c
    return xp.einsum("eij,ej->ei", gt_cov, r) * B1(a, b, c)[:, None]


def T2_cov(a, b, c):
    """Eq.(11)：\tilde{t}^{(2)}_{mu nu}，返回 (N,4,4)。"""
    a, b, c = _atleast2(a), _atleast2(b), _atleast2(c)
    gt_cov = Gt_cov(a)
    r = b - c
    t_r = xp.einsum("eij,ej->ei", gt_cov, r)
    return (xp.einsum("ei,ej->eij", t_r, t_r)
            - (1.0 / 3) * dot(t_r, t_r)[:, None, None] * gt_cov) * B2(a, b, c)[:, None, None]


def T3_cov(a, b, c):
    """Eq.(12)：\tilde{t}^{(3)}_{mu nu rho}，返回 (N,4,4,4)。"""
    a, b, c = _atleast2(a), _atleast2(b), _atleast2(c)
    gt_cov = Gt_cov(a)
    r = b - c
    t_r = xp.einsum("eij,ej->ei", gt_cov, r)
    gt_r_1 = xp.einsum("eij,ek->eijk", gt_cov, t_r)
    gt_r_2 = xp.einsum("ejk,ei->eijk", gt_cov, t_r)
    gt_r_3 = xp.einsum("eki,ej->eijk", gt_cov, t_r)
    rrr = xp.einsum("ei,ej,ek->eijk", t_r, t_r, t_r)
    return (rrr - (1.0 / 5) * dot(t_r, t_r)[:, None, None, None]
            * (gt_r_1 + gt_r_2 + gt_r_3)) * B3(a, b, c)[:, None, None, None]


def T4_cov(a, b, c):
    """Eq.(23)：\tilde{t}^{(4)}，返回 (N,4,4,4,4)。"""
    a, b, c = _atleast2(a), _atleast2(b), _atleast2(c)
    r = b - c
    p4 = P4(a)
    return xp.einsum("eijklmnop,em,en,eo,ep->eijkl", p4, r, r, r, r) \
        * B4(a, b, c)[:, None, None, None, None]


# ------------------------------------------------- 通用 \tilde t^{(l)}（任意 l）
#
# 论文 Eq.(124-136) 把 \tilde t^{(l)} 写成"r~ 的 l 重外积 + 无迹化修正"，
# 只列到 l=3；Eq.(258) 给出等价的一般形式：
#
#     \tilde t^{(L)}_{\mu_1...\mu_L}
#         = (-1)^L P^{(L)}_{\mu_1...\mu_L \mu'_1...\mu'_L} r^{\mu'_1}...r^{\mu'_L} B_L
#
# 而 `P^(L)` 的显式展开（Eq.208-253）本身就是"对指标配对求和"的结构。
# 把 P^(L) 作用到 r 的外积上，可以化简成下面这个**闭式**（无需显式构造
# 4^(2L) 的投影算子）：
#
#     t_{i_1...i_L} = sum_{k=0}^{floor(L/2)} (-1)^k (2L-2k-1)!! / (2L-1)!!
#                     * sum_{|S|=2k} (r.r)^k * prod_{pairs (u,v) in S} g~_{i_u i_v}
#                                     * prod_{j not in S} r~_{i_j}
#
# 其中 `g~ = Gt_cov(a)`、`r~ = g~ . (b-c)`、`r.r = r~_mu r~_nu G^{mu nu}`。
#
# 判据（见 `tests/test_covariant_general.py`）：
#   1. l=1..4 必须与写死的 `T1_cov..T4_cov` **逐位一致**（rel ~ 1e-16）；
#   2. 对任意一对指标缩并度规必须为 0（无迹）；
#   3. 同一指标的全对称（交换任意两个指标不变）。
#
# 这样 l>=5 也能算（原来 `T5_cov` 根本不存在），ρ₅ / K*₅ 才有协变表达式。
_TL_LET = "ijklmnopqrstuvwxyz"


def _double_factorial(n):
    """n!! —— 只用于**构造**（Python 整数，与事例无关）。"""
    r = 1
    while n > 0:
        r *= n
        n -= 2
    return r


def _matchings(items):
    """把 `items` 分成无序对的全部方式（递归；用于构造，不在热路径）。"""
    if not items:
        yield []
        return
    first = items[0]
    for i in range(1, len(items)):
        rest = items[1:i] + items[i + 1:]
        for sub in _matchings(rest):
            yield [(first, items[i])] + sub


def _t_terms(l):
    """`t_l` 的项表：[(系数, 配对数 k, 配对方案, 自由指标), ...]。

    与事例无关，可缓存；`_t_l_impl` 每次只做 einsum。
    """
    l = int(l)
    terms = []
    for k in range(l // 2 + 1):
        coef = (-1.0) ** k * _double_factorial(2 * l - 2 * k - 1) \
            / float(_double_factorial(2 * l - 1))
        for pairset in _combinations(range(l), 2 * k):
            free = tuple(i for i in range(l) if i not in pairset)
            for pairs in _matchings(list(pairset)):
                terms.append((coef, k, tuple(pairs), free))
    return terms


def _combinations(seq, r):
    return itertools.combinations(seq, r)


_T_TERMS_CACHE = {}


def _t_l_impl(l, a, b, c):
    a, b, c = _atleast2(a), _atleast2(b), _atleast2(c)
    l = int(l)
    gt = Gt_cov(a)
    tr = xp.einsum("eij,ej->ei", gt, b - c)
    rr = dot(tr, tr)                       # r~ . r~ （用度规）
    terms = _T_TERMS_CACHE.get(l)
    if terms is None:
        terms = _T_TERMS_CACHE[l] = _t_terms(l)

    out_sub = "e" + _TL_LET[:l]
    acc = 0.0
    for coef, k, pairs, free in terms:
        lhs = ["e%s%s" % (_TL_LET[u], _TL_LET[v]) for (u, v) in pairs]
        lhs += ["e%s" % _TL_LET[i] for i in free]
        ops = [gt] * len(pairs) + [tr] * len(free)
        if ops:
            t = xp.einsum(",".join(lhs) + "->" + out_sub, *ops)
        else:
            t = xp.ones((a.shape[0],))
        if k:
            t = t * rr.reshape((a.shape[0],) + (1,) * l) ** k
        acc = acc + coef * t
    return acc


def t_l(l, a, b, c):
    """通用 `\tilde t^{(l)}`（论文 Eq.258 / Eq.124-136），返回 `(N,4,...,4)`。

    对 l=1..4 与写死的 `T1_cov..T4_cov` 逐位一致；l=5 起是**新能力**
    （论文 Eq.158 之后把 B_5 注释掉了，本仓库补上 `B5`）。

    ⚠️ `l` 只受"构造项数"限制（l=6 有 76 项），不受原来"只到 4"的限制。
    """
    return _t_l_impl(l, a, b, c) * _BL[int(l)](a, b, c).reshape(
        (_atleast2(a).shape[0],) + (1,) * int(l))


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

    ⚠️ 在 jit 追踪里**不做缓存**：追踪期的 `a` 是 tracer，`tobytes()` /
    `array_equal()` 都拿不到具体值（前者直接报错），而且每个 batch 的 tracer
    身份都不同，缓存只会污染。返回 `None` 表示"这次不走缓存"。
    """
    if _B.is_tracer(a):
        return None, a
    arr = onp.asarray(_B.to_numpy(a), dtype=onp.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[0] > 1:
        first = arr[0]
        if onp.array_equal(arr, onp.broadcast_to(first, arr.shape)):
            arr = arr[:1]
    return (arr.shape, arr.tobytes()), arr


def _proj_cached(name, a, build):
    """对投影算子做记忆化；a 变化（如换 psi 方向）时自动失效。

    追踪中（jit）不缓存，直接把 host 常量交给 build —— XLA 会在编译期把
    它折成常量，等价于原版 `Tensor/FUNC.py` 的编译期折叠。
    """
    key, arr = _proj_key(a)
    if key is None:
        return build(arr)
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


# 缓存里存的是后端相关的数组，切后端时必须清掉。
_B.register_cache_clearer(clear_projection_cache)


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
    g_g_1 = xp.einsum("eij,ekl->eikjl", gt_cov, gt_cov)
    g_g_2 = xp.einsum("eil,ekj->eikjl", gt_cov, gt_cov)
    g_g_3 = xp.einsum("eik,ejl->eikjl", gt_cov, gt_cov)
    return 1.0 / 2 * (g_g_1 + g_g_2) - 1.0 / 3 * g_g_3


def _P3_impl(a):
    a = _atleast2(a)
    gt_cov = Gt_cov(a)
    g = gt_cov
    # 6 terms
    get_ggg = xp.einsum("eij,ekl,emn->eikmjln", g, g, g)
    get_ggg = get_ggg + xp.einsum("eij,ekn,eml->eikmjln", g, g, g)
    get_ggg = get_ggg + xp.einsum("eil,ekj,emn->eikmjln", g, g, g)
    get_ggg = get_ggg + xp.einsum("eil,ekn,emj->eikmjln", g, g, g)
    get_ggg = get_ggg + xp.einsum("ein,ekl,emj->eikmjln", g, g, g)
    get_ggg = get_ggg + xp.einsum("ein,ekj,eml->eikmjln", g, g, g)
    # 9 terms
    get_ggg_ = xp.einsum("eik,ejl,emn->eikmjln", g, g, g)
    get_ggg_ = get_ggg_ + xp.einsum("eik,eln,emj->eikmjln", g, g, g)
    get_ggg_ = get_ggg_ + xp.einsum("eik,ejn,eml->eikmjln", g, g, g)
    get_ggg_ = get_ggg_ + xp.einsum("eim,ejn,ekl->eikmjln", g, g, g)
    get_ggg_ = get_ggg_ + xp.einsum("eim,ejl,ekn->eikmjln", g, g, g)
    get_ggg_ = get_ggg_ + xp.einsum("eim,eln,ekj->eikmjln", g, g, g)
    get_ggg_ = get_ggg_ + xp.einsum("ekm,eln,eij->eikmjln", g, g, g)
    get_ggg_ = get_ggg_ + xp.einsum("ekm,ejl,ein->eikmjln", g, g, g)
    get_ggg_ = get_ggg_ + xp.einsum("ekm,ejn,eil->eikmjln", g, g, g)
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

    get_gggg = xp.zeros((n, 4, 4, 4, 4, 4, 4, 4, 4))
    for p in perms24:
        get_gggg = get_gggg + xp.einsum(_sub(p, "emnopijkl"), g, g, g, g)

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
    get_g_g_g_g = xp.zeros_like(get_gggg)
    for p in perms72:
        get_g_g_g_g = get_g_g_g_g + xp.einsum(_sub(p, "emnopijkl"), g, g, g, g)

    # 9 terms
    perms9 = [
        "ij,kl,mn,op", "ij,kl,mo,np", "ij,kl,mp,no",
        "ik,jl,mn,op", "ik,jl,mo,np", "ik,jl,mp,no",
        "il,jk,mn,op", "il,jk,mo,np", "il,jk,mp,no",
    ]
    get_gggg_ = xp.zeros_like(get_gggg)
    for p in perms9:
        get_gggg_ = get_gggg_ + xp.einsum(_sub(p, "emnopijkl"), g, g, g, g)

    return 1.0 / 24 * get_gggg - 1.0 / 84 * get_g_g_g_g + 1.0 / 105 * get_gggg_


# ------------------------------------------------------------------ 七条道
def _s1_vector(psi, phi, f2, Kp, Km, Pip, Pim):
    """Eq.(41) 修正版：纯 S=1 自旋矢量（见 FUNC.py::_s1_vector 的推导注释）。"""
    t2_cov = T2_cov(f2, Pip, Pim)
    t2_con = _raise_last(t2_cov, 2)
    t1_cov = T1_cov(phi, Kp, Km)
    t1_til = xp.einsum("eij,ej->ei", Gt_con(psi), t1_cov)
    W = xp.einsum("eij,ej->ei", t2_con, t1_til)
    sig = xp.einsum("eij,ei,ej->e", t2_con, psi, psi)
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
    return xp.einsum("eij,ej->ei", t2_con, t1_cov)


def _phif201(psi, phi, f2, Kp, Km, Pip, Pim):
    r"""Eq.(41) 修正版：<\phi f_2|01>。"""
    return _s1_vector(psi, phi, f2, Kp, Km, Pip, Pim)


def _phif221(psi, phi, f2, Kp, Km, Pip, Pim):
    r"""Eq.(42) 修正版：<\phi f_2|21>。"""
    W = _s1_vector(psi, phi, f2, Kp, Km, Pip, Pim)
    t2_cov = T2_cov(psi, phi, f2)
    t2_con = _raise_last(t2_cov, 2)
    return xp.einsum("eij,ej->ei", t2_con, W)


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
    return xp.einsum("eijklmn,ejk,elm,en->ei", p3_con, t2_phif2, t2_34, t1_12)


def _phif243(psi, phi, f2, Kp, Km, Pip, Pim):
    r"""Eq.(45)：<\phi f_2|43>。"""
    t4_cov = T4_cov(psi, phi, f2)
    t4_con = _raise_last(t4_cov, 4)
    t2_cov = T2_cov(f2, Pip, Pim)
    t1_cov = T1_cov(phi, Kp, Km)
    return xp.einsum("eijkl,ej,ekl->ei", t4_con, t1_cov, t2_cov)


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
        psi = xp.broadcast_to(PSI2S_DEFAULT, Kp.shape)
    return channel_fn(name)(psi, phi, f2, Kp, Km, Pip, Pim)
