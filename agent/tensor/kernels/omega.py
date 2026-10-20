#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""与 tf-pwa **无关**的螺旋度级联运行时（自带 CG / Wigner-D）。

这个文件是 `specgen` 代码生成器的**运行时**：它按一份链描述（`chain` dict）
算三样东西

    Omega(e, lam_A, lam_b, lam_c)   与 (l,s) 无关的角向对象
    K_i(e)                          逐道节点标量（位垒因子 x q^l）
    M_i                             常量 CG 矩阵

生成器把**本文件的源码原样拷进**产出目录（`runtime.py`），所以产出文件是
自足的：只依赖 `_backend` + numpy + math，不 import tf-pwa，也不 import
specgen。数组命名空间走 `_backend.xp`：jax 优先、无 jax 回退 numpy，逐事例
函数在 jax 下可被 `jit` 追踪。

链描述（`chain`）约定
---------------------
    chain = {
      "top": "psi2s",
      "nodes": {
        "psi2s": {"J": 1, "daughters": ["phi", "f2"],  "mass": 3.686},
        "phi":   {"J": 1, "daughters": ["Kp", "Km"],   "mass": 1.019, "l": 1},
        "f2":    {"J": 2, "daughters": ["Pip", "Pim"], "mass": 1.275, "l": 2},
        "Kp":    {"J": 0, "mass": 0.493677},
        ...
      },
      "finals": ["Kp", "Km", "Pip", "Pim"],   # 四动量轴顺序
      "index":  {"Kp": 0, "Km": 1, "Pip": 2, "Pim": 3},
    }

`l` 是**该节点子衰变**的轨道角动量（顶层节点的 l 逐道不同，写在道表里）。
四动量内部一律 (E, x, y, z)，度规 diag(1,-1,-1,-1)。
"""
import math

import numpy as onp

try:                      # 包内（正常路径）
    from . import _backend as _B
except ImportError:       # 被 emitter 拷成产物目录里的顶层 runtime.py
    import _backend as _B

xp = _B.xp

Q0 = 0.197321                       # BASE.py 的 Q0（R = 1）
# 度规是**与事例无关的常数**，用 host numpy 建：jax 数组不支持逐项赋值，
# 也没有理由把常数放到设备上（XLA 会在编译期折叠它）。
G_METRIC = onp.diag([1.0, -1.0, -1.0, -1.0])   # (E,x,y,z)
LAM_SLOTS = {0: [0], 1: [-1, 0, 1], 2: [-2, -1, 0, 1, 2],
             3: [-3, -2, -1, 0, 1, 2, 3], 4: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
             5: list(range(-5, 6)), 6: list(range(-6, 7))}

# Blatt-Weisskopf 分母多项式的系数（**降幂**），与 tf-pwa
# `breit_wigner.py::Bprime_polynomial` 逐项一致。`num` 是该行系数之和
# （= P_l(1)），也就是 `Bl(l,q) = sqrt(num / P_l(q^2/Q0^2))`。
#
# l <= 4 的 num 与仓库原有的 `BASE.py::B1..B4` 相同（1, 2, 13, 277, 12746）；
# l = 5 是新增的，系数取自 tf-pwa 写死的 L=5 行，并已用
# `|theta_l(iw)|^2`（反向 Bessel 多项式）独立复算过 —— 见
# `analyses/tensor_examples/probes/probe_j5_feasibility.py`。
#
# 超过 l=5 时 `Bl` 会现算系数（`_bprime_coeff`），不再写死。
_BPRIME_COEFF = {
    0: [1.0],
    1: [1.0, 1.0],
    2: [1.0, 3.0, 9.0],
    3: [1.0, 6.0, 45.0, 225.0],
    4: [1.0, 10.0, 135.0, 1575.0, 11025.0],
    5: [1.0, 15.0, 315.0, 6300.0, 99225.0, 893025.0],
}


# ------------------------------------------------------------------ CG / D
def _f(n):
    """n! 的整数参数版本；负参数对应 Racah 求和的边界，返回 0。"""
    if n < -1e-9:
        return 0.0
    return float(math.factorial(int(round(n))))


def cg(j1, m1, j2, m2, j3, m3):
    """<j1 m1; j2 m2 | j3 m3>，Condon-Shortley（与 sympy / tf-pwa 同约定）。"""
    if abs(m1 + m2 - m3) > 1e-9:
        return 0.0
    if j3 < abs(j1 - j2) - 1e-9 or j3 > j1 + j2 + 1e-9:
        return 0.0
    for j, m in ((j1, m1), (j2, m2), (j3, m3)):
        if abs(m) > j + 1e-9:
            return 0.0
    pre = math.sqrt((2 * j3 + 1) * _f(j1 + j2 - j3) * _f(j1 - j2 + j3)
                    * _f(-j1 + j2 + j3) / _f(j1 + j2 + j3 + 1))
    pre *= math.sqrt(_f(j1 + m1) * _f(j1 - m1) * _f(j2 + m2)
                     * _f(j2 - m2) * _f(j3 + m3) * _f(j3 - m3))
    s = 0.0
    for k in range(0, int(round(j1 + j2 - j3)) + 1):
        d = (_f(k) * _f(j1 + j2 - j3 - k) * _f(j1 - m1 - k)
             * _f(j2 + m2 - k) * _f(j3 - j2 + m1 + k) * _f(j3 - j1 - m2 + k))
        if d == 0.0:
            continue
        s += (-1.0) ** k / d
    return pre * s


def cg_matrix(ls_list, ja, jb, jc, lam_b, lam_c):
    """tf-pwa `HelicityDecay._get_cg_matrix` 的等价式（Chung Eq.12）。

        M_i[lam_A, lam_b, lam_c] =
            sqrt((2l+1)/(2ja+1)) * <jb lam_b; jc -lam_c | s, lam_b-lam_c>
                                 * <l 0; s lam_b-lam_c | ja, lam_b-lam_c>
    """
    m = len(ls_list)
    # 常量表：host numpy 建（逐项赋值），jax 侧当常数参与运算。
    ret = onp.zeros((m, len(lam_b), len(lam_c)))
    for i, (l, s) in enumerate(ls_list):
        for i1, lb in enumerate(lam_b):
            for i2, lc in enumerate(lam_c):
                ret[i, i1, i2] = (math.sqrt(2 * l + 1) / math.sqrt(2 * ja + 1)
                                  * cg(jb, lb, jc, -lc, s, lb - lc)
                                  * cg(l, 0, s, lb - lc, ja, lb - lc))
    return ret


def small_d_weight(j):
    """tf-pwa `dfun.small_d_weight`；这里的 j 是 **2J**。常量表，host numpy 建。"""
    ret = onp.zeros((j + 1, j + 1, j + 1))

    def f(x):
        return math.factorial(x >> 1)

    for m in range(-j, j + 1, 2):
        for n in range(-j, j + 1, 2):
            for k in range(max(0, n - m), min(j - m, j + n) + 1, 2):
                l = (2 * k + (m - n)) // 2
                sign = (-1) ** ((k + m - n) // 2)
                tmp = sign * math.sqrt(1.0 * f(j + m) * f(j - m) * f(j + n) * f(j - n))
                tmp /= f(j - m - k) * f(j + n - k) * f(k + m - n) * f(k)
                ret[l][(m + j) // 2][(n + j) // 2] = tmp
    return ret


def small_d_matrix(theta, j):
    a = xp.arange(0, j + 1)[None, :]
    ht = 0.5 * theta
    sc = xp.sin(ht)[:, None] ** a * xp.cos(ht)[:, None] ** (j - a)
    w = small_d_weight(j).reshape(j + 1, (j + 1) * (j + 1))
    return (sc @ w).reshape(-1, j + 1, j + 1)


def D_conj(alpha, beta, gamma, J):
    """D^{J*}_{m1,m2}(alpha,beta,gamma)，行 m1 = -J..J。"""
    m = xp.arange(-J, J + 1)
    d = small_d_matrix(beta, 2 * J)
    ea = xp.exp(1j * m[None, :] * alpha[:, None])
    eg = xp.exp(1j * m[None, :] * gamma[:, None])
    return ea[:, :, None] * d * eg[:, None, :]


def D_lambda(ang, J, la, lb, lc):
    """取 D^{J*}_{la, lb-lc}；la/lb/lc 是螺旋度列表，|lb-lc|>J 的元素为 0。

    逐事例函数，**不能用逐项赋值**（jax 数组不可变，jit 里更直接报错）。
    这里用两次 `take`（行、列各一次）做向量化 gather：把要取的 (行, 列)
    下标预先算成数组，一次性取出来。越界槽位先夹到合法范围、再用 `where`
    掩码置 0，等价于原来"只填满足 |lb-lc| <= J 的槽位"。
    """
    dc = D_conj(ang[0], ang[1], ang[2], J)          # (N, 2J+1, 2J+1)
    rows = onp.asarray([va + J for va in la])        # (na,)
    deltas = onp.asarray([vb - vc for vb in lb for vc in lc])     # (nb*nc,)
    valid = onp.abs(deltas) <= J
    # |delta| > J 的槽位要取 0，但下标会越界；先夹到合法范围再靠掩码清零。
    cols = onp.clip(deltas + J, 0, 2 * J)

    # (N, 2J+1, 2J+1) -> (N, na, nb*nc)：行取 rows、列取 cols
    picked = xp.take(dc, rows, axis=1)                       # (N, na, 2J+1)
    picked = xp.take(picked, cols, axis=2)                   # (N, na, nb*nc)
    picked = xp.where(xp.asarray(valid), picked, 0.0)
    n = dc.shape[0]
    return picked.reshape(n, len(la), len(lb), len(lc))


# ------------------------------------------------------------------ 运动学
def _s(p):
    return xp.einsum("...i,ij,...j->...", p, G_METRIC, p)


def q2_relative(a, b, c):
    """a -> b + c 的 breakup 动量平方（布局 (E,x,y,z)）。"""
    sa = _s(a)
    return (sa + _s(b) - _s(c)) ** 2 / (4.0 * sa) - _s(b)


def mass(p):
    return xp.sqrt(xp.maximum(_s(p), 0.0))


def _bprime_coeff(l):
    """`Bprime_polynomial` 的系数（降幂），l > 5 时现算。

    来源：tf-pwa `get_bprime_coeff` —— 反向 Bessel 多项式
    `theta_l(x)` 满足 `|theta_l(i w)|^2 = sum_i c_i w^(2i)`，
    这些 `c_i` 就是分母多项式的系数。这里用 `fractions.Fraction` 做
    **精确有理数**运算，再转 float，避免多项式乘法累积误差。
    """
    import fractions
    l = int(l)
    if l in _BPRIME_COEFF:
        return _BPRIME_COEFF[l]
    # theta_l(x) = sum_k co[k] * x^(l-k)，co[k] = (l+k)! / ((l-k)! k! 2^k)
    co = [fractions.Fraction(math.factorial(l + k),
                             math.factorial(l - k) * math.factorial(k) * 2 ** k)
          for k in range(l + 1)]
    # 代 x = i w：w^j 的系数（升幂）a_j = co[l-j] * i^j
    a = [complex(co[l - j]) * (1j) ** j for j in range(l + 1)]
    n = len(a)
    prod = [0j] * (2 * n - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(a):
            prod[i + j] += x * y.conjugate()
    # prod 是 w 的升幂系数；`|theta|^2` 只含偶次，取出来就是 z = w^2 的升幂
    asc = [prod[2 * i].real for i in range(l + 1)]
    # 统一成**降幂**（与 `_BPRIME_COEFF` 的表一致，`Bl` 的 Horner 依赖这个约定）
    return [float(c) for c in reversed(asc)]


def Bl(l, q):
    """Blatt-Weisskopf B_l(q)，R = 1。q 是 **q^2**。

    与 tf-pwa `Bprime` 的关系：本式是 `sqrt(P_l(1) / P_l(q^2/Q0^2))`，
    tf-pwa 是 `sqrt(P_l(q0^2 d^2) / P_l(q^2 d^2))`。两者**形状相同**
    （只差一个与 q 无关的常数因子，被 `g_i` 吸收）；`Q0` 与 `d` 的对应
    见 `documentation/tensor_examples_feasibility.md`。

    l 上限由 `_bprime_coeff` 决定 —— l <= 5 查表，更大现算。
    真正会崩的是 `small_d_weight` 的阶乘（J 约 50 以上，见
    `probe_j5_feasibility.py` 的实测），不是这里。
    """
    l = int(l)
    x0 = Q0 ** 2
    coeff = _bprime_coeff(l)
    num = sum(coeff)
    # Horner，从最高次开始；q 是 q^2，所以每项是 q^(l-i) * x0^i
    den = 0.0
    for i, c in enumerate(coeff):
        den = den * q + c * x0 ** i
    return xp.sqrt(num / den)


def node_barrier(l, Q):
    """q^l B_l(q^2) —— 一个节点对 K_i 的贡献。l = 0 时恒为 1。"""
    if l == 0:
        return xp.ones_like(Q)
    return Q ** (l / 2.0) * Bl(l, Q)


# ------------------------------------------------------- Chung Eq.52 协变项
def _cov_a(j, m):
    # Chung Eq.34: a^J(m) = (J+m)!(J-m)!/(2J)!  —— 注意分母是 (2J)! 不是 (2J+1)!
    return _f(j + m) * _f(j - m) / _f(2 * j)


def _cov_b(j, m, m0):
    mp = (j + m - m0) // 2
    mm = (j - m - m0) // 2
    return _f(j) / (_f(mp) * _f(m0) * _f(mm))


def covariant_hel_term(j, spins, gamma):
    """f^J_{|m|}(gamma)（Chung Eq.52），返回 (N, len(spins))。"""
    gamma = xp.asarray(gamma, dtype=xp.float64)
    cols = []
    for s in spins:
        m = abs(int(s))
        a = _cov_a(j, m)
        acc = xp.zeros_like(gamma)
        m0 = j - m
        while m0 >= 0:
            acc = acc + (a * _cov_b(j, m, m0) * 2 ** m0) * gamma ** m0
            m0 -= 2
        cols.append(acc)
    return xp.stack(cols, axis=-1)


# ------------------------------------------------------------------ 级联角
def _unit(v):
    return v / xp.linalg.norm(v, axis=-1, keepdims=True)


def _cross_unit(a, b):
    return _unit(xp.cross(a, b))


def _angle_from(self, x, y):
    return xp.arctan2((self * y).sum(-1), (self * x).sum(-1))


def angle_zx_z_getx(z1, x1, z2):
    """tf-pwa `EulerAngle.angle_zx_z_getx` 的逐行复刻。"""
    u_z1 = _unit(z1)
    u_z2 = _unit(z2)
    u_y1 = _cross_unit(z1, x1)
    u_x1 = _cross_unit(u_y1, z1)
    u_yr = _cross_unit(z1, z2)
    u_xr = _cross_unit(u_yr, z1)
    alpha = _angle_from(u_xr, u_x1, u_y1)
    beta = _angle_from(u_z2, u_z1, u_xr)
    u_x2 = _cross_unit(u_yr, u_z2)
    return alpha, beta, u_x2


def _boost(other, b):
    """把 other 按 3-vector b boost；other/约定 (T,X,Y,Z)。"""
    b2 = (b * b).sum(-1)
    g = 1.0 / xp.sqrt(1.0 - b2)
    bp = (b * other[:, 1:]).sum(-1)
    g2 = xp.where(b2 > 1e-12, (g - 1.0) / xp.where(b2 > 1e-12, b2, 1.0), 0.0)
    T = g * (other[:, 0] + bp)
    v = other[:, 1:] + (g2 * bp + g * other[:, 0])[:, None] * b
    return xp.stack([T, v[:, 0], v[:, 1], v[:, 2]], axis=-1)


def rest_vector(p_rest, p):
    return _boost(p, -p_rest[:, 1:] / p_rest[:, 0:1])


def _node_momentum(mom, chain, name):
    """递归求和得到节点 name 的四动量。"""
    node = chain["nodes"][name]
    ds = node.get("daughters")
    if not ds:
        return mom[:, chain["index"][name]]
    return _node_momentum(mom, chain, ds[0]) + _node_momentum(mom, chain, ds[1])


def chain_angles(mom, chain):
    """返回 {粒子名: (alpha,beta,gamma)}；只用每个节点的**第一个**子粒子角。"""
    top = chain["top"]
    nodes = chain["nodes"]
    A = _node_momentum(mom, chain, top)
    names = list(nodes.keys())
    rpA = {n: rest_vector(A, _node_momentum(mom, chain, n)) for n in names}
    rp = {}
    for n in names:
        for d in nodes[n].get("daughters", []):
            rp[d] = rest_vector(rpA[n], rpA[d])

    one = xp.ones((mom.shape[0], 3))
    set_x = {top: one * xp.array([1.0, 0.0, 0.0])}
    set_z = {top: one * xp.array([0.0, 0.0, 1.0])}
    ang = {}

    def do_node(core, outs):
        # ⚠️ bias 是**每个节点**的局部量，从 -pi 重新开始（tf-pwa 的约定）。
        bias = -xp.pi
        for j, prest in outs:
            z2 = prest[:, 1:4]
            al, be, x2 = angle_zx_z_getx(set_z[core], set_x[core], z2)
            set_x[j], set_z[j] = x2, z2
            al = (al - bias) % (2 * xp.pi) + bias
            ang[j] = (al, be, xp.zeros_like(al))
            bias -= xp.pi

    do_node(top, [(d, rpA[d]) for d in nodes[top]["daughters"]])
    for n in names:
        ds = nodes[n].get("daughters", [])
        if ds and n != top:
            do_node(n, [(d, rp[d]) for d in ds])
    return ang


# ------------------------------------------------------------------ Omega / K
def _node_decay_d(ang, chain, node, n_events):
    """`node -> 两体` 的衰变 D 矩阵贡献，返回 `(N, len(lam_node))`。

    末态节点（没有 daughters）不衰变，因子恒为 **1** —— 这正好等于
    `D^{0*}_{0,0}(...) == 1`，所以 `psi2s -> phi + pi`（`pi` 是末态）这类
    合法的三体拓扑不需要在调用处特判。

    以前这里无条件写 `nodes[node]["daughters"][0]`，于是那种拓扑会在
    `check` 通过之后运行时 `KeyError: 'daughters'`（审计 §2.2）。
    """
    nodes = chain["nodes"]
    ds = nodes[node].get("daughters")
    if not ds:
        return xp.ones((n_events, 1))
    J = nodes[node]["J"]
    return D_lambda(ang[ds[0]], J, LAM_SLOTS[J], [0], [0])[:, :, 0, 0]


def omega_from_momenta(mom, chain, covariant=True):
    """Omega[e, lam_A, lam_b, lam_c] —— 与 (l,s) 无关。

    每个节点贡献一个 D 矩阵；每个**子粒子**贡献一个 f^{J_d}_{|lam_d|}(gamma_d)
    （Chung Eq.52）。顶层节点的两个子粒子是共振态 phi/f2，故有非平凡的协变项；
    子节点的末态粒子自旋为 0，f^0_0 == 1。
    """
    top = chain["top"]
    nodes = chain["nodes"]
    nb, nc = nodes[top]["daughters"]
    Jb, Jc = nodes[nb]["J"], nodes[nc]["J"]
    lam_b, lam_c = LAM_SLOTS[Jb], LAM_SLOTS[Jc]
    JA = nodes[top]["J"]
    lam_A = LAM_SLOTS[JA]
    ang = chain_angles(mom, chain)

    D0 = D_lambda(ang[nb], JA, lam_A, lam_b, lam_c)
    D1 = _node_decay_d(ang, chain, nb, mom.shape[0])
    D2 = _node_decay_d(ang, chain, nc, mom.shape[0])

    if covariant:
        pA = _node_momentum(mom, chain, top)
        pb = _node_momentum(mom, chain, nb)
        pc = _node_momentum(mom, chain, nc)
        q2 = q2_relative(pA, pb, pc)
        fb = covariant_hel_term(Jb, lam_b, xp.sqrt(1.0 + q2 / mass(pb) ** 2))
        fc = covariant_hel_term(Jc, lam_c, xp.sqrt(1.0 + q2 / mass(pc) ** 2))
        D0 = D0 * fb[:, None, :, None] * fc[:, None, None, :]

    return (D0 * D1[:, None, :, None] * D2[:, None, None, :])


def node_factors(mom, chain, ls_list):
    """K_i(e) = prod_节点 q_node^{l_node} B_{l_node}(q_node)。-> (n_ch, N)

    顶层节点的 l 逐道取自 `ls_list`；子节点的 l 由 `chain` 给定（宇称 + 角动量
    唯一确定，见 specgen/physics.py::allowed_child_l）。

    **末态节点不贡献 K**：它不衰变，`l = 0` 时 `q^0 B_0 == 1`。审计 §2.2 的
    拓扑（`psi2s -> phi + pi`）在这里也曾因为无条件读 `nodes[n]["l"]` /
    `["daughters"]` 而崩。
    """
    top = chain["top"]
    nodes = chain["nodes"]
    pA = _node_momentum(mom, chain, top)
    nb, nc = nodes[top]["daughters"]
    common = xp.ones_like(mass(pA))
    for n in (nb, nc):
        ds = nodes[n].get("daughters")
        if not ds:
            continue                     # 末态：q^0 B_0 == 1
        l = int(nodes[n]["l"])
        pn = _node_momentum(mom, chain, n)
        d1, d2 = ds
        Q = q2_relative(pn, _node_momentum(mom, chain, d1),
                        _node_momentum(mom, chain, d2))
        common = common * node_barrier(l, Q)
    Qtop = q2_relative(pA, _node_momentum(mom, chain, nb),
                       _node_momentum(mom, chain, nc))
    # 用 stack 而不是 out[i] = ...：jax 数组不可变（jit 里逐项赋值直接报错）。
    return xp.stack([node_barrier(l, Qtop) * common for (l, _s) in ls_list], axis=0)


def assemble(M, Om, g, K):
    """W_i[e,lam_A] = g_i K_i(e) (M_i . Omega)。Om (N,nA,nb,nc) -> (n_ch,N,nA)"""
    P = xp.einsum("iab,enab->ien", M, Om)
    return g[:, None, None] * K[:, :, None] * P


# ------------------------------------------------------------------ IO
def momenta_from_npz(path, keys, start=0, n=None, order="x,y,z,E"):
    """读 Momentum_kk.npz（默认布局 (x,y,z,E)），返回 (N, n_part, 4) 的 (E,x,y,z)。

    IO 全在 host numpy 上做（切片 + stack），最后一次性搬到设备：jax 下
    `xp.asarray` 会把每个键都变成一次设备传输，逐键搬运比整体搬慢。
    """
    d = onp.load(path)
    sl = slice(start, None if n is None else start + n)
    arrs = [onp.asarray(d[k][sl], dtype=onp.float64) for k in keys]
    mom = onp.stack(arrs, axis=1)
    perm = [order.split(",").index(c) for c in ("E", "x", "y", "z")]
    return _B.to_device(mom[..., perm])
