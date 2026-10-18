#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""螺旋度级联内核：**逐节点**振幅 + **链级 einsum**，任意深度、任意逐节点 (l,s)。

为什么需要它
------------
`kernels/omega.py` 的装配式是

    W_i(e, lam_A) = g_i K_i(e) sum_{a,b} M_i[lam_A, a, b] Omega[e, lam_A, a, b]

其中 `M_i = cg_matrix(ls_list, ...)` 只由**顶层节点**的 (L,S) 决定，非顶层节点的
角动量只以标量位垒因子 `K_i` 的形式出现。后果是：两条道只要顶层 (L,S) 相同就
**逐位重合**。omega 可用的充要条件是

    所有非顶层节点都恰好只有 1 个允许的 (l,s)

`documentation/helicity_amplitude_roadmap.md` §2.3。b1 链（`psi->b1 pi`,
`b1->phi pi` 两个节点都允许 (0,1) 与 (2,1)）是第一条违反它的链：4 个独立振幅
被压成 2 条道。

本模块给出 tf-pwa `HelicityDecay` + `DecayChain.get_amp` 的等价实现（§3.1/§3.2）：

    A^(n)[lam_a, lam_b, lam_c]
        = D^{J_a*}_{lam_a, lam_b-lam_c}(alpha_n, beta_n, gamma_n)
          f^{J_b}_{|lam_b|}(gamma_b) f^{J_c}_{|lam_c|}(gamma_c)
          sum_{(l,s) in L_n} g_ls H^(n)_{(ls)}[lam_b, lam_c] q_n^l B_l(q_n)

    A_chain[lam_top] = sum_{内线 + 末态 lam} prod_n A^(n)[lam_n, lam_d1, lam_d2]

`H^(n)_{(ls)}` 就是 `omega.cg_matrix`（已与 tf-pwa `get_cg_matrix` 逐元素核对），
`D^{J*}` 就是 `omega.D_lambda`。
缺的只有三步：**不把子粒子螺旋度压成 0**（omega 的 `_node_decay_d` 写死
`[0],[0]`）、**按树缩并**（omega 写死 `D0*D1*D2`）与**逐节点的螺旋度参考系**
（见下节）。

逐节点螺旋度参考系（本内核与 omega 的第三个、也是最后一个差异）
------------------------------------------------------------------
omega 的 `chain_angles` 把**每个**粒子都从**顶层静止系**直接 boost，再在那个
系里量角度。对"子节点不再衰变"的链这是对的；一旦某个节点的子粒子本身还要
衰变（`psi->b1 pi` 的 `b1->phi pi`），就必须**逐层**构造：

> 每个节点 $n$ 的 $(\alpha_n,\beta_n,\gamma_n)$ 在它**母节点**的静止系里量；
> 走到 $n$ 时把 $n$ 的**整棵子树**按 $n$ 的四动量 boost 一次；`bias` 在**每个
> 节点**从 $-\pi$ 重新开始（tf-pwa `HelicityDecay` 的约定）。

omega 用 `chain_angles` 时 b₁ 的 4 条独立振幅只能对到 $\sim10^{-1}$；
换成 `node_angles` 后逐位到机器精度（§7 脚本 2 的判据 A1）。
$\phi f_2$ 那条链的 `phi` 是**末态**（不衰变），两种构造逐位相同 ——
所以这个修正在现有 13 份 spec 上是**无操作**的，只在 b₁ 这类链上生效。

与 omega 的两点差异
-------------------
1. **Chung Eq.52 的协变项 `f^J_{|m|}(gamma)` 不能省**（§4.4）。omega 只对顶层
   节点的两个子粒子乘它；本内核**逐节点**对每个子粒子乘（末态自旋 0 的
   `f^0_0 == 1`，所以 φf₀/φf₂ 的内层节点不受影响，而 `b1 -> phi pi` 的 φ 受影响）。
2. **不经过 EPS_A**。omega 与本内核都直接在螺旋度表象下构造角向对象；EPS_A 是
   *协变* 路线（`kernels/helicity.py`）用来把洛伦兹指标 μ 旋到 λ 的。判据 A1 就是
   `W_cascade[e, lam] == sum_mu EPS_A[lam, mu] T_covariant[e, mu]`（差一个整体常数）。

落盘口径
--------
输出 `(N, 3)` complex128，末轴行序与 cache 的 `EPS_A` 行序
`(lam = +1, 0, -1)` **逐位一致**；逐道堆成 `(n_ch, N, 3)` complex128 ——
**与今天的 helicity cache 逐位同格式**，拟合侧 `_load_cached_tensor`
零改动（§4.5）。

约定
----
四动量一律 `(E, x, y, z)`、度规 `diag(1,-1,-1,-1)`（与 `omega.py` 一致）。
`chain` 是 `spec.to_chain` 的输出，可以带逐节点的 `daughters` 覆盖（镜像树）。
"""
from __future__ import annotations

import string

import numpy as onp

try:                      # 包内（正常路径）
    from . import _backend as _B
    from . import omega as _RT
except ImportError:       # 被 emitter 拷成产物目录里的顶层模块后
    import _backend as _B
    import runtime as _RT

xp = _B.xp

_LETTERS = string.ascii_lowercase + string.ascii_uppercase


def _einsum(subscripts, *operands):
    return _B.einsum(subscripts, *operands)


# ------------------------------------------------------------------ 链工具
def tree_order(chain):
    """从顶层深度优先走一遍，返回粒子名列表（einsum 字母的稳定分配顺序）。"""
    nodes = chain["nodes"]

    def walk(n, out):
        out.append(n)
        for d in nodes[n].get("daughters", []):
            walk(d, out)

    out = []
    walk(chain["top"], out)
    return out


def decaying_nodes(chain):
    """链里所有会衰变的节点（按 `tree_order` 的顺序）。"""
    nodes = chain["nodes"]
    return [n for n in tree_order(chain) if nodes[n].get("daughters")]


def with_tree(chain, daughters):
    """返回一份 `daughters` 被逐节点覆盖的链副本（镜像树 / 旁观者交换用）。

    `daughters = {"psi2s": ["b1", "Pip"], "b1": ["phi", "Pim"]}`：只覆盖列出的
    节点，其余沿用原树。`index` / `finals` 不变 —— 覆盖的是**树**，不是四动量
    轴顺序。
    """
    if not daughters:
        return chain
    out = dict(chain)
    nodes = {}
    for nm, node in chain["nodes"].items():
        node = dict(node)
        if nm in daughters:
            node["daughters"] = list(daughters[nm])
        nodes[nm] = node
    out["nodes"] = nodes
    return out


# ------------------------------------------------------------- 逐节点参考系
def subtree(chain, node):
    """`node` 及其全部后代（含自己），深度优先。"""
    nodes = chain["nodes"]
    out = [node]
    for d in nodes[node].get("daughters", []):
        out += subtree(chain, d)
    return out


def node_angles(mom, chain):
    r"""逐节点螺旋度参考系下的欧拉角，返回 `{粒子名: (alpha,beta,gamma)}`。

    与 `omega.chain_angles` 的**唯一**区别：每层都在**母节点静止系**里量角。

    算法（`omega` 的工具函数原样复用）：

    1. 记 `m[n]` 为节点 $n$ 的四动量和（`_node_momentum`）；
    2. 顶层参考系 $z=(0,0,1)$、$x=(1,0,0)$；
    3. 按 `tree_order` 深度优先遍历每个衰变节点 $n$：把 $n$ 的**整棵子树**
       从当前（= $n$ 的母节点）系 boost 到 $n$ 的静止系 —— 于是子粒子的
       3-动量方向就落在"$n$ 的母系"里；对每个子粒子 $j$ 用
       `angle_zx_z_getx` 量出 $(\alpha_j,\beta_j)$，并把 $j$ 的 $z$ 轴设为
       $j$ 的 3-动量方向、$x$ 轴设为返回的 $x_2$；
    4. `bias` 每节点从 $-\pi$ 重新开始，每个子粒子减 $\pi$；
    5. 自旋为 0 的粒子（$J=0$）角分布平凡，$\gamma$ 恒取 0（tf-pwa 同）。

    返回的 $\gamma$ 恒为 0 —— 本形式里 $\gamma$ 只通过
    `covariant_hel_term` 的 $\gamma_d$（boost 因子）进入，不通过 Wigner-D。
    """
    nodes = chain["nodes"]
    top = chain["top"]
    order = tree_order(chain)
    n_ev = mom.shape[0]

    mom_in_parent = {n: _RT._node_momentum(mom, chain, n) for n in order}
    set_x = {top: xp.broadcast_to(xp.asarray([1.0, 0.0, 0.0]), (n_ev, 3))}
    set_z = {top: xp.broadcast_to(xp.asarray([0.0, 0.0, 1.0]), (n_ev, 3))}
    ang = {}
    for n in order:
        ds = nodes[n].get("daughters")
        if not ds:
            continue
        q = mom_in_parent[n]
        for x in subtree(chain, n):
            mom_in_parent[x] = _RT.rest_vector(q, mom_in_parent[x])
        bias = -xp.pi
        for j in ds:
            z2 = mom_in_parent[j][:, 1:4]
            al, be, x2 = _RT.angle_zx_z_getx(set_z[n], set_x[n], z2)
            set_x[j], set_z[j] = x2, z2
            al = (al - bias) % (2 * xp.pi) + bias
            ang[j] = (al, be, xp.zeros_like(al))
            bias = bias - xp.pi
    return ang


# ------------------------------------------------------------------ 逐节点振幅
def node_amplitude(mom, chain, node, ls_list, g=None, covariant=True, ang=None):
    """§4.1 的 `A^(n)[lam_a, lam_b, lam_c]`，形状 `(N, na, nb, nc)` complex。

    Parameters
    ----------
    mom      : (N, n_part, 4) 的 (E,x,y,z) 四动量
    chain    : `spec.to_chain` 的链描述
    node     : 节点名（必须有 daughters；末态节点没有振幅，不要传进来）
    ls_list  : 该节点允许的 (l,s) 列表，如 `[(0,1),(2,1)]`
    g        : 逐 (l,s) 的复耦合，None = 全 1；形状 `(len(ls_list),)`
    covariant: 是否乘 Chung Eq.52 的 `f^J_{|m|}(gamma)`（**生产必须为 True**）
    ang      : 预先算好的 `node_angles(mom, chain)`，None 则现算

    **不压子粒子螺旋度**：`lam_b` / `lam_c` 是完整的 `LAM_SLOTS[J]` 槽位，
    末轴是真正的 (lam_a, lam_b, lam_c) 三个指标（omega 的 `_node_decay_d` 把
    `lam_b=lam_c=[0]` 写死，那是这里被修掉的第二步）。
    """
    nodes = chain["nodes"]
    ds = nodes[node].get("daughters")
    if not ds:
        raise ValueError("节点 %s 不衰变，没有逐节点振幅" % node)
    if ang is None:
        ang = node_angles(mom, chain)

    Ja = int(nodes[node]["J"])
    Jb, Jc = int(nodes[ds[0]]["J"]), int(nodes[ds[1]]["J"])
    lam_a = _RT.LAM_SLOTS[Ja]
    lam_b, lam_c = _RT.LAM_SLOTS[Jb], _RT.LAM_SLOTS[Jc]

    # 1) 角向：D^{J_a*}_{lam_a, lam_b - lam_c}
    A = _RT.D_lambda(ang[ds[0]], Ja, lam_a, lam_b, lam_c)        # (N, na, nb, nc)

    # 2) CG 组合 H^(n)_{(ls)}[lam_b, lam_c]，再乘逐 (l,s) 的位垒因子
    ls_list = [(int(l), int(s)) for (l, s) in ls_list]
    M = _RT.cg_matrix(ls_list, Ja, Jb, Jc, lam_b, lam_c)         # (nls, nb, nc)
    pa = _RT._node_momentum(mom, chain, node)
    pb = _RT._node_momentum(mom, chain, ds[0])
    pc = _RT._node_momentum(mom, chain, ds[1])
    Q = _RT.q2_relative(pa, pb, pc)
    # q^l B_l(q^2)：l = 0 时恒为 1（node_barrier 已处理）
    K = xp.stack([_RT.node_barrier(l, Q) for (l, _s) in ls_list], axis=0)  # (nls, N)
    if g is None:
        g = onp.ones(len(ls_list), dtype=onp.complex128)
    g = xp.asarray(g, dtype=xp.complex128)
    # H[e, b, c] = sum_ls g_ls K[ls, e] M[ls, b, c]
    H = _einsum("l,le,lab->eab", g, K, M)
    A = A * H[:, None, :, :]

    # 3) Chung Eq.52 协变项，**逐节点作用在每个子粒子上**
    if covariant:
        mb, mc = _RT.mass(pb), _RT.mass(pc)
        gb = xp.sqrt(1.0 + Q / mb ** 2)      # gamma_d = sqrt(1 + q^2/m_d^2)
        gc = xp.sqrt(1.0 + Q / mc ** 2)
        A = A * _RT.covariant_hel_term(Jb, lam_b, gb)[:, None, :, None]
        A = A * _RT.covariant_hel_term(Jc, lam_c, gc)[:, None, None, :]
    return A


# ------------------------------------------------------------------ 链级缩并
def chain_amplitude(mom, chain, ls_map, g_map=None, covariant=True, ang=None):
    """§4.2 的链级 einsum，返回 `(N, 2 J_top + 1)` complex。

        A_chain[lam_top] = sum_{内线 + 末态 lam} prod_n A^(n)[lam_n, lam_d1, lam_d2]

    内线粒子的螺旋度字母在两个节点里各出现一次，`einsum` 自动求和 —— 这个循环
    对链的深度**没有任何假设**（深度 3、深度 5 都是同一段代码），这正是 omega
    写死 `D0*D1*D2` 所缺的那一步。

    ls_map : `{节点名: [(l,s), ...]}`，每个衰变节点都要有
    g_map  : `{节点名: 复耦合数组}`，可选
    """
    nodes = chain["nodes"]
    order = tree_order(chain)
    letters = {nm: _LETTERS[i] for i, nm in enumerate(order)}
    # 事例轴：所有逐节点振幅都有这一维，且**不求和**，必须显式写进下标。
    # 用树里没用到的下一个字母，避免与粒子字母撞车。
    batch = _LETTERS[len(order)]
    if ang is None:
        ang = node_angles(mom, chain)

    subs, ops = [], []
    for n in order:
        ds = nodes[n].get("daughters")
        if not ds:
            continue
        if n not in ls_map:
            raise KeyError("链级缩并需要节点 %s 的 (l,s) 列表；缺省值无法猜测" % n)
        A = node_amplitude(mom, chain, n, ls_map[n],
                           g=(g_map or {}).get(n), covariant=covariant, ang=ang)
        subs.append(batch + letters[n] + letters[ds[0]] + letters[ds[1]])
        ops.append(A)
    out = batch + letters[chain["top"]]
    return _einsum(",".join(subs) + "->" + out, *ops)


# ------------------------------------------------------------------ 逐道批量
def cascade_from_momenta(mom, chain, channels, covariant=True):
    """逐道算 `W_i(e, lam_A)`，返回 `(n_ch, N, 2 J_top + 1)` complex128。

    `channels` 是 `spec.cascade_channels(spec)` 的输出：每项
    `{"name", "ls": {节点: [(l,s),...]}, "tree": {节点: [子1,子2]}}`。
    每条道可以有自己的树（b1 的 `_1`/`_2` 就是旁观者 π 交换 —— 两组配对
    严格说是两棵树）。

    末轴行序与 cache 的 `EPS_A` 行序 `(lam=+1, 0, -1)` **逐位一致**，落盘
    不需要翻轴（见 `lambda_order` 的说明）。
    """
    out = []
    for ch in channels:
        c = with_tree(chain, ch.get("tree") or {})
        ang = node_angles(mom, c)
        A = chain_amplitude(mom, c, ch["ls"], covariant=covariant, ang=ang)
        out.append(A)
    return xp.stack(out, axis=0).astype(xp.complex128)


def lambda_order(J):
    """本内核末轴的 lambda 值，`LAM_SLOTS[J]`（即 `-J..J`）。

    **它已经就是 cache 要的行序**：落盘约定是 `EPS_A` 的行序
    `(lam = +1, 0, -1)`（承重约定，见 `kernels/helicity.py` 与
    `4_三条路线/eps_a_provenance.md`），而 `D_lambda(ang, J, lam_a, ...)`
    的第 i 行配 `lam_a[i] = LAM_SLOTS[J][i]`。

    对 J=1 两者都是 `[-1, 0, +1]` 与 `[+1, 0, -1]` —— **看起来相反，其实
    不是一回事**：`EPS_A` 的"第 0 行是 +1"说的是**行号 0 对应 λ=+1**，
    而本内核的行号 0 对应 `LAM_SLOTS[1][0] = -1`。实测两者**逐行 rel < 1e-13
    直接相等，不需要翻轴**（13 条道：phif2 5 条 + b1 8 条）。

    所以这里**不提供** `flip_lambda_axis`。曾经有过一个版本翻了轴，把 A1
    从 1e-13 变成 O(1) —— 那是把"行序约定"和"λ 值列表"混为一谈了。
    """
    return list(_RT.LAM_SLOTS[int(J)])


def channel_names(channels):
    return [c["name"] for c in channels]
