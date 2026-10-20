#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""链描述（TOML）的读入、校验与 (l,s) 自动枚举。

设计目标：**只写粒子参数与衰变链，不写代码**。`(l,s)` 由宇称守恒 + 角动量三角
自动枚举（也可以用 `[[channel]]` 显式覆盖），四动量轴顺序、质量、自旋全部从
TOML 读。格式见 `specs/*.toml` 与 README。

TOML 结构
---------
    [chain]
    name   = "phif2"            # 产物模块名后缀
    top    = "psi2s"            # 顶层母粒子
    output = "generated"        # 产物目录（相对 spec 文件）
    l_max  = 4                  # (l,s) 自动枚举的 l 上限

    [input]
    keys   = ["Kp", "Km", "Pip", "Pim"]   # npz 里的键，同时是四动量轴顺序
    layout = "x,y,z,E"                    # npz 每个键的布局

    [particles.psi2s]
    J = 1
    mass = 3.686
    parity = -1                 # eta（宇称），可选；给了才能自动枚举

    [particles.phi]
    J = 1
    mass = 1.019
    parity = -1
    daughters = ["Kp", "Km"]    # 树结构写在这里
    l = 1                       # 子衰变轨道角动量；省略则取宇称允许的最小值

    # 顶层节点的 daughters 也要写；其 l 由 [[channel]] / 自动枚举给出
    # 显式道（可选）：给了就完全按这个列表，不再自动枚举
    [[channel]]
    name = "phif201"
    l = 0
    s = 1

    # 可选：把若干道合成一个“族”，产出用同一个后缀（如 f0 族写 Omega0.npy）
    # family = "f0"
"""
from __future__ import annotations

import math
import os

try:                                     # python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:              # pragma: no cover
    import tomli as _toml                # type: ignore


# --------------------------------------------------------------------- 读入
class SpecError(ValueError):
    """TOML 描述不合法（拓扑 / 量子数 / 缺字段）。"""


def load_spec(path):
    """读入一份 TOML 链描述并校验，返回 dict。"""
    with open(path, "rb") as fh:
        raw = _toml.load(fh)
    spec = _normalise(raw, os.path.dirname(os.path.abspath(path)))
    validate(spec)
    return spec


def _as_ls_pair(v, where):
    """把 `[l, s]` / `(l, s)` 规整成 `(int, int)`。"""
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise SpecError("%s 要写成 [l, s]，收到 %r" % (where, v))
    return (int(v[0]), int(v[1]))


def _normalise_ls_map(d):
    """`{节点名 = [[l,s], ...]}` -> `{节点名: [(l,s), ...]}`。

    TOML 里写成

        [[channel]]
        name = "u_DD_1"
        ls = { psi2s = [[2,1]], b1 = [[2,1]], phi = [[1,0]] }

    `ls` 缺省为 `{}` —— 这时 helicity_cascade 表象会退回"顶层用道表的
    (l,s)、非顶层用宇称唯一值"的自动口径（等价于 omega 的可用条件）。
    """
    if not d:
        return {}
    if not isinstance(d, dict):
        raise SpecError("[[channel]].ls 必须是表 {节点名 = [[l,s], ...]}，收到 %r" % (d,))
    out = {}
    for node, lst in d.items():
        if isinstance(lst, (list, tuple)) and len(lst) == 2 \
                and not isinstance(lst[0], (list, tuple)):
            lst = [lst]                       # 允许简写 b1 = [2, 1]
        out[node] = [_as_ls_pair(x, "[[channel]].ls[%s]" % node) for x in lst]
    return out


def _normalise_tree(d):
    """`{节点名 = ["d1", "d2"]}` -> `{节点名: [d1, d2]}`（镜像树覆盖）。"""
    if not d:
        return {}
    if not isinstance(d, dict):
        raise SpecError("[[channel]].tree 必须是表 {节点名 = [d1, d2]}，收到 %r" % (d,))
    out = {}
    for node, ds in d.items():
        if not isinstance(ds, (list, tuple)) or len(ds) != 2:
            raise SpecError("[[channel]].tree[%s] 必须是恰好两个 daughter，收到 %r"
                            % (node, ds))
        out[node] = [str(x) for x in ds]
    return out


def _normalise(raw, base):
    ch = raw.get("chain") or {}
    if not ch.get("top"):
        raise SpecError("缺 [chain].top")
    name = ch.get("name") or ch["top"]
    out = ch.get("output") or "generated"
    out_abs = out if os.path.isabs(out) else os.path.normpath(os.path.join(base, out))

    inp = raw.get("input") or {}
    keys = list(inp.get("keys") or [])
    layout = (inp.get("layout") or "x,y,z,E").replace(" ", "")
    if not keys:
        raise SpecError("缺 [input].keys（npz 里的四动量键，同时决定轴顺序）")
    for c in layout.split(","):
        if c not in ("E", "x", "y", "z"):
            raise SpecError("[input].layout 只能含 E/x/y/z，收到 %r" % layout)

    particles = {}
    for nm, p in (raw.get("particles") or {}).items():
        d = dict(p)
        if "J" not in d:
            raise SpecError("粒子 %s 缺 J" % nm)
        d["J"] = int(d["J"])
        if d.get("daughters") is not None:
            d["daughters"] = list(d["daughters"])
        particles[nm] = d

    # [[channel]] 的扩展字段（全部可选）：
    #   family   把若干道合成一个"族"，产出用同一个后缀（如 f0 族写 Omega0.npy）
    #   note     写进产物注释的自由文本
    #   ls       逐节点 (l,s) 组合 `{节点名 = [[l,s], ...]}`（**只给
    #            helicity_cascade 表象用**；omega 侧仍只看顶层 l/s）
    #   tree     逐节点 daughters 覆盖 `{节点名 = ["d1","d2"]}`（镜像树）
    channels = []
    for c in (raw.get("channel") or []):
        if "name" not in c:
            raise SpecError("[[channel]] 缺 name")
        for req in ("l", "s"):
            if req not in c:
                raise SpecError("[[channel]] %s 缺 %s" % (c["name"], req))
        channels.append({
            "name": c["name"],
            "l": int(c["l"]),
            "s": int(c["s"]),
            "family": c.get("family", ""),
            "note": c.get("note", ""),
            "ls": _normalise_ls_map(c.get("ls")),
            "tree": _normalise_tree(c.get("tree")),
        })

    # [[basis]]：一份 TOML 要产出哪几种表象。省略 = 只出协变张量。
    bases = []
    for b in (raw.get("basis") or []):
        if "name" not in b:
            raise SpecError("[[basis]] 缺 name")
        bases.append({
            "name": b["name"],
            "kernel": b.get("kernel", b["name"]),
            "files": list(b.get("files") or []),
            "env": dict(b.get("env") or {}),
            "suffix_map": dict(b.get("suffix_map") or {}),
        })

    return {
        "name": name,
        "top": ch["top"],
        "l_max": int(ch.get("l_max", 4)),
        "output_dir": out_abs,
        "input_keys": keys,
        "input_layout": layout,
        "particles": particles,
        "channels": channels,
        "bases": bases,
        "raw": raw,
    }


# --------------------------------------------------------------------- 校验
def _children(spec, name):
    return spec["particles"].get(name, {}).get("daughters") or []


def validate(spec):
    """树结构、末态集合、自旋合法性。"""
    parts = spec["particles"]
    top = spec["top"]
    if top not in parts:
        raise SpecError("顶层粒子 %s 不在 [particles] 里" % top)

    # 前置能力边界（审计 §2.3）：超了要在**读 TOML 时**报错，而不是运行时
    # KeyError / 静默算错。边界**按表象分级**（2026-10-20 扩展 J=5/l=5）：
    #
    # | 表象 | 张量来源 | J / l 上限 |
    # |---|---|---|
    # | `helicity_cascade` / `omega` | `kernels/omega.py` 的通用原语 | 6 / 6 |
    # | `cartesian` / `helicity` | `kernels/covariant.py` 的 `t_l` | 6 / 6 |
    #
    # 级联路线的原语（`cg` / `small_d_weight` / `D_lambda` /
    # `covariant_hel_term`）本来就是通用的，`LAM_SLOTS` 与 `Bl` 扩到 6 之后
    # J=5/l=5 就能算（实测置换协变性残差 3e-15，见
    # `analyses/tensor_examples/probes/probe_j5_feasibility.py`）。
    #
    # 协变张量路线 2026-10-20 也放开到 6：`covariant.py` 新增了论文
    # Eq.(258) 的**通用** `t_l(l, a, b, c)`（l=1..4 与写死的 `T1_cov..T4_cov`
    # 逐位一致，l=5/6 是新能力）与 `B5`/`B6`。但**能算 ≠ 能 emit**：
    # 协变路线要把 `chains/<chain>/<channel>.py` 里的**手写**道函数内联进产物，
    # 所以某条链能不能出，取决于有没有人为它写道函数（三体置换族默认没有，
    # 只有 `helicity_cascade` 能出）。这是 emit 期的 LookupError，不是这里的
    # 上限问题 —— 两件事分开报，免得把"没写道函数"误诊成"J 超限"。
    #
    # 上限 6（而不是更大）是因为 `small_d_weight` 的阶乘在 float64 下溢出：
    # d 矩阵正交性 J=20 退到 4e-12、J=40 退到 2e-6、J=50 直接 NaN。
    _bases_all = spec["bases"] or [{"name": "omega", "kernel": "omega"}]
    _kernels = {(b.get("kernel") or b["name"]) for b in _bases_all}
    _tensor_route = bool(_kernels & {"cartesian", "helicity"})
    j_max, l_max_cap = 6, 6
    _why = ("两条路线的原语都已通用：级联走 LAM_SLOTS/Bl，协变走 t_l/B5；"
            "上限 6 是因为 small_d_weight 的阶乘在 float64 下溢出")
    for nm, p in sorted(parts.items()):
        if p["J"] > j_max:
            raise SpecError(
                "粒子 %s 的 J=%d 超出%s的上限 %d（%s）"
                % (nm, p["J"], "本表象" if _tensor_route else "级联内核",
                   j_max, _why))
        if "l" in p and p["daughters"] and int(p["l"]) > l_max_cap:
            raise SpecError(
                "节点 %s 的 l=%d 超出%s的上限 %d（%s）"
                % (nm, int(p["l"]), "本表象" if _tensor_route else "级联内核",
                   l_max_cap, _why))
    if spec["l_max"] > l_max_cap:
        raise SpecError("[chain].l_max=%d 超出上限 %d（%s）"
                        % (spec["l_max"], l_max_cap, _why))

    # 拓扑：从 top 出发走一遍，必须是二叉树且无环
    seen = set()
    order = []

    def walk(n):
        if n in seen:
            raise SpecError("粒子 %s 在链里出现两次（树必须无环）" % n)
        seen.add(n)
        order.append(n)
        ds = _children(spec, n)
        if len(ds) == 0:
            return
        if len(ds) != 2:
            raise SpecError("粒子 %s 有 %d 个 daughter，只支持两体衰变" % (n, len(ds)))
        for d in ds:
            if d not in parts:
                raise SpecError("daughter %s 不在 [particles] 里" % d)
            walk(d)

    walk(top)

    # 级联深度：**按表象分级**（见 documentation/tensor_examples_feasibility.md §4）。
    #
    # `kernels/omega.py` 的 `chain_angles` / `D_lambda` 逐节点读树、只算
    # `D0 * D1 * D2`，所以对 omega 表象来说深度 > 2 会**静默漏掉最内层的角分布**，
    # 必须在这里拒绝。
    #
    # 但协变/螺度路线**完全不读树**：`chains/<chain>/<channel>.py` 是按道名硬编码
    # 的公式（`grep -c "chain\|nodes" kernels/covariant.py` == 0）。所以只要这份
    # spec 不出 omega 表象，三层树（如例 5 的 `psi -> K*_2 pi, K*_2 -> phi pi,
    # phi -> K+K-`）是可以算的，不该被拒。
    #
    # 这与「`J<=4` / `l<=4` 是 omega 专属约束」是同一类问题：能力边界挂在表象上，
    # 不该挂在 spec 上。
    def _depth(n):
        ds = _children(spec, n)
        return 0 if not ds else 1 + max(_depth(d) for d in ds)

    depth = _depth(top)
    spec["depth"] = depth
    _bases = spec["bases"] or [{"name": "omega", "kernel": "omega"}]
    wants_omega = any((b.get("kernel") or b["name"]) == "omega" for b in _bases)
    def _tree_signature(spec_, tree_or_top):
        """树的**拓扑**签名：把每对 `父 -> (子1, 子2)` 排序后元组化。

        传节点名（顶层）时从 `spec_["particles"]` 现算 base 树；传 tree dict
        （来自 `[[channel]].tree`）时直接用它。只比拓扑，不比 `(l,s)` ——
        omega 只对拓扑敏感，`(l,s)` 走的是 `M`/`K`。
        """
        if isinstance(tree_or_top, str):
            def _ds(n):
                return _children(spec_, n)
            tree, stack = {}, [tree_or_top]
            while stack:
                n = stack.pop()
                d = _ds(n)
                if d:
                    tree[n] = tuple(d)
                    stack.extend(d)
        else:
            tree = {k: tuple(v) for k, v in tree_or_top.items() if v}
        return tuple(sorted(tree.items()))

    # 除了深度，omega 还**硬性假定"所有道共用同一棵树"**：
    # `omega.py` 的 `omega_from_momenta(mom, chain)` 只吃 **base chain**
    # （`to_chain(spec)`），返回一份逐道共用的 `Omega`。如果某些道的
    # `tree` 与 base chain 不同（置换族、旁观者交换），那些道就被静默算错。
    #
    # 这个坑比深度那个更危险：**emit 不报错、跑得出数**。实测：
    #
    #   | spec | 与 base 树不同的道 | 那些道的残差 |
    #   |---|---|---|
    #   | `ex1_pipipi0_rho`  | `u_rho_13` `u_rho_23`   | 0.30 / 0.27 |
    #   | `ex1_pipipi0_rho3` | `u_rho3_13` `u_rho3_23` | 1.06 / 1.36 |
    #   | `ex2_kkpi0_kstar`  | `u_kstar_23`            | 0.17 |
    #
    # 注意"顶层两个子节点都衰变"和"深度 <= 2"都**不是**正确判据 ——
    # 前者会把 `ex2_kkpi0_rho`（只有一条道、树与 base 相同、实测正确）误拒，
    # 后者拦不住三体（深度也是 2）。
    # 顺序：先报**深度**（更基本，`D0*D1*D2` 装不下），再报**树的拓扑**
    # （更隐蔽）。两者都成立时（如三层树 + 置换族），深度那条更直接。
    if depth > 2 and wants_omega:
        raise SpecError(
            "链的级联深度是 %d，但这份 spec 出了 omega 表象；omega 内核只算 "
            "D0*D1*D2（顶层 + 一层共振），更深的树会静默漏掉内层角分布。\n"
            "  要么把树压到 2 级，要么去掉 [[basis]] name=\"omega\" "
            "（协变/螺度路线按道名硬编码公式，不读树，深度不限）。"
            % depth)

    if wants_omega:
        _base_sig = _tree_signature(spec, top)
        # 只比**显式给了 tree** 的道：没给 tree 的道由 base 树补全，天然一致
        # （`cascade_channels` 里 `c["tree"]` 是 `{}`，`with_tree` 再补）。
        _bad = [c["name"] for c in spec["channels"]
                if c.get("tree")
                and _tree_signature(spec, c["tree"]) != _base_sig]
        if _bad:
            raise SpecError(
                "这份 spec 出了 omega 表象，但有 %d 条道的树与 base chain 不同"
                "（%s）。omega 只对 base chain 算**一份逐道共用**的 Omega，"
                "这些道会被**静默算错**（实测残差 0.17–1.36，且 emit 不报错）。\n"
                "  置换族 / 旁观者交换请用 helicity_cascade 表象。"
                % (len(_bad), "、".join(_bad)))

    leaves = [n for n in order if not _children(spec, n)]
    keys = spec["input_keys"]
    if sorted(leaves) != sorted(keys):
        raise SpecError("末态粒子 %s 与 [input].keys %s 不一致" % (leaves, keys))
    spec["leaves"] = leaves
    spec["index"] = {k: i for i, k in enumerate(keys)}

    # 非顶层内部节点必须有 l（或可由宇称推出）
    for n in order:
        ds = _children(spec, n)
        if len(ds) == 2 and n != top:
            if "l" not in parts[n]:
                parts[n]["l"] = minimal_l(spec, n)
        if len(ds) == 2:
            # a -> b + c: J_a = l + s，s 由 s_b + s_c 合成。
            # 所以需要存在 s ∈ [|s_b-s_c|, s_b+s_c] 与 l 使 |l-s| <= J_a <= l+s。
            ja = parts[n]["J"]
            sb, sc = parts[ds[0]]["J"], parts[ds[1]]["J"]
            l_here = parts[n].get("l")          # 顶层为 None（逐道不同）
            l_lo = 0 if l_here is None else l_here
            l_hi = spec["l_max"] if l_here is None else l_here
            ok = False
            for s in range(abs(sb - sc), sb + sc + 1):
                for l in range(l_lo, l_hi + 1):
                    if abs(l - s) <= ja <= l + s:
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                raise SpecError(
                    "节点 %s: J=%d 无法由 (l∈[%d,%d]) + (s∈[%d,%d]) 组成"
                    % (n, ja, l_lo, l_hi, abs(sb - sc), sb + sc))

    # 显式道表 vs 自动枚举：**交叉检查**（这是最有价值的一条校验）
    #
    # 以前 `resolve_channels` 是 `if 显式: 用它 else: 枚举`，两条路从不比对。
    # 后果是「漏一条道」「多一条宇称禁止的道」「(l,s) 标错」都会**静默通过**，
    # 而道表恰恰是 tensor cache 的行序 —— 错了不会报错，只会静默错位。
    #
    # 现在：只要 parity 齐全，就要求显式道表与枚举结果**逐项一致**（顺序可不同）。
    # 想故意只用子集（例如先只做 f0 族的两条），把 [chain].check_channels 设为
    # false 显式关掉，或写成注释说明。
    if spec["channels"] and spec["raw"].get("chain", {}).get("check_channels", True):
        try:
            expected = allowed_ls(spec)
        except SpecError:
            expected = None            # parity 不全，无法交叉检查，跳过
        if expected is not None:
            got = [(c["l"], c["s"]) for c in spec["channels"]]
            missing = [ls for ls in expected if ls not in got]
            extra = [ls for ls in got if ls not in expected]
            if missing or extra:
                raise SpecError(
                    "显式 [[channel]] 与宇称+三角自动枚举不一致：\n"
                    "  应有（%d 条）：%s\n"
                    "  实给（%d 条）：%s\n"
                    "  %s%s%s\n"
                    "道表决定 tensor cache 的行序，静默不一致会导致错位。"
                    "若确实只要子集，设 [chain].check_channels = false。"
                    % (len(expected), expected, len(got), got,
                       "缺 %s；" % missing if missing else "",
                       "多 %s；" % extra if extra else "",
                       "（注意：多出来的 (l,s) 可能被宇称/三角禁止）" if extra else ""))
    _validate_cascade_fields(spec)
    return spec


def _validate_cascade_fields(spec):
    """校验 `[[channel]].ls` / `.tree`（螺旋度级联表象用）。

    这两项是 §5-P0-1 新增的：`ls` 给逐节点 (l,s)，`tree` 给镜像树。它们只被
    `helicity_cascade` 表象读，但**必须在读 TOML 时报错**而不是运行时 KeyError。
    """
    parts = spec["particles"]
    decaying = set(decaying_nodes(spec))
    leaves = set(spec["leaves"])
    for c in spec["channels"]:
        for node, lst in (c.get("ls") or {}).items():
            if node not in parts:
                raise SpecError("[[channel]] %s 的 ls 里有未知节点 %r" % (c["name"], node))
            if node not in decaying:
                raise SpecError("[[channel]] %s 的 ls 给节点 %s 指定了 (l,s)，"
                                "但它没有 daughter（末态节点没有逐节点振幅）"
                                % (c["name"], node))
            if not lst:
                raise SpecError("[[channel]] %s 的 ls[%s] 是空列表" % (c["name"], node))
            # 宇称 + 三角必须允许（parity 齐全时才能查）
            try:
                ok = set(allowed_ls(spec, node))
            except SpecError:
                ok = None
            if ok is not None:
                bad = [x for x in lst if x not in ok]
                if bad:
                    raise SpecError(
                        "[[channel]] %s 给节点 %s 指定了宇称/三角不允许的 (l,s)=%s；"
                        "该节点允许的是 %s" % (c["name"], node, bad, sorted(ok)))
        for node, ds in (c.get("tree") or {}).items():
            if node not in decaying:
                raise SpecError("[[channel]] %s 的 tree 里有非衰变节点 %r"
                                % (c["name"], node))
            for d in ds:
                if d not in parts:
                    raise SpecError("[[channel]] %s 的 tree[%s] 里有未知粒子 %r"
                                    % (c["name"], node, d))
        if c.get("tree"):
            _check_mirror_tree(spec, c)


def _check_mirror_tree(spec, c):
    """镜像树必须**只是换配对**：整棵树的末态集合逐项不变、无环、两体。

    注意不能只比"某个节点的 daughter 集合" —— 顶层
    `psi -> [b1, Pim]` 换成 `psi -> [b1, Pip]` 时顶层 daughter 集合**确实变了**
    （Pim -> Pip），但 `b1` 那边同时从 `[phi, Pip]` 换成 `[phi, Pim]`，
    整棵树的末态集合 {Kp, Km, Pip, Pim} 不变。这才是正确的不变量。
    """
    overrides = c["tree"]
    base = {n: list(_children(spec, n)) for n in spec["particles"]}
    seen, leaves = set(), []

    def walk(n):
        if n in seen:
            raise SpecError("[[channel]] %s 的 tree 覆盖后有环：%s 出现两次"
                            % (c["name"], n))
        seen.add(n)
        ds = overrides.get(n, base.get(n, []))
        if len(ds) == 0:
            leaves.append(n)
            return
        if len(ds) != 2:
            raise SpecError("[[channel]] %s 的 tree 里节点 %s 有 %d 个 daughter，"
                            "只支持两体衰变" % (c["name"], n, len(ds)))
        for d in ds:
            walk(d)

    walk(spec["top"])
    if sorted(leaves) != sorted(spec["leaves"]):
        raise SpecError(
            "[[channel]] %s 的 tree 覆盖后末态集合是 %s，与 [input].keys %s 不一致；"
            "镜像树只能交换配对，不能改变末态粒子集合"
            % (c["name"], sorted(leaves), sorted(spec["leaves"])))


def _parity(spec, name):
    return spec["particles"][name].get("parity")


def minimal_l(spec, node):
    """宇称允许的**最小**轨道角动量：eta_a = eta_b eta_c (-1)^l。"""
    ds = _children(spec, node)
    ea = _parity(spec, node)
    eb, ec = _parity(spec, ds[0]), _parity(spec, ds[1])
    if None in (ea, eb, ec):
        raise SpecError("节点 %s 缺 parity，无法自动定 l；请在 TOML 里显式写 l"
                        % node)
    want = ea * eb * ec            # = (-1)^l
    for l in range(0, spec["l_max"] + 1):
        if abs(((-1) ** l) - want) < 1e-9:
            return l
    raise SpecError("节点 %s 找不到宇称允许的 l <= %d" % (node, spec["l_max"]))


# ------------------------------------------------------- (l,s) 自动枚举
def allowed_ls(spec, node=None):
    """宇称 + 角动量三角允许的全部 (l,s)，按 l 升序、s 升序。"""
    node = node or spec["top"]
    ds = _children(spec, node)
    ja = spec["particles"][node]["J"]
    jb, jc = spec["particles"][ds[0]]["J"], spec["particles"][ds[1]]["J"]
    ea = _parity(spec, node)
    eb, ec = _parity(spec, ds[0]), _parity(spec, ds[1])
    if None in (ea, eb, ec):
        raise SpecError("自动枚举 (l,s) 需要节点 %s 及其 daughter 的 parity；"
                        "请在 [[channel]] 里显式列出道" % node)
    want = ea * eb * ec            # = (-1)^l
    out = []
    for l in range(0, spec["l_max"] + 1):
        if abs(((-1) ** l) - want) > 1e-9:
            continue
        for s in range(abs(jb - jc), jb + jc + 1):
            if abs(l - s) <= ja <= l + s:
                out.append((l, s))
    if not out:
        raise SpecError("节点 %s 在 l<=%d 内没有宇称+三角允许的 (l,s)"
                        % (node, spec["l_max"]))
    return out


def resolve_channels(spec):
    """返回 [(name, l, s, family)]。显式 [[channel]] 优先，否则自动枚举。"""
    if spec["channels"]:
        ch = [(c["name"], c["l"], c["s"], c["family"]) for c in spec["channels"]]
    else:
        ch = [("%s_l%d_s%d" % (spec["name"], l, s), l, s, "")
              for (l, s) in allowed_ls(spec)]
    # 道名唯一
    names = [c[0] for c in ch]
    if len(set(names)) != len(names):
        raise SpecError("道名重复：%s" % names)
    return ch


# ------------------------------------------------- 螺旋度级联：逐节点 (l,s)
def cascade_channels(spec):
    """返回螺旋度级联内核要的逐道描述：

        [{"name":..., "ls": {节点: [(l,s), ...]}, "tree": {节点: [d1,d2]},
          "l":..., "s":..., "family":..., "note":...}, ...]

    逐节点 (l,s) 的取法（**优先级从高到低**）：

    1. `[[channel]].ls` 里显式写出的节点；
    2. 顶层节点：用该道的 `(l, s)`（即 `[[channel]]` / 自动枚举给出的那组）；
    3. 其余衰变节点：`allowed_ls(spec, node)` —— 宇称 + 三角允许的**全部**值。

    第 3 条是"新能力"的落点：b₁ 的 `b1 -> phi pi` 同时允许 `(0,1)` 与 `(2,1)`，
    所以一份 TOML 不再需要为每种"顶层波 × 内层波"组合手写一条道 —— 但**道表
    仍然决定 cache 的行序**，所以这里**不做**笛卡尔积扩张：一条 TOML 道 = 一条
    cache 行，内层多分波由 `g` 参数在拟合侧展开（与 tf-pwa `split_ls.py` 一致）。

    需要"每个组合一条道"时，在 TOML 里用 `[[channel]].ls` 显式列出，
    或在 `cascade_product_channels()` 里取笛卡尔积。
    """
    out = []
    chans = spec["channels"] or [
        {"name": n, "l": l, "s": s, "family": f, "note": "", "ls": {}, "tree": {}}
        for (n, l, s, f) in resolve_channels(spec)]
    for c in chans:
        ls = {}
        for node in decaying_nodes(spec):
            if node in c["ls"]:
                ls[node] = list(c["ls"][node])
            elif node == spec["top"]:
                ls[node] = [(c["l"], c["s"])]
            else:
                ls[node] = allowed_ls(spec, node)
        # 显式列出的节点即使不是衰变节点也保留（校验会在 validate 里报错）
        for node, v in c["ls"].items():
            ls.setdefault(node, list(v))
        out.append({"name": c["name"], "ls": ls, "tree": dict(c["tree"]),
                    "l": c["l"], "s": c["s"], "family": c["family"],
                    "note": c["note"]})
    return out


def decaying_nodes(spec):
    """树里所有有 daughter 的节点，深度优先（与 chain 的 tree_order 同序）。"""
    out = []

    def walk(n):
        ds = _children(spec, n)
        if not ds:
            return
        out.append(n)
        for d in ds:
            walk(d)

    walk(spec["top"])
    return out


def cascade_product_channels(spec, nodes=None):
    """逐节点 (l,s) 的**笛卡尔积**，每个组合一条道。

    这是 §2.2 计数律 `∏_n N_n` 的直接实现：b₁ 给 2×2 = 4 条（每棵树），
    φf₂ 给 5×1 = 5 条。返回与 `cascade_channels` 同形的列表，`name` 形如
    `u_DD_1__psi2s_l2s1__b1_l2s1`。

    `nodes` 可给出要展开的节点子集（默认：全部非顶层衰变节点 + 顶层）。
    """
    base = cascade_channels(spec)
    out = []
    for c in base:
        ns = list(nodes or [n for n in c["ls"] if len(c["ls"][n]) > 0])
        ns = [n for n in ns if len(c["ls"][n]) > 1]
        if not ns:
            out.append(c)
            continue
        import itertools
        for combo in itertools.product(*[c["ls"][n] for n in ns]):
            ls = {k: list(v) for k, v in c["ls"].items()}
            for n, v in zip(ns, combo):
                ls[n] = [v]
            tag = "__".join("%s_l%ds%d" % (n, v[0], v[1]) for n, v in zip(ns, combo))
            out.append(dict(c, name="%s__%s" % (c["name"], tag), ls=ls))
    return out


# --------------------------------------------------------------------- 导出
def to_chain(spec):
    """转成 runtime.py 需要的 chain dict。"""
    nodes = {}
    for nm, p in spec["particles"].items():
        d = {"J": int(p["J"]), "mass": float(p.get("mass", 0.0))}
        if p.get("daughters"):
            d["daughters"] = list(p["daughters"])
        if "l" in p and p["daughters"]:
            d["l"] = int(p["l"])
        nodes[nm] = d
    return {"top": spec["top"], "nodes": nodes, "index": dict(spec["index"]),
            "finals": list(spec["input_keys"])}


def describe(spec):
    """人类可读的一页摘要（CLI 打印 / 写进产物头部）。"""
    ch = resolve_channels(spec)
    parts = spec["particles"]
    lines = ["链 %s:  顶层 %s (J=%s)" % (spec["name"], spec["top"],
                                        parts[spec["top"]]["J"])]
    for nm in spec["leaves"]:
        p = parts[nm]
        lines.append("  末态 %-6s J=%s  m=%.6g  parity=%s"
                     % (nm, p["J"], p.get("mass", 0.0), p.get("parity")))
    for nm, p in parts.items():
        if p.get("daughters"):
            lines.append("  节点 %-6s J=%s  -> %s + %s   l=%s"
                         % (nm, p["J"], p["daughters"][0], p["daughters"][1],
                            p.get("l", "-" if nm == spec["top"] else "?")))
    lines.append("  道（%d 条，%s）：" % (len(ch), "显式" if spec["channels"] else "自动枚举"))
    for nm, l, s, fam in ch:
        lines.append("    %-16s (l,s)=(%d,%d)%s" % (nm, l, s,
                                                    "  family=%s" % fam if fam else ""))
    # 显式道表时把「枚举应有几条」也报出来，方便一眼确认完整性
    if spec["channels"]:
        try:
            expected = allowed_ls(spec)
        except SpecError:
            expected = None
        if expected is not None:
            if spec["raw"].get("chain", {}).get("check_channels", True):
                lines.append("  道表完整性：与宇称+三角枚举一致（%d 条）"
                             % len(expected))
            else:
                lines.append("  道表完整性：**已用 check_channels=false 关闭**"
                             "（枚举应为 %d 条）" % len(expected))
    return "\n".join(lines)
