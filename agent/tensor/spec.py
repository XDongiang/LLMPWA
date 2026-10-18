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

    channels = []
    for c in (raw.get("channel") or []):
        channels.append({"name": c["name"], "l": int(c["l"]), "s": int(c["s"]),
                         "family": c.get("family", "")})

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

    # 前置能力边界（审计 §2.3）：内核（Wigner-D / Blatt-Weisskopf / LAM_SLOTS）
    # 只到 J = 4，位垒因子 Bl 只到 l = 4。超了要在**读 TOML 时**报错，
    # 而不是运行时 KeyError。
    #
    # 另外只支持**两级级联**（顶层 + 一层共振），因为 runtime 只算
    # `D0 * D1 * D2`；更深的树在 validate 里拒绝，而不是静默算错。
    for nm, p in sorted(parts.items()):
        if p["J"] > 4:
            raise SpecError(
                "粒子 %s 的 J=%d 超出内核上限 4（LAM_SLOTS / Wigner-D 只到 J=4）"
                % (nm, p["J"]))
        if "l" in p and p["daughters"] and int(p["l"]) > 4:
            raise SpecError(
                "节点 %s 的 l=%d 超出内核上限 4（Blatt-Weisskopf Bl 只到 l=4）"
                % (nm, int(p["l"])))
    if spec["l_max"] > 4:
        raise SpecError("[chain].l_max=%d 超出内核上限 4" % spec["l_max"])

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

    # 级联深度：runtime 只算 `D0 * D1 * D2`（顶层 + 两个子节点各一层），
    # 所以**只支持两级级联**。三层的树（如 psi2s -> X + Y, X -> A + B,
    # A -> C + D）会静默漏掉最内层的角分布 —— 这里提前拒绝（审计 §2.3）。
    def _depth(n):
        ds = _children(spec, n)
        return 0 if not ds else 1 + max(_depth(d) for d in ds)

    depth = _depth(top)
    if depth > 2:
        raise SpecError(
            "链的级联深度是 %d，内核只支持 2 级（顶层 + 一层共振）；"
            "runtime 只算 D0*D1*D2，更深的树会静默漏掉内层角分布" % depth)

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
    return spec


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
