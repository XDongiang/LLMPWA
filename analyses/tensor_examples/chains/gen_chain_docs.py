#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""从 spec 自动生成链文档的**事实部分**，手工部分从 `.notes.md` 合并。

用法（仓库根）：

    python analyses/tensor_examples/chains/gen_chain_docs.py            # 全部写盘
    python analyses/tensor_examples/chains/gen_chain_docs.py --check    # 只查是否同步
    python analyses/tensor_examples/chains/gen_chain_docs.py --stdout phif2

## 为什么这样切

链文档里有两类内容：

* **事实**（会随 spec 改动而过期）：拓扑、逐节点 (l,s)、道表、表象可用性、
  计数律、emit 可行性、粒子表、树里的手写道清单。
  → 本脚本从 TOML **自动生成**，不手写。
* **叙述**（只有人知道）：这条链为什么重要、踩过什么坑、遗留问题、
  复现命令。→ 手写在 `chains/notes/<chain>.notes.md`，本脚本**原样嵌入**。

所以每份 `chains/<chain>.md` 都是**生成物**：不要直接编辑，
改 `notes/<chain>.notes.md` 或 spec，再重跑本脚本。

## 自动导出的事实

| 小节 | 来源 |
|---|---|
| 1. 拓扑 | `spec.to_chain` + `helicity_cascade.tree_order` |
| 2. 粒子表 | `spec["particles"]` |
| 3. 道表与逐节点 (l,s) | `spec.resolve_channels` / `cascade_channels` |
| 4. 计数律 | `spec.allowed_ls` 逐节点相乘 |
| 5. 表象与 emit 可行性 | `spec["bases"]` + 手写道目录是否存在 |
| 6. 深度与 omega 可用性 | `spec["depth"]` + spec.py 的分级规则 |

`--check` 会在生成物与磁盘不一致时**非零退出**，可以挂进 CI，
防止 spec 改了而文档没跟上。
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.tensor import spec as S                                   # noqa: E402
from agent.tensor.kernels import helicity_cascade as HC              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES = os.path.join(HERE, "notes")

# 要生成文档的链：显示名 -> (spec 路径, 手写道目录, 备注)
CHAINS = [
    ("phif0", "agent/tensor/specs/phif0.toml", "agent/tensor/chains/phif0"),
    ("phif2", "agent/tensor/specs/phif2.toml", "agent/tensor/chains/phif2"),
    ("b1pi", "analyses/tensor_examples/specs/ex3_b1pi.toml",
     "agent/tensor/chains/b1pi"),
    ("rhopi", "analyses/tensor_examples/specs/ex3_rhopi.toml",
     "agent/tensor/chains/rhopi"),
    ("rhof2_generic", "agent/tensor/specs/rhof2_generic.toml", None),
]


def _rel(p):
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


def _from_chains(target):
    """从 chains/ 目录到 `target`（仓库相对路径）的相对链接。

    用 `os.path.relpath` 而不是手数 `../` —— 手工版少一层，导致 5 条
    spec 链接全断（已由链接检查发现）。
    """
    return os.path.relpath(os.path.join(ROOT, target), HERE).replace(os.sep, "/")


# ----------------------------------------------------------------- 事实导出
def topology_block(spec, chain):
    """拓扑：树的文字图 + 深度 + 逐节点表。"""
    parts = spec["particles"]
    order = HC.tree_order(chain)
    dep = {spec["top"]: 0}

    def walk(n, k):
        for d in parts[n].get("daughters", []):
            dep[d] = k + 1
            walk(d, k + 1)

    walk(spec["top"], 0)
    depth = max(dep.values()) if dep else 0

    lines = []
    lines.append("```text")
    pt = parts[spec["top"]]
    ppar = pt.get("parity")
    ppar = ("%+d" % ppar) if isinstance(ppar, int) else str(ppar)
    lines.append("%s  J=%s  P=%s" % (spec["top"], pt["J"], ppar))
    for n in order[1:]:
        p = parts[n]
        ds = p.get("daughters")
        leaf = "" if ds else "   （末态）"
        par = p.get("parity")
        par = ("%+d" % par) if isinstance(par, int) else str(par)
        lines.append("%s├── %-10s J=%s  P=%s%s"
                     % ("│  " * (dep[n] - 1), n, p["J"], par,
                        ("  → %s + %s" % (ds[0], ds[1])) if ds else leaf))
    lines.append("```")
    lines.append("")
    lines.append("**深度 = %d**（最长路径上的边数）。" % depth)
    return "\n".join(lines), depth, dep


def particle_block(spec):
    parts = spec["particles"]
    lines = ["| 粒子 | $J$ | 宇称 | 质量 | 角色 | 允许的 $l$ |",
             "|---|---|---|---|---|---|"]
    for nm in HC.tree_order(S.to_chain(spec)):
        p = parts[nm]
        ds = p.get("daughters")
        role = "顶层" if nm == spec["top"] else ("节点" if ds else "末态")
        ls = "-"
        if ds:
            try:
                ls = ", ".join("%d" % l for l in
                               sorted({l for l, _ in S.allowed_ls(spec, nm)}))
            except S.SpecError:
                ls = "?"
        lines.append("| `%s` | %s | %s | %s | %s | %s |"
                     % (nm, p["J"], p.get("parity"),
                        ("%.6g" % p["mass"]) if p.get("mass") else "-",
                        role, ls))
    return "\n".join(lines)


def channel_block(spec):
    """道表：显式还是枚举，每条的顶层 (l,s) 与逐节点 (l,s)。"""
    explicit = bool(spec["channels"])
    try:
        casc = S.cascade_channels(spec)
    except S.SpecError as e:
        return "**道表解析失败**：%s" % e, 0
    lines = []
    lines.append("道表来源：**%s**（%d 条）。"
                 % ("显式 `[[channel]]`" if explicit else "自动枚举", len(casc)))
    lines.append("")
    # 逐节点列：用第一条道能看到的节点集合
    nodes = [n for n in casc[0]["ls"]] if casc else []
    head = ["道", "顶层 $(l,s)$"] + ["`%s`" % n for n in nodes if n != spec["top"]]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for c in casc:
        cells = ["`%s`" % c["name"], "(%d,%d)" % (c["l"], c["s"])]
        for n in nodes:
            if n == spec["top"]:
                continue
            v = c["ls"].get(n) or []
            cells.append("  ".join("(%d,%d)" % t for t in v) if v else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines), len(casc)


def counting_block(spec):
    """计数律 ∏_n N_n。"""
    nodes = S.decaying_nodes(spec)
    prod = 1
    lines = ["| 节点 | 允许的 $(l,s)$ | $N$ |", "|---|---|---|"]
    for n in nodes:
        try:
            ls = S.allowed_ls(spec, n)
        except S.SpecError as e:
            lines.append("| `%s` | **无法枚举**：%s | ? |" % (n, e))
            continue
        prod *= len(ls)
        lines.append("| `%s` | %s | %d |"
                     % (n, "  ".join("(%d,%d)" % t for t in ls), len(ls)))
    lines.append("| **合计** | | **%d** |" % prod)
    lines.append("")
    non_top = [n for n in nodes if n != spec["top"]]
    multi = [n for n in non_top if len(S.allowed_ls(spec, n)) > 1]
    if multi:
        lines.append("⚠️ 非顶层节点 %s **有多个** $(l,s)$ ⇒ omega 的信息容量"
                     "**不够**（它的 cache 里非顶层节点只有标量 $K_i$）。"
                     % "、".join("`%s`" % n for n in multi))
    else:
        lines.append("✅ 非顶层节点都只有 **1 个** $(l,s)$ ⇒ omega 的信息容量够用。")
    return "\n".join(lines), prod


def _try_emit(spec, basis):
    """**实测**某个表象能不能 emit（不靠"手写道目录存在"来推断）。

    写到临时目录 —— 生成文档不该在仓库里留产物。
    """
    import shutil
    import tempfile

    from agent.tensor import emitter as E
    s2 = dict(spec)
    s2["bases"] = [basis]
    tmp = tempfile.mkdtemp(prefix="chaindoc_")
    try:
        r = E.emit(s2, out_dir=tmp)
        return True, "%d 个脚本" % len(r["scripts"])
    except Exception as e:                                       # noqa: BLE001
        return False, "%s" % type(e).__name__
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def basis_block(spec, chain, hand_dir):
    """表象表 + emit 可行性（**实测**）。"""
    depth = spec["depth"]
    hand = []
    if hand_dir and os.path.isdir(os.path.join(ROOT, hand_dir)):
        hand = sorted(f[:-3] for f in os.listdir(os.path.join(ROOT, hand_dir))
                      if f.endswith(".py") and f != "__init__.py")
    names = [c[0] for c in S.resolve_channels(spec)]
    missing = [n for n in names if n not in hand] if hand_dir else names

    lines = ["| 表象 | kernel | 需要手写道 | 实测 `emit` | 说明 |",
             "|---|---|---|---|---|"]
    for b in spec["bases"] or [{"name": "omega", "kernel": "omega"}]:
        kern = b.get("kernel") or b["name"]
        need = kern in ("covariant", "helicity")
        ok, why = _try_emit(spec, b)
        lines.append("| `%s` | `%s` | %s | %s | %s |"
                     % (b["name"], kern, "是" if need else "否",
                        "✅" if ok else "❌", why))
    lines.append("")
    lines.append("（`emit` 一列是**实测**结果 —— 逐表象单独调 "
                 "`emitter.emit`，不是靠文件存在推断。）")
    lines.append("")
    if hand_dir:
        lines.append("手写道（`%s/`，%d 条）：%s"
                     % (_rel(os.path.join(ROOT, hand_dir)), len(hand),
                        ", ".join("`%s`" % h for h in hand) or "（无）"))
        if missing:
            lines.append("")
            lines.append("⚠️ 协变/螺度路线**缺** %d 条手写公式：%s —— "
                         "所以这两条路线的 `emit` 会报 `LookupError`，"
                         "只有 omega 能出产物。"
                         % (len(missing),
                            ", ".join("`%s`" % m for m in missing)))
    else:
        lines.append("⚠️ 本链**没有**手写道目录，所以 `cartesian`/`helicity` "
                     "两条路线的 `emit` 会报 `LookupError`；"
                     "只有 `omega` 能出产物。")
    lines.append("")
    lines.append("深度 = %d ⇒ omega %s。"
                 % (depth, "**可用**" if depth <= 2 else
                    "**不可用**（`spec.py` 的分级检查会直接报错，不是静默出错）"))
    return "\n".join(lines)


def generate(name, spec_path, hand_dir):
    spec = S.load_spec(os.path.join(ROOT, spec_path))
    chain = S.to_chain(spec)
    topo, depth, dep = topology_block(spec, chain)
    parts = particle_block(spec)
    chans, nch = channel_block(spec)
    counting, prod = counting_block(spec)
    bases = basis_block(spec, chain, hand_dir)

    notes_path = os.path.join(NOTES, "%s.notes.md" % name)
    if os.path.exists(notes_path):
        notes = open(notes_path, encoding="utf-8").read().rstrip()
    else:
        notes = ("<!-- 还没有手工笔记：在 chains/notes/%s.notes.md 里写 -->\n"
                 % name)

    out = []
    spec_rel = _rel(os.path.join(ROOT, spec_path))
    out.append("# 链 `%s` —— %s\n" % (name, _title(spec)))
    out.append("> **本文件是生成物**，不要直接编辑。\n"
               "> 事实部分由 [`gen_chain_docs.py`](gen_chain_docs.py) 从 "
               "[`%s`](%s) 自动导出；\n"
               "> 叙述部分来自 [`notes/%s.notes.md`](notes/%s.notes.md)。\n"
               "> 改完重跑：`python analyses/tensor_examples/chains/"
               "gen_chain_docs.py`\n"
               % (spec_rel, _from_chains(spec_rel), name, name))
    out.append("---\n")
    out.append("## 1. 拓扑（自动导出）\n")
    out.append(topo + "\n")
    out.append("## 2. 粒子表（自动导出）\n")
    out.append(parts + "\n")
    out.append("## 3. 道表与逐节点 $(l,s)$（自动导出）\n")
    out.append(chans + "\n")
    out.append("## 4. 计数律（自动导出）\n")
    out.append(counting + "\n")
    out.append("## 5. 表象与 `emit` 可行性（自动导出）\n")
    out.append(bases + "\n")
    out.append("---\n")
    out.append("## 6. 叙述：做了什么、还有什么问题（手工）\n")
    out.append(notes + "\n")
    return "\n".join(out)


def _title(spec):
    """一行式拓扑，只展开**衰变节点**（末态不展开），例如
    `psi2s → b1 π⁻ ; b1 → φ π⁺ ; φ → K⁺K⁻`。"""
    parts = spec["particles"]
    try:
        order = HC.tree_order(S.to_chain(spec))
    except Exception:                                            # noqa: BLE001
        return spec["name"]
    segs = []
    for n in order:
        ds = parts[n].get("daughters")
        if not ds:
            continue
        segs.append("%s → %s" % (n, " + ".join(ds)))
    return " ; ".join(segs) if segs else spec["name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="只检查磁盘上的文档是否与生成结果一致")
    ap.add_argument("--stdout", metavar="CHAIN", default=None,
                    help="打印到 stdout 而不写盘")
    a = ap.parse_args()

    os.makedirs(NOTES, exist_ok=True)
    bad = []
    for name, sp, hd in CHAINS:
        txt = generate(name, sp, hd)
        if a.stdout:
            if a.stdout == name:
                sys.stdout.write(txt)
            continue
        dst = os.path.join(HERE, "%s.md" % name)
        if a.check:
            old = open(dst, encoding="utf-8").read() if os.path.exists(dst) else ""
            if old != txt:
                bad.append(name)
            continue
        with open(dst, "w", encoding="utf-8") as f:
            f.write(txt)
        print("  写 %s" % _rel(dst))

    if a.check:
        if bad:
            print("**不同步**：%s\n  重跑 gen_chain_docs.py" % ", ".join(bad))
            return 1
        print("OK：%d 份链文档与 spec 同步" % len(CHAINS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
