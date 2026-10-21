#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`agent.tensor.spec`：TOML 读入、校验、(l,s) 自动枚举。

    python -m pytest agent/tensor/tests/test_spec.py
    python agent/tensor/tests/test_spec.py        # 直接跑（无 pytest 也能用）
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from agent.tensor import spec  # noqa: E402

_GOOD = '''
[chain]
name = "t"
top = "A"
l_max = 4
suffix = ""

[input]
keys = ["Kp", "Km", "Pip", "Pim"]
layout = "x,y,z,E"

[particles.A]
J = 1
mass = 3.686
parity = -1
daughters = ["B", "C"]

[particles.B]
J = 1
mass = 1.019
parity = -1
daughters = ["Kp", "Km"]
l = 1

[particles.C]
J = 2
mass = 1.275
parity = 1
daughters = ["Pip", "Pim"]
l = 2

[particles.Kp]
J = 0
mass = 0.493677
[particles.Km]
J = 0
mass = 0.493677
[particles.Pip]
J = 0
mass = 0.13957039
[particles.Pim]
J = 0
mass = 0.13957039
'''



_SPEC_PHIF2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "specs", "phif2.toml")

def _write(text):
    fd, p = tempfile.mkstemp(suffix=".toml")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    return p


def test_auto_enumeration():
    """宇称 + 角动量三角自动枚举出 phif2 族那 5 个 (l,s)。"""
    s = spec.load_spec(_write(_GOOD))
    assert spec.allowed_ls(s) == [(0, 1), (2, 1), (2, 2), (2, 3), (4, 3)], \
        spec.allowed_ls(s)
    ch = spec.resolve_channels(s)
    assert [(c[1], c[2]) for c in ch] == [(0, 1), (2, 1), (2, 2), (2, 3), (4, 3)]


def test_auto_enumeration_is_generic():
    """换一组量子数（a1 -> rho pi, rho -> pi pi）也照样枚举，不是硬编码 phif2。

    A(J=1, eta=+1) -> B(J=1, eta=-1) + C(J=1, eta=-1)，B -> K K (l=1)，
    C -> pi pi (l=1)：顶层宇称 eta_A = eta_B eta_C (-1)^l = (+1)(-1)^l，
    要等于 +1，故 l 取偶。
    """
    t = _GOOD.replace('name = "t"', 'name = "a1"')
    t = t.replace('J = 2\nmass = 1.275', 'J = 1\nmass = 0.775')
    t = t.replace('parity = 1\ndaughters = ["Pip", "Pim"]',
                  'parity = -1\ndaughters = ["Pip", "Pim"]')
    t = t.replace('parity = -1\ndaughters = ["B", "C"]',
                  'parity = 1\ndaughters = ["B", "C"]')
    # C 的子衰变 l 由 2 改 1（J_C = 1，S 波不允许），B 保持 l=1
    t = t.replace('daughters = ["Pip", "Pim"]\nl = 2',
                  'daughters = ["Pip", "Pim"]\nl = 1')
    s = spec.load_spec(_write(t))
    ls = spec.allowed_ls(s)
    assert ls, ls
    assert all(l % 2 == 0 for (l, _s) in ls), ls   # eta_A=+1 -> l 偶


def test_explicit_channels_win():
    """显式道表优先于自动枚举（只要集合一致，或显式关掉交叉检查）。

    这里刻意只留一条道，所以必须 `check_channels = false` —— 否则新的
    交叉检查（见下一条测试）会拒绝它，这正是我们想要的行为。
    """
    t = _GOOD.replace("l_max = 4", "l_max = 4\ncheck_channels = false")
    t += '\n[[channel]]\nname = "only"\nl = 2\ns = 3\n'
    s = spec.load_spec(_write(t))
    assert spec.resolve_channels(s) == [("only", 2, 3, "")]


def test_j_over_limit_is_readable_error():
    """J 超过级联内核上限（6）必须在读 TOML 时报错，而不是运行时 KeyError。

    上限在 2026-10-20 从 4 扩到 6（`LAM_SLOTS` / `Bl` / `spec.py` 一起改），
    所以这里用 7 触发。J=5 现在**合法**（ρ₅），见
    `test_j5_is_allowed_on_cascade_but_rejected_on_tensor_route`。
    """
    t = _GOOD.replace("J = 2\nmass = 1.275", "J = 7\nmass = 1.275")
    try:
        spec.load_spec(_write(t))
    except spec.SpecError as e:
        assert "J=7" in str(e) and "上限 6" in str(e), str(e)
    else:
        raise AssertionError("J>6 未报错")


def test_l_over_limit_is_readable_error():
    """l / `l_max` 超过级联内核上限（6）必须报错。"""
    # (a) 节点 l=7 撞上限（`l_max` 也要放大，否则先撞三角枚举那条）
    t = _GOOD.replace("l = 2\n\n[particles.Kp]", "l = 7\n\n[particles.Kp]")
    t = t.replace("l_max = 4", "l_max = 7")
    try:
        spec.load_spec(_write(t))
    except spec.SpecError as e:
        assert "上限 6" in str(e), str(e)
    else:
        raise AssertionError("l>6 未报错")

    # (b) `[chain].l_max` 自己超限
    t2 = _GOOD.replace("l_max = 4", "l_max = 9")
    try:
        spec.load_spec(_write(t2))
    except spec.SpecError as e:
        assert "l_max" in str(e) and "上限 6" in str(e), str(e)
    else:
        raise AssertionError("l_max>6 未报错")


def test_j5_allowed_on_both_routes():
    """J=5 / l=5 在**两条路线上都放行**（2026-10-20 扩展）。

    级联路线：`LAM_SLOTS` + `Bl` 扩到 6。
    协变张量路线：`covariant.py` 新增论文 Eq.(258) 的通用 `t_l` 与 `B5`，
    所以 J=5 不再被上限挡住。

    ⚠️ "validate 放行" ≠ "能 emit"：协变路线要内联手写道函数，三体置换族
    默认没有（那是 emit 期的 LookupError，由
    `tests/test_ex12_permutation.py::test_emit_end_to_end` 覆盖）。
    """
    t = _GOOD.replace("J = 2\nmass = 1.275", "J = 5\nmass = 1.275")
    t = t.replace("l = 2\n\n[particles.Kp]", "l = 5\n\n[particles.Kp]")
    t = t.replace("l_max = 4", "l_max = 5")

    # 1) 默认（级联）路线：J=5/l=5 放行
    s = spec.load_spec(_write(t))
    assert s["particles"]["C"]["J"] == 5
    assert s["particles"]["C"]["l"] == 5

    # 2) 协变张量路线：同样放行
    s2 = dict(s)
    s2["bases"] = [{"name": "helicity", "kernel": "helicity"}]
    spec.validate(s2)

    # 3) 但 J=7 在两条路线上都要被拒（上限 6 是因为 small_d_weight 溢出）
    #
    # `load_spec` 自己就会 validate，所以 J=7 在**读 TOML 时**就炸 ——
    # 这正是我们要的（而不是等到运行时 KeyError）。
    t7 = t.replace("J = 5\nmass = 1.275", "J = 7\nmass = 1.275")
    try:
        spec.load_spec(_write(t7))
    except spec.SpecError as e:
        assert "J=7" in str(e) and "上限 6" in str(e), str(e)
    else:
        raise AssertionError("J=7 未被拒绝")

    # 协变路线同样在 validate 层被拒（显式改 bases 再 validate）
    s6 = spec.load_spec(_write(t))
    s6 = dict(s6)
    s6["particles"] = dict(s6["particles"])
    s6["particles"]["C"] = dict(s6["particles"]["C"], J=7)
    s6["bases"] = [{"name": "helicity", "kernel": "helicity"}]
    try:
        spec.validate(s6)
    except spec.SpecError as e:
        assert "J=7" in str(e) and "上限 6" in str(e), str(e)
    else:
        raise AssertionError("J=7 未被拒绝（协变路线）")


def test_leaf_key_mismatch():
    t = _GOOD.replace('keys = ["Kp", "Km", "Pip", "Pim"]',
                      'keys = ["Kp", "Km", "Pip", "Pim", "X"]')
    try:
        spec.load_spec(_write(t))
    except spec.SpecError as e:
        assert "不一致" in str(e), str(e)
    else:
        raise AssertionError("末态/键不一致未报错")


def test_cycle_detected():
    t = _GOOD.replace('daughters = ["Pip", "Pim"]', 'daughters = ["Pip", "Km"]')
    try:
        spec.load_spec(_write(t))
    except spec.SpecError as e:
        assert "两次" in str(e) or "不一致" in str(e), str(e)
    else:
        raise AssertionError("重复节点未报错")


def test_leaf_daughter_topology_is_accepted():
    """`psi2s -> phi + pi`（pi 是末态）是合法拓扑，validate 不该拒绝。

    审计 §2.2：这种拓扑以前 `check` 通过、运行时 `KeyError: 'daughters'`。
    现在 runtime 把末态节点的衰变因子当成 1（== D^{0*}_{0,0}）。
    """
    t = _GOOD.replace('daughters = ["B", "C"]', 'daughters = ["B", "Pim"]')
    t = t.replace('keys = ["Kp", "Km", "Pip", "Pim"]',
                  'keys = ["Kp", "Km", "Pim"]')
    t = t.replace('''
[particles.C]
J = 2
mass = 1.275
parity = 1
daughters = ["Pip", "Pim"]
l = 2

''', '')
    s = spec.load_spec(_write(t))
    ch = spec.to_chain(s)
    assert ch["nodes"]["Pim"].get("daughters") is None
    assert "l" not in ch["nodes"]["Pim"]
    assert ch["nodes"]["B"]["l"] == 1
    from agent.tensor.kernels import omega as RT
    assert RT._node_decay_d({}, ch, "Pim", 3).shape == (3, 1)
    assert (RT._node_decay_d({}, ch, "Pim", 3) == 1).all()


def test_three_level_cascade_rejected():
    """审计 §2.3：内核只算 D0*D1*D2，三层树必须提前拒绝而不是静默算错。"""
    t = '''
[chain]
name = "deep"
top = "A"
l_max = 4

[input]
keys = ["f1", "f2", "f3", "f4"]
layout = "x,y,z,E"

[particles.A]
J = 1
mass = 3.686
parity = -1
daughters = ["B", "C"]

[particles.B]
J = 1
mass = 1.019
parity = -1
daughters = ["f1", "f2"]
l = 1

[particles.C]
J = 1
mass = 0.775
parity = -1
daughters = ["D", "f3"]
l = 1

[particles.D]
J = 1
mass = 0.5
parity = -1
daughters = ["f2b", "f4"]
l = 1

[particles.f1]
J = 0
mass = 0.49
[particles.f2]
J = 0
mass = 0.49
[particles.f2b]
J = 0
mass = 0.49
[particles.f3]
J = 0
mass = 0.14
[particles.f4]
J = 0
mass = 0.14
'''
    try:
        spec.load_spec(_write(t))
    except spec.SpecError as e:
        assert "深度" in str(e) or "2 级" in str(e), str(e)
    else:
        raise AssertionError("三层级联未被拒绝")


def test_explicit_channels_cross_checked_against_enumeration():
    """显式 [[channel]] 必须与自动枚举一致（漏/多/标错都要报错）。

    这条校验的价值最高：道表是 tensor cache 的**行序**，静默不一致不会崩，
    只会让拟合结果错位。
    """
    import re as _re

    full = open(_SPEC_PHIF2, encoding="utf-8").read()
    head = _re.split(r"^\[\[channel\]\]", full, flags=_re.M)[0]
    basis = '\n[[basis]]\nname = "cartesian"\nkernel = "covariant"\n'
    names = ["phif201", "phif221", "phif222", "phif223", "phif243"]
    ls = [(0, 1), (2, 1), (2, 2), (2, 3), (4, 3)]

    def mk(chans):
        """返回 TOML **文本**（不含 [[channel]] 之外的原有道表）。"""
        body = "".join('[[channel]]\nname = "%s"\nl = %d\ns = %d\n' % (n, l, s)
                       for n, (l, s) in chans)
        return head + body + basis

    # 完整 -> 通过
    spec.load_spec(_write(mk(list(zip(names, ls)))))

    # 少一条 / 多一条 / 标错 -> 都要拒绝
    bad = {
        "漏一条": [(names[i], ls[i]) for i in (0, 1, 3, 4)],
        "漏两条": [(names[i], ls[i]) for i in (0, 1, 4)],
        "多一条宇称禁止": list(zip(names, ls)) + [("x", (1, 1))],
        "l/s 标错": list(zip(names, [(0, 1), (2, 1), (2, 2), (4, 3), (4, 3)])),
    }
    for label, chans in bad.items():
        try:
            spec.load_spec(_write(mk(chans)))
        except spec.SpecError as e:
            assert "不一致" in str(e), (label, e)
        else:
            raise AssertionError("道表 %s 竟然通过了" % label)

    # 顺序不同但集合相同 -> 通过（顺序由 TOML 决定，是承重的）
    spec.load_spec(_write(mk(list(zip(names, [(2, 3), (0, 1), (4, 3), (2, 1), (2, 2)])))))

    # 显式关掉（只要子集）
    t = mk([(names[0], ls[0])]).replace("l_max = 4",
                                       "l_max = 4\ncheck_channels = false")
    s = spec.load_spec(_write(t))
    assert spec.resolve_channels(s) == [("phif201", 0, 1, "")]


def test_basis_section_parsed():
    t = _GOOD + '''
[[basis]]
name = "helicity"
kernel = "helicity"
env = { TENSOR_BASIS = "helicity", HELICITY_LAYER = "1" }
'''
    s = spec.load_spec(_write(t))
    assert len(s["bases"]) == 1
    assert s["bases"][0]["kernel"] == "helicity"
    assert s["bases"][0]["env"]["HELICITY_LAYER"] == "1"




def test_three_daughter_node_rejected():
    """§2.5 缺陷：三体节点 `A -> b+c+d` 必须**显式**拒绝，而不是被静默当成占位树。

    文档例 4 的 $\\omega^\\mu$（418–430 行）是一个**单节点三体**对象：

    $$\\omega^\\mu=\\epsilon^{\\mu\\nu\\lambda\\sigma}p_1^\\nu p_2^\\lambda p_0^\\sigma[\\cdots]$$

    它不能分解成两体树，因为 $\\epsilon^{\\mu\\nu\\lambda\\sigma}
    p_\\omega{}_\\nu p_\\rho{}_\\lambda p_\\pi{}_\\sigma\\equiv0$
    （$p_\\omega=p_\\rho+p_\\pi$，全反对称三缩并）。

    现在 `specs/ex4_omega_kk.toml` 只能写二叉占位树绕过去。本测试**锁住**
    这条缺陷：哪天有人实现了真三体节点，这个测试会失败 —— 那时应改成
    "接受三体节点"并更新
    `documentation/tensor_examples_feasibility.md` §2.5。
    """
    t = '''
[chain]
name = "threebody"
top = "omega"
l_max = 4

[input]
keys = ["p1", "p2", "p0"]
layout = "x,y,z,E"

[particles.omega]
J = 1
mass = 1.5
parity = -1
daughters = ["p1", "p2", "p0"]

[particles.p1]
J = 0
mass = 0.13957039
[particles.p2]
J = 0
mass = 0.13957039
[particles.p0]
J = 0
mass = 0.1349768
'''
    try:
        spec.load_spec(_write(t))
    except spec.SpecError as e:
        assert "个 daughter" in str(e) and "两体" in str(e), str(e)
    else:
        raise AssertionError(
            "三体节点（3 个 daughter）未被拒绝 —— 若这是有意支持的，"
            "请更新 test_three_daughter_node_rejected 与 feasibility §2.5")


def test_channel_tree_may_introduce_unreachable_node():
    """§2.3.1：`[[channel]].tree` 把顶层换成 base 树里**不可达**的节点时必须通过。

    例 6 的 base 树是 `psi -> phi + f2`（$f_2$ 分支），而 $f_0$ 分支的道用
    `tree = { psi2s = ["phi","f0"], f0 = [...] }` 把顶层换成 $f_0$ ——
    `f0` 在 base 树里根本不可达。旧实现用 base 树算"哪些节点在衰变"，
    于是报"给末态节点指定 (l,s)"。真正该看的是**覆盖后**的树。
    """
    t = '''
[chain]
name = "mirror_top"
top = "psi2s"
l_max = 4
check_channels = false

[input]
keys = ["Kp", "Km", "Pip", "Pim", "Pip2", "Pim2"]
layout = "x,y,z,E"

[particles.psi2s]
J = 1
mass = 3.686
parity = -1
daughters = ["phi", "f2"]

[particles.phi]
J = 1
mass = 1.019461
parity = -1
l = 1
daughters = ["Kp", "Km"]

[particles.f2]
J = 2
mass = 1.9
parity = 1
l = 2
daughters = ["sigA", "sigB"]

[particles.f0]
J = 0
mass = 1.9
parity = 1
l = 0
daughters = ["sigA", "sigB"]

[particles.sigA]
J = 0
mass = 0.5
parity = 1
l = 0
daughters = ["Pip", "Pim"]

[particles.sigB]
J = 0
mass = 0.5
parity = 1
l = 0
daughters = ["Pip2", "Pim2"]

[particles.Kp]
J = 0
mass = 0.493677
[particles.Km]
J = 0
mass = 0.493677
[particles.Pip]
J = 0
mass = 0.13957039
[particles.Pim]
J = 0
mass = 0.13957039
[particles.Pip2]
J = 0
mass = 0.13957039
[particles.Pim2]
J = 0
mass = 0.13957039

[[channel]]
name = "c_f2"
l = 2
s = 2
ls = { psi2s = [[2, 2]], f2 = [[2, 0]] }
tree = { psi2s = ["phi", "f2"], f2 = ["sigA", "sigB"] }

[[channel]]
name = "c_f0"
l = 0
s = 1
ls = { psi2s = [[0, 1]], f0 = [[0, 0]] }
tree = { psi2s = ["phi", "f0"], f0 = ["sigA", "sigB"] }

[[basis]]
name = "cartesian"
kernel = "covariant"
'''
    s = spec.load_spec(_write(t))
    assert sorted(c["name"] for c in s["channels"]) == ["c_f0", "c_f2"]
    # 逐道按覆盖后的树算衰变节点：f0 道必须认出 f0 是衰变节点
    assert "f0" in spec._decaying_with(s, {"psi2s": ["phi", "f0"],
                                          "f0": ["sigA", "sigB"]})


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print("  OK   %s" % fn.__name__)
        except Exception as e:                                  # noqa: BLE001
            bad += 1
            print("  FAIL %s: %s" % (fn.__name__, e))
    print("%s: %d/%d" % ("PASS" if not bad else "**FAIL**", len(fns) - bad, len(fns)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
