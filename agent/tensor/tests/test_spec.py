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
    """J > 4 必须在读 TOML 时报错，而不是运行时 KeyError。"""
    t = _GOOD.replace("J = 2\nmass = 1.275", "J = 7\nmass = 1.275")
    try:
        spec.load_spec(_write(t))
    except spec.SpecError as e:
        assert "J=7" in str(e) and "上限 4" in str(e), str(e)
    else:
        raise AssertionError("J>4 未报错")


def test_l_over_limit_is_readable_error():
    t = _GOOD.replace("l = 2\n\n[particles.Kp]", "l = 6\n\n[particles.Kp]")
    try:
        spec.load_spec(_write(t))
    except spec.SpecError as e:
        assert "上限 4" in str(e), str(e)
    else:
        raise AssertionError("l>4 未报错")


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
