#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""内核的纯数值单测（不需要四动量文件）。

审计 §2.4 指出 `specgen`（现 `agent/tensor/kernels/omega.py`）**零测试**。
这里补上三类不依赖数据的性质检验：

1. `cg()` 的闭式与对称性、正交性；
2. `small_d_matrix` 的正交性 `sum_n d^j_{mn} d^j_{m'n} == delta_{mm'}`；
3. `cg_matrix` 的 shape 与 `allowed_ls` 的一致性。

    python agent/tensor/tests/test_kernels.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))

from agent.tensor.kernels import omega as RT       # noqa: E402


def test_cg_closed_forms():
    """已知闭式：<j 0; j 0|0 0> = (-1)^j/sqrt(2j+1)；<1/2 ±1/2; ...> 等。"""
    for j in range(0, 5):
        got = RT.cg(j, 0, j, 0, 0, 0)
        want = ((-1) ** j) / math.sqrt(2 * j + 1)
        assert abs(got - want) < 1e-12, (j, got, want)
    # <1 1; 1 -1|0 0> = 1/sqrt(3)
    assert abs(RT.cg(1, 1, 1, -1, 0, 0) - 1 / math.sqrt(3)) < 1e-12
    # 三角不等式之外必须严格为 0
    assert RT.cg(1, 0, 1, 0, 3, 0) == 0.0
    assert RT.cg(1, 0, 1, 0, 1, 1) == 0.0      # m 不守恒


def test_cg_orthogonality():
    """sum_{m1,m2} <j1 m1; j2 m2|J M>^2 == 1（对每个合法的 J, M）。"""
    for j1, j2 in ((0, 0), (1, 1), (1, 2), (2, 2), (3, 2)):
        for J in range(abs(j1 - j2), j1 + j2 + 1):
            for M in range(-J, J + 1):
                s = 0.0
                for m1 in range(-j1, j1 + 1):
                    m2 = M - m1
                    if abs(m2) > j2:
                        continue
                    s += RT.cg(j1, m1, j2, m2, J, M) ** 2
                assert abs(s - 1.0) < 1e-12, (j1, j2, J, M, s)


def test_cg_row_orthogonality():
    """固定 M 时，不同的 J 行正交（只在 |M| <= J 的 J 上比较）。

    注意 J 的范围必须同时满足三角不等式 |j1-j2| <= J <= j1+j2 与 |M| <= J，
    否则 cg 直接返回 0（这正是第一次写这条测试时的错处）。
    """
    j1 = j2 = 2
    lo, hi = abs(j1 - j2), j1 + j2
    for M in range(-hi, hi + 1):
        valid = [J for J in range(lo, hi + 1) if abs(M) <= J]
        for J in valid:
            for Jp in valid:
                s = 0.0
                for m1 in range(-j1, j1 + 1):
                    m2 = M - m1
                    if abs(m2) > j2:
                        continue
                    s += RT.cg(j1, m1, j2, m2, J, M) * RT.cg(j1, m1, j2, m2, Jp, M)
                want = 1.0 if J == Jp else 0.0
                assert abs(s - want) < 1e-12, (J, Jp, M, s)


def test_small_d_orthogonality():
    """d^j(theta) 是正交矩阵：d d^T == I。

    `small_d_matrix` 的约定（照抄 tf-pwa）：第一个参数是**角度数组**，
    第二个参数是 **2J**；返回 `(n_theta, 2J+1, 2J+1)`，行列下标都是
    m, m' = -J..J（按 m+J 索引）。
    """
    for j in (1, 2, 3, 4):
        for theta in (0.0, 0.3, 1.1, 2.5, math.pi):
            d = RT.small_d_matrix(np.array([theta]), 2 * j)[0]   # (2j+1, 2j+1)
            prod = d @ d.T
            assert np.abs(prod - np.eye(2 * j + 1)).max() < 1e-12, (j, theta)


def test_small_d_at_zero_is_identity():
    for j in (0, 1, 2, 3):
        d = RT.small_d_matrix(np.array([0.0]), 2 * j)[0]
        assert np.abs(d - np.eye(2 * j + 1)).max() < 1e-12, j


def test_cg_matrix_shape_matches_ls():
    """`cg_matrix` 的 shape 必须是 (len(ls), 2jb+1, 2jc+1)。"""
    cases = (
        ([(0, 1)], 1, 1, 0),
        ([(2, 3)], 1, 1, 2),
        ([(0, 1), (2, 1), (2, 3)], 1, 1, 2),
        ([(0, 1), (2, 1), (2, 2), (2, 3), (4, 3)], 1, 1, 2),
    )
    for ls, ja, jb, jc in cases:
        M = RT.cg_matrix(ls, ja, jb, jc, RT.LAM_SLOTS[jb], RT.LAM_SLOTS[jc])
        assert M.shape == (len(ls), 2 * jb + 1, 2 * jc + 1), (ls, M.shape)
        assert np.isfinite(M).all(), ls


def test_cg_matrix_full_phif2():
    """phif2 的五条道：shape 与道表一一对应，且全部有限。"""
    ls = [(0, 1), (2, 1), (2, 2), (2, 3), (4, 3)]
    M = RT.cg_matrix(ls, 1, 1, 2, RT.LAM_SLOTS[1], RT.LAM_SLOTS[2])
    assert M.shape == (5, 3, 5), M.shape
    assert np.isfinite(M).all()


def test_blatt_weisskopf_limits():
    """l = 0 时 B_0 == 1、node_barrier == 1；l >= 1 在 q -> 0 时 -> 0。"""
    q = np.array([0.0, 0.01, 1.0, 10.0])
    assert np.allclose(RT.Bl(0, q), 1.0)
    assert np.allclose(RT.node_barrier(0, q), 1.0)
    for l in (1, 2, 3, 4):
        assert RT.Bl(l, np.array([0.0]))[0] > 0.0
        assert abs(RT.node_barrier(l, np.array([0.0]))[0]) < 1e-12, l
    try:
        RT.Bl(5, q)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("l=5 应当 NotImplementedError")


def test_lam_slots_cover_spins():
    for J in range(0, 5):
        assert RT.LAM_SLOTS[J] == list(range(-J, J + 1)), J


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
    print("%s: %d/%d" % ("PASS" if not bad else "**FAIL**",
                         len(fns) - bad, len(fns)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
