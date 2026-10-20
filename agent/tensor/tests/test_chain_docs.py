#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""链文档与 spec 的同步检查（把 `gen_chain_docs.py --check` 挂进测试）。

`analyses/tensor_examples/chains/<chain>.md` 是**生成物**：事实部分从 spec
自动导出。spec 改了而文档没重生成时，文档会**静默过期** —— 这个测试就是
防这个。

    python agent/tensor/tests/test_chain_docs.py

失败时：

    python analyses/tensor_examples/chains/gen_chain_docs.py
"""
from __future__ import annotations

import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)

GEN = os.path.join(_ROOT, "analyses", "tensor_examples", "chains",
                   "gen_chain_docs.py")


def test_generator_exists():
    assert os.path.exists(GEN), GEN


def test_docs_in_sync_with_specs():
    """5 份链文档必须与当前 spec 完全一致。"""
    r = subprocess.run([sys.executable, GEN, "--check"],
                       capture_output=True, text=True, cwd=_ROOT)
    assert r.returncode == 0, (
        "链文档与 spec 不同步：\n%s%s\n  重跑：python "
        "analyses/tensor_examples/chains/gen_chain_docs.py"
        % (r.stdout, r.stderr))


def test_every_chain_has_notes():
    """生成器登记**每一条**链都要有手工笔记。

    链名从 `gen_chain_docs.CHAINS` 读，不硬编码 —— 否则新增链时这个测试
    会静默漏检（它本来就是为了防"文档缺叙述"）。
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen_chain_docs", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    notes = os.path.join(_ROOT, "analyses", "tensor_examples", "chains",
                         "notes")
    have = {f[:-len(".notes.md")] for f in os.listdir(notes)
            if f.endswith(".notes.md")}
    want = {name for name, _sp, _hd in mod.CHAINS}
    missing = sorted(want - have)
    assert not missing, "缺 notes/<chain>.notes.md：%s" % ", ".join(missing)
    # 生成物也要在盘上
    for name in want:
        assert os.path.exists(os.path.join(
            _ROOT, "analyses", "tensor_examples", "chains",
            "%s.md" % name)), "缺 chains/%s.md（跑 gen_chain_docs.py）" % name


def test_index_lists_every_chain():
    """`chains/README.md` 的「一张表看全」必须列出**每一条**链。

    这张索引表是**手工维护**的（生成器只管 `<chain>.md`，不管 README），
    所以最容易悄悄漏行 —— 加了新链、跑完生成器、README 却忘了加。
    2026-10-20 就漏过 7 条（例 1/2 的 ex1_*/ex2_*），所以补这条守卫。
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen_chain_docs", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    idx = os.path.join(_ROOT, "analyses", "tensor_examples", "chains",
                       "README.md")
    text = open(idx, encoding="utf-8").read()
    missing = sorted(name for name, _sp, _hd in mod.CHAINS
                     if "(%s.md)" % name not in text)
    assert not missing, (
        "chains/README.md 的索引表漏了：%s\n"
        "（新链要手工加进「一张表看全」）" % ", ".join(missing))


def test_no_broken_links():
    """生成物里的相对链接必须都能解析（曾因手数 `../` 少一层断过 5 条）。"""
    import glob
    import re
    base = os.path.join(_ROOT, "analyses", "tensor_examples", "chains")
    bad = []
    for f in sorted(glob.glob(os.path.join(base, "*.md"))):
        for m in re.finditer(r"\]\(([^)#]+?)(?:#[^)]*)?\)", open(f).read()):
            t = m.group(1)
            if t.startswith(("http://", "https://")):
                continue
            if not os.path.exists(os.path.normpath(os.path.join(base, t))):
                bad.append("%s -> %s" % (os.path.basename(f), t))
    assert not bad, "断链：\n  " + "\n  ".join(bad)


def test_check_detects_stale_doc():
    """反向验证：故意改一份文档，`--check` 必须报不同步。

    这条是防"检查器永远返回 OK"的假绿。
    """
    import shutil
    import tempfile

    dst = os.path.join(_ROOT, "analyses", "tensor_examples", "chains",
                       "phif2.md")
    with tempfile.TemporaryDirectory() as td:
        bak = os.path.join(td, "phif2.md")
        shutil.copyfile(dst, bak)
        try:
            with open(dst, "a", encoding="utf-8") as f:
                f.write("\n<!-- 故意弄脏 -->\n")
            r = subprocess.run([sys.executable, GEN, "--check"],
                               capture_output=True, text=True, cwd=_ROOT)
            assert r.returncode != 0, "改脏了文档，--check 却说 OK"
            assert "不同步" in r.stdout, r.stdout
        finally:
            shutil.copyfile(bak, dst)
    # 复原后必须回到 OK
    r = subprocess.run([sys.executable, GEN, "--check"],
                       capture_output=True, text=True, cwd=_ROOT)
    assert r.returncode == 0, r.stdout + r.stderr


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    ok = 0
    for n, f in tests:
        try:
            f()
            print("  OK   %s" % n)
            ok += 1
        except AssertionError as e:
            print("  FAIL %s: %s" % (n, e))
        except Exception as e:                                   # noqa: BLE001
            print("  ERR  %s: %s: %s" % (n, type(e).__name__, e))
    print("%s: %d/%d" % ("PASS" if ok == len(tests) else "**FAIL**",
                         ok, len(tests)))
    return 0 if ok == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
