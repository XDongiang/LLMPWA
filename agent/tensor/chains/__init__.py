#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`chain.<name>/chain.<channel>.py` 存储层。

用户指定的存储约定（原话）：

    协变张量部分代码可以由 `chain.name/chain.channel.py` 的方式储存，生成时组装。

也就是**一条道一个文件**，按链名分目录：

    chains/
      phif0/                 chain.name = "phif0"（f0 族）
        __init__.py
        phif001.py           chain.channel = "phif001"
        phif021.py
      phif2/
        __init__.py
        phif201.py
        phif221.py
        phif222.py
        phif223.py
        phif243.py

每个 `chain.<channel>.py` 导出一个与文件同名的函数

    def <channel>(psi, phi, f2, Kp, Km, Pip, Pim) -> (N, 4)

`emitter` 在生成时把道表里点到的那些文件拼成一个自足的脚本；`load_chain()`
则在进程内动态装配（测试 / 直跑用）。
"""
from __future__ import annotations

import importlib
import os
import pkgutil

HERE = os.path.dirname(os.path.abspath(__file__))


def list_chains():
    """返回所有 chain 名（子包名），排序。"""
    return sorted(m.name for m in pkgutil.iter_modules([HERE])
                  if m.ispkg)


def list_channels(chain):
    """返回某条链下的全部道名（模块名），排序。"""
    pkg = importlib.import_module("%s.%s" % (__name__, chain))
    d = os.path.dirname(os.path.abspath(pkg.__file__))
    return sorted(m.name for m in pkgutil.iter_modules([d])
                  if not m.ispkg and not m.name.startswith("_"))


def load_channel(chain, channel):
    """导入并返回 `chain.channel.py` 里的同名函数。"""
    mod = importlib.import_module("%s.%s.%s" % (__name__, chain, channel))
    fn = getattr(mod, channel, None)
    if fn is None:
        raise AttributeError("%s.%s 里没有定义 %s()" % (__name__, chain, channel))
    return fn


def load_chain(chain):
    """返回 {channel: callable} —— 生成时组装用的道表。"""
    return {c: load_channel(chain, c) for c in list_channels(chain)}


__all__ = ["list_chains", "list_channels", "load_channel", "load_chain"]
