#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数值后端：**jax 优先，无 jax 时自动回退 numpy**。

为什么要有这一层
----------------
生产把 tensor cache 的预计算与拟合统一在 jax 上（GPU 相对 CPU 有 5–21× 的
实测提升，见 `documentation/tensor_cache_generation.md` §7）。但生成器
（`python -m agent.tensor emit/check/show`）本身不该被 jax 绑住：它只读 TOML、
拼源码，在没有 jax 的机器上也要能跑。

于是所有数值内核只认这一层的 `xp`：

    xp.einsum / xp.stack / ...    数组命名空间（jax.numpy 或 numpy）
    jit(fn)                       jax.jit；numpy 下是恒等装饰器
    to_numpy(x) / to_device(x)    跨后端搬运
    is_tracer(x)                  是否处于 jit 追踪中
    einsum(...)                   后端感知的收缩（numpy 侧带 optimal 路径缓存）

选择规则
--------
    TENSOR_BACKEND=auto     （默认）有 jax 就用 jax，否则 numpy
    TENSOR_BACKEND=jax      强制 jax；没有 jax 直接报错（不静默降级）
    TENSOR_BACKEND=numpy    强制 numpy（对拍 / 无 GPU 环境）

jax 一旦选定就打开 `jax_enable_x64` —— float32 会让相对误差掉到 1e-7，而生产
cache 的验收线是 1e-12。**必须在创建任何 jax 数组之前设置**，所以本模块把
配置收敛到 `configure()`，并在第一次触碰 `xp` 时惰性执行。

惰性：`emit` / `check` / `show` 不碰数值代码，因此不会 import jax（省掉 ~1 s
启动，也让无 jax 的机器能出代码）。

产物自足
--------
`emitter` 会把本文件原样拷进产物目录（与 `covariant.py` / `runtime.py` 一起），
所以生成的脚本不依赖 `agent.tensor` 包。
"""
from __future__ import annotations

import os
import threading

import numpy as onp

__all__ = ["xp", "jit", "to_numpy", "to_device", "is_tracer", "backend_name",
           "configure", "einsum", "is_jax", "register_cache_clearer"]

_ENV = "TENSOR_BACKEND"
_LOCK = threading.RLock()

_state = {"backend": None, "jax": None, "ns": None}
_clearers = []


# ------------------------------------------------------------------ 后端选择
def _pick(requested):
    """把请求解析成实际后端名；`jax` 请求失败时**报错**而不是降级。"""
    if requested in ("auto", "jax"):
        try:
            import jax  # noqa: F401
            return "jax"
        except ImportError:
            if requested == "jax":
                raise ImportError(
                    "TENSOR_BACKEND=jax 但环境里没有 jax。装 jax 或用 "
                    "TENSOR_BACKEND=numpy 显式走 numpy 回退。")
            return "numpy"
    if requested == "numpy":
        return "numpy"
    raise ValueError("TENSOR_BACKEND 只能是 auto / jax / numpy，收到 %r"
                     % (requested,))


def configure(backend=None, force=False):
    """确定后端（幂等），jax 侧顺带打开 float64。返回后端名。

    Parameters
    ----------
    backend : str | None
        None -> 读环境变量 `TENSOR_BACKEND`（默认 auto）；显式给值时忽略环境。
    force : bool
        即使已经配置过也重新解析（测试里在同一进程内切换后端用）。

    切换后端会清空各内核注册的记忆化缓存（投影算子、收缩路径），因为它们
    存的是后端相关的数组。
    """
    with _LOCK:
        requested = (backend or os.environ.get(_ENV, "auto")).strip().lower()
        if _state["backend"] is not None and not force:
            return _state["backend"]

        name = _pick(requested)
        if name == "jax":
            # 必须在 import jax 之前设，否则 XLA 会预占整块显存。
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            import jax
            jax.config.update("jax_enable_x64", True)
            _state["jax"] = jax
            _state["ns"] = jax.numpy
        else:
            _state["jax"] = None
            _state["ns"] = onp
        _state["backend"] = name

        for fn in _clearers:
            try:
                fn()
            except Exception:       # 缓存清理失败不该拖垮配置
                pass
        return name


def register_cache_clearer(fn):
    """登记一个"切换后端时清空"的回调（投影算子缓存等）。"""
    _clearers.append(fn)


def backend_name():
    return configure()


def is_jax():
    return backend_name() == "jax"


def _ns():
    configure()
    return _state["ns"]


# ------------------------------------------------------------------ 数组命名空间
class _LazyNamespace:
    """惰性 `xp`：第一次取属性时才决定后端并 import jax。

    这样模块 import 本身是免费的，`emit`/`check` 不会把 jax 拖进来。
    """

    __slots__ = ()

    def __getattr__(self, name):
        return getattr(_ns(), name)

    def __dir__(self):
        return dir(_ns())

    def __repr__(self):
        try:
            return "<xp backend=%s>" % backend_name()
        except Exception:
            return "<xp backend=unresolved>"


xp = _LazyNamespace()


# ------------------------------------------------------------------ jit / 搬运
def jit(fn=None, **kwargs):
    """`jax.jit`；numpy 后端下是恒等装饰器。可当 `jit(f)` 或 `@jit` 用。"""
    if not is_jax():
        if fn is None:
            return lambda f: f
        return fn
    j = _state["jax"]
    if fn is None:
        return lambda f: j.jit(f, **kwargs)
    return j.jit(fn, **kwargs)


def is_tracer(x):
    """x 是否处于 jit 追踪中（numpy 后端恒为 False）。"""
    if not is_jax():
        return False
    return isinstance(x, _state["jax"].core.Tracer)


def to_numpy(x):
    """搬回 host numpy（jax 下做 device_get）。"""
    if is_jax():
        return onp.asarray(_state["jax"].device_get(x))
    return onp.asarray(x)


def to_device(x):
    """搬到计算设备（jax 下 device_put；numpy 下就是 asarray）。"""
    if is_jax():
        return _state["jax"].device_put(x)
    return onp.asarray(x)


def tree_to_numpy(tree):
    """对 dict/list/tuple 递归做 `to_numpy`。"""
    if isinstance(tree, dict):
        return {k: tree_to_numpy(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(tree_to_numpy(v) for v in tree)
    return to_numpy(tree)


# ------------------------------------------------------------------ 收缩
_EINSUM_PATHS = {}


def _np_einsum(subscripts, *operands):
    """numpy 侧的 `einsum` + 缓存的 optimal 收缩路径。

    只对 >=3 个操作数的收缩做路径搜索（2 个操作数时 numpy 已是最优）。
    jax 侧不需要这层：XLA 自己做收缩规划，且 `jnp.einsum` 不接受 numpy
    `einsum_path` 返回的那种路径列表（元素含字符串，会 TypeError）。
    """
    if len(operands) < 3:
        return onp.einsum(subscripts, *operands)
    path = _EINSUM_PATHS.get(subscripts)
    if path is None:
        path, _info = onp.einsum_path(subscripts, *operands, optimize="optimal")
        _EINSUM_PATHS[subscripts] = path
    return onp.einsum(subscripts, *operands, optimize=path)


def einsum(subscripts, *operands):
    """后端感知的 `einsum`。"""
    if is_jax():
        return _state["ns"].einsum(subscripts, *operands)
    return _np_einsum(subscripts, *operands)


def clear_einsum_cache():
    _EINSUM_PATHS.clear()


register_cache_clearer(clear_einsum_cache)
