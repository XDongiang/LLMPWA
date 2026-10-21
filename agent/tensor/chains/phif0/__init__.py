r"""chain.name = "phif0" —— psi(2S) -> phi f_0 -> (K+K-)(pi+pi-)。

f0 自旋为 0，S 只能来自 phi，故 s 恒为 1、l 取偶数。
"""
from .phif001 import phif001  # noqa: F401
from .phif021 import phif021  # noqa: F401

NAME = "phif0"
CHANNELS = ["phif001", "phif021"]
SUFFIX = "0"          # Omega0.npy / K0.npy / M0.npy / g0.npy
LS = [(0, 1), (2, 1)]
