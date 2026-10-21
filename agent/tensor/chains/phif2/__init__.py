r"""chain.name = "phif2" —— psi(2S) -> phi f_2 -> (K+K-)(pi+pi-)。"""
from .phif201 import phif201  # noqa: F401
from .phif221 import phif221  # noqa: F401
from .phif222 import phif222  # noqa: F401
from .phif223 import phif223  # noqa: F401
from .phif243 import phif243  # noqa: F401

NAME = "phif2"
CHANNELS = ["phif201", "phif221", "phif222", "phif223", "phif243"]
SUFFIX = ""           # Omega.npy / K.npy / M.npy / g.npy
LS = [(0, 1), (2, 1), (2, 2), (2, 3), (4, 3)]
