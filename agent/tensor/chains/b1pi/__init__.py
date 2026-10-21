r"""chain.name = "b1pi" —— psi(2S) -> b1 pi -> (phi pi) pi -> (K+K-)(pi+pi-)。

中间共振态 X = b1(1^{+-})，由 phi + pi 重建。psi->b1pi 与 b1->phi pi 都可以
既有 S 波又有 D 波，所以有**四个**独立振幅（文档 391–416 行）：

    u_SS  psi->b1pi S 波  x  b1->phi pi S 波   gtilde_(123) . t1_(12)
    u_SD  psi->b1pi S 波  x  b1->phi pi D 波   t2_(b1 phi pi) . t1_(12)
    u_DS  psi->b1pi D 波  x  b1->phi pi S 波   T2_(psi b1 pi) . gtilde . t1_(12)
    u_DD  psi->b1pi D 波  x  b1->phi pi D 波   T2_(psi b1 pi) . t2_(b1 phi pi) . t1_(12)

粒子编号（与文档一致）：K+=1，K-=2，pi+=3，pi-=4

    u_*_1   b1 = phi + Pip = (1,2,3)   旁观者 = Pim = 粒子 4
    u_*_2   b1 = phi + Pim = (1,2,4)   旁观者 = Pip = 粒子 3

⚠️ `_1`/`_2` 的下标在 rho 与 b1 两个族里**含义相反**（照抄参考实现的命名）：

    rho 族  _1 -> 旁观者是粒子 3 (pi+)
    b1  族  _1 -> 旁观者是粒子 4 (pi-)

顶层 (l,s) 只有 (0,1) 与 (2,1) 两组，而道有 4 条 —— **(l,s) 不是这里的完备标签**，
完备标签是"顶层波 x 内层波"（SS/SD/DS/DD）。所以 spec 里要设
`check_channels = false`，并且本族**不能**用 omega 表象（omega 的 CG 矩阵
需要 (l,s) 唯一确定一条道）。
"""
from .u_DD_1 import u_DD_1  # noqa: F401
from .u_DD_2 import u_DD_2  # noqa: F401
from .u_DS_1 import u_DS_1  # noqa: F401
from .u_DS_2 import u_DS_2  # noqa: F401
from .u_SD_1 import u_SD_1  # noqa: F401
from .u_SD_2 import u_SD_2  # noqa: F401
from .u_SS_1 import u_SS_1  # noqa: F401
from .u_SS_2 import u_SS_2  # noqa: F401

NAME = "b1pi"
CHANNELS = ["u_SS_1", "u_SS_2", "u_SD_1", "u_SD_2",
            "u_DS_1", "u_DS_2", "u_DD_1", "u_DD_2"]
# 顶层 (l,s)：SS/DS 是 S 波 (0,1)，SD/DD 是 D 波 (2,1)
LS = [(0, 1), (0, 1), (2, 1), (2, 1), (0, 1), (0, 1), (2, 1), (2, 1)]
