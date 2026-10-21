r"""chain.name = "rhopi" —— psi(2S) -> rho' pi -> (phi pi) pi -> (K+K-)(pi+pi-)。

中间共振态 X = rho'(1^{--})，由 phi + pi 重建。psi->rho'pi 与 rho'->phi pi
都限于 P 波，所以只有**一个**独立振幅（文档 373 行）；`_1`/`_2` 两条道是
"哪一个 pi 是旁观者"的两种取法，不是两个独立分波。

粒子编号（与文档一致）：K+=1，K-=2，pi+=3，pi-=4

    u_rho_1   rho' = phi + Pim = (1,2,4)   旁观者 = Pip = 粒子 3
    u_rho_2   rho' = phi + Pip = (1,2,3)   旁观者 = Pim = 粒子 4

⚠️ `_1`/`_2` 的下标在 rho 与 b1 两个族里**含义相反**（照抄参考实现的命名）：

    rho 族  _1 -> 旁观者是粒子 3 (pi+)
    b1  族  _1 -> 旁观者是粒子 4 (pi-)

改动道函数时请以函数体里的 `rho = ...` / `b1 = ...` 为准，不要靠下标猜。

顶层 (l,s) 恒为 (1,1)（宇称 + 三角唯一给出），故本族不参与
`allowed_ls` 的多道枚举 —— spec 里要设 `check_channels = false`。
"""
from .u_rho_1 import u_rho_1  # noqa: F401
from .u_rho_2 import u_rho_2  # noqa: F401

NAME = "rhopi"
CHANNELS = ["u_rho_1", "u_rho_2"]
LS = [(1, 1), (1, 1)]
