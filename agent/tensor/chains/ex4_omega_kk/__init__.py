r"""chain.name = "ex4_omegakk" —— $\psi(2S)\to\omega K^+K^-\to(\pi^+\pi^-\pi^0)K^+K^-$。

文档 `document/ψ衰变到介子的分波分析协变张量形式.md` 418–430 行，例 4。

拓扑（**深度 3**）：

    psi2s -> omega + X          X = f_0 或 f_2（KK 那一对）
    omega -> rho + Pi0          （三条 ρ 置换道，写在 omega^mu 里）
    X     -> Kp + Km

⚠️ 文档的 $\omega^\mu$ 是一个**单节点三体**对象：它把 $\pi^0\pi^+\pi^-$ 三个
动量一起塞进 $\epsilon^{\mu\nu\lambda\sigma}$。本链把它写成一条**手写协变道**
（`u_omega_*` 里的 `_omega_mu`），因此 `tree` 只是为了让四动量名字能解析，
物理上并不是"$\omega\to\rho\pi$ 的两体级联" —— 两体节点的 ε 恒为零
（见 `analyses/tensor_examples/chains/notes/ex4_omega.notes.md`）。

这条链**只有协变/螺度表象**：三体节点的 tree 表示不了，级联内核算不了。
"""
