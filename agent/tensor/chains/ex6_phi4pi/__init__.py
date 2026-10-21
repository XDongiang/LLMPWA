r"""chain.name = "ex6_phi4pi" —— $\psi(2S)\to\phi\pi^+\pi^-\pi^+\pi^-\to K^+K^-\pi^+\pi^-\pi^+\pi^-$。

文档 624–795 行，例 6。

末态编号照文档：$K^+\to1$、$K^-\to2$、$\pi^+\to3$、$\pi^-\to4$、$\pi^+\to5$、
$\pi^-\to6$。$f_J$ 来自 $\pi^+\pi^-\pi^+\pi^-$，$\phi$ 来自 $K^+K^-$。

## 三条子分支（全部实现，34 条道）

| 子分支 | 文档行 | 道数 | A1（螺度 vs 级联） | 状态 |
|---|---|---|---|---|
| $f_J\to\sigma\sigma$ | 630–674 | 14 | **6.7e-15** | ✅ 可用 |
| $f_J\to\rho\rho$ | 676–730 | 10 | 0.65–1.09 | ⚠️ **通不过** |
| $f_2\to f_2(1270)\sigma$ | 732–775 | 10 | 0.66–1.25 | ⚠️ **通不过** |

## 为什么 σσ 能精确复现

$\sigma$ 是**标量**，所以文档的

$$
T^{(f_2)\mu\nu}_{(\sigma\sigma)}
=\tilde t^{(2)\mu\nu}_{(\sigma_{34}\sigma_{56})}f^{(\sigma)}_{(34)}f^{(\sigma)}_{(56)}
+(\text{另一配对})
$$

每一项就是一个普通的**两体** $J=2$ 算符：
`T2_ss(frame, a1, a2, b1, b2) = T2_cov(frame, a1+a2, b1+b2)`。

## 为什么 ρρ 与 f₂(1270)σ 通不过（已知，未解决）

公式是**按文档逐式直译**的：

$$
T^{(f_2)\mu\nu}_{(\rho\rho)}=P^{(2)\mu\nu\alpha\beta}(p_{f_2})
\Bigl[\tilde t^{(1)}_{(34)\alpha}\tilde t^{(1)}_{(56)\beta}+\cdots\Bigr],
\qquad
T^{(f_2)\mu\nu}_{(f_2\sigma)}=P^{(2)\mu\nu\alpha\beta}(p_{f_2})
\Bigl[\tilde t^{(2)}_{(34)\alpha\beta}+\tilde t^{(2)}_{(56)\alpha\beta}\Bigr]
$$

（`kernels/covariant.py::T2_rr` / `T2_f2s`。）直译本身没错，但**与逐节点
螺旋度级联对不上**：A1 残差 0.65–1.25，不是常数倍的差别。

原因：$1\otimes1\to2$ 与 $2\otimes0\to2$ 的**无迹化必须在耦合所在系
（$f_2$ 系）里做**，而 $P^{(2)}(p_{f_2})$ 作用在**实验室系**的张量上，
缺少"先 boost 到 $f_2$ 系、去迹、再 boost 回来"这一步的等价物。
这与 `ex5_kst2` 的 `u_PP_*`/`u_FP_*` 是同一类问题（那边已修，见
`agent/tensor/chains/kstpi/CORRECTION.md`），但**嵌套秩 2** 的正确写法
还没推出来。

试过但都不行的变体（残差 0.19–0.57）：加 $\tilde g(p_{f_2})$ 横向投影、
对称化 $u_\mu v_\nu+v_\mu u_\nu$、`T2_from_pair`、把 $P^{(2)}$ 换成
$f_2$ 系里的无迹化。详见
`analyses/tensor_examples/chains/notes/ex6_phi4pi.notes.md`。

## 这 20 条道怎么记账

**不是**"缺道"（道在、能 emit、L1/L2/L3 全过），而是"公式还没对"。
所以在两处显式列成**已知失配**，且两处都断言"豁免项必须确实失配"：

* `probes/verify_all_specs.py::A1_KNOWN`
* `agent/tensor/tests/test_handwritten_complete.py::A1_EXEMPT`

哪天修好了，这两个断言会失败并提醒删表。**不要拿这 20 条道去拟合。**
它们的作用是：把文档公式钉在代码里（改公式能立刻看出影响）+ 给未来
修无迹化参考系留下对照基线。
"""
