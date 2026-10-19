### 这条链是例 5 的 K*₂ 部分

$\psi\to K^+\pi^-K^-\pi^+$，文档 431–623 行。拓扑
$\psi\to K^*_2\pi\to(\phi\pi)\pi\to(K^+K^-)(\pi\pi)$ —— **深度 3**，
与 `b1pi` 同构（把 $b_1$ 换成 $K^*_2$）。

10 条道已从参考实现 `Tensor/FUNC.py` 逐式移植到
[`agent/tensor/chains/kstpi/`](../../../agent/tensor/chains/kstpi/)，
**与 jax 逐式对拍通过**（rel ≈ 1e-15，见
[`probes/verify_ex5_port.py`](../probes/verify_ex5_port.py)）。

### 移植时踩到的坑：jax 默认 float32

第一次对拍 10/10 全部"失败"，rel 在 1e-6…1e-8。看着像移植错了，
其实是 **jax 默认 float32**（$\varepsilon\approx1.2\times10^{-7}$）。

必须在**导入 `FUNC` 之前**：

```python
import jax
jax.config.update("jax_enable_x64", True)
```

（`test_covariant_vs_jax.py` 里早有这一步，我漏抄了。）
打开 x64 后立刻 10/10 到 1e-15。

⚠️ **这个坑同样污染了级联可表示性探针** —— 见 §"级联判定"。

### 为什么拆成两份 spec

$K^*_0$ 的 $J=0$ 与 $K^*_2$ 的 $J=2$ **冲突**：一个 `[particles.Kstar]`
条目只能有一个 $J$。协变道函数本身不读 $J$（公式硬编码），所以"能不能算"
不受影响，但拓扑校验需要一个确定的 $J$，只能拆。

本文件是 $K^*_2$ 的 8 条；$K^*_0$ 的 2 条在
[`ex5_kst0`](ex5_kst0.md)。

### 与 `b1pi` 同样的设计瑕疵

8 条道是**两棵树**（旁观者 $\pi$ 不同），靠逐道 `tree` 覆盖塞进一份 TOML。
协变张量道函数完全不读树，所以能共存；但真实拟合里"两棵树是否共享
传播子/参数"仍未处理。见 [`b1pi`](b1pi.md) 的同名小节。

### 级联判定：本链 4 / 8 命中

用与 `b1pi` 相同的手法穷举「树 × 顶层 $(l,s)$ × 内层 $(l,s)$」，
判据是"$W$ 是否落在该 $K^*$ 自旋下全部允许 $(l,s)$ 组合张成的子空间里"：

```bash
python analyses/tensor_examples/probes/probe_ex5_identify.py
python analyses/tensor_examples/probes/probe_ex5_identify.py --control
```

| 道 | 顶层 $(l,s)$ | 内层 $(l,s)$ | rel | 结论 |
|---|---|---|---|---|
| `u_PF_1` | (1,2) | (3,1) | 1.8e-14 | ✅ 是级联 |
| `u_PF_2` | (1,2) | (3,1) | 3.7e-15 | ✅ 是级联 |
| `u_FF_1` | (3,2) | (3,1) | 6.3e-15 | ✅ 是级联 |
| `u_FF_2` | (3,2) | (3,1) | 3.7e-15 | ✅ 是级联 |
| `u_PP_1` | (1,2) | (1,1) | 6.7e-03 | ❌ 不是 |
| `u_PP_2` | (1,2) | (1,1) | 1.7e-01 | ❌ 不是 |
| `u_FP_1` | (3,2) | (1,1) | 5.7e-03 | ❌ 不是 |
| `u_FP_2` | (3,2) | (1,1) | 2.3e-01 | ❌ 不是 |

**本链 4 / 8。** 规律很清楚：内层是 **F 波**（$l=3$）的四条全部精确命中；
内层是 **P 波**（$l=1$）且 $K^*$ 自旋为 2 的四条落在级联子空间之外。

⚠️ 别把两个数搞混：探针输出的是**例 5 全部 10 条道**的结果（**6 / 10**），
它把 `ex5_kst0` 的 2 条也算进去了。本链只有 8 条，所以是 **4 / 8**。
$K^*_0$ 那 2 条见 [`ex5_kst0`](ex5_kst0.md)。

也就是说：$K^*_0$ 的内层**也是** P 波却命中 —— 所以不是"P 波一律不行"，
而是 $(J_{K^*}=2,\ l_{\text{内}}=1)$ 这个特定组合不行。

#### 阳性对照（这个判据可信的前提）

同一套机器跑 `b1pi` 的 8 条道：**8/8 到 6e-15**。
所以"4/8"不是探针坏了 —— 见
[`probes/probe_ex5_identify.py`](../probes/probe_ex5_identify.py) 的 `control()`。

#### 已知的坑（会伪造出"未命中"）

探针第一版**没有**打开 jax x64，于是拿 float32 的参考实现比 float64 的
级联，得到一堆 1e-6…1e-9 的**假未命中**，还伪造出一个"`_1` 族全败、
`_2` 族全中"的假不对称。改用已移植的协变内核（本仓库、float64）后
不对称消失。

⚠️ **教训**：跨库对拍时，精度设置是承重约定。

### 还有什么问题

| 项 | 状态 |
|---|---|
| **$u_{PP}$/$u_{FP}$ 为何不是级联** | ❓ **未解决**。四条失败的道都含 $P^{(2)}_{\psi}$ 横向投影 + 内层 P 波。是"投影算符带来额外张量结构"还是别的，未定 |
| 与文档 Eq 的逐条对应 | ⚠️ **未敲定**。文档 431–623 行给的是 $\langle K^*K^*_J|LS\rangle$ 一族，而参考实现按自己的 $P/F$、$l/r$ 命名。移植以**参考实现为准** |
| $\psi\to K^*_2\pi$ 的宇称 | ⚠️ spec 里写 $K^*_2$ 为 $2^-$（宇称+三角要求 $l$ 为奇数）。若实际取 $2^+$（$l$ 偶），道表要重列 |
| $K^*$ 质量 | ⚠️ spec 写 1.43，**只是占位**；协变道函数对任意四动量都成立 |
| 真实样本 / 拟合侧 | ❌ 未做（与例 3 相同） |
| 节点级 Eq.52 消融 | ❌ 未做 |

### 复现

```bash
cd <LLMPWA>

# 移植正确性（需要 jax + Tensor/，注意 x64）
analyses/kk_dis/.venv/bin/python analyses/tensor_examples/probes/verify_ex5_port.py

# 纯 numpy 回归（永远能跑）
python agent/tensor/tests/test_ex5_channels.py

# 级联可表示性 + 阳性对照
python analyses/tensor_examples/probes/probe_ex5_identify.py
python analyses/tensor_examples/probes/probe_ex5_identify.py --control

# 端到端生成
python -m agent.tensor check analyses/tensor_examples/specs/ex5_kst2.toml
python -m agent.tensor emit  analyses/tensor_examples/specs/ex5_kst2.toml
```

### 相关

* [链 `ex5_kst0`](ex5_kst0.md) —— 同一移植的 $J=0$ 部分
* [链 `b1pi`](b1pi.md) —— 同构拓扑，级联验收的主力
* [链索引](README.md)
