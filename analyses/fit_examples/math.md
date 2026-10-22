# 当前 likelihood 的数学结构

> 本文写的是 `common/likelihood.py` **现在**在算的东西 —— 不是它将来可能算什么。
> 每一条等式都在 `probe/verify_math.py` 里被独立重算过（不拿 `jax.grad` 验
> `jax.grad`），跑一次就知道文档和代码有没有漂移：
>
> ```bash
> cd LLMPWA
> analyses/kk_dis/.venv/bin/python analyses/fit_examples/probe/verify_math.py
> # => 全部通过   （5 个 case/表象组合共 44 项 + 跨表象 7 项 = 51 项）
> ```
>
> **范围**：这里只有"张量缓存 → 振幅 → 强度 → 似然 → 梯度"。没有传播子、
> 没有先验、没有份额罚项 —— 那些是拟合层面的事，本阶段刻意留空（§10）。

---

## 记号

| 符号 | 含义 | 代码 |
|---|---|---|
| $R$ | 共振族（一份 spec） | `Case.families` |
| $\mathcal{C}_R$ | 族 $R$ 的道表，$n_R = \lvert\mathcal{C}_R\rvert$ | `[[channel]]` |
| $e$ | 事例 | 张量的第 1 轴 |
| $i$ | 道 | 张量的第 0 轴 |
| $k$ | 末轴槽位 | 第 2 轴 |
| $N_d, N_{mc}$ | data / MC 事例数 | `--n-data` / `--n-mc` |
| $g_i$ | 道 $i$ 的复耦合 | `const + i·theta` |
| $T_i(e,\mu)$ | 协变张量缓存，$\mu=x,y,z$ | `cartesian` |
| $W_i(e,\lambda)$ | 级联螺旋度缓存，$\lambda=+1,0,-1$ | `helicity_cascade` |

两个约定是**承重**的，写错了不会报错、只会算出错数：

* 协变侧末轴是**洛伦兹指标** $\mu$，横向分量固定取 $(x,y)$ —— 与 $k$ 无关；
* 级联侧末轴是**螺旋度槽位** $\lambda$，横向取 $(+1,-1)$。

---

## 1. 参数化

每道一个复耦合，拆成两个**实**参数：

$$g_i \;=\; c_i + \mathrm{i}\,t_i,\qquad c_i, t_i\in\mathbb{R}$$

拟合向量 $x=(c_1,t_1,c_2,t_2,\dots)$，长度 $2\sum_R n_R$。

**为什么不用复参数直接拟合**：$x\in\mathbb{R}^{2n}$ 时 $`\texttt{jax.grad}`$ 与
$`\texttt{jvp}`$ 都在实空间里，没有复梯度的共轭约定问题。这一点在写 HVP 时
省掉整整一类 bug（README §6 记的那个 `grad_vs_fd_rel ≈ 1.5` 就是共轭约定造成的）。

固定项（`fixed = true`）不参与拟合，但从文件取值参与计算：

$$x_{\text{full}} \;=\; \mathrm{scatter}(x_{\text{free}};\ \text{fixed 项用文件里的值})$$

---

## 2. 振幅

**族内**先对道求和，**族间**相干相加：

$$A_R(e,k)\;=\;\sum_{i\in\mathcal{C}_R} g_i\, W_i(e,k),\qquad
A(e,k)\;=\;\sum_{R} f_R\, A_R(e,k)$$

* $k$ 对所有族是**同一个**末轴（协变侧都是 $\mu$，级联侧都是 $\lambda$）——
  这是"同一批末态、不同共振族能相干"的前提，与 `kk_dis` 的
  `_total_amplitude` 同构；
* $f_R$ 是传播子/线形因子。**当前恒等于 1**（§10 的空位）。

协变侧存的是实张量，但 $g_i$ 是复数，所以 $A_\mu\in\mathbb{C}$。

---

## 3. 横向投影（$e^+e^-$ 初态的强度）

$$I(e)\;=\;\sum_{k\in\mathcal{K}}\bigl\lvert A(e,k)\bigr\rvert^2,
\qquad
\mathcal{K}=\begin{cases}
\{x,y\} & \text{协变}\\ \{+1,-1\} & \text{级联}
\end{cases}$$

**取 $\lvert A\rvert^2$ 而不是 $\mathrm{Re}^2$**：$g_i$ 复，$\mathrm{Re}$ 会丢掉
虚部。在纯实耦合时两者恰好相等 —— 于是这个 bug 会被掩盖，直到你用上复相位。

---

## 4. 跨表象的等价性

$\mathrm{EPS\_A}$ 把协变张量线性地变成螺旋度振幅：

$$W_i(e,\lambda) \;=\; \sum_{\mu} (\mathrm{EPS\_A})_{\lambda\mu}\, T_i(e,\mu),
\qquad
\mathrm{EPS\_A}=
\begin{pmatrix}
\tfrac{1}{\sqrt2} & -\tfrac{\mathrm{i}}{\sqrt2} & 0\\
0 & 0 & 1\\
-\tfrac{1}{\sqrt2} & -\tfrac{\mathrm{i}}{\sqrt2} & 0
\end{pmatrix}$$

行序 $\lambda=+1,0,-1$。因为变换是**逐个 $e$ 线性**的，对 $i$ 的求和可以交换：

$$I_{\text{casc}}(e)
= \sum_{\lambda=\pm1}\Bigl\lvert\sum_i g_i W_i(e,\lambda)\Bigr\rvert^2
= \sum_{\mu=x,y}\Bigl\lvert\sum_i g_i T_i(e,\mu)\Bigr\rvert^2
= I_{\text{cart}}(e)$$

这是**逐事例的恒等式**，不是统计意义上的近似。实测：

| 场景 | 残差 |
|---|---|
| `psi` 静止 + 中间态在壳（合成数据） | ~1e-15 |
| tf-pwa 四体相空间（中间态离壳） | ~1e-7 |
| `ex3_rhopi`、`ex6_rr`、`ex6_f2s` | 0.3 – 1.0（**形状不同**，恒等式不成立） |

前两档是"同一振幅、只差逐道常数 $c_i$"（§5）；第三档是两个**不同**振幅，
数学上就不该相等（README「已知边界」）。

### 4.1 逐道标定常数

同一个 spec 的两份 cache 不是同一个数组，而是

$$W_i(e,\lambda) \;=\; c_i \cdot C_i(e,\lambda)$$

$c_i$ 是逐道的复常数。从数据反解（最小二乘）：

$$c_i \;=\; \frac{\langle C_i \mid \mathrm{EPS\_A}\,T_i\rangle}{\langle C_i \mid C_i\rangle}$$

`case.py build <case> --only calibrate` 把它写进 `tensors/index.toml`，
`likelihood.py` 在读盘时**乘进级联侧**（把 $c_i$ 乘进张量而不是耦合：
这样拟合出的 $g_i$ 就是"以该表象的 cache 为单位"的耦合）。

> ⚠️ 方向：$c$ 的定义是 $W = c\cdot C$。要让两侧可比，必须让级联侧乘 $c$。
> 乘反方向会差 $c^2$，强度上差 $c^4$ —— 实测能到 $10^6$ 倍，很显眼但极易被
> 当成"物理不同"。

**标定残差本身是判据**：$\lvert W_i - c_i C_i\rvert/\lvert W_i\rvert$ 的最大值

| 残差 | 结论 | 处理 |
|---|---|---|
| $\lesssim10^{-9}$ | 逐位同一振幅 | 标定 |
| $10^{-9}\sim10^{-6}$ | 同一振幅，残差来自中间态离壳 | 标定 |
| $>10^{-6}$ | **形状不同** | **不标定**，标为不可比并排除 |

第三档要是硬标定，等于用最小二乘把一个真实差异"摊平"成一个假的常数因子。
实测 `psi_phi4pi` 那样会得到 $I_{\text{rel}}\sim10^4$ 的纯噪声。

---

## 5. 似然

扩展似然（extended），**当前只有两项**：

$$\boxed{
-\ln\mathcal{L}(x)\;=\;
\underbrace{\sum_{e\in\text{data}}\bigl(-\ln I(e)\bigr)}_{\text{形状项}}
\;+\;
\underbrace{N_d\,\ln\!\Bigl(\frac{1}{N_{mc}}\sum_{e\in\text{mc}} I(e)\Bigr)}_{\text{归一化项}}
}$$

代码里两处都有 `max(·, tiny)` 保护（$I$ 可能下溢到 0）。

* **形状项**：data 上的 $-\ln I$。$I$ 是"未归一化的强度"，绝对大小无关，
  只有逐事例的相对分布有信息；
* **归一化项**：$N_d\ln\langle I\rangle_{mc}$。$\langle I\rangle_{mc}$ 是
  MC 上的平均强度，充当积分 $\int I\,\mathrm{d}\Phi$ 的估计 —— 这就是
  "接受-拒绝"归一化的标准做法；
* 没有因子 2（不是 $-\ln$ 似然的两倍），所以 $\partial^2/\partial x^2$ 的
  曲率可以直接当 Fisher 信息用。

### 5.1 一个重要的退化：默认 data 与 MC 是同一批事例

`--n-data` / `--n-mc` 默认取 0，代码取 $N_d=N_{mc}=N$ 整份缓存，
于是 `t_data` 与 `t_mc` **逐位相同**。此时

$$-\ln\mathcal{L}\;=\;\sum_e\bigl(\ln I(e)\bigr)+N\ln\bar I
\;=\;N\Bigl[\ln \bar I-\overline{\ln I}\Bigr]
\;=\;N\ln\frac{\mathrm{AM}(I)}{\mathrm{GM}(I)}$$

即 $N$ 倍的 "算术平均 / 几何平均" 的对数。由 AM–GM 不等式，这个量 $\ge 0$，
且**只依赖 $e$ 上的分布**、与整体缩放无关。

实测（`psi_kkpipi`，$N=20000$）：$-\ln\mathcal{L}=19744.5924$，
$N\ln(\mathrm{AM}/\mathrm{GM})=19744.5924$，$-\ln\mathcal{L}/N=0.98723$。

> 这不是 bug，但它意味着**默认参数下的似然不是在"拟合数据"**，而是在最小化
> 一个自归一化的形状差异。演示链路够用；真要拟合就得传不重叠的
> `--n-data` / `--n-mc`（代码切的是两段不重叠的区间）。

---

## 6. 整体缩放与相位对称性

$I$ 是 $A$ 的二次型，$A$ 对 $g$ 线性，所以对**任意复数** $s$：

$$g\;\to\;s\,g\ \Longrightarrow\
A\to sA\ \Longrightarrow\
I\to\lvert s\rvert^2 I$$

形状项变化 $-N_d\ln\lvert s\rvert^2$，归一化项变化 $+N_d\ln\lvert s\rvert^2$，
**精确抵消**：

$$-\ln\mathcal{L}(s\,g)\;=\;-\ln\mathcal{L}(g)\qquad \forall s\in\mathbb{C}$$

实测：$s$ 从 $10^{-4}$ 到 $10^{3}$、相位任意，$\lvert\Delta\mathcal{L}\rvert\le10^{-10}$。

这带来两个**精确的零方向**（不是数值病态，是解析的）：

| 方向 | 实参表示 | 含义 |
|---|---|---|
| $\partial\mathcal{L}/\partial\ln\lvert g\rvert=0$ | $g\cdot x$（$x$ 为实参向量） | 整体模长 |
| $\partial\mathcal{L}/\partial\arg g=0$ | $J\,g$，$J$ 为逐道 $90^\circ$ 旋转 | 整体相位 |

因此 $g^\top H g=0$、$(Jg)^\top H(Jg)=0$，Hessian 有**至少 2 个零本征值**，
$-\ln\mathcal{L}$ 在这些方向上是常数。数值验证（`verify_math.py`）：

| 量 | 典型值 |
|---|---|
| $\nabla\mathcal{L}\cdot x$ | $\le10^{-11}$ |
| $\nabla\mathcal{L}\cdot Jx$ | $\le10^{-11}$ |
| $x^\top H x$ | $\le10^{-11}$（而 $\lvert H\rvert_{\max}\sim10^{5}$）|

**对拟合的后果**：这是一个**规范冗余**，不是一个平坦的物理方向。

* 用 L-BFGS-B 之类只用梯度的优化器：梯度在这些方向上为 0，不会被它带偏，
  但收敛后 $g$ 的整体模长/相位是没有意义的 —— 只有**比值** $g_i/g_j$
  （等价地：道的相对份额与相对相位）有物理意义；
* 用 Newton-CG / HVP 求解时，Hessian 奇异，必须加阻尼 $\lambda I$ 或显式
  固定一个道（把它的 $c,t$ 冻结）；
* 报"参数误差"时要用伪逆，或者先投影掉这两个方向，否则协方差矩阵发散。

> ⚠️ 需要注意的还有 $\lvert s\rvert$ 之外的**部分**对称性：如果某个族只有
> 一个道，或某两个族的道表完全成比例，就会出现**更多的**零方向。上面的
> 2 个是任何 case 都有的下界。

---

## 7. 梯度闭式（`jax.grad` 就是在算这个）

记横向槽位集合 $\mathcal{K}$，$A^d_e=A(e,k)$ 在 data 上，$A^{mc}_e$ 在 MC 上。

**第一步**：$\partial I(e)/\partial g_i^*$。由 $I=\sum_k A_kA_k^*$、
$A_k=\sum_i g_iW_{i,k}$，有 $\partial A_k/\partial g_i^*=0$、
$\partial A_k^*/\partial g_i^*=W_{i,k}^*$：

$$\frac{\partial I(e)}{\partial g_i^*}\;=\;\sum_{k\in\mathcal{K}} A_k\,W_{i,k}^*
\;\equiv\; u_{e,i}$$

**第二步**：链式。注意 $-\ln\mathcal{L}$ 对 $g^*$ 的偏导（Wirtinger，
$\partial/\partial g^*$ 保持 $g$ 不变）：

$$v_i \;=\; \frac{\partial(-\ln\mathcal{L})}{\partial g_i^*}
\;=\;\sum_{e\in d}\frac{-1}{I_e}\,u^{d}_{e,i}
\;+\;\frac{N_d}{\langle I\rangle_{mc}}\cdot\frac{1}{N_{mc}}
\sum_{e\in mc}u^{mc}_{e,i}$$

**第三步**：回到实参数。$g_i=c_i+\mathrm{i}t_i$，$\partial g_i/\partial c_i=1$、
$\partial g_i/\partial t_i=\mathrm{i}$，于是

$$\boxed{
\frac{\partial(-\ln\mathcal{L})}{\partial c_i}=2\,\mathrm{Re}\,v_i,
\qquad
\frac{\partial(-\ln\mathcal{L})}{\partial t_i}=-2\,\mathrm{Im}\,v_i
}$$

因子 2 来自 $g$ 与 $g^*$ 两侧各贡献一次；负号来自 $\mathrm{i}$ 的共轭。
**这两处是手推时最容易错的地方**，而且它们的"暴露条件"不一样：

| 错误 | `psi_omega_kk`（单族） | `psi_kkpipi`（4 族） |
|---|---|---|
| 符号写成 $+2\mathrm{Im}$ | rel $=1.59$ ✗ | rel $=0.62$ ✗ |
| $A$ 用**本族**而不是**总**振幅 | rel $=9.9\times10^{-15}$ ✓（假通过） | rel $=0.56$ ✗ |

第二行是真正危险的：单族时"本族振幅 $\equiv$ 总振幅"，错误被**结构性地**
掩盖，只有多族 case 能发现。所以 **`psi_omega_kk` 不能用来验梯度**，
必须至少用一个多族 case。`verify_math.py` 因此固定跑 `psi_kkpipi`
（4 族）与 `psi_phi4pi`（3 族）。

实测闭式 vs `jax.grad`：

| case / 表象 | $\max\lvert g_{\text{cl}} - g_{\text{jax}}\rvert/\max\lvert g_{\text{jax}}\rvert$ |
|---|---|
| `psi_kkpipi` / cartesian | 3.1e-14 |
| `psi_kkpipi` / helicity_cascade | 3.3e-14 |
| `psi_phi4pi` / helicity_cascade | 7.7e-15 |
| `psi_omega_kk` / cartesian | 9.8e-15 |
| `psi_pipipi0` / cartesian | 3.1e-15 |

---

## 8. 为什么梯度自检要和中心差分比

`jax.grad` 的闭式对上了，只说明**导数算对了**；还得确认 `jax.grad` 的实现
没问题。所以 `--check` 做两层：

1. **grad vs FD**：$\bigl(g(x+\epsilon v)-g(x-\epsilon v)\bigr)/2\epsilon$，$\epsilon=10^{-6}$；
2. **HVP vs FD**：$\mathrm{jvp}(\nabla\mathcal{L},v)$ 对 $\nabla\mathcal{L}$ 再做差分。

参数是实数，所以第 1 步直接对 $x$ 扰动即可，不需要分别扰 Re/Im ——
那是"复参数直接拟合"才需要的额外处理（见注 2）。

实测全部 case：grad $\le4.6\times10^{-9}$，HVP $\le6.1\times10^{-6}$。

> **注 1**：HVP 的误差比 grad 大三个数量级，原因不在实现，而在**差分本身
> 的条件数**。HVP 是二阶量，中心差分的截断误差与舍入误差（$\propto\eta/\epsilon$）
> 都被再放大一次；$6.1\times10^{-6}$ 已经是 $\epsilon=10^{-6}$ 下的合理值。
>
> **注 2**：README §4.2 记的那条教训（复参数 FD 必须分别扰 Re/Im，且
> `jax.grad` 返回 $\mathrm{d}f/\mathrm{d}\mathrm{Re}-\mathrm{i}\,\mathrm{d}f/\mathrm{d}\mathrm{Im}$）
> 是**早期原型** `probe/proto_likelihood.py` 用复参数直接拟合时踩的。
> 现在 $g_i=c_i+\mathrm{i}t_i$ 从一开始就是实参数，这一类问题在结构上不存在；
> 代价是参数数目翻倍（$n_{\text{free}}=2\sum_R n_R$）。

---

## 9. 与 `kk_dis` 生产似然的差别

| 项 | `kk_dis/run/likelihood_function.py` | 本文 |
|---|---|---|
| 形状项 $-\sum\ln I$ | ✓ | ✓ |
| 归一化项 $N_d\ln\langle I\rangle_{mc}$ | ✓ | ✓ |
| 份额罚项 $1000(\sum f_R-1.1)^2$ | ✓（可关） | ✗ |
| 高斯先验 $\sum(x-x_0)^2/2\sigma^2$ | ✓（可关） | ✗ |
| 传播子 $f_R(s)$（BW / flatté） | ✓ | 恒等 1 |
| 逐族振幅 | 四类（$f_0,f_2,\dots$） | 任意族（读 spec） |

所以本文件的 $\mathcal{L}$ 是 `kk_dis` 的**结构骨架**：把中间三项摘掉、
把第四项换成占位。摘掉的那三项都落在 `_family_propagator` 与 `likelihood()`
两个点上，接回来不需要动张量缓存与梯度推导。

---

## 10. 空位在哪（下次要接什么）

| 空位 | 位置 | 接法 |
|---|---|---|
| 传播子 $f_R(s)$ | `_family_propagator(fam, s=None)` | 需要逐事例的 $s$。现在签名里没有 $s$，所以只能返回常数 —— 这是**故意的**：先证明链路，再谈线形 |
| 先验 | `likelihood()` | 加 $\sum(x-x_0)^2/2\sigma^2$；`free_params.toml` 已有 `error` 字段（现在是占位 0.001） |
| 份额罚项 | `likelihood()` | 需要逐族强度 $\int I_R$ 与总强度之比 |
| 冻结 | `free_params.toml` 的 `fixed` | 现在全 `false`；固定一个道同时也是**消除 §6 规范冗余**的手段 |

---

## 附：符号一览

$$\begin{aligned}
A_R(e,k) &= \sum_{i\in\mathcal{C}_R} g_i\, W_i(e,k) &&\text{族内对道求和}\\
A(e,k) &= \sum_R f_R\, A_R(e,k) &&\text{族间相干}\\
I(e) &= \sum_{k\in\mathcal{K}} \lvert A(e,k)\rvert^2 &&\text{横向强度}\\
-\ln\mathcal{L} &= -\sum_{d}\ln I + N_d\ln\langle I\rangle_{mc} &&\text{似然}\\
v_i &= \partial(-\ln\mathcal{L})/\partial g_i^* &&\text{Wirtinger 梯度}\\
\partial_{c_i} &= 2\mathrm{Re}\,v_i,\quad \partial_{t_i}=-2\mathrm{Im}\,v_i &&\text{实梯度}
\end{aligned}$$
