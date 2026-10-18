# 螺旋度表象振幅的正确表达形式 —— 能力边界与拓展路线

> **这份文档记录一件事**：我们现在的螺旋度表象（`TENSOR_BASIS=helicity`）在
> **生成侧**有一道明确的能力边界，边界之外的道算不出来。下一步的重心是从
> **tf-pwa** 找到螺旋度振幅的一般表达形式，据此把边界推开。
>
> 相关文档：
> [`tensor_cache_generation.md`](tensor_cache_generation.md)（框架用法与验收）、
> [`tensor_examples_feasibility.md`](tensor_examples_feasibility.md)（待移植道清单）。
>
> 结论的判据全部是**实测**的，脚本见本文各节的「复现」行。

---

## 0. 结论速览

| 项 | 结论 |
|---|---|
| 边界在哪一层 | **生成侧**。拟合侧对 `(l,s)` 与树结构完全无感知（§2.4），用 cartesian/helicity 时零改动 |
| 根本限制 | omega 的 cache 格式 `Omega + K + M + g` **无处安放"逐节点 (l,s)"**。这是格式限制，不是算法没写全 |
| 触发条件 | **任一个非顶层节点有 ≥2 个允许的 (l,s)**。ρ′ 只有 1 个 → 可用；b₁ 有 2 个 → 失效 |
| 计数律 | 独立螺旋度振幅数 $=\prod_{\text{节点}} N_{\text{节点}}$，**不是** $N_{\text{顶层}}$ |
| 第二限制 | `omega_from_momenta` 只算 `D0*D1*D2`，深度 >2 时最内层角分布静默丢失（独立问题） |
| tf-pwa 的答案 | **逐节点** $H$（指标 $\lambda_a,\lambda_b,\lambda_c$）+ **链级 einsum** 收缩内线螺旋度；与深度无关 |
| 好消息 | 我们**已经有** `cg_matrix`（与 tf-pwa `get_cg_matrix` 逐元素一致）和树通用的 `chain_angles`；缺的是"别把子粒子螺旋度压成 0" + "链级缩并" |
| 落地口径 | 新内核**直接按道**算 $W_i(e,\lambda_A)$，cache 形状 `(n_ch, N, 3)` complex —— **与今天的 helicity cache 逐位同格式，拟合侧零改动** |

---

## 1. 边界的两个来源（先分清）

「omega 不能用」有两个**互相独立**的原因，修好任一个都不够。

### 1.1 原因一：逐节点 `(l,s)` 无法表达（根本）

omega 的装配式（`agent/tensor/kernels/omega.py:401`）：

$$
W_i(e,\lambda_A)=g_i\,K_i(e)\sum_{a,b}M_i[\lambda_A,a,b]\,\Omega[e,\lambda_A,a,b]
$$

`M_i` 来自 `cg_matrix(ls_list, ...)`（`omega.py:85`），而 `ls_list` 是**顶层节点**的
$(L,S)$ 列表。于是：

* 道的身份由行号 $i$ 承载；
* $M_i$ 只由**顶层** $(L,S)$ 决定；
* 非顶层节点的角动量只以标量 $K_i$（位垒因子）的形式出现。

**后果**：两条道只要顶层 $(L,S)$ 相同，$M_i$ 就逐位相同，$K_i$ 也相同，
于是 $\Omega$ 完全相同 → **道在 cache 里逐位重合**。

对 b₁（`chains/b1pi/`）：SS 与 DS 顶层都是 $(L,S)=(0,1)$；SD 与 DD 顶层都是 $(2,1)$。
实测 8 条道里 **12/28 对**逐位重合。

### 1.2 原因二：只算两层（独立）

`omega_from_momenta`（`omega.py:338`）只取三个因子：

```python
D0 = D_lambda(ang[nb], JA, lam_A, lam_b, lam_c)   # 顶层节点
D1 = _node_decay_d(ang, chain, nb, mom.shape[0])  # 第一个子节点
D2 = _node_decay_d(ang, chain, nc, mom.shape[0])  # 第二个子节点
return (D0 * D1[:, None, :, None] * D2[:, None, None, :])   # omega.py:367
```

链更深时，第三层及以后的角分布**根本不进公式**。

`_node_decay_d`（`omega.py:320`）本身还有第二处硬编码（`:335`）：

```python
return D_lambda(ang[ds[0]], J, LAM_SLOTS[J], [0], [0])[:, :, 0, 0]
#                                               ^^^^^^^^ 子粒子螺旋度被钉死为 0
```

对 $\phi\to K^+K^-$、$f_2\to\pi^+\pi^-$ 这是**对的**（末态自旋为 0）。
对 $b_1\to\phi\pi$ 是**错的**（$\phi$ 自旋 1，$\lambda_\phi$ 被丢掉了）。

### 1.3 实测：把最内层转 37°，omega 毫无反应

绕 $\phi$ 方向转动 $\phi\to K^+K^-$ 的衰变平面（保持 $\phi$ 四动量与总四动量逐位不变）：

```
自检：phi 不变 rel = 1.57e-17
自检：总四动量不变 rel = 7.53e-18

Omega（omega 路线）变化 rel = 0.000e+00     ← 完全不敏感
T（协变路线）变化   rel = 4.877e-01        ← 明显变化
```

**复现**：本文 §7 脚本 1。

---

## 2. 判据与计数律（把边界量化）

### 2.1 决定性判据：自由复系数最小二乘

「omega 能不能表示这条道」有一个**不依赖归一化约定**的判据：
把真实振幅 $A$ 用 omega 的基底 $\{W_k\}$ 做自由复线性组合拟合，看残差。

$$
\min_{c}\ \bigl\|A-\textstyle\sum_k c_k W_k\bigr\|\Big/\|A\|
$$

残差 ~ 机器精度 → 可表示；残差 O(1) → 表示不了。

实测（48 事例，seed 11）：

| 链 | omega 基底维数 | 真实振幅数 | 残差 |
|---|---|---|---|
| φf₂（`phif2.toml`） | 5 | 5 | $3.7\times10^{-15}$ … $9.9\times10^{-15}$ → **可表示** |
| b₁（`ex3_b1pi.toml`） | **2** | **4** | $9.8\times10^{-1}$ … $9.9\times10^{-1}$ → **不可表示** |

**复现**：§7 脚本 2。这个判据把"看起来像不一致"的归一化差异排除掉了 ——
上表用的是自由复系数，任何整体常数都被吸收。

### 2.2 计数律

对每个节点 $a\to b+c$，允许的 $(l,s)$ 由**宇称** $P_a=P_bP_c(-1)^l$ 与
**角动量三角** $|l-s|\le J_a\le l+s$ 唯一枚举。独立振幅数是**逐节点个数的乘积**：

$$
\boxed{\ N_{\text{独立振幅}}=\prod_{\text{节点 }n}N_n\ }
$$

实测枚举：

| 链 | 顶层节点 | 非顶层节点 | 独立振幅数 | omega |
|---|---|---|---|---|
| ρ′（例 3） | $\psi\to\rho'\pi$：$(1,1)$ → 1 个 | $\rho'\to\phi\pi$：$(1,1)$ → 1 个 | $1\times1=1$ | ✅ |
| b₁（例 3） | $\psi\to b_1\pi$：$(0,1),(2,1)$ → 2 个 | $b_1\to\phi\pi$：$(0,1),(2,1)$ → 2 个 | $2\times2=4$ | ❌ |
| φf₀（现有） | $\psi\to\phi f_0$：$(0,1),(2,1)$ → 2 个 | $f_0\to\pi\pi$：$(0,0)$ → 1 个 | $2\times1=2$ | ✅ |
| φf₂（现有） | $\psi\to\phi f_2$：5 个 | $f_2\to\pi\pi$：$(2,0)$ → 1 个 | $5\times1=5$ | ✅ |

**复现**：§7 脚本 3。

### 2.3 omega 的隐含不变量

把 §2.2 的乘积和 omega 实际用的 $N_{\text{顶层}}$ 对照，边界就一句话：

$$
\text{omega 可用}\iff \prod_n N_n=N_{\text{顶层}}\iff
\textbf{所有非顶层节点都恰好只有 1 个允许的 }(l,s)
$$

| 链 | 顶层 | 非顶层 | 乘积 | 道数 == (l,s) 数？ |
|---|---|---|---|---|
| φf₀ | 2 | ×1 | 2 | 是 → omega 可用 |
| φf₂ | 5 | ×1 | 5 | 是 → omega 可用 |
| ρ′ | 1 | ×1 | 1 | 是 → omega 可用 |
| b₁ | 2 | ×2 | **4** | **否 → omega 失效** |

这条不变量解释了为什么"到现在才撞上"：现有的 φf₀/φf₂ 恰好满足它
（$f_0/f_2\to\pi\pi$ 只有唯一分波），ρ′ 也恰好满足（两个节点都只有 1 个）。
**b₁ 是第一个违反它的链。**

### 2.4 拟合侧无关

拟合侧对 omega cache 只做一件事（`analyses/kk_dis/run/base_functions.py:334`）：

```python
P = onp.einsum("iab,enab->ien", M, Om)
W = g[:, None, None] * K[:, :, None] * P
```

```
$ grep -n "ls_list\|l_max\|nodes\|daughters\|chain\[" analyses/kk_dis/run/base_functions.py
(空)
```

它把 `Omega/K/M/g` 读进来、缩并、乘耦合与传播子，**不知道每条道对应哪个
$(l,s)$、树有几层**。而且 `_load_cached_tensor`（`base_functions.py:339`）按
`TENSOR_BASIS` 分派：`cartesian` / `helicity` 两条路径对"道是怎么来的"毫无假设。

**所以：能力边界挂在表象上，与拟合侧无关。** 这也正是深度分级检查放在
`spec.py`（生成侧）的原因。

---

## 3. tf-pwa 的答案

tf-pwa 有**两条并行的路**，我们的协变路线对应它的 (b)，螺旋度路线对应 (a)。

### 3.1 逐节点的螺旋度振幅（`HelicityDecay`）

`amp/core.py:682` 的类文档给出完整公式：

$$
A=H_{\lambda_B,\lambda_C}^{A\to B+C}\,
D^{J_A*}_{\lambda_A,\lambda_B-\lambda_C}(\varphi,\theta,0)
$$

$$
H_{\lambda_B,\lambda_C}^{A\to B+C}=
\sum_{ls}g_{ls}\sqrt{\frac{2l+1}{2J_A+1}}
\langle l\,0;\,s\,\delta|J_A\,\delta\rangle
\langle J_B\,\lambda_B;\,J_C\,{-\lambda_C}|s\,\delta\rangle
\,q^l B_l'(q,q_0,d),
\qquad \delta=\lambda_B-\lambda_C
$$

实现分三步：

| 步骤 | 位置 | 产出 |
|---|---|---|
| CG 矩阵 | `amp/core.py:879` `get_cg_matrix()` | $M_{(ls)}[\lambda_B,\lambda_C]$，常量 |
| 质量/位垒 | `amp/core.py:1045` `get_ls_amp()` | $g_{ls}q^lB_l'$，形状 `(N, n_ls)` |
| 组装 $H$ | `amp/core.py:969` `get_helicity_amp()` | $H=\sum_{ls}m_{ls}M_{(ls)}$，形状 `(N,1,\lambda_B,\lambda_C)` |
| 乘 $D$ | `amp/core.py:1173` `get_D_matrix_term()` | $D^{J_A*}_{\lambda_A,\lambda_B-\lambda_C}$，形状 `(N,\lambda_A,\lambda_B,\lambda_C)` |

**关键**：$H$ **不依赖 $\lambda_A$** —— $\lambda_A$ 只通过 $D$ 的
$(\lambda_A,\lambda_B-\lambda_C)$ 下标进入，且 $|\lambda_B-\lambda_C|>J_A$ 的槽位为 0
（`dfun.py:98` `Dfun_delta_v2`）。所以 $H$ 的指标是 $(\lambda_B,\lambda_C)$，
$D$ 补上 $\lambda_A$。

### 3.2 链级组装：einsum 收缩内线螺旋度

`amp/core.py:1426` `DecayChain.get_amp()` 的核心：

```python
for i in self:                          # 遍历链里**每一个** decay
    indices.append(i.amp_index(base_map))
    amp_d.append(i.get_amp(data_c[i], data_p, all_data=all_data))
...
ret = einsum(idx_s, *amp_d)
```

`amp_index`（`core.py:655`）返回 `[core, out1, out2]` 三个螺旋度下标；
`base_map`（`core.py:1656`）给每个粒子分配一个 einsum 字母。
内线粒子的字母在两个节点里各出现一次 → **einsum 自动把它求和掉**。

$$
A_{\text{chain}}[\lambda_{\text{top}},\{\lambda_{\text{finals}}\}]
=\sum_{\text{内线 }\lambda}\ \prod_{n}A^{(n)}[\lambda_n,\lambda_{d_1(n)},\lambda_{d_2(n)}]
$$

**这个循环对链的深度没有任何假设** —— 深度 3、深度 5 都是同一段代码。
这正是我们需要的东西。

### 3.3 完整的子粒子螺旋度（`helicity_inner_full`）

`core.py:661` / `:670` 的 `n_helicity_inner()` / `list_helicity_inner()`：
默认用粒子自己的 `spins` 列表，打开 `helicity_inner_full` 后用完整的
$2J+1$ 个螺旋度槽（`-J…J`）。我们的 `LAM_SLOTS`（`omega.py:50`）就是后者。

### 3.4 逐节点多 $(l,s)$ 与共享参数（`split_ls.py`）

`amp/split_ls.py:14` `ParticleDecayLS` / `:79` `ParticleBWRLS`：
一个节点可以带**多个** $(l,s)$，每个一个复耦合 $g_{ls}$，用极角
$\theta_i$ 参数化归一化（$\sum_i\gamma_i^2=1$），并共用同一个
Breit-Wigner 的 $m_0,\Gamma_0$。

这正是我们讨论过的 `shared_state` 结构在**生成侧**的对应物：
**一个节点内多个分波共享一条共振的传播子，但耦合各自独立。**

### 3.5 我们已有的 vs 缺的

| 需要的东西 | 我们有没有 | 位置 |
|---|---|---|
| CG 矩阵 $M_{(ls)}[\lambda_B,\lambda_C]$ | ✅ **有，且逐元素对** | `omega.py:85` `cg_matrix` |
| 逐节点角度（树通用） | ✅ **有，且树通用** | `omega.py:283` `chain_angles` |
| 完整螺旋度槽 $D^{J*}_{\lambda_a,\lambda_b,\lambda_c}$ | ✅ **有**（`D_lambda` 已收三个列表） | `omega.py:139` |
| 逐节点位垒因子 $q^lB_l'$ | ✅ 有（但 $l$ 是单值） | `omega.py:370` `node_factors` |
| **不把子粒子螺旋度压成 0** | ❌ **缺** | `omega.py:335` 写死 `[0],[0]` |
| **链级 einsum 缩并内线** | ❌ **缺** | `omega.py:367` 写死三个因子 |
| **逐节点 $(l,s)$ 列表** | ❌ **缺** | `spec.py:355` 只存单个整数 |
| Chung Eq.52 协变项 $f^J_{|m|}(\gamma)$ | ✅ **有**（**必需**，见 §4.4） | `omega.py:216` `covariant_hel_term` |
| 逐道直接算 $W_i$（不经过通用 $\Omega$） | ❌ **缺** | 现在只有手写协变公式 |

**验证 `cg_matrix` 已经对了**：用 `phif2` 的 5 个 $(l,s)$ 复算
`agent/tensor/kernels/omega.py:85`，与
`3_报告/tfpwa_helicity_check.md` §3.1 记录的 tf-pwa 实测值比较：

```
(0,1)  最大差 = 4.05e-07   OK
(2,1)  最大差 = 4.45e-07   OK
(2,2)  最大差 = 2.90e-07   OK
```

差值量级 ~$10^{-7}$ 来自报告只列到 6 位小数。**我们的 `cg_matrix` 就是
tf-pwa 的 `get_cg_matrix`（Chung Eq.12）。**

---

## 4. 目标：正确的表达形式

### 4.1 逐节点振幅

对链里每个节点 $n$（母 $a$，子 $b,c$），定义

$$
\boxed{
A^{(n)}[\lambda_a,\lambda_b,\lambda_c]
=D^{J_a*}_{\lambda_a,\lambda_b-\lambda_c}(\alpha_n,\beta_n,\gamma_n)\,
f^{J_b}_{|\lambda_b|}(\gamma_b)\,f^{J_c}_{|\lambda_c|}(\gamma_c)
\sum_{(l,s)\in\mathcal{L}_n}
g^{(n)}_{ls}\,H^{(n)}_{(ls)}[\lambda_b,\lambda_c]\,
q_n^{\,l}B_l'(q_n)
}
$$

其中 $f^J_{|m|}$ 是 Chung Eq.52 的协变项（§4.4，**必需**），$\gamma_d$ 是子粒子的
boost 因子。$H^{(n)}_{(ls)}$ 是 CG 组合：

$$
H^{(n)}_{(ls)}[\lambda_b,\lambda_c]=
\sqrt{\frac{2l+1}{2J_a+1}}
\langle l\,0;\,s\,\delta|J_a\,\delta\rangle
\langle J_b\,\lambda_b;\,J_c\,{-\lambda_c}|s\,\delta\rangle,
\qquad \delta=\lambda_b-\lambda_c
$$

这正是 `omega.py:85` `cg_matrix` 算的东西（§3.5 已验证与 tf-pwa 逐元素一致）。

$\mathcal{L}_n$ 是**该节点**允许的 $(l,s)$ 集合。道的身份是
**逐节点 $(l,s)$ 的一个组合**：

$$
\text{道}\ \equiv\ \bigl((l,s)_{n_1},(l,s)_{n_2},\dots\bigr),
\qquad
\text{道数}=\prod_n|\mathcal{L}_n|
$$

### 4.2 链级缩并

$$
A_{\text{chain}}[\lambda_{\text{top}},\{\lambda_{\text{finals}}\}]
=\sum_{\text{内线 }\lambda}\ \prod_n A^{(n)}[\lambda_n,\lambda_{d_1(n)},\lambda_{d_2(n)}]
$$

顶层再与初始态横向极化收缩（我们的 `EPS_A` 干这件事），得到
$W_i(e,\lambda_A)$，$\lambda_A=+1,0,-1$。

### 4.3 b₁ 的具体形式（对照文档 Eq.47–50）

文档里的 $U^\mu_{b_1SS}$ 等四个协变张量，在螺旋度语言里就是
$\mathcal{L}_{\psi}=(L,S)$、$\mathcal{L}_{b_1}=(l,s)$ 的四种组合：

| 道 | 顶层 $(L,S)$ | $b_1$ 节点 $(l,s)$ |
|---|---|---|
| `u_SS_*` | $(0,1)$ | $(0,1)$ |
| `u_SD_*` | $(0,1)$ | $(2,1)$ |
| `u_DS_*` | $(2,1)$ | $(0,1)$ |
| `u_DD_*` | $(2,1)$ | $(2,1)$ |

实测这四个逐节点组合 $H_1[(L,S)]\otimes H_2[(l,s)]$ 张成**秩 4**；
只按顶层分组只给**秩 2**。这就是 omega 丢掉的那 2 维。

**复现**：§7 脚本 4。

### 4.4 Chung Eq.52 协变项是**必需**的（不是可选项）

§4.1 的 $A^{(n)}$ 里没有写 Eq.52 的 $f^{J_d}_{|\lambda_d|}(\gamma_d)$。
**它不能省。** 判据是「逐事件比值是否为常数」：若 omega 与真实振幅只差一个
整体常数，逐事件比值应当逐位不变。

实测（64 事例，seed 11）：

| 链 | 道 | `covariant=True` 比值散布 | `covariant=False` 比值散布 |
|---|---|---|---|
| φf₂ | `phif201` | $1.5\times10^{-13}$ | **$2.7\times10^{+1}$** |
| φf₂ | `phif221` | $2.2\times10^{-13}$ | **$6.1\times10^{+0}$** |
| φf₂ | `phif222` | $1.3\times10^{-13}$ | **$3.2\times10^{+1}$** |

打开时比值是常数（机器精度）；关掉后**连常数都不是** —— 它不是一个可吸收的
整体归一化，而是逐事例的动力学因子。

> **这个区别只有用「逐事件比值」才看得出。** 我一开始用 5 基底最小二乘拟合
> 5 条道，两种设置残差都是 $\sim10^{-15}$ —— 因为每条道 144 个约束、5 个自由
> 复系数（10 个实自由度），**过定但恰好可解**，残差对基底是否"正确"不敏感。
> 同理，§2.1 里 b₁ 的判据之所以有效，是因为基底维数（2）**少于**真实振幅数（4）。
> **做这个判断必须用逐事件比值。**
>
> **复现**：§7 脚本 5。

所以新内核的 $A^{(n)}$ 必须包含（`omega.py:216` `covariant_hel_term`，可直接复用）：

$$
f^{J}_{|m|}(\gamma)=\sum_{m_0=J-|m|,\,J-|m|-2,\dots\ge0}
\frac{J!}{\bigl(\tfrac{J+|m|-m_0}{2}\bigr)!\;m_0!\;\bigl(\tfrac{J-|m|-m_0}{2}\bigr)!}
\cdot\frac{(J+|m|)!(J-|m|)!}{(2J)!}\;2^{m_0}\gamma^{m_0}
$$

其中 $\gamma_d=\sqrt{1+q^2/m_d^2}$ 是子粒子的 boost 因子。**它逐节点作用在每个
子粒子上**；末态自旋 0 的粒子 $f^0_0=1$，所以 φf₀/φf₂ 的内层节点不受影响，
而 $b_1\to\phi\pi$ 的 $\phi$（自旋 1）受影响。

### 4.5 为什么这个口径最省事

新内核**逐道直接算** $W_i(e,\lambda_A)$，落盘形状 `(n_ch, N, 3)` complex
—— **与今天的 helicity cache 逐位同格式**。

| | 今天 | 拓展后 |
|---|---|---|
| cache 形状 | `(n_ch, N, 3)` complex | **不变** |
| 拟合侧 `_load_cached_tensor` | 读 `<name>_hel.npy` | **不变** |
| 拟合侧强度 | $I=\sum_{\lambda=\pm1}|W_\lambda|^2$ | **不变** |
| 能算的道 | 深度 2 且非顶层唯一 $(l,s)$ | **任意深度、任意逐节点 $(l,s)$** |

**代价**：cache 不再"与 $(l,s)$ 无关"，每道一份（与路线 3 现状一致），
失去了 omega 那种"预计算一份 $\Omega$ 服务所有道"的复用。这是**正确的取舍**：
$\Omega$ 的复用性本来就是用表达能力换来的，而表达能力在 b₁ 这类链上已经不够用。

---

## 5. 落地步骤

### P0 —— 打通表达形式（最小可验证）

1. **链描述支持逐节点 $(l,s)$ 列表**
   `agent/tensor/spec.py:355` 现在写 `d["l"] = int(p["l"])`（单值）。
   改为同时保留 `d["ls"] = [(l,s), ...]`；`l` 单值保留作为兼容。
   TOML 侧 `[[channel]]` 增加逐节点组合的表达（如
   `nodes = {b1 = [0,1], phi = [1,1]}`），缺省时沿用现有自动枚举。

2. **新增螺旋度级联内核** `agent/tensor/kernels/helicity_cascade.py`
   - `node_amplitude(mom, chain, node, ls_list)` —— §4.1 的 $A^{(n)}$
     （复用 `D_lambda`，**不**压子粒子螺旋度；
     复用 `covariant_hel_term`，**不能省** Eq.52 的 $f^J_{|m|}(\gamma)$，见 §4.4）
   - `chain_amplitude(mom, chain, channel)` —— §4.2 的 einsum 缩并
   - 复用 `chain_angles`（树通用）与 `cg_matrix`（已验证）
   - 输出 `(N, 3)` complex，末轴 $\lambda_A=+1,0,-1$
   - **不要**复制 `_node_decay_d`（`omega.py:335` 把子粒子螺旋度写死为 `[0],[0]`）

3. **emitter / basis 接入**
   新增表象 `helicity_cascade`（或作为 `helicity` 的生成侧升级），
   `cache_basis="helicity"`，落盘格式不变。

### P1 —— 验收（这是最强的一条）

**与协变路线逐道对拍。** `agent/tensor/kernels/covariant.py` 里那 10 条
ρ′/b₁ 道是**手写公式**、与 omega 完全独立的实现。对每条道：

$$
W_i^{\text{cascade}}(e,\lambda)\ \overset{?}{=}\
\sum_\mu \mathrm{EPS\_A}[\lambda,\mu]\,T_i^{\text{covariant}}(e,\mu)
$$

这条判据**能同时抓住** CG 约定错、螺旋度槽错、缩并顺序错 ——
比"重算一遍自己"强得多。现有 `tests/test_covariant_vs_jax.py` 已经是这个
模式（24 个对象，rel < 1e-12），扩展它即可。

### P2 —— 用新能力移植例 5

例 5（`ψ→K⁺π⁻K⁻π⁺`）的 10 个道（`u_kst0_r/l` + `u_PP/PF/FP/FF_r/l`）
内部是**深度 3 + 多分波**，正是这次拓展的目标场景。
移植后应能**纯自动**产出，不再需要手写公式。

### P3 —— 反向核对：能否简化现有手写道

新的通用内核应当能**重现** `covariant.py` 里那 7 条 legacy 道与 10 条
ρ′/b₁ 道。若能，则手写公式可以退化为回归测试的参照物，而不是生产路径。

---

## 6. 验收判据汇总

| 编号 | 判据 | 阈值 | 抓什么 |
|---|---|---|---|
| A1 | 新内核 vs `covariant.py` 逐道对拍（经 `EPS_A`） | rel < 1e-12 | CG 约定、螺旋度槽、缩并顺序 |
| A2 | 逐节点组合的秩 = $\prod_n N_n$ | 整数精确 | 分波是否被吞掉 |
| A3 | b₁ 8 条道两两不重合 | 全部 rel > 1e-6 | 退化回归 |
| A4 | 深度 3 链对最内层角分布敏感 | 转动后 rel > 0.1 | 最内层是否参与 |
| A5 | 与 jax 原版 `FUNC.py` 对拍（例 5 移植后） | rel < 1e-12 | 移植正确性 |
| A6 | 现有 13 份 spec TOML 全部 `check` 通过 | 无异常 | 向后兼容 |
| A7 | cache 形状与拟合侧 `_load_cached_tensor` 兼容 | 形状/ dtype 精确 | 拟合侧零改动 |

A1/A4 是**新**判据，A6/A7 是**不许退化**的判据。

---

## 7. 复现脚本

全部用 `analyses/kk_dis/.venv/bin/python` 在仓库根运行。

**脚本 1 —— 最内层敏感性（§1.3）**

```python
import numpy as np, sys
sys.path.insert(0, "."); sys.path.insert(0, "agent/tensor/tests")
import test_ex3_channels as T
from agent.tensor import spec as S
from agent.tensor.kernels import omega as RT, covariant as COV

def rot3(v, axis, ang):
    ax = axis / np.linalg.norm(axis, axis=1, keepdims=True)
    c, s = np.cos(ang), np.sin(ang)
    d = (v * ax).sum(1, keepdims=True)
    return v * c + np.cross(ax, v) * s + ax * d * (1 - c)

def rel(a, b): return float(np.abs(a-b).max() / max(np.abs(b).max(), 1e-300))

momd = T.physical(n=16, seed=3)
s = S.load_spec("analyses/tensor_examples/specs/ex3_b1pi.toml")
chain = S.to_chain(s)
mom_o = np.stack([momd[k][:, [3, 0, 1, 2]] for k in ["Kp","Km","Pip","Pim"]], axis=1)
ang = np.deg2rad(37.0)

phi3 = mom_o[:,0,1:4] + mom_o[:,1,1:4]
r    = mom_o[:,0,1:4] - mom_o[:,1,1:4]
r2   = rot3(r, phi3, ang)
mom_r = mom_o.copy()
mom_r[:,0,1:4] = (phi3 + r2) / 2
mom_r[:,1,1:4] = (phi3 - r2) / 2

Om0 = RT.omega_from_momenta(mom_o, chain, covariant=True)
Om1 = RT.omega_from_momenta(mom_r, chain, covariant=True)
print("Omega 变化 rel = %.3e" % rel(Om1, Om0))     # 0.000e+00

T0 = COV.compute_event("u_DD_1", momd)
momd_r = {k: v.copy() for k, v in momd.items()}
phi3c = momd["Kp"][:,:3] + momd["Km"][:,:3]
rc = momd["Kp"][:,:3] - momd["Km"][:,:3]
r2c = rot3(rc, phi3c, ang)
momd_r["Kp"][:,:3] = (phi3c + r2c) / 2
momd_r["Km"][:,:3] = (phi3c - r2c) / 2
print("T 变化 rel = %.3e" % rel(COV.compute_event("u_DD_1", momd_r), T0))  # 4.877e-01
```

**脚本 2 —— 最小二乘可表示性（§2.1）**

```python
import numpy as np, sys
sys.path.insert(0, "."); sys.path.insert(0, "agent/tensor/tests")
import test_ex3_channels as T
from agent.tensor import spec as S
from agent.tensor.kernels import omega as RT, covariant as COV
from agent.tensor.kernels.helicity import EPS_A

momd = T.physical(n=48, seed=11)
mom2 = np.stack([momd[k][:, [3,0,1,2]] for k in ["Kp","Km","Pip","Pim"]], axis=1)

def true_amp(nm):
    Tc = COV.compute_event(nm, momd)
    return np.einsum("lm,em->el", EPS_A, Tc[..., :3]).ravel()

def resid(A, basis):
    B = np.stack(basis, axis=1)
    c, *_ = np.linalg.lstsq(B, A, rcond=None)
    return float(np.linalg.norm(A - B @ c) / np.linalg.norm(A))

# φf2：可表示
s = S.load_spec("analyses/kk_dis/task/20261017_多表象拟合流程标准化_多GPU/"
                "bundle/route4/tensor/specs/phif2.toml")
ch = S.to_chain(s)
Om = RT.omega_from_momenta(mom2, ch, covariant=True)
ls = [(0,1),(2,1),(2,2),(2,3),(4,3)]
M  = RT.cg_matrix(ls, 1, 1, 2, [-1,0,1], [-2,-1,0,1,2])
K  = RT.node_factors(mom2, ch, ls)
W  = RT.assemble(M, Om, np.ones(5), K)
for nm in ["phif201","phif221","phif222","phif223","phif243"]:
    print("  %-9s 残差 = %.3e" % (nm, resid(true_amp(nm), [W[i].ravel() for i in range(5)])))

# b1：不可表示
s2 = S.load_spec("analyses/tensor_examples/specs/ex3_b1pi.toml")
ch2 = S.to_chain(s2)
Om2 = RT.omega_from_momenta(mom2, ch2, covariant=True)
ls2 = [(0,1),(2,1)]
M2  = RT.cg_matrix(ls2, 1, 1, 0, [-1,0,1], [0])
K2  = RT.node_factors(mom2, ch2, ls2)
W2  = RT.assemble(M2, Om2, np.ones(2), K2)
for nm in ["u_SS_1","u_SD_1","u_DS_1","u_DD_1"]:
    print("  %-9s 残差 = %.3e" % (nm, resid(true_amp(nm), [W2[i].ravel() for i in range(2)])))
```

**脚本 3 —— 逐节点枚举与计数律（§2.2）**

```python
def allowed_ls(Ja, Pa, Jb, Pb, Jc, Pc, lmax=4):
    return [(L, S) for L in range(lmax+1)
            if Pa == Pb*Pc*(-1)**L
            for S in range(abs(Jb-Jc), Jb+Jc+1)
            if abs(L-S) <= Ja <= L+S]

for nm, nodes in [
    ("rho'", [("psi->rho'pi", 1,-1, 1,-1, 0,-1), ("rho'->phipi", 1,-1, 1,-1, 0,-1)]),
    ("b1",   [("psi->b1pi",   1,-1, 1,+1, 0,-1), ("b1->phipi",   1,+1, 1,-1, 0,-1)]),
    ("phif0",[("psi->phif0",  1,-1, 1,-1, 0,+1), ("f0->pipi",    0,+1, 0,-1, 0,-1)]),
    ("phif2",[("psi->phif2",  1,-1, 1,-1, 2,+1), ("f2->pipi",    2,+1, 0,-1, 0,-1)]),
]:
    prod = 1
    print("\n%s" % nm)
    for label, *q in nodes:
        ls = allowed_ls(*q); prod *= len(ls)
        print("   %-14s %d 个: %s" % (label, len(ls), "  ".join("(%d,%d)" % t for t in ls)))
    print("   => 独立振幅数 = %d" % prod)
```

**脚本 4 —— 逐节点组合的秩（§4.3）**

```python
import numpy as np, sys
sys.path.insert(0, ".")
from agent.tensor.kernels.omega import cg_matrix
LAM = {0:[0], 1:[-1,0,1], 2:[-2,-1,0,1,2]}
H1 = cg_matrix([(0,1),(2,1)], 1, 1, 0, LAM[1], LAM[0])[:, :, 0]   # psi -> b1 pi
H2 = cg_matrix([(0,1),(2,1)], 1, 1, 0, LAM[1], LAM[0])[:, :, 0]   # b1  -> phi pi
M4 = np.stack([np.outer(H1[a], H2[b]).ravel()
               for a in range(2) for b in range(2)])
print("逐节点组合张成的秩 =", np.linalg.matrix_rank(M4))   # 4
```

**脚本 5 —— 逐事件比值判据（§4.4，协变项必需）**

```python
import numpy as np, sys
sys.path.insert(0, "."); sys.path.insert(0, "agent/tensor/tests")
import test_ex3_channels as T
from agent.tensor import spec as S
from agent.tensor.kernels import omega as RT, covariant as COV
from agent.tensor.kernels.helicity import EPS_A

momd = T.physical(n=64, seed=11)
mom = np.stack([momd[k][:, [3,0,1,2]] for k in ["Kp","Km","Pip","Pim"]], axis=1)
s = S.load_spec("analyses/kk_dis/task/20261017_多表象拟合流程标准化_多GPU/"
                "bundle/route4/tensor/specs/phif2.toml")
ch = S.to_chain(s)
ls = [(0,1),(2,1),(2,2),(2,3),(4,3)]
M  = RT.cg_matrix(ls, 1, 1, 2, [-1,0,1], [-2,-1,0,1,2])
K  = RT.node_factors(mom, ch, ls)

for nm, i in [("phif201", 0), ("phif221", 1), ("phif222", 2)]:
    Wc = np.einsum("lm,em->el", EPS_A, COV.compute_event(nm, momd)[..., :3])
    for cov in (True, False):
        Om = RT.omega_from_momenta(mom, ch, covariant=cov)
        W  = RT.assemble(M, Om, np.ones(5), K)[i]
        mask = np.abs(Wc).max(1) > 1e-6
        r = Wc[mask] / W[mask]; r = r / r[0]
        print("%-9s covariant=%-5s 比值散布 = %.3e" % (nm, cov, np.abs(r - 1).max()))
# covariant=True  -> ~1e-13（常数，只差归一化）
# covariant=False -> O(1)..O(10)（连常数都不是）
```

---

## 8. 风险与边界

| 项 | 说明 |
|---|---|
| **barrier 的 $q_0$ 依赖** | 我们的 $B_l(Q)$ 归一化在 $Q=0$ 侧、$Q_0=0.197321$；tf-pwa 是 $B_l'(q_0)=1$。`3_报告/tensor_jax_helicity_rewrite.md` §2.3 指出：只有当 $q_0,d$ 是**常量**时两者比值才是逐通道常数。真实拟合里共振质量自由 → 若 $q_0$ 跟着自由质量走，简并会被打破。**部署前必须确认** |
| **EPS_A 行序** | 承重约定（`4_三条路线/eps_a_provenance.md`）：第三行取正号或换 $(1,+i)$ 支，重叠度从 1.000000 掉到 0.62–0.67。新内核必须沿用同一 `EPS_A` |
| **$\lambda_A=0$ 槽** | 强度只对 $\lambda=\pm1$ 求和，但 cache 里保留 3 槽（拟合侧要能复现 `EPS_A` 的 0 行）。新内核必须存 3 槽 |
| **cache 体积** | 逐道一份、复数，比 omega 大（omega 一份 $\Omega$ 服务所有道）。这是换取表达能力的代价，需在 spec 里显式声明 |
| **张量生成器的上游依赖** | `Tensor/`（jax 参考）只在 gitignore 的 `task/` 树里；跨分析复用需要先确认 provenance/license（`4_三条路线/audit_provenance.md`）。本次拓展**不**依赖它 —— `cg_matrix` 已自带 |
| **不在本次范围** | 置换求和模型（例 1/例 2）、辐射扇区（例 7–11，秩 2 振幅 + Coulomb 横向投影） |

---

## 9. 一句话

**omega 的能力边界是一条可以精确写下来的不变量**
（"所有非顶层节点恰好只有 1 个允许的 $(l,s)$"），
**b₁ 是第一个违反它的链**；
**tf-pwa 的逐节点 $H$ + 链级 einsum 就是边界之外的正确答案**，
而我们已经有了它需要的全部零件（`cg_matrix` 已验证、`chain_angles` 树通用、
`D_lambda` 已收完整螺旋度槽）——
**缺的只是"别把子粒子螺旋度压成 0"和"按树缩并"这两步。**
