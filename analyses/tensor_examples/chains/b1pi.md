# 链 `b1pi` —— ψ(2S) → b₁ π → (φ π) π → (K⁺K⁻)(π⁺π⁻)

> 状态：**验收主力链**。深度 3、非顶层多分波 —— 把框架的三个坑一起暴露。
> spec：[`analyses/tensor_examples/specs/ex3_b1pi.toml`](../specs/ex3_b1pi.toml)
> 道：`agent/tensor/chains/b1pi/`（8 条）
> 本文档写于 2026-10-19。

---

## 1. 这条链是什么

```
ψ(2S)  J=1  P=-1
  ├── b₁  J=1  P=+1  → φ + π      （**S 波与 D 波都可**，l=0 或 2）
  │     └── φ  J=1  P=-1  → K⁺ K⁻ （P 波，l=1）
  └── π⁻                            （旁观者）
```

三个要点，每一个都对应框架的一个坑：

1. **深度 3**（ψ → b₁π → φπ → KK）。omega 只算 `D0*D1*D2`，漏掉 φ 的角分布。
2. **非顶层多分波**：$b_1\to\phi\pi$ 有 **2 个**允许的 $(l,s)$ —— $(0,1)$ 与 $(2,1)$
   （$b_1$ 是 $1^{+-}$，宇称允许 S 波与 D 波）。omega 的 cache 里非顶层节点
   只有标量 $K_i$，**无处安放**这个信息。
3. **中间共振还衰变**：φ 自己还要衰变成 $K^+K^-$，所以逐节点螺旋度参考系
   必须逐层构造 —— 这是最隐蔽的坑（见 §4.3）。

粒子编号：`K⁺=1`、`K⁻=2`、`π⁺=3`、`π⁻=4`。

## 2. 做了什么

### 2.1 八条道（协变张量）

完备标签是**"顶层波 × 内层波"**：

| 道 | 文档公式 | 顶层 $(l,s)$ | 内层 $(l,s)$ | 旁观者 |
|---|---|---|---|---|
| `u_SS_1` | Eq.(47) | $(0,1)$ | $(0,1)$ | $\pi^-$ |
| `u_SD_1` | Eq.(47) | $(0,1)$ | $(2,1)$ | $\pi^-$ |
| `u_DS_1` | Eq.(47) | $(2,1)$ | $(0,1)$ | $\pi^-$ |
| `u_DD_1` | Eq.(47) | $(2,1)$ | $(2,1)$ | $\pi^-$ |
| `u_SS_2` | Eq.(48) | $(0,1)$ | $(0,1)$ | $\pi^+$ |
| `u_SD_2` | Eq.(48) | $(0,1)$ | $(2,1)$ | $\pi^+$ |
| `u_DS_2` | Eq.(48) | $(2,1)$ | $(0,1)$ | $\pi^+$ |
| `u_DD_2` | Eq.(48) | $(2,1)$ | $(2,1)$ | $\pi^+$ |

（Eq.47–50 分别对应 SS / SD / DS / DD 四种组合；`_1`/`_2` 是旁观者交换。）

⚠️ **`_1`/`_2` 的下标在两个族里含义相反**（照抄参考实现的命名）：
`rhopi` 族 `_1` 的旁观者是粒子 3（$\pi^+$），`b1pi` 族 `_1` 的旁观者是
粒子 4（$\pi^-$）。**这不是笔误**，是历史命名。

⚠️ **`_1` 与 `_2` 是两棵不同的树**（旁观者 $\pi$ 交换）：

```
_1:  ψ → b₁ + π⁻ ,  b₁ → φ + π⁺
_2:  ψ → b₁ + π⁺ ,  b₁ → φ + π⁻
```

所以 `[particles.b1].daughters` 只能写**一组**配对（spec 里写的是 `_1` 那组），
`_2` 那些道靠逐道 `tree = {...}` 覆盖。这在协变路线下没问题 ——
协变张量的道函数**完全不读树**（`grep -c "chain\|nodes" kernels/covariant.py` == 0），
树只被 omega / 级联内核读。但见 §3.3。

### 2.2 两个关键判据（A1 / A3）

b₁ 是**唯一**能让 A1 真正发力的链。

| 判据 | 内容 | 结果 |
|---|---|---|
| **A1** | 级联 vs 协变逐道对拍 | **6.5e-15 … 2.5e-14**（8 条全过；$n=500$）|
| **A2** | 逐节点组合的秩 = $\prod_n N_n$ | **4 = 2×2**（只按顶层分组是 2）|
| **A3** | 8 条道两两不重合 | min over 28 pairs = **4.23e-01**（$n=500$；$n=2000$ 时 3.22e-01）|

两条随样本变的判据（A1/A3）数字会浮动，但**结论不变**：
A1 恒在机器精度，A3 恒远大于阈值。原始 stdout 见
`analyses/kk_dis/task/20261019_螺旋度级联内核_审阅/material/evidence/02_check_ex3.txt`。

A3 的意义：omega 时代 8 条道只映射到 2 组 $(l,s)$，所以 28 对里有 **12 对
逐位重合**。级联内核让 8 条道全部线性独立。

### 2.3 A4：对最内层衰变平面的敏感性

绕 φ 方向把 $\phi\to K^+K^-$ 的衰变平面转 37°，保持 $p_\phi$ 与总四动量不变：

| 道 | 级联 rel | 协变 rel |
|---|---|---|
| `u_SS_1` | 3.852e-01 | 3.852e-01 |
| `u_SD_1` | 2.564e-01 | 2.564e-01 |
| `u_DS_1` | 3.724e-01 | 3.724e-01 |
| `u_DD_1` | 4.516e-01 | 4.516e-01 |

级联与协变**逐位同值**，而 **omega 的 $\Omega$ 对这一转动完全不敏感**
（roadmap §1.3）—— 这就是深度限制的直接证据。

### 2.4 顺带修正的一个既有 bug

spec 初稿里 `u_SD_1` 的 `l = 2`、`u_DS_1` 的 `l = 0` 是**写反的**
（SD = 顶层 S 波 + 内层 D 波，顶层 $l$ 应为 0）。用穷举把每条协变道对到
逐节点组合实测确认：

```
u_SS_1   top=(0,1) inner=(0,1)  rel=5.081e-15
u_SD_1   top=(0,1) inner=(2,1)  rel=4.600e-15
u_DS_1   top=(2,1) inner=(0,1)  rel=5.857e-15
u_DD_1   top=(2,1) inner=(2,1)  rel=5.747e-15
```

旧的 `l` 值只被 omega 的 $K_i$ 用到，而这份 spec 不出 omega，所以此前没暴露。

### 2.5 螺旋度级联表象的接入

spec 里加了第三条 basis：

```toml
[[basis]]
name   = "helicity_cascade"
kernel = "helicity_cascade"
files  = ["b1pi_casc.npy"]
env    = { TENSOR_BASIS = "helicity", HELICITY_LAYER = "1" }
```

每条道带 `ls`（逐节点 $(l,s)$）与 `tree`（镜像树）。落盘形状
$(8,N,3)$ complex128，与 `helicity` cache **逐位同格式**，
拟合侧 `_load_cached_tensor` 零改动（判据 A7）。

## 3. 还有什么问题

### 3.1 三个坑的现状

| 坑 | 状态 |
|---|---|
| 非顶层多分波（omega 容量不足） | ✅ 级联内核解决 |
| 深度 3（omega 截断） | ✅ 级联内核解决（omega 对本链**不可用**）|
| 逐节点参考系 | ✅ 已修正，但**只是经验验证**，不是形式推导（§3.2）|

### 3.2 参考系构造：经验验证 ≠ 形式推导

这是**审阅意见的核心质疑**，必须如实写在这里。

`node_angles` 的正确性由"与协变振幅逐位对拍"确认（A1）。但
[`verify.py`](../../kk_dis/task/20261019_螺旋度级联内核_审阅/material/verify.py)
里的独立复算虽然不 import 内核，用的却是**同一套算法**（逐层 boost、
每节点重置 `bias = -π`）。所以它不是对"这个 frame 约定是否正确"的独立检验。

**没有排除的替代**：

* 不重置 `bias`（全程从 $-\pi$ 递减）；
* **Wigner rotation** 修正（tf-pwa 的 `HelicityDecay` 里有
  `get_boost_matrix` 的痕迹，本次没对照）；
* φ 的**子粒子顺序**（`K⁺` 先还是 `K⁻` 先）；
* 第三种等价构造（不同方位角起点）。

**要证明**，需要与 tf-pwa 的 `HelicityDecay` 在同一四动量、同一子粒子顺序下
逐节点比较 D 矩阵 / Wigner rotation / bias / 整链振幅。**未做。**

### 3.3 一份 spec 装两棵树 —— 设计上不干净

b₁ 的 8 条道是**两棵不同的树**，靠逐道 `tree = {...}` 覆盖塞进一份 TOML。
这能工作，但：

* `[particles.b1].daughters` 只描述其中一棵，另一棵靠覆盖 —— 读者容易误解；
* **真实拟合里两棵树是否共享传播子/参数**，本次完全没涉及；
* 更干净的做法是**一份 TOML 一棵树**（像 `ex3_rhopi` / `ex3_b1pi` 分开那样），
  或让 spec 层原生支持"多棵树"。

**这是待改进项**，不是 bug。

### 3.4 A3 的局限

A3 的阈值 1e-6 只排除**测试点上逐位相等**，**不保证线性独立**。
而且 28 对里混了"不同 spectator 树"与"不同分波"两种差异，
所以 min = 0.423 这个数**不能**单独证明"8 条道在拟合里稳定可分辨"。

### 3.5 未做的事

| 项 | 状态 |
|---|---|
| **A5：与 jax 原版 `FUNC.py` 对拍** | ❌ **未做**（需要例 5 移植）。这是最大的缺口 |
| 真实拟合 / `_load_cached_tensor` / toy likelihood | ❌ 未做。A7 只验形状/dtype |
| 节点级 Eq.52 消融 | ⚠️ 只做了整链开关（关掉 → rel 0.201），**没做**分节点消融 + 质量扫描 |
| barrier 的 $q_0$ 依赖 | ⚠️ 未解决。A1 用的是固定 $Q_0=0.197321$、两边同一个 $B_l$，所以不影响 A1 的结论；但真实拟合里会显现 |
| 13 份旧 spec 的旧/新 cache 逐位回归 | ⚠️ 只实测了 3 份生产 spec（深度 2）是 no-op；没做逐位对照 |

## 4. 三个坑的细节

### 4.1 坑一：非顶层分波被压成标量

omega 的 cache 格式是 $\Omega + K + M + g$：非顶层节点的角动量只以**标量**
$K_i$ 出现，$(l,s)$ 只在顶层的 $M$ 里。所以 $b_1\to\phi\pi$ 的 S/D 波之别
**无处可放**。

计数律：独立振幅数 $=\prod_{\text{节点}} N_{\text{节点}}$。
b₁ 是 $2\times 2=4$（乘旁观者交换 = 8），而 omega 只能给 2 组 $(l,s)$。

### 4.2 坑二：只算 `D0*D1*D2`

`omega_from_momenta` 逐节点读树，但只算顶层 + 一层共振的 D 矩阵乘积。
深度 3 时最内层（$\phi\to KK$）的角分布被**静默丢掉**。

现在由 `spec.py` 的深度分级检查**强制报错**，而不是静默出错。
所以 b₁ 这份 spec **不出** omega 表象 —— 不是遗漏，是被禁止。

### 4.3 坑三：逐节点参考系（最隐蔽）

omega 的 `chain_angles` 把**每个**粒子都从**顶层**静止系直接 boost：

```python
rpA = {n: rest_vector(A, _node_momentum(mom, chain, n)) for n in names}  # A = 顶层
rp  = {d: rest_vector(rpA[n], rpA[d]) for ...}
```

对"子节点不再衰变"的链（φf₀ / φf₂ / ρ′）这是**对的**。一旦某个节点的子粒子
本身还要衰变（b₁ → φπ，φ 还要衰变），就必须逐层构造：每个节点的角度在它
**母节点**的静止系里量，走到 $n$ 时把 $n$ 的**整棵子树**按 $p_n$ boost 一次。

**实测差别落在哪里**（`evidence/04_frame_noop_check.txt`）：

```
depth 1  b1        diff=0.000e+00
depth 1  Pim       diff=0.000e+00
depth 2  phi       diff=0.000e+00
depth 2  Pip       diff=0.000e+00
depth 3  Kp        diff=9.301e-01   <-- 不同
depth 3  Km        diff=9.301e-01   <-- 不同
```

**顶层与 b₁ 节点的角一样**；差别出在 **φ 的子粒子**（$K^\pm$）——
因为 $p_\phi$ 的静止系在两种构造里不同（旧构造从 ψ 系一次 boost 到 φ 系，
新构造先到 b₁ 系再到 φ 系）。φ 自旋为 1，它的 D 矩阵对 $K^\pm$ 的角度敏感，
所以这一差别是**承重的**。

影响：`chain_angles` 下 rel ≈ 0.10 … 0.15，`node_angles` 下 ≈ 1e-14。

**准确的适用范围**：差别只在链的**最大深度 ≥ 3** 时出现。
φf₀/φf₂/`rhof2_generic`（深度 2）上两种构造**逐位相同**。

## 5. 复现

```bash
cd <LLMPWA>

# 端到端（生成 → 跑 → 三层自检 + A1/A3/A7）
python analyses/tensor_examples/check_ex3.py --n 500

# 只看 b1 那棵树
python analyses/tensor_examples/check_ex3.py --n 500 --tree ex3_b1pi

# 框架测试
python agent/tensor/tests/test_ex3_channels.py      # 8 项
python agent/tensor/tests/test_helicity_cascade.py  # 13 项

# 判据 A1/A2/A3/A4 的独立复算（不 import 本仓库）
python analyses/kk_dis/task/20261019_螺旋度级联内核_审阅/material/verify.py

# 诊断探针
python analyses/tensor_examples/probes/probe_a1_sensitivity.py   # 错误注入
python analyses/tensor_examples/probes/probe_rho_decomposition.py # 逐节点分解
```

## 6. 相关文档

* [链 `rhopi`](rhopi.md) —— 同拓扑、单分波、但协变式不是级联
* [链 `phif2`](phif2.md) —— 深度 2 的对照
* [链索引](README.md)
* 审阅记录：`analyses/kk_dis/task/20261019_螺旋度级联内核_审阅/gen/fragments/`
