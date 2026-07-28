# 从 `combined_likelihood` 到 `calculate/component` 的数学过程

## 1. 总 likelihood

代码中：

```python
combined_likelihood(args, jax_data)
    = data_likelihood_kk(args, jax_data)
      + data_size * log(mc_likelihood_kk(args, jax_data))
```

数学上写为：

$$
\mathcal{L}_{\rm comb}(\theta)
=
\mathcal{L}_{\rm data}(\theta)
+
N_{\rm data}\log \mathcal{N}_{\rm MC}(\theta)
$$

其中：

- \(\theta\)：所有拟合参数；
- \(N_{\rm data}\)：真实数据事件数；
- \(\mathcal{L}_{\rm data}\)：data negative log-likelihood；
- \(\mathcal{N}_{\rm MC}\)：MC 给出的归一化积分项。

## 2. data 部分

对每个 data 事件 \(e\)，总振幅由四类振幅相干相加：

$$
\mathcal{A}_{e k}
=
\mathcal{A}^{f_0,{\rm Flatte980}}_{e k}
+
\mathcal{A}^{f_0,{\rm BW}}_{e k}
+
\mathcal{A}^{f_2,{\rm Flatte1270}}_{e k}
+
\mathcal{A}^{f_2,{\rm BW}}_{e k}
$$

其中 \(k\) 是振幅张量最后一维的分量指标。在当前 `kk` 拟合中，
`phif0_kk` 和 `phif2_kk` 的实际 shape 最后一维均为 2，因此四类振幅最终都在共同的
\(k=1,2\) 空间中相干相加。不同的是基/helicity 指标 \(i\)：`phif0_kk` 有 2 个基分量，
`phif2_kk` 有 5 个基分量。

事件强度为：

$$
I_e(\theta)
=
\sum_k
\left|
\mathcal{A}_{e k}(\theta)
\right|^2
$$

所以 data negative log-likelihood 为：

$$
\mathcal{L}_{\rm data}(\theta)
=
-
\sum_{e\in{\rm data}}
\log I_e(\theta)
+
C_{\rm penalty}(\theta)
$$

其中 penalty 来自 `data_step_function`：

$$
C_{\rm penalty}(\theta)
=
\lambda
\left(
f_{\rm total}-1.1
\right)^2
+
\sum_r
\frac{
\left(x_r-x_{r,0}\right)^2
}{
2\sigma_r^2
}
$$

代码中：

$$
\lambda = 1000
$$


## 3. MC 归一化项

MC 样本上同样构造总振幅。这里的 MC 归一化使用 `mc_*` 样本的全部事件，
不是 `truth_*` 的前 150000 个事件。`truth_*` 只用于 fraction penalty。
在当前 `load_data()` 设置 `n_repeat = 1` 时，若 `mc_truth/*.npy` 为 1200000 个事件，
则 \(\mathcal{N}_{\rm MC}\) 使用全部 1200000 个 MC 事件。

$$
\mathcal{A}^{\rm MC}_{e k}
=
\sum_\alpha
\mathcal{A}^{\alpha}_{e k}
$$

其中 \(\alpha\) 遍历四类振幅：

$$
\alpha
\in
\{
f_0{\rm -Flatte980},
f_0{\rm -BW},
f_2{\rm -Flatte1270},
f_2{\rm -BW}
\}
$$

MC 归一化为平均强度：

$$
\mathcal{N}_{\rm MC}(\theta)
=
\frac{1}{N_{\rm MC}}
\sum_{e\in{\rm MC}}
\sum_k
\left|
\mathcal{A}^{\rm MC}_{e k}(\theta)
\right|^2
$$

因此总目标函数可以写成：

$$
\boxed{
\mathcal{L}_{\rm comb}(\theta)
=
-
\sum_{e\in{\rm data}}
\log I_e(\theta)
+
C_{\rm penalty}(\theta)
+
N_{\rm data}
\log
\mathcal{N}_{\rm MC}(\theta)
}
$$

## 4. `calculate_*` 的通用振幅结构

设输入的角分布/动力学张量为：

$$
T_{i e k}
$$

其中：

- \(i\)：输入振幅基或 helicity/LS 分量指标；
- \(e\)：事件指标；
- \(k\)：末态张量分量指标。

对当前数据：

- `phif0_kk`：\(i=1,2\)，\(k=1,2\)；
- `phif2_kk`：\(i=1,\dots,5\)，\(k=1,2\)。

所以 `phif0` 与 `phif2` 的基/helicity 数不同，但被耦合求和后都得到
\(\mathcal{A}_{e k}\)，并在相同的末维 \(k=1,2\) 上相加。

复耦合常数由 `Amplitude_param_const` 和 `Amplitude_param_theta` 构造：

$$
c_{\ell i}
=
a_{\ell i}
+
i\phi_{\ell i}
$$

代码对应：

```python
const_ph = dplex_dconstruct(Amplitude_param_const, Amplitude_param_theta)
```

先做耦合加权：

$$
B_{\ell e k}
=
\sum_i
T_{i e k}
c_{\ell i}
$$

然后乘上传播子 \(P_{\ell e}\)。

若该振幅只有一个子态，可写为：

$$
\mathcal{A}_{e k}
=
B_{e k}
P_e
$$

若有多个子态 \(\ell\)，`calculate_*` 会对 \(\ell\) 相干求和：

$$
\mathcal{A}_{e k}
=
\sum_\ell
B_{\ell e k}
P_{\ell e}
$$

而 `component_*` 不对 \(\ell\) 求和，保留每个分量：

$$
\mathcal{A}^{\rm comp}_{\ell e k}
=
B_{\ell e k}
P_{\ell e}
$$

因此：

- `calculate_*` 返回总振幅，用于 likelihood；
- `component_*` 返回分振幅，用于 fraction 计算。

## 5. 传播子

### 5.1 Breit-Wigner

普通 Breit-Wigner 在代码中为 `BW`：

$$
BW(m,\Gamma;s)
=
\frac{1}{
m^2-s-im\Gamma
}
$$

其中 \(s\) 对应 `phi_kk` 或 `f_kk`。

### 5.2 `BW_flatte980`

`calculate_BW_flatte980` / `component_BW_flatte980` 使用：

$$
P_e
=
BW(m_\phi,\Gamma_\phi;s_{\phi,e})
\cdot
Flatte_{980}(m,g_{\pi\pi},r_g;s_{f,e})
$$

其中：

$$
g_{KK}
=
r_g g_{\pi\pi}
$$

$$
\rho_{KK}(s)
=
\sqrt{
\left|
1-\frac{4m_K^2}{s}
\right|
}
$$

$$
\rho_{\pi\pi}(s)
=
\sqrt{
\left|
1-\frac{4m_\pi^2}{s}
\right|
}
$$

Flatte 形式为：

$$
Flatte_{980}(s)
=
\frac{1}{
m^2-s
-
i
\left(
g_{\pi\pi}\rho_{\pi\pi}(s)
+
g_{KK}\rho_{KK}(s)
\right)
}
$$

### 5.3 `BW_BW`

`calculate_BW_BW` / `component_BW_BW` 使用：

$$
P_{\ell e}
=
BW(m_\phi,\Gamma_\phi;s_{\phi,e})
\cdot
BW(m_\ell,\Gamma_\ell;s_{f,e})
$$

其中 \(\ell\) 对应多个 \(f_0\) 或 \(f_2\) 共振态。

### 5.4 `BW_flatte1270`

`calculate_BW_flatte1270` / `component_BW_flatte1270` 使用：

$$
P_e
=
BW(m_\phi,\Gamma_\phi;s_{\phi,e})
\cdot
Flatte_{1270}(m,\Gamma;s_{f,e})
$$

代码中的 `flatte1270` 定义为：

$$
q_r^2
=
\frac{1}{4}m^2
-
0.0194792
$$

$$
b_r^2
=
q_r^2
\left(q_r^2+0.1825\right)
+
0.033306
$$

$$
g_{11270}
=
m\Gamma
\frac{
b_r^2
}{
\left(q_r^2\right)^{2.5}
}
$$

对事件 \(e\)：

$$
q_e^2
=
\frac{1}{4}s_{f,e}
-
0.0194792
$$

$$
b_e^2
=
q_e^2
\left(q_e^2+0.1825\right)
+
0.033306
$$

$$
g_1(s_{f,e})
=
g_{11270}
\frac{
\left(q_e^2\right)^{2.5}
}{
b_e^2
}
$$

于是：

$$
Flatte_{1270}(s)
=
\frac{
m\Gamma
}{
s-m^2
+
i g_1(s)
}
$$

## 6. fraction 约束

truth 样本上，代码先用 `component_*` 得到各个分量：

$$
\mathcal{A}^{\alpha}_{\ell e k}
$$

其中：

$$
\alpha
\in
\{
f_0{\rm -Flatte980},
f_0{\rm -BW},
f_2{\rm -Flatte1270},
f_2{\rm -BW}
\}
$$

总分母使用所有分量的相干和：

$$
D
=
\sum_{e,k}
\left|
\sum_\alpha
\sum_\ell
\mathcal{A}^{\alpha}_{\ell e k}
\right|^2
$$

每类 fraction 的分子为该类分量的模平方和：

$$
F_\alpha
=
\sum_{\ell,e,k}
\left|
\mathcal{A}^{\alpha}_{\ell e k}
\right|^2
$$

因此：

$$
f_\alpha
=
\frac{F_\alpha}{D}
$$

总 fraction 明确定义为四类 fraction 的和：

$$
f_{\rm total}
=
\sum_\alpha
f_\alpha
$$

即：

$$
f_{\rm total}
=
f_{f_0{\rm -Flatte980}}
+
f_{f_0{\rm -BW}}
+
f_{f_2{\rm -Flatte1270}}
+
f_{f_2{\rm -BW}}
$$

需要注意，这里的 \(D\) 是所有分量先相干求和再取模平方的总强度，而
\(F_\alpha\) 是每一类自身分量的非相干模平方和。因此各 \(f_\alpha\) 的和
不必严格等于 1，代码中用目标值 1.1 对 \(f_{\rm total}\) 加约束。

然后进入约束项：

$$
C_{\rm frac}
=
1000
\left(
f_{\rm total}-1.1
\right)^2
$$

## 7. 分布式求和含义

在该分布式脚本中，事件轴被 shard 到多个设备/进程。

数学公式中的：

$$
\sum_e
$$

和：

$$
\frac{1}{N}\sum_e
$$

在代码中分别对应 `jnp.sum` 和 `jnp.mean`。由于数组是 sharded array，JAX/XLA 会自动插入跨设备的 all-reduce，因此数学上仍然等价于对全数据集求和或求平均。

