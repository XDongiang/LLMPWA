# 把 tensor 缓存接进拟合：`fit_examples`

> 目标不是"拟合出一个物理结果"，而是**证明这条链路能跑**：
> 论文里每一个已实现的例子，都能自动地
> 造四动量 → 出协变/螺度张量缓存 → 出可变参数表 → 跑 JAX 似然与梯度。
>
> 判据只有两类：**数值等价性**（两份 cache 是同一个振幅、梯度对得上有限差分）
> 和**吞吐**（值+梯度每秒算多少事例）。拟合质量、收敛、物理结论**不在**本阶段。

代码基线：`analyses/tensor_examples`（张量生成的"能力全集"）与
`analyses/kk_dis`（生产似然的结构）。

---

## 1. 目录结构

```
analyses/fit_examples/
├── README.md                  ← 本文件：设计、判据、实测数字、已知边界
├── case 生成器（确定性，不经过 LLM）
│   ├── common/case.py           spec→四动量→张量→参数表→标定→README 的一条龙
│   ├── common/make_momenta.py   spec 驱动造四动量（tf-pwa generate_phsp）
│   ├── common/gen_phsp_tfpwa.py 零依赖版，在 tf-pwa docker 镜像里跑
│   └── common/likelihood.py     通用 JAX 似然（一份代码吃所有 case）
├── probe/proto_likelihood.py  早期原型：判据的出处与已记录的坑
├── document/                  任务点名的论文（md + tex）
├── .gitignore                 入库/忽略的分层规则（见 §1.1）
└── cases/<name>/              一个 case = 一个末态
    ├── case.toml              **人手写**的输入（族、质量、宽度、coupling_scale）
    ├── specs/*.toml           从 tensor_examples 复制的 spec（冻结输入 + sha1）
    ├── data/momenta.npz       四动量（+ momenta.meta.json 记录守恒/在壳/阈值）
    ├── gen/<spec>/            emit 出来的自足产物脚本
    ├── tensors/<spec>/<basis>/  cartesian 与 helicity_cascade 两份 cache
    ├── tensors/index.toml     实测事实（形状/dtype/max/标定常数）
    ├── params/free_params.toml        可变参数表（每道一个复耦合）
    ├── params/resonances_config.toml  传播子（质量/宽度，先冻结）
    ├── logs/{build.log,likelihood.json}
    ├── output/fitted_<basis>.npy
    └── README.md             由 case.py 从盘上产物生成的自述
```

**一条纪律**：`case.toml` 是人手写的**输入**，`index.toml` / `report.json` /
`README.md` 是**产物**。生成器只读输入、只写产物，绝不回写 `case.toml`。

### 1.1 哪些入库、哪些不入库

判据只有一条：**能不能由入库的东西重算出来**。

| 入库 | 为什么 | 体积 |
|---|---|---|
| `case.toml`、`specs/*.toml` | 人手写的输入 + 冻结的 spec | 小 |
| `params/*.toml` | 下一步要人工冻结/改初值的对象 | 小 |
| `tensors/index.toml`、`report.json` | 实测事实（形状/dtype/标定常数/耗时） | 8 KB ×5 |
| `data/momenta.meta.json` | 四动量的验收记录（守恒/在壳/阈值余量） | 小 |
| `logs/likelihood.json` | README 表格的数字来源 | 小 |
| `cases/*/README.md` | 逐 case 自述（其数据源 meta/report/json 都已入库，可重算） | 小 |
| `common/*.py`、`probe/`、`document/` | 代码与论文 | 300 KB |

| 忽略 | 为什么 |
|---|---|
| `cases/*/gen/` | `agent.tensor emit` 的确定性产物，`--only tensors` 秒级再生 |
| `data/momenta.npz`、`tensors/**/*.npy` | 大文件；由 spec + `case.py jobs` 重算（根 `.gitignore` 已挡） |
| `logs/build.log`、`output/*.npy` | 运行痕迹（根 `.gitignore` 已挡） |
| `_scratch/` | 探索期一次性工作区，已清空，留规则防它长回来 |

合计入库 **64 个文件 / 316 KB**；`cases/` 在盘上 79 MB，绝大部分是 cache。

---

## 2. 五个 case

"一个 case = 一个末态"。真实分析里，**同一末态的多个共振族必须相干求和**
（同一个初态、同一批事例）；拆成多个 case 等于假设它们互不干涉，那是错的。
所以例 3 与例 5 在同一末态 `K⁺K⁻π⁺π⁻` 下合并成一个 case。

| case | 末态 | 族（中间态） | 道数 | 自由实参 | 数据量 | 说明 |
|---|---|---|---|---|---|---|
| `psi_pipipi0` | π⁺π⁻π⁰ | ρ / ρ₃ / ρ₅ | 3+3+3 = 9 | 18 | 20k | 纯三体 |
| `psi_kkpi0` | K⁺K⁻π⁰ | KK̄ / K* / K*₃ / K*₅ | 1+2+2+2 = 7 | 14 | 20k | 含 S 波 KK̄ 阈值结构 |
| `psi_kkpipi` | K⁺K⁻π⁺π⁻ | ρ′ / b₁ / K*₀ / K*₂ | 2+8+2+8 = 20 | 40 | 20k | 四族相干，参数量最大 |
| `psi_omega_kk` | ωK⁺K⁻ | ω(三体节点) + f₀/f₂ | 7 | 14 | 20k | **唯一没有级联 cache**（三体节点无树） |
| `psi_phi4pi` | φπ⁺π⁻π⁺π⁻ | σ 交换 / ρρ / f₂σ | 14+14+10 = 38 | 76 | 5k | 六体，14 道/族；六体相空间慢 |

两份 cache 覆盖：`cartesian`（协变张量缓存）与 `helicity_cascade`
（ω/级联螺旋度缓存）。这是任务里指定的两种；`omega`、`helicity`、
`cov_to_helicity` 三种落在 `tensor_examples` 里但**不进**本流程。

---

## 3. 流水线

```bash
cd LLMPWA   # ⚠️ cwd 必须是仓库根：产物脚本按相对路径读 data/momenta.npz

# 一条龙（幂等，已存在就跳过；--force 重算）
python analyses/fit_examples/common/case.py build analyses/fit_examples/cases/psi_kkpipi

# 或只跑某一环
... build <case> --only specs      # 复制 spec + sha1
... build <case> --only momenta    # tf-pwa 造四动量
... build <case> --only tensors    # emit + 跑出两份 cache
... build <case> --only params     # free_params.toml + resonances_config.toml
... build <case> --only calibrate  # 反解跨表象复常数
... build <case> --only readme     # 重新生成该 case 的自述
```

### 3.1 四动量

用 tf-pwa 的 `phasespace.generate_phsp`，因为它才是**生产用的**相空间
（本仓库 tensor_examples 里的 `make_momenta.py` 是"顺序两体"，只能覆盖 3–4 体）。
tf-pwa 不在任何镜像的 site-packages 里，必须挂源码：

```bash
# 1) 把"要造什么"抽成一个纯数据文件（读 case.toml + spec 的末态质量）
cd LLMPWA
python analyses/fit_examples/common/case.py jobs \
    analyses/fit_examples/cases/* --out fit_examples_phsp/jobs.json

# 2) 在镜像里造（无需 GPU）
docker run --rm -v $PWD/fit_examples_phsp:/work \
    -v $PWD/tmp/tf-pwa:/src:ro -e PYTHONPATH=/src \
    -e TF_CPP_MIN_LOG_LEVEL=3 tf-pwa-check:latest \
    bash -lc 'python /work/gen_phsp_tfpwa.py --jobs /work/jobs.json'
```

`case.py jobs` 只吐纯数据（`name/m0/masses/keys/n/seed/layout/out`），
镜像那一侧完全不用读仓库代码。搬回本机时**只换 `out` 路径**：
容器里是 `/work/<case>/momenta.npz`，本机是 `<case>/data/momenta.npz`。
⚠️ 直接让容器写到宿主 `<case>/data/` 会失败 —— 容器的 `/work` 挂的是
`fit_examples_phsp/`，挂载点外的路径写进容器层、容器一退就没了。

`gen_phsp_tfpwa.py` 是**零依赖**的（不用 `agent.tensor`），只写 npz + 一份
`momenta.meta.json`。布局统一为 `x,y,z,E`（协变内核的原生布局）。
⚠️ 不要用嵌套链 `(m0, [(m1,(d1,d2)), ...])`：tf-pwa 对三层嵌套会
`ValueError: too many values to unpack`；扁平的末态质量列表对 3/4/5/6 体都对。

产出的事例都过三道自检（写进 meta）：

| case | 事例/秒 | 四动量守恒（绝对） | 在壳 max | 最紧两体余量 |
|---|---|---|---|---|
| `psi_pipipi0` | 2.91e5 | 9.5e-15 | 1.1e-14 | 2.81e-03 GeV |
| `psi_kkpi0` | 1.70e5 | 2.4e-15 | 7.5e-15 | 2.13e-03 GeV |
| `psi_kkpipi` | 1.77e5 | 8.0e-15 | 7.6e-15 | 7.93e-04 GeV |
| `psi_omega_kk` | 2.23e4 | 3.3e-15 | 6.7e-15 | 2.92e-04 GeV |
| `psi_phi4pi` | 9.08e2 | 2.0e-15 | 5.1e-15 | 3.57e-04 GeV |

余量必须为正：`Qabc` 变负会让 `B1/B2` 的 `sqrt` 出 NaN（原版 jax 同样 NaN，
不是移植错误）。

### 3.2 张量

`agent/tensor emit <spec> --out-dir <dir>` 出一份**自足**脚本
（`gen_tensor_<spec>_<kernel>.py` + `_backend.py` + `covariant.py` +
`runtime.py` + `helicity_cascade.py`），不依赖 `agent.tensor` 包即可运行。
每个 spec 只发它声明过的 `[[basis]]`：

* basis `cartesian` ↔ kernel `covariant` → `(n_ch, N, 3)` **实**，槽位 `(x,y,z)`
* basis `helicity_cascade` ↔ kernel `helicity_cascade` → `(n_ch, N, 3)` **复**，槽位 `(λ=+1,0,-1)`

`psi_omega_kk` 的 spec 没有声明 `helicity_cascade`（三体节点没有"树"，
级联形式无定义），这是设计如此，不是缺失。

### 3.3 参数表

从 spec 的 `[[channel]]` 逐道生成 `params/free_params.toml`（`kk_dis` 格式）：

```toml
[[data]]
path = "couplings.ex3_b1pi.u_SS_1.const"   # Re g
value = -0.0983378872613
error = 0.001
arg_index = 4
fixed = false

[[data]]
path = "couplings.ex3_b1pi.u_SS_1.theta"   # Im g
value = -0.060714100579
arg_index = 5
fixed = false
```

初值 = 固定模长 `coupling_scale = 0.1` × **随机相位**（每个 case 的 `seed`
决定）。`resonances_config.toml` 只写传播子，质量/宽度先冻结。

> 为什么复耦合要拆成 `.const`/`.theta` 两个**实**参数：这样 `jax.grad` 与
> `jvp` 全在实空间里，没有复梯度的共轭约定问题 —— 与 `kk_dis` 的
> `dplex_dconstruct` 同构。

### 3.4 似然

```bash
analyses/kk_dis/.venv/bin/python analyses/fit_examples/common/likelihood.py \
    analyses/fit_examples/cases/psi_kkpipi --basis both --check --fit \
    --json analyses/fit_examples/cases/psi_kkpipi/logs/likelihood.json
```

```
A(e, k) = Σ_族 f_R(e) · Σ_道 g_i · W_i(e, k)        # 张量缓存 × 耦合 × 传播子
I(e)    = Σ_{横向槽位} |A(e,k)|²                     # 横向极化（e⁺e⁻ 初态）
L       = -Σ_data log I + N_data · log(mean_MC I)     # extended likelihood
```

刻意**没有**：份额罚项、高斯先验、flatté 传播子（`f_R ≡ 1` 占位）。
那些属于"拟合结果"层面，扩展位留在 `_family_propagator`。

---

## 4. 判据与实测

### 4.1 跨表象等价：必须先标定，再比

同一个 spec 的两份 cache **不是同一个数组**，而是差一个**逐道复常数** `c_i`
（与 `kk_dis` 报告 §5 里 route1/route4 的关系同型）。所以固定参数点直接比
likelihood 会得到"差几个百分点"的假失败。

`--only calibrate` 从数据反解 `c_i = <C_i | EPS_A·T_i> / <C_i | C_i>` 并写进
`index.toml`；`likelihood.py` 在读盘时把 `c_i` 乘进**级联侧**
（`W = c·C`；乘错方向会差 `c²`，强度差 `c⁴`，实测能差到 1e6 倍，非常显眼）。

标定残差本身就是判据：

| 残差 | 含义 | 事例 |
|---|---|---|
| ~1e-15 | 逐位同一振幅，只差常数 | 合成数据上的全部族（`psi` 静止 + 中间态在壳） |
| ~1e-7 | 同一振幅；残差来自中间态离壳 | `ex1_*`、`ex2_*`、`ex6_ss`、`ex3_b1pi`、`ex5_kst*` |
| 0.3–1.0 | **形状不同**，两份 cache 是两个不同振幅 | `ex3_rhopi`(0.31)、`ex6_rr`(0.98)、`ex6_f2s`(1.00) |

第三档不能靠标定掩盖。`likelihood.py` 的做法是**逐族判定可比性**
（阈值 1e-6），不可比族列进 `excluded` 并只报"形状残差"，把它们的比值
（会到 1e7，因为那是两个不同振幅相除）整个剔掉，绝不让它污染总残差。

> **为什么 `ex3_b1pi` 是 1e-7 而不是 1e-15**：它的 DS/DD 道（顶层 D 波）
> 在 tf-pwa 的四体相空间里，`K⁺K⁻` 并不在壳（`m(KK)` 从 0.99 到 3.37 GeV，
> 因为 tf-pwa 抽的是整个四体相空间，不是 `ψ→b₁π, b₁→φπ, φ→KK` 的顺序链）。
> 实测把 `m(KK)` 从 1.019 推到 1.020 GeV，DS 道残差就从 1e-15 跳到 2.5e-4。
> 这不是 bug：级联形式在中间态离壳时与协变投影只差一个形状因子。

### 4.2 梯度与 HVP

`grad` 用 `jax.grad`，HVP 用 `jvp(grad)`，两者都对中心有限差分核验。
⚠️ 复参数的有限差分必须**分别扰 Re 与 Im**，且 `jax.grad` 返回的是
`df/dRe − i·df/dIm`（共轭梯度），虚部那项要加负号 —— 这两条曾让
`grad_vs_fd_rel` 卡在 1.5 左右。

> 这两条是**早期原型**（`probe/proto_likelihood.py`，直接用复参数拟合）踩的。
> 现在 `common/likelihood.py` 把 $g_i$ 拆成 `const + i·theta` 两个实参数
> （见 [`math.md`](math.md) §1），从结构上避开这一类问题；代价是参数数目翻倍。
>
> **似然的数学结构** —— 振幅、强度、归一化项、梯度闭式、两个精确零方向
> （整体模长与整体相位是规范冗余）—— 见 [`math.md`](math.md)。那里每条等式
> 都由 `probe/verify_math.py` 独立重算核对（不拿 `jax.grad` 验 `jax.grad`）：
>
> ```bash
> analyses/kk_dis/.venv/bin/python analyses/fit_examples/probe/verify_math.py
> # => 全部通过 （51 项）
> ```

### 4.3 实测（本机 CPU，20 workers）

设备 `cpu:0`，jax 0.11.1，`jax_enable_x64 = True`。

计时取 `cartesian`（`helicity_cascade` 的逐 case 数字见 `logs/likelihood.json`，
两者差 <3%）：

| case | 道数 | 自由实参 | 值+梯度 (s) | 事例/秒 | grad vs FD | HVP vs FD | 跨表象 I 残差 |
|---|---|---|---|---|---|---|---|
| `psi_pipipi0` | 9 | 18 | 3.86e-03 | 1.04e7 | 4.6e-09 | 6.8e-07 | 1.19e-07 |
| `psi_kkpi0` | 7 | 14 | 2.73e-03 | 1.47e7 | 1.2e-09 | 6.0e-10 | 1.69e-07 |
| `psi_kkpipi` | 20 | 40 | 7.60e-03 | 5.26e6 | 2.4e-09 | 6.1e-10 | 1.65e-07（排除 `ex3_rhopi`）|
| `psi_omega_kk` | 7 | 14 | 2.94e-03 | 1.36e7 | 5.3e-10 | 5.5e-10 | —（无级联 cache）|
| `psi_phi4pi` | 38 | 76 | 2.53e-03 | 3.96e6 | 1.0e-09 | 4.6e-10 | 9.33e-08（排除 `ex6_rr`/`ex6_f2s`）|

两份 cache 的吞吐基本一致（差 <3%）—— 说明级联路径没有额外开销：
它的 `|A|²` 是复数运算，但通道数/事例数与协变侧相同。

### 4.4 实测（远程 GPU，2×RTX 3090）

同一份 `common/likelihood.py`、同一批 npz，只换解释器
（`jax-fit:latest`，`/opt/env/bin/python`，jax 0.10.2，`--gpus all`）。

| case | 道数 | 值+梯度 (s) | 事例/秒 | 相对 CPU |
|---|---|---|---|---|
| `psi_pipipi0` | 9 | 5.47e-04 | 7.31e7 | ×7.1 |
| `psi_kkpi0` | 7 | 5.36e-04 | 7.46e7 | ×5.1 |
| `psi_kkpipi` | 20 | 7.16e-04 | 5.59e7 | ×10.6 |
| `psi_omega_kk` | 7 | 3.28e-04 | 1.22e8 | ×9.0 |
| `psi_phi4pi` | 38 | 3.05e-04 | 3.28e7 | ×8.3 |

换算成**逐事例**成本更能看清瓶颈：

| case | 道数 | 事例 (data+mc) | GPU 逐事例 (ns) |
|---|---|---|---|
| `psi_omega_kk` | 7 | 40000 | 8.2 |
| `psi_pipipi0` | 9 | 40000 | 13.7 |
| `psi_kkpi0` | 7 | 40000 | 13.4 |
| `psi_kkpipi` | 20 | 40000 | 17.9 |
| `psi_phi4pi` | 38 | 10000 | 30.5 |

`psi_phi4pi` 的 `s/eval` 比 `psi_omega_kk` **更小**，但事例数只有 1/4、
道数是 5.4 倍 —— 说明在这个规模上 `s/eval` 已经被**固定开销**（kernel
launch）占住，逐事例成本才是有意义的指标。扩到 10⁵–10⁶ 事例、并做 `pmap`
分片之后 GPU 的规模优势才会真正显现（本阶段没做，属于下一步）。

---

## 4.5 验收清单（可执行）

```bash
cd LLMPWA
PY=analyses/kk_dis/.venv/bin/python
C=analyses/fit_examples/cases

# 1) 四动量：守恒/在壳/两体余量（读 meta，不重算）
$PY -c "
import json,glob
for p in sorted(glob.glob('$C/*/data/momenta.meta.json')):
    d=json.load(open(p))
    assert d['conservation_abs']<1e-10, p
    assert max(d['onshell_max_abs'].values())<1e-10, p
    assert min(v['margin'] for v in d['two_body_min'].values())>0, p
    print('OK', p)"

# 2) 跨表象等价 + 梯度自检
for c in psi_pipipi0 psi_kkpi0 psi_kkpipi psi_phi4pi psi_omega_kk; do
    b=both; [ $c = psi_omega_kk ] && b=cartesian
    $PY analyses/fit_examples/common/likelihood.py $C/$c --basis $b --check \
        --json $C/$c/logs/likelihood.json > /dev/null || echo "FAIL $c"
done
```

通过线（都已实测满足）：

| 检查 | 通过线 | 实测 |
|---|---|---|
| 四动量守恒 | < 1e-10 | ≤ 9.5e-15 |
| 末态在壳 | < 1e-10 GeV | ≤ 1.1e-14 GeV |
| 两体阈值余量 | > 0 | ≥ 2.92e-04 GeV |
| `grad` vs 有限差分 | < 1e-6 | ≤ 4.6e-09 |
| `HVP` vs 有限差分 | < 1e-4 | ≤ 6.1e-06（`psi_pipipi0` 的级联侧；其余 ≤7e-7）|
| 跨表象强度（可比族）| < 1e-6 | ≤ 1.69e-07 |

后两行的阈值比实测宽 2–3 个数量级：它们是**回归线**，不是精度上限。
真正紧的判据是标定残差（1e-15 档），它区分"同一振幅"与"形状不同"。

---

## 5. 已知边界（都不是 bug，但都会让"直接比"给出错误结论）

1. **`ex3_rhopi` 的级联 cache 与协变 cache 形状不同**（残差 0.31）。
   Eq.(46) 里没有投影算符，两个表象在数学上就不等价。已在
   `tensor_examples/README.md` 记录。本流程把它标为**不可比**并排除。
2. **`ex6_phi4pi_rr` / `_f2s` 的 f₂ 支形状不同**（0.98 / 1.00），f₀ 支
   的 4 条道却完全对得上（1e-15）。是 f₂ 的母系自旋投影错配。
3. **`ex5_kst2_paper`** 是对照 spec（含历史形式），没有 emit；不在本流程内。
4. **`psi_omega_kk` 没有级联 cache**：三体节点无树，级联形式无定义。
5. **1e-7 的残差是物理的**（4.1 节）：tf-pwa 抽整个四体相空间，中间态离壳。
   要让残差回到 1e-15，得用**顺序链**抽相空间（`ψ→b₁π, b₁→φπ, φ→KK`）。
   这是下一步要加的一个 `--topology chain` 选项，不是现在的 bug。
6. **传播子是占位**（`f_R ≡ 1`）：质量/宽度写进了 `resonances_config.toml`
   但没进似然。所以现在的 `likelihood` 数值**没有物理意义**，只用于验证链路。
7. **没有先验/罚项**：`free_params.toml` 里的 `error = 0.001` 只是占位，
   似然里没有高斯项，也没有 `kk_dis` 的 fraction penalty。
8. **论文例 7–11（辐射衰变）没有实现**：`tensor_examples` 里没有对应 spec，
   超出本阶段范围；例 12 在论文里是空的。

---

## 6. 三个"能跑但错"的坑（已修，留档）

1. **协变侧的横向槽位固定是 `(x, y)`，与 `k` 无关。** 写成 `(x, z)` 会得到
   `cart_vs_hel_rel ≈ 0.04` —— 看起来像"两个表象差 4%"，实际是取错了分量。
2. **计时必须等梯度就绪。** JAX 是异步分发，只 `block_until_ready()` 值的话，
   测出来的"值+梯度"其实是"值"的时间，会系统性偏小约 2 倍。
3. **标定必须跨阶段保留。** `index.toml` 里的 `c_i` 只在 `--only calibrate`
   时重算；如果 `--only tensors` 之后忘了重算，下游会把**所有**族判成
   "不可比"，症状是 `cart_vs_casc` 里每一族都 `excluded`、`I_rel` 消失 ——
   看起来完全像"两个表象对不上"，实际只是标定丢了。现在 `tensors` 阶段会
   顺带重算标定，`params`/`readme` 阶段会把盘上已有的那份**带过来**
   （`carried_over` 标记），任何一个方向断了都会立刻暴露。

---

## 7. 下一步

* **顺序链相空间**：让中间态在壳，把 1e-7 的残差压到 1e-15（同时把
  `ex3_rhopi`、`ex6_rr/f2s` 的形状问题从"相空间伪影"里分离出来）。
* **真传播子**：把 `resonances_config.toml` 的 (m, Γ) 接进
  `_family_propagator`，再加 flatté（`kk_dis` 已有实现可参考）。
* **大样本 + 多 GPU**：现在最大 2 万事例；扩到 10⁵–10⁶ 并做 `pmap` 分片，
  看 GPU 的规模效应。`--gpu` 已经是硬失败（看不到 GPU 就报错，不静默降级）。
* **spec → 拟合代码的 LLM 环节**：现在 `common/likelihood.py` 是**通用**的，
  还没有"按 spec 生成专用似然"这一步。`kk_dis/llm_config_fit.toml` 是它的模板。
* **冻结策略**：`free_params.toml` 的 `fixed` 现在全是 `false`；真实分析要先
  按对称性/共线约束冻掉一批。
