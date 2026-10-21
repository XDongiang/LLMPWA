# 修正说明：$u_{PP}$ / $u_{FP}$ 的内层无迹化参考系

> **适用链**：`kstpi`（例 5，$\psi\to K^*_2\pi\to(\phi\pi)\pi\to(K^+K^-)(\pi\pi)$）
> **适用道**：`u_PP_1`、`u_PP_2`、`u_FP_1`、`u_FP_2`（共 4 条）
> **结论**：参考实现历史形式（下称"历史形式"）把 $1\otimes1\to2$ 的**无迹化**交给了 $P^{(2)}(p_\psi)$
> —— 那是 $\psi$ 系的度规，而耦合发生在 $K^*_2$ 系。修正后 A1 残差从
> $10^{-2}\sim10^{-1}$ 降到 $\sim10^{-15}$。
>
> **术语**：`*_paper.py` 里的是**参考实现**（`FUNC.py`）的逐式移植，
> 我们**没有**核对过论文原文，所以全文称"历史形式"而非"论文原式"。
>
> **已过独立审阅（2026-10-21）**：审阅接受核心数值结果，但指出我们原先
> 有四处过度声称（"根因可证"、"闭式常数是独立证据"、对照只测了一条、
> "论文原式"命名）。本文件已据其修订，关键处标 ⚠️。
> 审阅全文见 `analyses/kk_dis/task/20260929_螺旋度耦合振幅推导/5_生成器日志/review_kst2_frame_fix/gen/fragments/kst2_frame_fix_review.md`。

---

## 1. 一句话结论

$K^*_2$ 的**自旋 2** 由**两个秩-1 分析器**（内层 $\tilde t^{(1)}_{(K^*_2\pi)}$
与经 $\tilde g_{K^*_2}$ 搬运的 $\tilde t^{(1)}_{(\phi\pi)}$）耦合而来时，
$1\otimes1\to2$ 要去掉自旋 0 的**迹**。这个迹必须用
$\tilde g(p_{K^*_2})$（**耦合所在系**的度规）来取；
历史形式用 $\tilde g(p_\psi)$ 取，于是留下一个**事件依赖的**偏差。

---

## 2. 公式对照

### 2.1 历史形式（= 当前 `*_paper.py`）

论文在 431–623 行给出的 $KK^*_2$ 类公式（如 Eq.(511)）用的是
$\epsilon$ + $P^{(2)}(p_X)$ + 内层 $\epsilon\cdot p_X$ 的写法。
参考实现把它化简成 $P^{(2)}_{\psi}$ 直接收缩的形式：

$$
U^\mu_{PP}
= P^{(2)}_{\psi}{}^{\mu jkl}\;
  \tilde T^{(1)}_{(K^*_2\pi)\,j}\;
  \tilde t^{(1)}_{(\phi\pi)\,k}\;
  \tilde g_{K^*_2}{}^{lm}\;
  \tilde t^{(1)}_{(KK)\,m}
\tag{原式}
$$

其中（论文 Eq.(10)、Eq.(20)）

$$
\tilde t^{(1)}_{\mu}=\tilde g_{\mu\nu}(p_a)r^\nu B_1(Q_{abc}),\qquad
P^{(2)}_{\mu\nu\mu'\nu'}(p_a)=
\tfrac12(\tilde g_{\mu\mu'}\tilde g_{\nu\nu'}+\tilde g_{\mu\nu'}\tilde g_{\nu\mu'})
-\tfrac13\tilde g_{\mu\nu}\tilde g_{\mu'\nu'}
$$

**问题**：内层 $\tilde t^{(1)}_{k}\tilde g^{lm}\tilde t^{(1)}_{m}$
是一个**秩-2 但不无迹**的张量。原式指望 $P^{(2)}_{\psi}$ 顺手把迹去掉 ——
但 $P^{(2)}_{\psi}$ 用的是 $\tilde g(p_\psi)$，去的是 **$\psi$ 系**的迹。
耦合实际发生在 $K^*_2$ 系，两系不同（$K^*_2$ 在 $\psi$ 静止系里是动的，
$|\mathbf p|\approx0.51$ GeV；实测 $\max|\tilde g(p_{K^*_2})-\tilde g(p_\psi)|=0.34$，
2000 事例冻结样本）。

> ⚠️ **这一节的边界（2026-10-21 独立审阅后加）**：
> $P^{(2)}(a)$ 对称且相对 $\tilde g(a)$ 无迹，所以
> $P^{(2)}_{\psi}\cdot(u\otimes v)=P^{(2)}_{\psi}\cdot T^{(2)}_{\psi}(u,v)$。
> 这说明原式**在该收缩路径下等价于**"$\psi$ 系去迹"—— 是**代数等价**，
> **不是**"唯一根因"的证明。满足 A1 的构造不唯一
> （$T^{(2)}_{K^*_2}+A_{[kl]}+\alpha(e)\tilde g(\psi)_{kl}$ 同样满足）。
> 我们只声称：$K^*_2$ 系去迹**足以**消除所见失配。
> 详见 `review_kst2_frame_fix/gen/fragments/kst2_frame_fix_review.md`。

### 2.2 修正版（= 当前生产文件）

把内层显式地构造成**在 $K^*_2$ 系里无迹**的秩-2 张量：

$$
T^{(2)}_{K^*_2}{}_{kl}
= \tfrac12\Big(\tilde t^{(1)}_{k}\,\tilde v_{l}
              +\tilde t^{(1)}_{l}\,\tilde v_{k}\Big)
-\tfrac13\,\tilde g_{K^*_2\,kl}\;
  \tilde g_{K^*_2}^{mn}\,\tilde t^{(1)}_{m}\tilde v_{n},
\qquad
\tilde v_{l}\equiv \tilde g_{K^*_2\,lm}\,\tilde t^{(1)}_{(KK)}{}^{m}
\tag{修正}
$$

于是 $PP$ 的振幅变成

$$
U^\mu_{PP}
= P^{(2)}_{\psi}{}^{\mu jkl}\;
  \tilde T^{(1)}_{(K^*_2\pi)\,j}\;
  T^{(2)}_{K^*_2}{}_{kl}
$$

$FP$（顶层是 $F$ 波 $\tilde T^{(3)}$）同理，只是把顶层算符换掉、
再把 $T^{(2)}_{K^*_2}$ 的两个指标升上去：

$$
U^\mu_{FP}
= \tilde T^{(3)}{}^{\mu jk}_{\psi}\;
  P^{(2)}_{\psi\,jklm}\;
  T^{(2)}_{K^*_2}{}^{lm}
$$

### 2.3 为什么对称化是必需的（但数值上不改变结果）

$\tilde g$ 缩并只看到对称部分，所以"非对称去迹"与"对称去迹"**数值相同**
（实测两条路径 rel 都到 $\sim10^{-14}$，互相逐位一致到 $10^{-16}$）。
但写成对称形式才与 $1\otimes1\to2$ 的 CG 分解一致，也才对任意后续收缩稳健 ——
所以落地用对称形式（`T2_from_pair` 里就是对称的）。

---

## 3. 代码对照

两种形式都在本目录里，**用道名切换**：

| 形式 | 文件 | 道名（写进 spec 用这个） | A1 |
|---|---|---|---|
| 历史形式 | `u_PP_1_paper.py` | `u_PP_1_paper` | ❌ $6.99\times10^{-3}$ |
| **修正版（生产）** | `u_PP_1.py` | `u_PP_1` | ✅ $7.17\times10^{-15}$ |

### 3.1 历史形式（`u_PP_1_paper.py`）

```python
def u_PP_1_paper(psi, phi, f2, Kp, Km, Pip, Pim):
    kst2 = Kp + Km + Pim
    p2psi_cov = P2(psi)                          # P^(2)_psi
    p2psi_con = _raise_last(p2psi_cov, 4)
    gt_ks_cov = Gt_cov(kst2)                     # g~_{K*_2}
    t1_top_cov = T1_cov(psi, kst2, Pip)          # T~(1)_{(K*_2 4)}
    t1_in_cov = T1_cov(kst2, phi, Pim)           # t~(1)_{(phi 4)}
    t1_12_cov = T1_cov(phi, Kp, Km)              # t~(1)_{(12)}
    t1_12_con = _metric_axis(t1_12_cov, 1)
    #              ↓ 内层秩 2 但**不无迹**：去迹被隐式交给 P2(psi)（错的系）
    return _einsum("eijkl,ej,ek,elm,em->ei",
                   p2psi_con, t1_top_cov, t1_in_cov, gt_ks_cov, t1_12_con)
```

### 3.2 修正版（`u_PP_1.py`）

```python
def u_PP_1(psi, phi, f2, Kp, Km, Pip, Pim):
    kst2 = Kp + Km + Pim
    p2psi_con = _raise_last(P2(psi), 4)          # P^(2)_psi（顶层投影）
    gt_ks_cov = Gt_cov(kst2)                     # g~_{K*_2}
    t1_top_cov = T1_cov(psi, kst2, Pip)
    t1_in_cov = T1_cov(kst2, phi, Pim)
    t1_12_cov = T1_cov(phi, Kp, Km)
    t1_12_con = _metric_axis(t1_12_cov, 1)
    gphi = _einsum("elm,em->el", gt_ks_cov, t1_12_con)   # v = g~ · t~(1)_{(12)}
    #              ↓ 在 K*_2 系里无迹 —— **修正的那一步**
    t2_in_cov = T2_from_pair(kst2, t1_in_cov, gphi)
    return _einsum("eijkl,ej,ekl->ei", p2psi_con, t1_top_cov, t2_in_cov)
```

### 3.3 新增的内核原语

修正引入了一个通用原语 `T2_from_pair`（与 `T1_cov`/`T2_cov`/`T3_cov` 并列）：

```python
def T2_from_pair(frame, u, v):
    r"""把两个秩-1 分析器耦合成 J=2，**无迹化在 `frame` 系里做**：
        T^(2)_{uv} = 1/2(u_u v_v + v_u u_v) - 1/3 g~(frame)_{uv} g~(frame)^{ab} u_a v_b
    """
    gt_cov = Gt_cov(frame)
    gt_con = _raise_last(gt_cov, 2)
    sym = 0.5 * (xp.einsum("ei,ej->eij", u, v) + xp.einsum("ei,ej->eij", v, u))
    tr = xp.einsum("eij,ei,ej->e", gt_con, u, v)
    return sym - (1.0 / 3) * xp.einsum("eij,e->eij", gt_cov, tr)
```

与既有的 `T2_cov(a, b, c)` 的区别：`T2_cov` 的两个因子都是同一个
$\tilde g(a)(b-c)$；`T2_from_pair` 允许 $u$、$v$ 是**任意**两个秩-1 分析器，
参考系 `frame` 单独给出 —— 这正是本例需要的自由度。

---

## 4. 为什么只有这 4 条道有问题

$K^*_2$ 的 8 条道按"顶层波 × 内层波"分四族，**命中与否只由内层波决定**：

| 族 | 顶层 | 内层 | 内层是否自带正确系的无迹化 | A1 |
|---|---|---|---|---|
| `u_PF_*` | P | **F** | ✅ `T3_cov(kst2, ...)` 自带 $K^*_2$ 系的 $\tilde g$ | ✅ |
| `u_FF_*` | F | **F** | ✅ 同上 | ✅ |
| `u_PP_*` | P | **P** | ❌ 裸外积，去迹靠 $P^{(2)}(p_\psi)$ | ❌ |
| `u_FP_*` | F | **P** | ❌ 同上 | ❌ |

内层 F 波用 `T3_cov(kst2, phi, π)`，它的定义里就有
$\tilde r_\mu\tilde r_\nu\tilde r_\lambda-\frac15(\tilde r\cdot\tilde r)
(\tilde g\tilde r+\ldots)$，其中的 $\tilde g$ **就是** $\tilde g(p_{K^*_2})$ ——
无迹化已经在正确的系里做完，$P^{(2)}(p_\psi)$ 只剩顶层的作用，不再引入错配。

**为什么 $K^*_0$ 的内层也是 P 波却命中**：$J_{K^*}=0$ ⇒ 内层
$(l,s)=(1,1)$ 耦合成 **0**，是标量，**根本没有秩-2 无迹化这一步**。

---

## 5. 验收结果

三形式对拍（历史形式 / 修正版 / 级联），实测 2026-10-21，seed 7/11/23：

| 道 | 历史形式 rel | 修正版 rel | 两者之差 |
|---|---|---|---|
| `u_PP_1` | $2.00\times10^{-2}\sim2.47\times10^{-2}$ | $\le4.1\times10^{-14}$ | $\ge2.5\times10^{-2}$ |
| `u_PP_2` | $1.90\times10^{-1}\sim2.66\times10^{-1}$ | $\le5.9\times10^{-15}$ | $\ge2.0\times10^{-1}$ |
| `u_FP_1` | $1.40\times10^{-2}\sim2.09\times10^{-2}$ | $\le9.8\times10^{-15}$ | $\ge2.7\times10^{-2}$ |
| `u_FP_2` | $2.40\times10^{-1}\sim2.70\times10^{-1}$ | $\le6.4\times10^{-15}$ | $\ge2.1\times10^{-1}$ |

判据 A1 = 经 `EPS_A` 投到螺旋度后，与该道的级联振幅相差一个**全局复常数**
（常数本身不必是 1）。三条要点：

1. **历史形式不满足 A1**（rel $10^{-2}\sim10^{-1}$，且随事件变 —— 不是常数）；
2. **修正版满足 A1**（到机器精度）；
3. **两者确实不同**（差 $\ge2.5\times10^{-2}$）—— 所以不是空改。

A5（与 jax 参考实现逐位对拍）落在 `*_paper.py` 上，10/10 到 $\sim10^{-15}$；
`*_paper.py` 是逐式移植，**没有**被本次修正改动。

> ⚠️ **A1 通过的边界（2026-10-21 独立审阅后加）**：
> A1 每道允许一个复常数（2 个实自由度），四道共 8 个。
> 它能判出**事件依赖**的失配（历史形式的残差随运动学变），
> 但**能隐藏**每道的相位与归一化错误。
> 所以"修正版通过 A1"**不等于**"修正版在真实拟合中更优" ——
> 后者需要生成两套缓存做多道 toy 拟合，**尚未做**。
> 另外：修正后的常数恰为闭式（$c_{PP}=3/40$、$c_{FP}=\sqrt6/64$），
> 但审阅者指出这**不是**支持修正的证据（把正确张量乘 $k$ 仍得闭式 $3/(40k)$）。

---

## 6. 怎么生成"论文形式"

`emitter` 按 **`channel.name`** 去本目录下找同名 `.py` 文件
（`<channel.name>.py` 里有一个同名函数），取出函数体并内联。
所以**不用改任何代码**，只要在 spec 里把道名写成 `*_paper`，
就能生成历史形式的产物：

```toml
[[channel]]
name = "u_PP_1_paper"          # ← 决定用哪个文件
l = 1
s = 2
ls = { psi2s = [[1, 2]], Kstar = [[1, 1]], phi = [[1, 0]] }
tree = { psi2s = ["Kstar", "Pip"], Kstar = ["phi", "Pim"] }
```

把 4 条 `_paper` 道与 4 条修正道写进**同一份 spec**，就能得到一份
"两种形式共存"的产物，便于逐道对照。

⚠️ 名字里的 `_paper` 后缀**没有魔法** —— 换成任何名字都行，只要本目录下
有同名文件。约定 `*_paper` 只是为了可读性。

---

## 7. 为什么生产用修正版

* **物理**：$1\otimes1\to2$ 的无迹化必须在该耦合所在系里做；用 $\psi$ 系的
  $\tilde g$ 会留下一个**事件依赖的**偏差（不是常数，不能被拟合的自由
  复系数吸收）。这正是 A1 的残差"随样本变"的原因。
* **与级联一致**：修正后 4 条道对 `helicity_cascade` 的 A1 到机器精度，
  于是"协变 / 螺度 / 级联"三条路线在同一份 spec 上互相印证。
* **与 `phif2` 的 (41)(42) 缺陷同型**：那是同一个机制（母系自旋投影错配，
  修法是把投影搬到正确的系并去掉迹）在另一条链上的表现。本例是它在
  $K^*_2$ 内层上的再现。
* **A5 仍然成立**：逐式移植原样保留在 `*_paper.py`，与 jax 的对拍落在
  它身上；生产文件与历史形式的差则作为"修正确实改了东西"的阳性对照。
