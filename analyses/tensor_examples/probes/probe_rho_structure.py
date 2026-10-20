#!/usr/bin/env python3
"""Is the covariant rho' channel a 2-node cascade, or a top-tensor x scalar?

本探针回答"ρ′ 的 Eq.(46) 到底是不是逐节点级联"的**结构**问题，与
`probe_rho_rank.py`（子空间维数/主角）、`probe_rho_qfactor.py`（事件依赖
标量因子）、`probe_rho_decomposition.py`（逐节点分解）互为独立证据。

四问（全部在同一个事例集上）：

  1. `u_rho` 作为 4-矢量，是否逐事例正比于顶层张量 `T1_cov(psi, rho, pi)`？
  2. `W_cov` 是否逐事例正比于 `W_top`（即"顶层张量 × 标量"）？
  3. 完整的逐节点级联（逐节点 (l,s) 缩并）与 `W_cov` 差多少？
  4. 同一条链的 **omega** 口径与 `W_cov` 差多少？

⚠️ 历史：本文件曾 `import harness`（`_work/scratch/harness.py`），而那个模块
**从未入库**、所在目录已删除，且用到的 `harness.recursive_angles` /
`harness.chain_amp` 也早已改名。所以它长期处于"跑不起来"的状态，且没有任何
文档/测试引用它。这里改用**现行**内核 API（`HC.node_angles` /
`HC.chain_amplitude` / `HC.with_tree`）重写，保持原来的四问不变。
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "agent", "tensor", "tests"))

import test_ex3_channels as T                                       # noqa: E402
from agent.tensor import spec as S                                  # noqa: E402
from agent.tensor.kernels import covariant as COV                   # noqa: E402
from agent.tensor.kernels import helicity_cascade as HC             # noqa: E402
from agent.tensor.kernels import omega as RT                        # noqa: E402
from agent.tensor.kernels.helicity import EPS_A                     # noqa: E402
from agent.tensor.kernels.covariant import T1_cov                   # noqa: E402

np.set_printoptions(precision=5, linewidth=200)
N = 24
momd = T.physical(n=N, seed=11)
mom = np.stack([momd[k][:, [3, 0, 1, 2]]
                for k in ["Kp", "Km", "Pip", "Pim"]], axis=1)

SPEC = os.path.join(ROOT, "analyses", "tensor_examples", "specs", "ex3_rhopi.toml")


def Wcov(ch):
    return np.einsum("lm,em->el", EPS_A, COV.compute_event(ch, momd)[..., :3])


def rel(A, B):
    den = np.vdot(A, A)
    if np.abs(den) == 0:
        return float("nan")
    s = np.vdot(B, A) / den
    return float(np.abs(B - s * A).max() / np.abs(B).max())


def c_scalar(A, B):
    """逐事例的最佳复标量 c(e)，使 B[e] ≈ c(e) A[e]。

    A 在某个事例上恒零时返回 nan（**不要**返回 0 —— 那会把"无法比较"
    伪装成"完全不成比例"）。
    """
    num = np.einsum("ei,ei->e", np.conj(A), B)
    den = np.einsum("ei,ei->e", np.conj(A), A)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(den) == 0, np.nan, num / den)


for ch, who in (("u_rho_1", "Pim"), ("u_rho_2", "Pip")):
    spec = S.load_spec(SPEC)
    chain0 = S.to_chain(spec)
    # 逐道取它自己的镜像树（`_1` 的旁观者是 Pip，`_2` 的是 Pim）
    cc = [x for x in S.cascade_channels(spec) if x["name"] == ch][0]
    c = HC.with_tree(chain0, cc["tree"])
    ang = HC.node_angles(mom, c)

    rho = momd["Kp"] + momd["Km"] + momd[who]
    psi = np.broadcast_to(np.asarray(COV.PSI2S_DEFAULT, float), rho.shape)
    # 顶层张量：psi -> rho' + (旁观者 pi)
    Tt = T1_cov(psi, rho, momd["Pim"] if who == "Pip" else momd["Pip"])
    u = COV.compute_event(ch, momd)

    print("=" * 100)
    print("%s : rho' = phi + %s，旁观者 = %s" % (ch, who, "Pip" if who == "Pim" else "Pim"))

    # 1) u 作为 4-矢量是否正比于顶层张量（逐事例复标量）。
    #    |c| == 0 是**结论**不是退化：说明 u ⟂ T1_cov 逐事例成立。
    rr = c_scalar(Tt, u)
    res = np.abs(u - rr[:, None] * Tt).max() / np.abs(u).max()
    cosang = np.abs(np.einsum("ei,ei->e", np.conj(Tt), u)) / (
        np.linalg.norm(Tt, axis=1) * np.linalg.norm(u, axis=1))
    print("   1  u =? c(e) * T1_cov(psi, rho, pi)      resid = %.3e   |c| %.4f..%.4f"
          % (res, np.nanmin(np.abs(rr)), np.nanmax(np.abs(rr))))
    print("      （|cos∠(T1_cov, u)| max = %.3e —— 若 ~0 则两者正交，" \
          "问题 1 不构成检验）" % np.nanmax(cosang))

    # 2) W_cov 是否 = c(e) * W_top（"顶层张量 × 标量"）
    Wt = np.einsum("lm,em->el", EPS_A, Tt[..., :3])
    Wc = Wcov(ch)
    cw = c_scalar(Wt, Wc)
    res2 = np.abs(Wc - cw[:, None] * Wt).max() / np.abs(Wc).max()
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = np.nanmax(np.abs(cw / cw[0] - 1))
    print("   2  W_cov =? c(e) * W_top                 resid = %.3e   "
          "|c/c0-1|max = %.3e" % (res2, spread))

    # 3) 完整逐节点级联（现行 API）
    A = HC.chain_amplitude(
        mom, c, {"psi2s": [(1, 1)], "rhoprime": [(1, 1)], "phi": [(1, 0)]}, ang=ang)
    print("   3  cascade vs W_cov                      rel   = %.3e" % rel(A, Wc))

    # 4) 只有顶层张量（不做内层缩并）
    print("   4  W_top vs W_cov                        rel   = %.3e" % rel(Wt, Wc))

    # 5) omega 口径（同一条链；omega 只算 D0*D1*D2）。
    #    必须走 `RT.assemble` —— 手写 "enab,ab->en" 会**静默广播**成错误结果
    #    （M 的 m 轴被当成了 lam_b），所以这里用内核自己的缩并。
    top, nb, nc = chain0["top"], *chain0["nodes"][chain0["top"]]["daughters"]
    M = RT.cg_matrix([(1, 1)], chain0["nodes"][top]["J"],
                     chain0["nodes"][nb]["J"], chain0["nodes"][nc]["J"],
                     RT.LAM_SLOTS[chain0["nodes"][nb]["J"]],
                     RT.LAM_SLOTS[chain0["nodes"][nc]["J"]])
    Wom = RT.assemble(M, RT.omega_from_momenta(mom, c, covariant=True),
                      np.ones(1), np.ones((1, mom.shape[0])))[0]
    print("   5  omega vs W_cov                        rel   = %.3e" % rel(Wom, Wc))
