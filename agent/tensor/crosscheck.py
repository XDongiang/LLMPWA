#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨算法一致性检验：协变张量路线 vs Omega 路线。

两条路线是**独立实现**的：

    route1 / cartesian   T_i(e,mu)          <- kernels/covariant.py（Chung 协变张量）
    route3 / helicity    W_i = EPS_A . T_i  <- 同一份 T_i 旋到螺度
    route4 / omega       Omega/K/M/g/norm   <- kernels/omega.py（级联角 + CG 装配）

它们必须给出同一个物理量，只差**逐道的复常数** c_i：

    W_i^{cov}(e,lam) = c_i * g_i * K_i(e) * (M_i . Omega)_e,lam        （逐 e 成立）

c_i 是逐道的整体归一化（耦合常数 + 归一化 + 相位），由拟合吸收，所以**不能**
要求 c_i = 1；能要求的是 c_i **与事例无关**（即残差是数值噪声而不是结构性错误）。

这条恒等式是整份 cache 最强的自检：
* 任一条路线的约定写错（EPS_A 行序/符号、度规、轴序、psi 口径）-> 残差爆炸；
* 某条道的 (l,s) 与道函数不匹配 -> 该道单独爆；
* CG 矩阵 M 的轴序反了 -> 全部爆。

不需要四动量：只看已生成的 cache 文件即可，所以可以放在 `run` 之后直接跑。

    from agent.tensor import crosscheck
    r = crosscheck.check_dirs(cartesian_dir, omega_dir)
    print(r.report())

约定：cartesian 侧读 `<chain>_kk.npy`（(n_ch,N,3或4)），omega 侧读
`Omega{suf}.npy` / `K{suf}.npy` / `M{suf}.npy` / `g{suf}.npy`。
"""
from __future__ import annotations

import os

import numpy as np

# 与 kernels/helicity.py 的 EPS_A 必须一致（这里只用来做检验，
# 不参与生产；不一致的话这条检验自己就会失败，属于预期行为）。
EPS_A = np.array([[1 / np.sqrt(2), -1j / np.sqrt(2), 0],
                  [0, 0, 1],
                  [-1 / np.sqrt(2), -1j / np.sqrt(2), 0]], dtype=np.complex128)

# 判定阈值。实测（2000 事例、真实样本）残差 ~1e-9 ~ 1e-12；
# 结构性错误（约定写错）是 1e-1 量级，所以 1e-6 有巨大余量。
DEFAULT_TOL = 1e-6


class CrossCheckError(Exception):
    pass


def _precompute(T):
    """T (..., >=3) -> W (..., 3)，与 kernels/helicity.precompute 相同。

    保留复数 dtype：`already_rotated=True` 时传进来的就是复数 W。
    """
    T = np.asarray(T)
    if T.shape[-1] < 3:
        raise CrossCheckError(
            "cartesian cache 最后一维是 %d，需要 >=3（x,y,z）；"
            "丢了 z 就无法复现 lambda=0 那一行" % T.shape[-1])
    return np.einsum("lm,...m->...l", EPS_A, T[..., :3].astype(np.complex128,
                                                              copy=False))


def check_arrays(T, Omega, K, M, g=None, names=None, tol=DEFAULT_TOL,
                 min_frac=1e-3, already_rotated=False):
    """核心检验：逐道算 c_i 与残差。

    Parameters
    ----------
    T      : (n_ch, N, >=3)  协变张量（cartesian cache）
    Omega  : (N, nA, nb, nc)
    K      : (n_ch, N)
    M      : (n_ch, nA, nb, nc)
    g      : (n_ch,) or None（None 视为全 1）
    names  : 道名，用于报告
    tol    : 相对残差上限
    min_frac : 参与拟合的事例比例下限（避免只用几个大数拟合）
    already_rotated : True 表示 `T` 已经是螺度振幅 W（读了 `_hel.npy`），
                      不要再乘一次 EPS_A。

    Returns
    -------
    dict: {channel: {...}} + 汇总键 "max_rel"、"ok"、"worst"
    """
    T = np.asarray(T)
    Omega = np.asarray(Omega)
    K = np.asarray(K)
    M = np.asarray(M)
    n_ch = T.shape[0]
    if g is None:
        g = np.ones(n_ch)
    g = np.asarray(g, dtype=np.float64).reshape(-1)

    if names is None:
        names = ["ch%d" % i for i in range(n_ch)]

    # 形状自洽。注意 M 收缩的是 Omega 的**后两个**轴 (nb,nc)：
    #   P[i,e,nA] = sum_{a,b} M[i,a,b] * Omega[e,nA,a,b]
    if K.shape != (n_ch, T.shape[1]):
        raise CrossCheckError("K 形状 %s 与 (n_ch=%d, N=%d) 不符"
                              % (K.shape, n_ch, T.shape[1]))
    if M.shape[0] != n_ch or M.shape[1:] != Omega.shape[2:]:
        raise CrossCheckError(
            "M 形状 %s 与 n_ch=%d / Omega 的后两轴 (nb,nc)=%s 不符"
            % (M.shape, n_ch, Omega.shape[2:]))
    if Omega.shape[1] != T.shape[2]:
        raise CrossCheckError(
            "Omega 的 nA=%d 与 cartesian cache 的螺旋度维 %d 不符"
            % (Omega.shape[1], T.shape[2]))

    W_cov = T if already_rotated else _precompute(T)      # (n_ch, N, 3)
    P = np.einsum("iab,enab->ien", M, Omega)              # (n_ch, N, nA)
    W_omg = g[:, None, None] * K[:, :, None] * P          # (n_ch, N, nA)

    out = {}
    for i, nm in enumerate(names):
        a, b = W_cov[i], W_omg[i]                         # (N, nA)
        scale = np.abs(b).max()
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            out[nm] = {"c": None, "spread": np.inf, "rel": np.inf,
                       "n_used": 0, "note": "有非有限值"}
            continue
        if scale <= 0:
            out[nm] = {"c": None, "spread": np.inf, "rel": np.inf,
                       "n_used": 0, "note": "Omega 侧恒为 0"}
            continue
        # 只用模长够大的分量做拟合（小到 1e-12 的分量信噪比太低）
        mask = np.abs(b) > max(min_frac * scale, 1e-300)
        n_used = int(mask.sum())
        if n_used < 10:
            out[nm] = {"c": None, "spread": np.inf, "rel": np.inf,
                       "n_used": n_used, "note": "可用分量太少"}
            continue
        # 最小二乘的复常数：c = <a,b> / <b,b>（对 b 取共轭）
        num = np.vdot(b[mask], a[mask])
        den = np.vdot(b[mask], b[mask])
        c = num / den
        resid = np.abs(a - c * b).max() / max(np.abs(a).max(), 1e-300)
        spread = np.abs(a[mask] / b[mask] - c).max() / max(abs(c), 1e-300)
        out[nm] = {"c": complex(c), "spread": float(spread),
                   "rel": float(resid), "n_used": n_used, "note": ""}

    rels = {nm: v["rel"] for nm, v in out.items() if v["rel"] == v["rel"]}
    worst = max(rels, key=rels.get) if rels else None
    out["max_rel"] = max(rels.values()) if rels else float("inf")
    out["worst"] = worst
    out["ok"] = bool(rels) and out["max_rel"] < tol
    out["tol"] = tol
    out["names"] = list(names)
    return out


# --------------------------------------------------------------- 读文件
def _find(d, cands):
    for c in cands:
        p = os.path.join(d, c)
        if os.path.exists(p):
            return p
    return None


def load_dirs(cartesian_dir, omega_dir, chain="phif2", suffix="",
              cart_name=None):
    """从两个产物目录读 cache。返回 (T, Omega, K, M, g, already_rotated)。

    优先读 `<chain>_kk.npy`（协变张量，还要乘 EPS_A）；没有才退回
    `<chain>_hel.npy`（已经是螺度振幅 W，`already_rotated=True`，不能再乘）。
    """
    if cart_name:
        cpath, rotated = os.path.join(cartesian_dir, cart_name), (
            "_hel." in cart_name)
    else:
        cpath = _find(cartesian_dir, ["%s_kk.npy" % chain])
        rotated = False
        if cpath is None:
            cpath = _find(cartesian_dir, ["%s_hel.npy" % chain])
            rotated = cpath is not None
    if cpath is None or not os.path.exists(cpath):
        raise CrossCheckError(
            "%s 里找不到 %s_kk.npy（或 %s_hel.npy）"
            % (cartesian_dir, chain, chain))
    opath = _find(omega_dir, ["Omega%s.npy" % suffix])
    kpath = _find(omega_dir, ["K%s.npy" % suffix])
    mpath = _find(omega_dir, ["M%s.npy" % suffix])
    gpath = _find(omega_dir, ["g%s.npy" % suffix])
    if not all((opath, kpath, mpath)):
        raise CrossCheckError(
            "%s 里缺 Omega/K/M（后缀 %r）" % (omega_dir, suffix))
    T = np.load(cpath)
    Omega = np.load(opath)
    K = np.load(kpath)
    M = np.load(mpath)
    g = np.load(gpath) if gpath else None
    return T, Omega, K, M, g, rotated


def check_dirs(cartesian_dir, omega_dir, chain="phif2", suffix="",
               names=None, tol=DEFAULT_TOL):
    T, Omega, K, M, g, rotated = load_dirs(cartesian_dir, omega_dir,
                                           chain=chain, suffix=suffix)
    return check_arrays(T, Omega, K, M, g, names=names, tol=tol,
                        already_rotated=rotated)


def report(res, title="跨算法一致性（covariant/EPS_A  vs  Omega）"):
    """把 check_arrays 的结果排成人读文本。"""
    lines = [title]
    for nm in res.get("names", []):
        v = res[nm]
        if v["c"] is None:
            lines.append("  %-10s **失败**  %s" % (nm, v["note"]))
            continue
        flag = "OK " if v["rel"] < res["tol"] else "**FAIL**"
        lines.append("  %-10s %s  c=%12.6f%+.6fj  rel=%.3e  分量=%d"
                     % (nm, flag, v["c"].real, v["c"].imag, v["rel"],
                        v["n_used"]))
    lines.append("  最大相对残差 %.3e（阈值 %.0e）-> %s"
                 % (res["max_rel"], res["tol"],
                    "PASS" if res["ok"] else "**FAIL**"))
    return "\n".join(lines)
