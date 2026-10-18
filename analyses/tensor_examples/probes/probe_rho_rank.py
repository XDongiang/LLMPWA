#!/usr/bin/env python3
r"""Is the rho' "not a cascade" conclusion robust, or an artefact of a
rank-deficient A_top?

A referee objected: for rho' the top node is  psi -> rho'(1) + pi(0), so the
3x3 matrix A_top has a null direction and `v = lstsq(A_top, W)` is NOT unique.
The report's phrasing "residual ~0 proves v is unique" is therefore wrong.

What is actually true (this probe):
  * A_top is rank 2 with a CLEAN gap (s3/s1 ~ 1e-17): the null direction is
    exactly the lam_res = 0 slot, which carries zero CG weight.
  * W_cov lies in the IMAGE of A_top (reconstruction residual ~3e-16), so v is
    unique **on the image**, which is what the R test compares.
  * The direct least-squares residual (no inversion at all) is 0.272 -- this is
    the load-bearing evidence and it is unaffected by the rank issue.

It then attacks the referee's alternative hypothesis: could a DIFFERENT
intermediate-state projector (the inner epsilon contracting p_rho' instead of
p_psi) make Eq.(46) a cascade?  Answer: no.
"""
from __future__ import annotations

import os
import sys

import numpy as onp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
for _p in (ROOT,
           os.path.join(ROOT, "analyses", "kk_dis", "task",
                        "20261019_螺旋度级联内核_审阅", "material")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import verify as V                                                  # noqa: E402
from agent.tensor.kernels import helicity_cascade as HC             # noqa: E402


SLICE = os.path.join(ROOT, "analyses", "kk_dis", "task",
                     "20261019_螺旋度级联内核_审阅", "material", "data",
                     "review_slice.npz")


def load():
    d = onp.load(SLICE)
    keys = ("Kp", "Km", "Pip", "Pim")
    momd = {k: onp.asarray(d[k], float) for k in keys}
    mom = onp.stack([momd[k][:, [3, 0, 1, 2]] for k in keys], axis=1)
    return momd, mom


def main():
    momd, mom = load()
    n = mom.shape[0]

    print("=" * 92)
    print("1  A_top 的秩与奇异值（rho' 顶层 psi -> rho'(1) + pi(0)）")
    ch = V.rho_chain(("rhoprime", "Pip"), ("phi", "Pim"))
    ang = V.node_angles(mom, ch)
    A_top = V.node_amplitude(mom, ch, "psi2s", [(1, 1)], ang)[:, :, :, 0]
    sv = onp.linalg.svd(A_top, compute_uv=False)
    print("   s3/s1:  中位数 = %.3e   最大 = %.3e"
          % (onp.median(sv[:, 2] / sv[:, 0]), (sv[:, 2] / sv[:, 0]).max()))
    print("   s2/s1:  中位数 = %.3e   最小 = %.3e"
          % (onp.median(sv[:, 1] / sv[:, 0]), (sv[:, 1] / sv[:, 0]).min()))
    print("   => 秩 = 2，且间隙干净（s3/s1 ~ 1e-17）")

    # which slot is the null direction?
    _, _, Vt = onp.linalg.svd(A_top[0])
    null = Vt[2]
    print("   零方向 = %s   （|null_1| = %.3e）"
          % (onp.array2string(null, precision=4), abs(null[1])))
    print("   => 零方向**就是** lam_res = 0 那一槽（它没有 CG 权重）")

    print()
    print("=" * 92)
    print("2  W_cov 是否落在 A_top 的像空间里（决定 v 在像上是否唯一）")
    Wc = V.w_cov("u_rho_1", momd)
    res = []
    for e in range(n):
        x, *_ = onp.linalg.lstsq(A_top[e], Wc[e], rcond=None)
        res.append(onp.linalg.norm(A_top[e] @ x - Wc[e])
                   / max(onp.linalg.norm(Wc[e]), 1e-300))
    print("   ||A_top v - W||/||W||:  中位数 = %.3e   最大 = %.3e"
          % (onp.median(res), max(res)))
    print("   => W 在像里，所以 v 的**像空间分量**是唯一的（零方向不参与）")

    print()
    print("=" * 92)
    print("3  R 检验只在 lam_res = +-1 两槽上做（跳过零方向那一槽）")
    v = onp.stack([onp.linalg.lstsq(A_top[e], Wc[e], rcond=None)[0]
                   for e in range(n)])
    A_res = V.node_amplitude(mom, ch, "rhoprime", [(1, 1)], ang)[:, :, :, 0]
    A_phi = V.node_amplitude(mom, ch, "phi", [(1, 0)], ang)[:, :, 0, 0]
    vc = onp.einsum("eab,eb->ea", A_res, A_phi)
    print("   |v| 逐槽最大：", onp.array2string(onp.abs(v).max(0), precision=4))
    print("   |v_casc| 逐槽最大：", onp.array2string(onp.abs(vc).max(0), precision=4))
    for k in (0, 2):
        a, b = vc[:, k], v[:, k]
        m = ((onp.abs(a) > 1e-9 * onp.abs(vc).max())
             & (onp.abs(b) > 1e-9 * onp.abs(v).max()))
        r = b[m] / a[m]
        print("   lam_res=%+d  可用 %d 事例   |R/R0-1|max = %.3e"
              % (k - 1, m.sum(), onp.abs(r / r[0] - 1).max()))

    print()
    print("=" * 92)
    print("4  直接最小二乘（**不**做任何反演）—— 这是承重证据")
    for label, chname, dsp, dsb, tl in (
            ("b1  u_DD_1", "u_DD_1", ("b1", "Pim"), ("phi", "Pip"), [(2, 1)]),
            ("rho u_rho_1", "u_rho_1", ("rhoprime", "Pip"),
             ("phi", "Pim"), [(1, 1)])):
        c = (V.b1_chain(dsp, dsb) if chname.startswith("u_DD")
             else V.rho_chain(dsp, dsb))
        a = V.node_angles(mom, c)
        resn = c["nodes"][c["top"]]["daughters"][0]
        basis = []
        for il in ([(0, 1), (2, 1)] if chname.startswith("u_DD") else [(1, 1)]):
            basis.append(V.chain_amplitude(
                mom, c, {"psi2s": tl, resn: [il], "phi": [(1, 0)]},
                a).ravel())
        B = onp.stack(basis, 1)
        A = V.w_cov(chname, momd).ravel()
        coef, *_ = onp.linalg.lstsq(B, A, rcond=None)
        print("   %-14s 基底 %d 维   残差 = %.3e"
              % (label, len(basis),
                 onp.linalg.norm(A - B @ coef) / onp.linalg.norm(A)))
    print("   => rho' 的 0.272 与反演无关，秩亏不影响它")

    print()
    print("=" * 92)
    print("5  审稿人的替代假设：内层 epsilon 换成 p_rho'（不同中间态投影）")
    print("   rho' -> phi pi 的允许 (l,s)：")
    # enumerate allowed (l,s): Ja=1, Pa=-1; phi Jb=1 Pb=-1; pi Jc=0 Pc=-1
    allowed = []
    for L in range(5):
        if -1 != (-1) * (-1) * (-1) ** L:
            continue
        for S in range(abs(1 - 0), 1 + 0 + 1):
            if abs(L - S) <= 1 <= L + S:
                allowed.append((L, S))
    print("      允许的 (l,s) =", allowed, "  -> 只有一项（计数律：1 个独立振幅）")
    print("      所以换 (l,s) 这条路不存在 —— 内层只有 P 波一项")

    print()
    print("   用 p_rho' 而不是 p_psi 的横向投影重建 Eq.(46)（手写变体）")
    # rebuild the inner epsilon with p_rho' and see if it matches the cascade
    def rho_variant(use_rho_momentum):
        rho = momd["Kp"] + momd["Km"] + momd["Pim"]
        psi = onp.broadcast_to(V.PSI_XYZ_E, momd["Kp"].shape)
        inner_p = rho if use_rho_momentum else psi
        leci_mu = V._metric_axis(V.leci_four(), 0)
        return V._einsum(
            "ijkl,ej,ek,lmno,em,en,eo->ei",
            leci_mu, psi, V._metric_axis(V.T1_cov(psi, rho, momd["Pip"]), 1),
            V._leci_con(), V._metric_axis(inner_p, 1),
            V.T1_cov(rho, momd["Kp"] + momd["Km"], momd["Pim"]),
            V.T1_cov(momd["Kp"] + momd["Km"], momd["Kp"], momd["Km"]))

    W_psi = rho_variant(False)
    W_rho = rho_variant(True)
    Wref = V.ch_u_rho_1(onp.broadcast_to(V.PSI_XYZ_E, momd["Kp"].shape),
                        momd["Kp"] + momd["Km"],
                        momd["Pip"] + momd["Pim"],
                        momd["Kp"], momd["Km"], momd["Pip"], momd["Pim"])
    print("      与 chains_rhopi/u_rho_1.py 的逐位一致性 rel = %.3e"
          % (onp.abs(W_psi - Wref).max() / onp.abs(Wref).max()))

    c = V.rho_chain(("rhoprime", "Pip"), ("phi", "Pim"))
    a = V.node_angles(mom, c)
    # NOTE: the cascade output is ALREADY in EPS_A row order (report §4), so it
    # must be compared to EPS_A . T_cov DIRECTLY -- applying EPS_A again is the
    # double-application bug.
    Wcasc = V.chain_amplitude(mom, c, {"psi2s": [(1, 1)],
                                       "rhoprime": [(1, 1)],
                                       "phi": [(1, 0)]}, a)
    for lbl, W4 in (("内层 epsilon 用 p_psi", W_psi),
                    ("内层 epsilon 用 p_rho'", W_rho)):
        W = onp.einsum("lm,em->el", V.EPS_A, W4[..., :3])
        s = onp.vdot(Wcasc, W) / onp.vdot(W, W)
        r = float(onp.abs(Wcasc - s * W).max() / onp.abs(Wcasc).max())
        print("      %-26s  vs 级联  rel = %.3e" % (lbl, r))
    print("   => 两种投影都不是级联（0.24 / 0.22），换动量不解决问题")

    print()
    print("=" * 92)
    print("6  反过来：级联在**协变道张成的空间**里的残差")
    print("   协变侧只有 2 条道（u_rho_1 / u_rho_2，两个旁观者取法）。")
    print("   如果 Eq.(46) 真是级联，级联就应落在它们张成的空间里。")
    basis = []
    for nm in ("u_rho_1", "u_rho_2"):
        T = V.COV[nm](onp.broadcast_to(V.PSI_XYZ_E, momd["Kp"].shape),
                      momd["Kp"] + momd["Km"],
                      momd["Pip"] + momd["Pim"],
                      momd["Kp"], momd["Km"], momd["Pip"], momd["Pim"])
        basis.append(onp.einsum("lm,em->el", V.EPS_A, T[..., :3]).ravel())
    B = onp.stack(basis, 1)
    A = Wcasc.ravel()
    coef, *_ = onp.linalg.lstsq(B, A, rcond=None)
    print("   级联在 2 维协变基底里的残差 = %.3e"
          % (onp.linalg.norm(A - B @ coef) / onp.linalg.norm(A)))
    print("   （对照：b1 的级联在 8 维协变基底里应为机器精度）")


if __name__ == "__main__":
    main()
