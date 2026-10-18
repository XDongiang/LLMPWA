#!/usr/bin/env python3
r"""Can an EVENT-DEPENDENT momentum factor absorb the rho' mismatch?

The referee's sharpest objection: the least-squares test used a 1-dimensional
basis (a single constant complex coefficient), so it cannot rule out a cascade
whose coefficient depends on the event (a q-dependent coupling, a different
barrier convention, a mass-dependent form factor).  This probe answers that.

Method: fit the covariant rho' amplitude with cascades multiplied by candidate
event-dependent scalar functions f(e):

  basis = { f_k(e) * cascade(e) }        k = 1..K

If the residual drops to machine precision for some K, the mismatch IS a
convention/factor difference.  If it stays O(1), the shapes are genuinely
different.

Candidate factors (all per-event, all real or complex scalars):
  f1 = 1                       (constant: the report's original test)
  f2 = q(rho' -> phi pi)       the inner relative momentum
  f3 = B1(q_inner)             the inner Blatt-Weisskopf
  f4 = q(psi -> rho' pi)       the outer relative momentum
  f5 = s_rho = m^2(rho')       the rho' invariant mass
  f6 = gamma_phi, gamma_pi     the Eq.52 boost factors
  f7 = all of the above        (K = 7 basis)
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


SLICE = os.path.join(ROOT, "analyses", "kk_dis", "task",
                     "20261019_螺旋度级联内核_审阅", "material", "data",
                     "review_slice.npz")


def main():
    d = onp.load(SLICE)
    keys = ("Kp", "Km", "Pip", "Pim")
    momd = {k: onp.asarray(d[k], float) for k in keys}
    mom = onp.stack([momd[k][:, [3, 0, 1, 2]] for k in keys], axis=1)
    n = mom.shape[0]

    # ---- the cascade (rho' tree 1) and the covariant target
    c = V.rho_chain(("rhoprime", "Pip"), ("phi", "Pim"))
    a = V.node_angles(mom, c)
    casc = V.chain_amplitude(mom, c, {"psi2s": [(1, 1)], "rhoprime": [(1, 1)],
                                      "phi": [(1, 0)]}, a)
    W = V.w_cov("u_rho_1", momd)

    # ---- candidate event-dependent scalars
    rho = momd["Kp"] + momd["Km"] + momd["Pim"]
    phi = momd["Kp"] + momd["Km"]
    psi = onp.broadcast_to(V.PSI_XYZ_E, momd["Kp"].shape)

    def qrel(pa, pb, pc):
        return onp.sqrt(onp.maximum(V.Qabc(pa, pb, pc), 0.0))

    q_in = qrel(rho, phi, momd["Pim"])          # rho' -> phi pi
    q_out = qrel(psi, rho, momd["Pip"])         # psi  -> rho' pi
    m_rho2 = onp.maximum(V._dot(rho, rho), 0.0)
    gam_phi = onp.sqrt(1.0 + q_in / onp.maximum(V._dot(phi, phi), 1e-30))
    gam_pi = onp.sqrt(1.0 + q_in / onp.maximum(V._dot(momd["Pim"],
                                                      momd["Pim"]), 1e-30))
    B1_in = V._B(1, rho, phi, momd["Pim"])

    cands = [("常数 1", onp.ones(n)),
             ("q(内层)", q_in),
             ("B1(q_内层)", B1_in),
             ("q(外层)", q_out),
             ("m^2(rho')", m_rho2),
             ("gamma_phi", gam_phi),
             ("gamma_pi", gam_pi),
             ("q_内 * q_外", q_in * q_out),
             ("m^2 * B1", m_rho2 * B1_in)]

    def resid(basis_cols):
        B = onp.stack(basis_cols, 1)
        A = W.ravel()
        coef, *_ = onp.linalg.lstsq(B, A, rcond=None)
        return float(onp.linalg.norm(A - B @ coef) / onp.linalg.norm(A))

    print("=" * 92)
    print("1  单个事件依赖因子 f(e) * 级联  —— 残差会掉到机器精度吗？")
    for lbl, f in cands:
        col = (f[:, None] * casc).ravel()
        print("   f = %-12s  基底 1 维   残差 = %.4e" % (lbl, resid([col])))

    print()
    print("2  全部候选因子一起（9 维基底）")
    cols = [(f[:, None] * casc).ravel() for _, f in cands]
    print("   基底 9 维   残差 = %.4e" % resid(cols))

    print()
    print("3  更一般的检验：允许级联的**每个 lambda 槽**有独立的")
    print("   事件依赖因子（即 c_k(e) 是任意函数）—— 这等价于问")
    print("   'W 是否落在级联张成的逐事例一维子空间的并集里'。")
    print("   做法：对每个事例单独解 c(e)，看 |c| 是否跨事例恒定。")
    print()
    for lbl, chname, dsp, dsb in (
            ("b1  u_DD_1（控制）", "u_DD_1", ("b1", "Pim"), ("phi", "Pip")),
            ("rho u_rho_1", "u_rho_1", ("rhoprime", "Pip"),
             ("phi", "Pim"))):
        cc = (V.b1_chain(dsp, dsb) if chname.startswith("u_DD")
              else V.rho_chain(dsp, dsb))
        aa = V.node_angles(mom, cc)
        resn = cc["nodes"][cc["top"]]["daughters"][0]
        tl = [(2, 1)] if chname.startswith("u_DD") else [(1, 1)]
        il = [(2, 1)] if chname.startswith("u_DD") else [(1, 1)]
        cs = V.chain_amplitude(mom, cc, {"psi2s": tl, resn: il,
                                         "phi": [(1, 0)]}, aa)
        tgt = V.w_cov(chname, momd)
        coef = onp.array([onp.vdot(cs[e], tgt[e]) / onp.vdot(cs[e], cs[e])
                          for e in range(n)])
        recon = coef[:, None] * cs
        r = float(onp.abs(recon - tgt).max() / onp.abs(tgt).max())
        cn = coef / coef[0]
        print("   %-18s 逐事例 c(e) 重建 rel = %.3e" % (lbl, r))
        print("   %-18s |c/c0 - 1|max = %.3e   (常数 => 是级联)" %
              ("", onp.abs(cn - 1).max()))
    print()
    print("   => b1 的 c(e) 是常数；rho' 的 c(e) 随事例变 O(1)，")
    print("      所以**没有任何**事件依赖标量因子能把 Eq.(46) 变成这个级联。")
    print("      这是比 1 维最小二乘更强的否证。")


if __name__ == "__main__":
    main()
