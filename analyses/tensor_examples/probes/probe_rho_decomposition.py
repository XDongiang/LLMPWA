#!/usr/bin/env python3
"""Decisive decomposition using the KERNEL's own node amplitudes.

  A_top[e,la,lr]  = node_amplitude(top)   (la = lam_psi, lr = lam_resonance)
  v[e,lr]         = solve(A_top, W_cov)   (what the covariant amplitude says the
                                           intermediate-helicity content is)
  v_casc[e,lr]    = sum_lp A_res[e,lr,lp] A_phi[e,lp]     (the cascade prediction)

Ratio R = v / v_casc.  Constant across events  ->  the covariant channel IS the
cascade.  Non-constant -> the covariant channel is not a naive cascade.
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "agent", "tensor", "tests"))
import test_ex3_channels as T
from agent.tensor import spec as S
from agent.tensor.kernels import helicity_cascade as HC
from agent.tensor.kernels import covariant as COV
from agent.tensor.kernels.helicity import EPS_A

np.set_printoptions(precision=4, suppress=True, linewidth=210)
N = 24
momd = T.physical(n=N, seed=11)
mom = np.stack([momd[k][:, [3, 0, 1, 2]] for k in ["Kp", "Km", "Pip", "Pim"]], axis=1)


def run(sp_path, chname, tree, top_ls, inner_ls, phi_ls=(1, 0)):
    spec = S.load_spec(os.path.join(ROOT, sp_path))
    c = HC.with_tree(S.to_chain(spec), tree)
    ang = HC.node_angles(mom, c)
    nodes = c["nodes"]; top = c["top"]
    res = nodes[top]["daughters"][0]
    phi = nodes[res]["daughters"][0]
    Wc = np.einsum("lm,em->el", EPS_A, COV.compute_event(chname, momd)[..., :3])

    A_top = HC.node_amplitude(mom, c, top, [top_ls], ang=ang)[:, :, :, 0]
    A_res = HC.node_amplitude(mom, c, res, [inner_ls], ang=ang)[:, :, :, 0]
    A_phi = HC.node_amplitude(mom, c, phi, [phi_ls], ang=ang)[:, :, 0, 0]

    v = np.stack([np.linalg.lstsq(A_top[e], Wc[e], rcond=None)[0] for e in range(N)])
    res_top = max(np.linalg.norm(A_top[e] @ v[e] - Wc[e]) for e in range(N)) / np.abs(Wc).max()
    v_casc = np.einsum("eab,eb->ea", A_res, A_phi)
    Wcasc = np.einsum("eab,eb->ea", A_top, v_casc)
    print("=" * 100)
    print("%s  top_ls=%s inner_ls=%s" % (chname, top_ls, inner_ls))
    print("  top lstsq residual = %.3e   (0 => A_top spans W_cov)" % res_top)
    print("  cascade W vs W_cov  rel = %.3e" % (np.abs(Wcasc - Wc).max() / np.abs(Wc).max()))
    for k in range(A_top.shape[2]):
        a, b = v_casc[:, k], v[:, k]
        m = (np.abs(a) > 1e-9 * np.abs(v_casc).max()) & (np.abs(b) > 1e-9 * np.abs(v).max())
        if m.sum() < 3:
            print("  lam_res=%+d  skip (%d usable)" % (RT_LAM(c, top, k), m.sum())); continue
        r = b[m] / a[m]; rn = r / r[0]
        print("  lam_res=%+d  |R| = %.4f..%.4f  |R/R0-1|max = %.3e  arg(R)/pi = %s"
              % (RT_LAM(c, top, k), np.abs(r).min(), np.abs(r).max(), np.abs(rn - 1).max(),
                 np.array2string(np.angle(r) / np.pi, precision=3)))


def RT_LAM(c, top, k):
    J = int(c["nodes"][c["nodes"][top]["daughters"][0]]["J"])
    return k - J


print("### CONTROL: b1 u_DD_1  (even l at both nodes)")
run("analyses/tensor_examples/specs/ex3_b1pi.toml", "u_DD_1",
    {"psi2s": ["b1", "Pim"], "b1": ["phi", "Pip"]}, (2, 1), (2, 1))
print()
print("### rho' u_rho_1  (odd l at both nodes)")
run("analyses/tensor_examples/specs/ex3_rhopi.toml", "u_rho_1",
    {"psi2s": ["rhoprime", "Pip"], "rhoprime": ["phi", "Pim"]}, (1, 1), (1, 1))
