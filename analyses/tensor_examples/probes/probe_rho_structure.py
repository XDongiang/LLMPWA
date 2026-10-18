#!/usr/bin/env python3
"""Is the covariant rho' channel a 2-node cascade, or a top-tensor x scalar?"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT,"agent","tensor","tests"))
sys.path.insert(0, os.path.join(ROOT,"analyses/tensor_examples/_work/scratch"))
import test_ex3_channels as T
from agent.tensor import spec as S
from agent.tensor.kernels import omega as RT, covariant as COV
from agent.tensor.kernels.helicity import EPS_A
from agent.tensor.kernels.covariant import T1_cov, T2_cov, metric
import harness as H

np.set_printoptions(precision=5, linewidth=200)
N = 24
momd = T.physical(n=N, seed=11)
mom = np.stack([momd[k][:, [3,0,1,2]] for k in ["Kp","Km","Pip","Pim"]], axis=1)

def Wcov(ch):
    return np.einsum("lm,em->el", EPS_A, COV.compute_event(ch, momd)[..., :3])

def rel(A, B):
    s = np.vdot(B, A)/np.vdot(A, A)
    return float(np.abs(B - s*A).max()/np.abs(B).max())

def spread(A, B):
    r = np.zeros(A.shape, complex)
    for l in range(A.shape[1]):
        r[:, l] = B[:, l]/A[:, l]
    return r

for ch, tree, who in (("u_rho_1", {"psi2s":["rhoprime","Pip"],"rhoprime":["phi","Pim"]}, ("Kp","Km","Pim")),
                      ("u_rho_2", {"psi2s":["rhoprime","Pim"],"rhoprime":["phi","Pip"]}, ("Kp","Km","Pip"))):
    spec = S.load_spec(os.path.join(ROOT,"analyses/tensor_examples/specs/ex3_rhopi.toml"))
    chain = S.to_chain(spec); c = H.HC.with_tree(chain, tree)
    rho = momd["Kp"] + momd["Km"] + momd[who[2]]
    psi = np.broadcast_to(np.asarray(COV.PSI2S_DEFAULT, float), rho.shape)
    Tt = T1_cov(psi, rho, momd["Pip"] if who[2]=="Pim" else momd["Pim"])
    u = COV.compute_event(ch, momd)
    # 1) is u proportional to Tt as a 4-vector, per event?
    rr = np.array([np.vdot(Tt[e], u[e])/np.vdot(Tt[e], Tt[e]) for e in range(N)])
    res = np.abs(u - rr[:, None]*Tt).max()/np.abs(u).max()
    print("="*100)
    print("%s : u =? c(e) * T1_cov(psi, rho, pi)   resid=%.3e   |c| range %.4f..%.4f" %
          (ch, res, np.abs(rr).min(), np.abs(rr).max()))
    Wt = np.einsum("lm,em->el", EPS_A, Tt[..., :3])
    Wc = Wcov(ch)
    # 2) is W_cov = c(e) W_top (rank-1, lambda-independent scalar)?
    cw = np.array([np.vdot(Wt[e], Wc[e])/np.vdot(Wt[e], Wt[e]) for e in range(N)])
    print("   W_cov =? c(e) W_top           resid=%.3e" % (np.abs(Wc - cw[:, None]*Wt).max()/np.abs(Wc).max()))
    # 3) full cascade
    for an, af in (("chain_angles", RT.chain_angles), ("recursive", H.recursive_angles)):
        A = H.chain_amp(mom, c, {"psi2s":[(1,1)], "rhoprime":[(1,1)], "phi":[(1,0)]}, af(mom, c))
        print("   cascade(%s) vs W_cov        rel=%.3e" % (an, rel(A, Wc)))
    # 4) W_top alone vs W_cov
    print("   W_top vs W_cov                rel=%.3e" % rel(Wt, Wc))
    # 5) omega for the same 2-level chain
    om = RT.omega_from_momenta(mom, c, covariant=True)
    Wom = np.einsum("enab,ab->en", om, RT.cg_matrix([(1,1)], 1, 1, 0, RT.LAM_SLOTS[1], [0])[:, :, 0])
    print("   omega vs W_cov                rel=%.3e" % rel(Wom, Wc))
