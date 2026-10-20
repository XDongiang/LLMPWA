#!/usr/bin/env python3
"""Identify each covariant b1 channel's (top_ls, inner_ls) empirically."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT,"agent","tensor","tests"))
import test_ex3_channels as T
from agent.tensor import spec as S
from agent.tensor.kernels import covariant as COV
from agent.tensor.kernels import helicity_cascade as HC
from agent.tensor.kernels.helicity import EPS_A

N = 24
momd = T.physical(n=N, seed=11)
mom = np.stack([momd[k][:, [3,0,1,2]] for k in ["Kp","Km","Pip","Pim"]], axis=1)
spec = S.load_spec(os.path.join(ROOT,"analyses/tensor_examples/specs/ex3_b1pi.toml"))
chain0 = S.to_chain(spec)

def rel(A, B):
    s = np.vdot(B, A)/np.vdot(A, A)
    return float(np.abs(B - s*A).max()/np.abs(B).max())

def Wcov(ch):
    return np.einsum("lm,em->el", EPS_A, COV.compute_event(ch, momd)[..., :3])

for ch in ["u_SS_1","u_SD_1","u_DS_1","u_DD_1"]:
    obs = None
    for tree_side in ("phi+Pip", "phi+Pim"):
        if tree_side == "phi+Pip":
            tree = {"psi2s":["b1","Pim"],"b1":["phi","Pip"]}
        else:
            tree = {"psi2s":["b1","Pip"],"b1":["phi","Pim"]}
        c = HC.with_tree(chain0, tree); ang = HC.node_angles(mom, c)
        for tl in [(0,1),(2,1)]:
            for il in [(0,1),(2,1)]:
                A = HC.chain_amplitude(mom, c, {"psi2s":[tl],"b1":[il],"phi":[(1,0)]}, ang=ang)
                r = rel(A, Wcov(ch))
                if obs is None or r < obs[0]:
                    obs = (r, tree_side, tl, il)
    Wc = Wcov(ch)
    print("%-8s  best: tree=%-8s top=%s inner=%s  rel=%.3e   (|W|=%.3e)"
          % (ch, obs[1], obs[2], obs[3], obs[0], np.abs(Wc).max()))
