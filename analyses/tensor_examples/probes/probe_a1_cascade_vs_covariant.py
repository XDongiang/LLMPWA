#!/usr/bin/env python3
"""End-to-end: spec.cascade_channels -> helicity_cascade -> compare with covariant.py."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT,"agent","tensor","tests"))
import test_ex3_channels as T
from agent.tensor import spec as S
from agent.tensor.kernels import covariant as COV
from agent.tensor.kernels import helicity_cascade as HC
from agent.tensor.kernels.helicity import EPS_A

N = 32
momd = T.physical(n=N, seed=11)
mom = np.stack([momd[k][:, [3,0,1,2]] for k in ["Kp","Km","Pip","Pim"]], axis=1)

def rel(A, B):
    s = np.vdot(B, A)/np.vdot(A, A)
    return float(np.abs(B - s*A).max()/np.abs(B).max())

for sp_path in ("analyses/tensor_examples/specs/ex3_b1pi.toml",
                "analyses/tensor_examples/specs/ex3_rhopi.toml",
                "agent/tensor/specs/phif2.toml"):
    spec = S.load_spec(os.path.join(ROOT, sp_path))
    chain0 = S.to_chain(spec)
    print("=" * 92); print(sp_path, " depth=", spec["depth"])
    cache = {}
    for c in S.cascade_channels(spec):
        ch = HC.with_tree(chain0, c["tree"])
        ang = HC.node_angles(mom, ch)
        A = HC.chain_amplitude(mom, ch, c["ls"], ang=ang)
        cache[c["name"]] = A
        Wc = np.einsum("lm,em->el", EPS_A, COV.compute_event(c["name"], momd)[..., :3])
        print("  %-8s rel=%.3e" % (c["name"], rel(A, Wc)))
    ks = list(cache)
    worst = min(rel(cache[ks[i]], cache[ks[j]])
                for i in range(len(ks)) for j in range(i+1, len(ks)))
    print("  min pairwise rel = %.4e" % worst)
