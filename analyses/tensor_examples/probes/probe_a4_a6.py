#!/usr/bin/env python3
"""A4 (depth-3 inner-plane sensitivity) and A6 (all specs check/emit)."""
import glob, os, subprocess, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "agent", "tensor", "tests"))
import test_ex3_channels as T
from agent.tensor import spec as S
from agent.tensor.kernels import helicity_cascade as HC
from agent.tensor.kernels import omega as RT, covariant as COV
from agent.tensor.kernels.helicity import EPS_A

def rot3(v, axis, ang):
    ax = axis / np.linalg.norm(axis, axis=1, keepdims=True)
    c, s = np.cos(ang), np.sin(ang)
    d = (v * ax).sum(1, keepdims=True)
    return v * c + np.cross(ax, v) * s + ax * d * (1 - c)

def rel(a, b):
    s = np.vdot(b, a) / np.vdot(a, a)
    return float(np.abs(b - s * a).max() / max(np.abs(b).max(), 1e-300))

print("=" * 96)
print("A4: depth-3 chain must be sensitive to the INNER decay plane (rotate phi->KK by 37 deg)")
print("=" * 96)
N = 32
momd = T.physical(n=N, seed=3)
# rotate the K+K- plane about the phi direction, keeping p_phi and the total fixed
phi3 = momd["Kp"][:, :3] + momd["Km"][:, :3]
r = momd["Kp"][:, :3] - momd["Km"][:, :3]
r2 = rot3(r, phi3, np.deg2rad(37.0))
momd_r = {k: v.copy() for k, v in momd.items()}
momd_r["Kp"][:, :3] = (phi3 + r2) / 2
momd_r["Km"][:, :3] = (phi3 - r2) / 2
tot0 = sum(momd[k] for k in ("Kp", "Km", "Pip", "Pim"))
tot1 = sum(momd_r[k] for k in ("Kp", "Km", "Pip", "Pim"))
print("  self-check: p_phi unchanged rel = %.3e" % rel(momd_r["Kp"] + momd_r["Km"], momd["Kp"] + momd["Km"]))
print("  self-check: total   unchanged rel = %.3e" % rel(tot1, tot0))

def to_mom(d):
    return np.stack([d[k][:, [3, 0, 1, 2]] for k in ("Kp", "Km", "Pip", "Pim")], 1)

spec = S.load_spec(os.path.join(ROOT, "analyses/tensor_examples/specs/ex3_b1pi.toml"))
chain0 = S.to_chain(spec)
for cc in S.cascade_channels(spec):
    ch = HC.with_tree(chain0, cc["tree"])
    A0 = HC.chain_amplitude(to_mom(momd), ch, cc["ls"], ang=HC.node_angles(to_mom(momd), ch))
    A1 = HC.chain_amplitude(to_mom(momd_r), ch, cc["ls"], ang=HC.node_angles(to_mom(momd_r), ch))
    Wc0 = np.einsum("lm,em->el", EPS_A, COV.compute_event(cc["name"], momd)[..., :3])
    Wc1 = np.einsum("lm,em->el", EPS_A, COV.compute_event(cc["name"], momd_r)[..., :3])
    print("  %-8s cascade rel = %.3e   covariant rel = %.3e  %s"
          % (cc["name"], rel(A1, A0), rel(Wc1, Wc0),
             "OK" if rel(A1, A0) > 0.1 else "**NOT SENSITIVE**"))

print()
print("=" * 96)
print("A6: every spec TOML in the repo must load + emit without error")
print("=" * 96)
pats = sorted(glob.glob(os.path.join(ROOT, "agent/tensor/specs/*.toml"))
              + glob.glob(os.path.join(ROOT, "analyses/*/specs/*.toml")))
bad = 0
for p in pats:
    try:
        sp = S.load_spec(p)
        ch = S.cascade_channels(sp)
        print("  OK   %-46s depth=%d  %d ch  %d cascade ch"
              % (os.path.relpath(p, ROOT), sp["depth"], len(sp["channels"]), len(ch)))
    except Exception as e:
        bad += 1
        print("  FAIL %-46s %s: %s" % (os.path.relpath(p, ROOT), type(e).__name__, e))
print("  %d/%d specs OK" % (len(pats) - bad, len(pats)))
