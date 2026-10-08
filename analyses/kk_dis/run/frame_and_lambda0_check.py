#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frame / beam-axis audit and lambda = 0 budget  (report section 5 item 0 + P2-1).

Consumes a 3-component (x, y, z) tensor cache produced by run/gen_tensor3.py and
answers the two questions the referee said were still open:

  1. Is the "psi at rest / beam along z" premise actually true for this sample?
     The whole `T_x^2 + T_y^2` reading depends on it.  We check
        (a) |p_psi| / E_psi for the reconstructed psi = sum of the four finals;
        (b) how much the intensity would move if the beam were along x / y /
            a random direction, i.e. I_pipe / I_transverse(n).
  2. How big is the lambda = 0 term that the 2-component cache throws away?
     (it is exactly |T_z|^2 in the psi rest frame, up to the normalisation of
      the helicity basis).

Usage:
    python run/frame_and_lambda0_check.py --tensor-dir data/tensor3_mc1M \
        --npz data/mc_truth/Momentum_kk.npz
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as onp

METRIC = onp.array([-1.0, -1.0, -1.0, 1.0])       # (x, y, z, t)
NAMES = ["phif001", "phif021", "phif201", "phif221", "phif222", "phif223", "phif243"]
F0 = ["phif001", "phif021"]
F2 = ["phif201", "phif221", "phif222", "phif223", "phif243"]


def transverse_basis(n):
    n = onp.asarray(n, dtype=onp.float64)
    n = n / onp.linalg.norm(n)
    helper = onp.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else onp.array([1.0, 0.0, 0.0])
    e1 = onp.cross(n, helper)
    e1 = e1 / onp.linalg.norm(e1)
    e2 = onp.cross(n, e1)
    return e1, e2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor-dir", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="output/frame_and_lambda0.json")
    args = ap.parse_args()

    T = {}
    for tag, names in (("phif0", F0), ("phif2", F2)):
        arr = onp.load(os.path.join(args.tensor_dir, "%s_kk.npy" % tag), mmap_mode="r")
        for i, nm in enumerate(names):
            T[nm] = onp.asarray(arr[i], dtype=onp.float64)
            print("%-8s %s  max|T| = %.4e" % (nm, T[nm].shape, onp.abs(T[nm]).max()))

    report = {}

    # ---------------------------------------------------------------- (1a) ---
    d = onp.load(args.npz)
    n = min(T[NAMES[0]].shape[0], d["Kp"].shape[0])
    tot = sum(onp.asarray(d[k][:n], dtype=onp.float64)
              for k in ("Kp", "Km", "Pip", "Pim"))
    p_sp = onp.linalg.norm(tot[:, :3], axis=-1)
    E = tot[:, 3]
    print("\n[1a] psi kinematics from the four final-state momenta (n=%d)" % n)
    print("     |p_psi| / E_psi : median %.3e   max %.3e   p90 %.3e"
          % (onp.median(p_sp / E), (p_sp / E).max(), onp.percentile(p_sp / E, 90)))
    print("     E_psi           : median %.6f   min %.6f   max %.6f"
          % (onp.median(E), E.min(), E.max()))
    report["psi_beta_median"] = float(onp.median(p_sp / E))
    report["psi_beta_max"] = float((p_sp / E).max())
    report["psi_mass_median"] = float(onp.median(onp.sqrt(onp.maximum(
        -(tot[:, 0] ** 2 + tot[:, 1] ** 2 + tot[:, 2] ** 2) + tot[:, 3] ** 2, 0))))

    # ---------------------------------------------------------------- (1b) ---
    print("\n[1b] I_pipe = T_x^2 + T_y^2  vs  the two directions transverse to n")
    print("     (ratio median over %d events; 1.000000 <=> the beam is along z)" % n)
    dirs = {"z (pipeline assumption)": (0, 0, 1), "x": (1, 0, 0), "y": (0, 1, 0),
            "tilted (1,1,0.3)": (1, 1, 0.3)}
    rows = {}
    for nm in NAMES:
        t = T[nm][:n, :3]
        ipipe = t[:, 0] ** 2 + t[:, 1] ** 2
        row = {}
        for label, nvec in dirs.items():
            e1, e2 = transverse_basis(nvec)
            iphys = (t @ e1) ** 2 + (t @ e2) ** 2
            with onp.errstate(divide="ignore", invalid="ignore"):
                r = onp.where(iphys > 0, ipipe / onp.maximum(iphys, 1e-300), onp.nan)
            row[label] = float(onp.nanmedian(r))
        rows[nm] = row
    hdr = "%-9s " % "channel" + " ".join("%-24s" % k for k in dirs)
    print("     " + hdr)
    for nm in NAMES:
        print("     %-9s " % nm + " ".join("%-24.6f" % rows[nm][k] for k in dirs))
    report["beam_axis_ratio"] = rows

    # ----------------------------------------------------------------- (2) ---
    print("\n[2] lambda = 0 budget: |T_z|^2 / |T|^2  (per event, median)")
    zshare = {}
    for nm in NAMES:
        t = T[nm][:n]
        s = (t ** 2).sum(-1)
        with onp.errstate(divide="ignore", invalid="ignore"):
            frac = onp.where(s > 0, t[:, 2] ** 2 / onp.maximum(s, 1e-300), 0.0)
        zshare[nm] = float(onp.median(frac))
        print("     %-9s z-share %.4f   (x %.4f, y %.4f, t %.4f)"
              % (nm, zshare[nm],
                 float(onp.median(t[:, 0] ** 2 / onp.maximum(s, 1e-300))),
                 float(onp.median(t[:, 1] ** 2 / onp.maximum(s, 1e-300))),
                 0.0))
    report["z_share_median"] = zshare
    print("     -> enabling lambda = 0 would add this fraction of intensity; on an")
    print("        unpolarised transverse psi it must NOT be added (P2-1 is about")
    print("        being ABLE to model it, not about changing the default).")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nwritten %s" % args.out)


if __name__ == "__main__":
    main()
