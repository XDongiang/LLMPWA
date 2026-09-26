#!/usr/bin/env python3
"""Make data/fit/partial-wave overlay plots from draw-stage weights.

Run from the analysis top-level directory with::
    python run/draw_plot_script.py
"""
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = None
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ANALYSIS_ROOT / "output" / "pictures"
WEIGHT_PATH = ANALYSIS_ROOT / "output" / "draw" / "weight.npz"
TRUTH_WEIGHT_PATH = ANALYSIS_ROOT / "output" / "draw" / "weight_truth.npz"
PLOT_VARIABLES = ["phi_kk", "f_kk", "b123_kk", "b124_kk"]
MODE_RESONANCES = {
    "phif0_kk_BW_flatte980": ["phif0_980"],
    "phif0_kk_BW_BW": ["phif0_1710", "phif0_2470"],
    "phif2_kk_BW_BW": ["phif2_1270", "phif2_1525", "phif2_2150", "phif2_2340"],
}


def _import_root():
    global ROOT
    if ROOT is None:
        import ROOT as root
        root.gROOT.SetBatch(True)
        ROOT = root
    return ROOT


def load_analysis_data():
    """Load real-data / MC arrays for plotting (pure numpy; no jax/base_functions).

    Mirrors the weight stage: real data comes from data/real_data, MC evaluation +
    truth samples from data/mc_truth; only phif0_kk / phif2_kk amplitude tensors
    need the MC-based normalization (same convention as base_functions.normalize_data).
    """
    base = ANALYSIS_ROOT / "data"
    real = base / "real_data"
    mc = base / "mc_truth"
    data = {}
    for var in ("phi_kk", "f_kk", "b123_kk", "b124_kk"):
        data["data_" + var] = np.load(str(real / (var + ".npy")))
        data["mc_" + var] = np.load(str(mc / (var + ".npy")))
    for var in ("phif0_kk", "phif2_kk"):
        data["data_" + var] = np.load(str(real / (var + ".npy")))
        data["mc_" + var] = np.load(str(mc / (var + ".npy")))
    # amplitude normalization factor from the MC evaluation sample
    for var in ("phif0_kk", "phif2_kk"):
        mc_arr = data["mc_" + var]
        regular = 1.0 / np.mean(np.sqrt(np.sum(mc_arr ** 2, axis=-1)), axis=1)
        data["data_" + var] = np.einsum("c,cek->cek", regular, data["data_" + var])
        data["mc_" + var] = np.einsum("c,cek->cek", regular, data["mc_" + var])
    return data


def _load_npz(path):
    if not path.exists():
        return {}
    with np.load(str(path), allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def load_weights():
    weights = _load_npz(WEIGHT_PATH)
    truth = _load_npz(TRUTH_WEIGHT_PATH)
    if "all_mods_wt" not in weights:
        raise FileNotFoundError("missing output/draw/weight.npz or all_mods_wt")
    return weights, truth


def available_components(weights):
    """Return (mode, basis, key) entries, tolerating absent modes/bases."""
    found = []
    pattern = re.compile(r"^(.*)_(\d+)$")
    for key in weights:
        if key == "all_mods_wt":
            continue
        match = pattern.match(key)
        if not match:
            continue
        mode, basis = match.group(1), int(match.group(2))
        if mode in MODE_RESONANCES:
            found.append((mode, basis, key))
    return sorted(found, key=lambda item: (list(MODE_RESONANCES).index(item[0]), item[1]))


def _flat_scalar(array):
    array = np.asarray(array)
    if array.ndim != 1:
        raise ValueError("plot variable must be one-dimensional, got shape %s" % (array.shape,))
    return np.asarray(array, dtype=float)


def _histogram(root, name, title, values, weights=None, bins=60, limits=None):
    if limits is None:
        finite = np.asarray(values)[np.isfinite(values)]
        if finite.size == 0:
            limits = (0.0, 1.0)
        else:
            lo, hi = float(np.min(finite)), float(np.max(finite))
            if not hi > lo:
                delta = max(abs(lo) * 0.01, 1.0)
                limits = (lo - delta, hi + delta)
            else:
                pad = 0.02 * (hi - lo)
                limits = (lo - pad, hi + pad)
    hist = root.TH1D(name, title, bins, float(limits[0]), float(limits[1]))
    hist.SetDirectory(0)
    vals = np.asarray(values, dtype=float)
    good = np.isfinite(vals)
    if weights is None:
        for value in vals[good]:
            hist.Fill(float(value))
    else:
        wt = np.asarray(weights, dtype=float)
        count = min(vals.size, wt.size)
        good = good[:count] & np.isfinite(wt[:count])
        for value, weight in zip(vals[:count][good], wt[:count][good]):
            hist.Fill(float(value), float(weight))
    return hist


def _scale_to_data(hist, weights, data_size):
    total = float(np.sum(np.asarray(weights, dtype=float)))
    if np.isfinite(total) and total != 0.0:
        hist.Scale(float(data_size) / total)
    return total


def _fit_fraction(truth, key):
    total = np.asarray(truth.get("all_mods_wt", []), dtype=float)
    component = np.asarray(truth.get(key, []), dtype=float)
    denominator = float(np.sum(total))
    if component.size == 0 or denominator == 0.0:
        return 0.0
    return float(np.sum(component) / denominator)


def draw_variable(var, weights, truth, data=None, bins=60, output_dir=OUTPUT_DIR):
    """Draw one observable and return a dict of truth fit fractions."""
    root = _import_root()
    if data is None:
        data = load_analysis_data()
    data_values = _flat_scalar(data["data_" + var])
    mc_values = _flat_scalar(data["mc_" + var])
    total_weights = np.asarray(weights["all_mods_wt"], dtype=float).ravel()
    n = min(mc_values.size, total_weights.size)
    mc_values, total_weights = mc_values[:n], total_weights[:n]
    data_size = data_values.size
    finite_all = np.concatenate((data_values[np.isfinite(data_values)], mc_values[np.isfinite(mc_values)]))
    limits = None
    if finite_all.size:
        limits = (float(np.min(finite_all)), float(np.max(finite_all)))
        if limits[0] == limits[1]:
            limits = (limits[0] - 1.0, limits[1] + 1.0)

    canvas = root.TCanvas("c_" + var, "Data and fit: " + var, 900, 600)
    canvas.SetGrid()
    hist_data = _histogram(root, "data_" + var, var, data_values, bins=bins, limits=limits)
    hist_fit = _histogram(root, "fit_" + var, var, mc_values, total_weights, bins=bins, limits=limits)
    _scale_to_data(hist_fit, total_weights, data_size)
    hist_data.SetMarkerStyle(20)
    hist_data.SetMarkerSize(0.65)
    hist_data.SetLineColor(root.kBlack)
    hist_data.SetTitle(";" + var + ";Events")
    hist_fit.SetLineColor(root.kRed + 1)
    hist_fit.SetLineWidth(3)
    hist_fit.SetFillStyle(0)
    hist_data.Draw("E1")
    hist_fit.Draw("HIST SAME")

    legend = root.TLegend(0.56, 0.52, 0.89, 0.89)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.AddEntry(hist_data, "Data", "lep")
    legend.AddEntry(hist_fit, "Fit total", "l")
    fractions = {}
    colors = [root.kBlue + 1, root.kGreen + 2, root.kMagenta + 1, root.kOrange + 7,
              root.kCyan + 1, root.kViolet + 1, root.kAzure + 2, root.kRed - 4]
    keep_alive = [hist_data, hist_fit, legend]
    fraction_text = []
    for index, (mode, basis, key) in enumerate(available_components(weights)):
        component_weights = np.asarray(weights[key], dtype=float).ravel()[:n]
        hist_component = _histogram(root, "comp_%s_%s_%s" % (var, mode, basis), var,
                                    mc_values, component_weights, bins=bins, limits=limits)
        _scale_to_data(hist_component, component_weights, data_size)
        hist_component.SetLineColor(colors[index % len(colors)])
        hist_component.SetLineWidth(2)
        hist_component.SetLineStyle(1 + (index // len(colors)) % 3)
        hist_component.Draw("HIST SAME")
        resonance_names = MODE_RESONANCES.get(mode, [])
        if basis < len(resonance_names):
            label = resonance_names[basis]
        else:
            label = "%s[basis%d]" % (mode, basis)
        fractions[key] = _fit_fraction(truth, key)
        fraction_text.append("%s: %.3f" % (label, fractions[key]))
        legend.AddEntry(hist_component, "%s (%.3f)" % (label, fractions[key]), "l")
        keep_alive.append(hist_component)

    latex = root.TLatex()
    latex.SetNDC(True)
    latex.SetTextSize(0.028)
    latex.DrawLatex(0.14, 0.86, "Fit fractions (truth)")
    for line_no, text in enumerate(fraction_text):
        latex.DrawLatex(0.14, 0.82 - 0.035 * line_no, text)
    legend.Draw()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas.SaveAs(str(output_dir / ("partial_mods_%s.png" % var)))
    canvas._plot_objects = keep_alive
    return fractions


def main(variables=None):
    os.chdir(str(ANALYSIS_ROOT))
    data = load_analysis_data()
    weights, truth = load_weights()
    variables = PLOT_VARIABLES if variables is None else list(variables)
    produced = []
    for var in variables:
        if "data_" + var not in data or "mc_" + var not in data:
            continue
        draw_variable(var, weights, truth, data=data)
        produced.append(var)
    return produced


if __name__ == "__main__":
    main()
