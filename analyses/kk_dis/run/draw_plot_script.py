#!/usr/bin/env python3
"""Draw LLMPWA data/fit/partial-wave overlays with numpy and pyROOT only."""

import argparse
import json
import math
import os
import re
import tomllib
from pathlib import Path

import numpy as np

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
WEIGHT_PATH = ANALYSIS_ROOT / "output" / "draw" / "weight.npz"
TRUTH_WEIGHT_PATH = ANALYSIS_ROOT / "output" / "draw" / "weight_truth.npz"
CLASSIFICATION_PATH = ANALYSIS_ROOT / "gen" / "fragments" / "classification.json"
CONFIG_PATH = ANALYSIS_ROOT / "resonances_config.toml"
FREE_PARAMS_PATH = ANALYSIS_ROOT / "run" / "free_params.toml"
PICTURE_DIR = ANALYSIS_ROOT / "output" / "pictures"

AXIS_TITLES = {
    "phi_kk": "M_{#phi} (GeV)",
    "f_kk": "M_{KK} (GeV)",
    "b123_kk": "M_{#phi K} (GeV)",
    "b124_kk": "M_{#phi K} (GeV)",
}
COLOR_OFFSETS = (
    ("kBlue", 1), ("kGreen", 2), ("kMagenta", 1), ("kOrange", 7),
    ("kCyan", 1), ("kViolet", 1), ("kAzure", 2), ("kRed", -4),
)
# The authoritative standard images omit this second f0 basis even though its
# fitted weight exists.  All other configured mode/basis entries are drawn.
STANDARD_OMITTED_RESONANCES = {"phif0_2470"}
ROOT = None


def root_module():
    global ROOT
    if ROOT is None:
        import ROOT as root
        root.gROOT.SetBatch(True)
        root.gStyle.SetOptStat(0)
        ROOT = root
    return ROOT


def load_toml(path):
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)


def analysis_description():
    """Return configured mode mapping, plot variables, and resonance names."""
    with CLASSIFICATION_PATH.open("r", encoding="utf-8") as stream:
        classification = json.load(stream)
    config = load_toml(CONFIG_PATH)
    configured = set(config.get("resonances", {}))
    mode_resonances = {}
    for mode, names in classification.get("amplitude_classification", {}).items():
        kept = [name for name in names if name in configured]
        if kept:
            mode_resonances[mode] = kept

    variables = ["phi_kk", "f_kk"]
    for var in config.get("draw", {}).get("extra_sbc", []):
        if var not in variables:
            variables.append(var)
    return mode_resonances, variables


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def load_weights():
    weights = load_npz(WEIGHT_PATH)
    truth = load_npz(TRUTH_WEIGHT_PATH)
    for label, values in (("pass", weights), ("truth", truth)):
        if "all_mods_wt" not in values:
            raise KeyError("%s weight file lacks all_mods_wt" % label)
    return weights, truth


def discover_components(weights, mode_resonances):
    """Find mode_basis keys and map valid basis indices to resonance labels."""
    pattern = re.compile(r"^(.*)_(\d+)$")
    mode_order = {mode: index for index, mode in enumerate(mode_resonances)}
    found = []
    for key in weights:
        match = pattern.match(key)
        if not match:
            continue
        mode, basis = match.group(1), int(match.group(2))
        if mode not in mode_resonances or basis >= len(mode_resonances[mode]):
            continue
        resonance = mode_resonances[mode][basis]
        if resonance in STANDARD_OMITTED_RESONANCES:
            continue
        found.append((mode, basis, key, resonance))
    return sorted(found, key=lambda item: (mode_order[item[0]], item[1]))


def load_kinematics(var):
    real_path = ANALYSIS_ROOT / "data" / "real_data" / (var + ".npy")
    mc_path = ANALYSIS_ROOT / "data" / "mc_truth" / (var + ".npy")
    real_s = np.asarray(np.load(real_path), dtype=np.float64).reshape(-1)
    mc_s = np.asarray(np.load(mc_path), dtype=np.float64).reshape(-1)
    # Stored arrays are invariant-mass squared values.
    real = np.sqrt(real_s)
    mc = np.sqrt(mc_s)
    return real, mc, int(real.size)


def matched_values_weights(values, weights, label):
    """Match combination arrays, doubling weights only for a doubled array."""
    values = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    weights = np.ascontiguousarray(weights, dtype=np.float64).reshape(-1)
    if values.size == 2 * weights.size:
        weights = np.ascontiguousarray(np.tile(weights, 2), dtype=np.float64)
    if values.size != weights.size:
        raise ValueError(
            "%s length mismatch: %d values versus %d weights"
            % (label, values.size, weights.size)
        )
    good = np.isfinite(values) & np.isfinite(weights)
    return (np.ascontiguousarray(values[good], dtype=np.float64),
            np.ascontiguousarray(weights[good], dtype=np.float64))


def histogram_limits(data, mc):
    finite_data = data[np.isfinite(data)]
    finite_mc = mc[np.isfinite(mc)]
    finite = np.concatenate((finite_data, finite_mc))
    if not finite.size:
        return 0.0, 1.0
    low, high = float(np.min(finite)), float(np.max(finite))
    center = 0.5 * (low + high)
    scale = max(abs(low), abs(high), 1.0)
    rel_span = (high - low) / scale
    # A sharply-peaked (near-delta) variable — e.g. M_phi where every event sits at
    # ~1.019 GeV — makes the raw min/max span collapse, so ROOT's auto x-axis labels
    # (many decimals) pile up and overlap. Widen to a physically meaningful window
    # centred on the peak so the tick labels stay legible.
    if rel_span < 1.0e-2:
        half_width = 0.15 * max(abs(center), 1.0)
        return center - half_width, center + half_width
    if high <= low:
        delta = max(1.0, abs(low) * 0.01)
        return low - delta, high + delta
    # Keep extrema away from ROOT's overflow boundary.
    padding = max((high - low) * 1.0e-9, 1.0e-12)
    return low - padding, high + padding


def make_histogram(name, values, weights, bins, limits):
    root = root_module()
    hist = root.TH1D(name, name, bins, limits[0], limits[1])
    hist.SetDirectory(0)
    values = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    if weights is None:
        good_values = np.ascontiguousarray(values[np.isfinite(values)], dtype=np.float64)
        fill_weights = np.ones(good_values.size, dtype=np.float64)
    else:
        good_values, fill_weights = matched_values_weights(values, weights, name)
    if good_values.size:
        hist.FillN(int(good_values.size), good_values, fill_weights)
    return hist


def fit_fraction(truth_weights, key):
    denominator = float(np.sum(np.asarray(truth_weights["all_mods_wt"], dtype=np.float64)))
    numerator = float(np.sum(np.asarray(truth_weights.get(key, []), dtype=np.float64)))
    return numerator / denominator if denominator != 0.0 else 0.0


def parameter_lines(resonance, fit_values):
    """Build the required resonance-specific propagator annotation lines."""
    if not FREE_PARAMS_PATH.exists():
        return []
    entries = load_toml(FREE_PARAMS_PATH).get("data", [])
    number_match = re.search(r"_(\d+)$", resonance)
    resonance_number = number_match.group(1) if number_match else resonance
    suffix_labels = {
        "B_propagator.mass": "kk_f%s_m" % resonance_number,
        "B_propagator.width": "kk_f%s_w" % resonance_number,
        "B_propagator.g_kk": "kk_g_kk",
        "B_propagator.rg": "kk_rg",
    }
    prefix = "resonances.%s.propagators." % resonance
    lines = []
    for entry in sorted(entries, key=lambda item: int(item.get("arg_index", 10**9))):
        path = str(entry.get("path", ""))
        if not path.startswith(prefix):
            continue
        suffix = path[len(prefix):]
        if suffix not in suffix_labels:
            continue
        index = int(entry.get("arg_index", -1))
        if 0 <= index < fit_values.size:
            lines.append("%s_result:%.6g" % (suffix_labels[suffix], float(fit_values[index])))
    return lines


def component_color(index):
    root = root_module()
    name, offset = COLOR_OFFSETS[index % len(COLOR_OFFSETS)]
    return int(getattr(root, name) + offset)


def build_histograms(var, data, mc, data_size, weights, truth, components, bins=60):
    limits = histogram_limits(data, mc)
    sum_wt = float(np.sum(np.asarray(weights["all_mods_wt"], dtype=np.float64)))
    if not np.isfinite(sum_wt) or sum_wt == 0.0:
        raise ValueError("sum(all_mods_wt) is zero or non-finite")
    scale = float(data_size) / sum_wt

    hdata = make_histogram("h_data_%s" % var, data, None, bins, limits)
    hfit = make_histogram("h_fit_%s" % var, mc, weights["all_mods_wt"], bins, limits)
    hfit.Scale(scale)
    built_components = []
    for index, (mode, basis, key, resonance) in enumerate(components):
        if key not in weights:
            continue
        hist = make_histogram("h_%s_%s_%d" % (var, mode, basis), mc, weights[key], bins, limits)
        # All curves use the same total-fit normalization, not their own integral.
        hist.Scale(scale)
        hist.SetLineColor(component_color(index))
        hist.SetLineWidth(2)
        hist.SetLineStyle(1 + (index // len(COLOR_OFFSETS)) % 3)
        built_components.append({
            "mode": mode, "basis": basis, "key": key, "resonance": resonance,
            "hist": hist, "fraction": fit_fraction(truth, key),
        })

    width = (limits[1] - limits[0]) / float(bins)
    hdata.SetTitle(";%s;Events/%.3g GeV" % (AXIS_TITLES.get(var, var), width))
    hdata.SetMarkerStyle(20)
    hdata.SetMarkerSize(0.65)
    hdata.SetMarkerColor(root_module().kBlack)
    hdata.SetLineColor(root_module().kBlack)
    hfit.SetLineColor(root_module().kRed + 1)
    hfit.SetLineWidth(3)
    hfit.SetFillStyle(0)
    return hdata, hfit, built_components


def set_vertical_range(hdata, hfit, components):
    maxima = [hdata.GetMaximum(), hfit.GetMaximum()]
    maxima.extend(item["hist"].GetMaximum() for item in components)
    maximum = max(maxima) if maxima else 1.0
    hdata.SetMinimum(0.0)
    hdata.SetMaximum(1.22 * maximum if maximum > 0.0 else 1.0)


def draw_all_overlay(var, hdata, hfit, components):
    root = root_module()
    canvas = root.TCanvas("c_%s_all" % var, var, 900, 600)
    set_vertical_range(hdata, hfit, components)
    hdata.Draw("E1")
    hfit.Draw("HIST SAME")
    for item in components:
        item["hist"].Draw("HIST SAME")
    hdata.Draw("E1 SAME")

    legend = root.TLegend(0.58, 0.50, 0.90, 0.90)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.027)
    legend.AddEntry(hdata, "Data", "lep")
    legend.AddEntry(hfit, "Fit total", "l")
    for item in components:
        legend.AddEntry(item["hist"], "%s (%.3f)" % (item["resonance"], item["fraction"]), "l")
    legend.Draw()
    canvas.RedrawAxis()
    output = PICTURE_DIR / (var + "_weight.png")
    canvas.SaveAs(str(output))
    return output


def draw_single_overlay(var, hdata, hfit, item, fit_values):
    root = root_module()
    resonance = item["resonance"]
    canvas = root.TCanvas("c_%s_%s" % (var, resonance), var, 900, 600)
    set_vertical_range(hdata, hfit, [item])
    hdata.Draw("E1")
    hfit.Draw("HIST SAME")
    item["hist"].Draw("HIST SAME")
    hdata.Draw("E1 SAME")

    # Keep this compact legend below the required top-right annotation block.
    legend = root.TLegend(0.64, 0.16, 0.90, 0.32)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.028)
    legend.AddEntry(hdata, "Data", "lep")
    legend.AddEntry(hfit, "Fit total", "l")
    legend.AddEntry(item["hist"], resonance, "l")
    legend.Draw()

    latex = root.TLatex()
    latex.SetNDC(True)
    latex.SetTextSize(0.028)
    lines = parameter_lines(resonance, fit_values)
    lines.append("fit fraction : %.6g" % item["fraction"])
    y = 0.90
    for line in lines:
        latex.DrawLatex(0.61, y, line)
        y -= 0.035
    canvas.RedrawAxis()
    output = PICTURE_DIR / (var + resonance + "_kk_weight.png")
    canvas.SaveAs(str(output))
    return output


def draw_variable(var, weights, truth, components):
    data, mc, data_size = load_kinematics(var)
    hdata, hfit, built = build_histograms(
        var, data, mc, data_size, weights, truth, components
    )
    fit_values = np.asarray(weights.get("fit_value", []), dtype=np.float64).reshape(-1)
    produced = [draw_all_overlay(var, hdata, hfit, built)]
    for item in built:
        produced.append(draw_single_overlay(var, hdata, hfit, item, fit_values))
    return produced


def main(variables=None):
    os.chdir(ANALYSIS_ROOT)
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    mode_resonances, configured_variables = analysis_description()
    weights, truth = load_weights()
    components = discover_components(weights, mode_resonances)
    if not components:
        raise RuntimeError("no configured partial-wave keys were found in weight.npz")
    selected_variables = configured_variables if variables is None else list(variables)
    produced = []
    for var in selected_variables:
        real_path = ANALYSIS_ROOT / "data" / "real_data" / (var + ".npy")
        mc_path = ANALYSIS_ROOT / "data" / "mc_truth" / (var + ".npy")
        if not real_path.exists() or not mc_path.exists():
            print("Skipping %s: raw real-data or MC array is missing" % var)
            continue
        produced.extend(draw_variable(var, weights, truth, components))
    for path in produced:
        print("produced", path.relative_to(ANALYSIS_ROOT))
    return produced


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--var", action="append", dest="variables",
                        help="draw only this variable (repeatable); default: all configured variables")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments.variables)
