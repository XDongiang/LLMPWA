#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# SECTION: draw_plot_imports

import numpy as onp
import os
import re
import json
import logging
import pandas as pd
import ROOT
from ROOT import TH1D, TCanvas, gStyle, TLegend, TLatex

# SECTION: PATH_CONFIG
import sys
foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)

# SECTION: LOGGING_CONFIG
def setup_logging():
    """Setup logging configuration"""
    with open("config/logconfig_plot.json", "r") as config_file:
        LOGGING_CONFIG = json.load(config_file)
        logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger("fit")

# SECTION: draw_plot_resonance_template


def draw_single_resonance_phif0_980(var_name, data_arr, mc_arr, all_wt, all_truth_wt, data_size):
    """画 phif0_980 共振态的 data vs fit 叠加图，返回 fit fraction。"""
    fit_result_wt = all_wt["all_mods_wt"]
    sum_wt = onp.sum(fit_result_wt)
    sum_truth_wt = onp.sum(all_truth_wt["all_mods_wt"])

    max_value = mc_arr.max() + 0.15
    min_value = mc_arr.min() - 0.15

    # 数据直方图
    hist_data = TH1D("data_" + var_name, var_name + " data", 100, min_value, max_value)
    for i in range(data_arr.shape[0]):
        hist_data.Fill(data_arr[i])

    # 总拟合直方图
    hist_fit = TH1D("fit_" + var_name, var_name + " fit", 100, min_value, max_value)
    for i in range(mc_arr.shape[0]):
        hist_fit.Fill(mc_arr[i], fit_result_wt[i])
    hist_fit.Scale(data_size / sum_wt)

    # phif0_980 分波直方图（键名：phif0_kk_BW_flatte980_0）
    comp_wt = all_wt["phif0_kk_BW_flatte980_0"]
    hist_comp = TH1D("phif0_980_" + var_name, "phif0_980 partial wave", 100, min_value, max_value)
    for i in range(mc_arr.shape[0]):
        hist_comp.Fill(mc_arr[i], comp_wt[i])
    hist_comp.Scale(data_size / sum_wt)

    # fit fraction（用 truth MC 计算）
    frac = onp.sum(all_truth_wt["phif0_kk_BW_flatte980_0"]) / sum_truth_wt

    # 画图
    os.makedirs("output/pictures/partial_mods_pictures", exist_ok=True)
    c = TCanvas("phif0_980_" + var_name, "phif0_980 " + var_name, 900, 600)
    gStyle.SetOptStat(0)
    hist_data.SetMarkerStyle(20)
    hist_data.SetMarkerSize(0.8)
    hist_data.Draw("E1")
    hist_fit.SetLineColor(ROOT.kRed)
    hist_fit.SetLineWidth(2)
    hist_fit.Draw("HIST SAME")
    hist_comp.SetLineColor(ROOT.kBlue)
    hist_comp.SetLineWidth(2)
    hist_comp.SetLineStyle(2)
    hist_comp.Draw("HIST SAME")
    legend = TLegend(0.65, 0.70, 0.92, 0.92)
    legend.AddEntry(hist_data, "Data", "ep")
    legend.AddEntry(hist_fit, "Fit total", "l")
    legend.AddEntry(hist_comp, "phif0_980", "l")
    legend.Draw()
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.03)
    latex.DrawLatex(0.15, 0.85, "fit fraction: {:.4f}".format(frac))
    c.SaveAs("output/pictures/partial_mods_pictures/phif0_980_{}.png".format(var_name))

    return frac


# SECTION: draw_plot_main_template


if __name__ == "__main__":
    ROOT.gROOT.SetBatch(True)
    logger = setup_logging()
    os.makedirs("output/pictures/partial_mods_pictures", exist_ok=True)

    # 加载权重文件
    all_wt = onp.load("output/draw/weight.npz")
    all_truth_wt = onp.load("output/draw/weight_truth.npz")

    # 加载运动学变量（sqrt 转换为质量）
    data_f_kk = onp.sqrt(onp.load("data/real_data/f_kk.npy"))
    mc_f_kk = onp.sqrt(onp.load("data/mc_truth/f_kk.npy"))
    data_phi_kk = onp.sqrt(onp.load("data/real_data/phi_kk.npy"))
    mc_phi_kk = onp.sqrt(onp.load("data/mc_truth/phi_kk.npy"))
    # extra_sbc 变量（1D，直接 sqrt）
    data_b123_kk = onp.sqrt(onp.load("data/real_data/b123_kk.npy"))
    mc_b123_kk = onp.sqrt(onp.load("data/mc_truth/b123_kk.npy"))

    fit_fraction_coll = []

    # 遍历 sbc 变量
    for var_name, data_arr, mc_arr in [
        ("f_kk", data_f_kk, mc_f_kk),
        ("phi_kk", data_phi_kk, mc_phi_kk),
    ]:
        data_size = data_arr.shape[0]
        frac = draw_single_resonance_phif0_980(var_name, data_arr, mc_arr, all_wt, all_truth_wt, data_size)
        fit_fraction_coll.append({"mod_name": "phif0_980", "var": var_name, "fraction": frac})

    # 遍历 extra_sbc 变量（权重需要复制）
    for var_name, data_arr, mc_arr in [
        ("b123_kk", data_b123_kk, mc_b123_kk),
    ]:
        # b.* 变量每个事例有两个组合，权重复制
        doubled_wt = {key: onp.append(all_wt[key], all_wt[key]) for key in all_wt.files}
        doubled_truth_wt = {key: onp.append(all_truth_wt[key], all_truth_wt[key]) for key in all_truth_wt.files}
        doubled_mc = onp.append(mc_arr[:all_wt["all_mods_wt"].shape[0]], mc_arr[:all_wt["all_mods_wt"].shape[0]])
        data_size = data_arr.shape[0]
        frac = draw_single_resonance_phif0_980(var_name, data_arr, doubled_mc, doubled_wt, doubled_truth_wt, data_size)
        fit_fraction_coll.append({"mod_name": "phif0_980", "var": var_name, "fraction": frac})

    # 输出 fit fraction 表格
    os.makedirs("output/draw", exist_ok=True)
    with open("output/draw/fit_fraction_table.json", "w") as f:
        json.dump(fit_fraction_coll, f, indent=2)
    df = pd.DataFrame(fit_fraction_coll)
    with open("output/draw/fit_fraction_table.md", "w") as f:
        f.write(df.to_markdown())
    with open("output/draw/fit_fraction_table.latex", "w") as f:
        f.write(df.to_latex(escape=False))
    logger.info("画图完成，结果已保存至 output/pictures/partial_mods_pictures/")