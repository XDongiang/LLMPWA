#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SECTION: LOGGING_CONFIG
def setup_logging():
    """Setup logging configuration"""
    with open("config/logconfig_fit.json", "r") as config_file:
        LOGGING_CONFIG = json.load(config_file)
        logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger("fit")


# SECTION: draw_weight_functions

def weight_kk(args):
    params = extract_parameters(args)
    comp_phif0_kk_BW_BW = component_BW_BW(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['phif0_kk_BW_BW_mass'], params['phif0_kk_BW_BW_width'], data_f_kk,
        data_phif0_kk, params['phif0_kk_BW_BW_const'], params['phif0_kk_BW_BW_theta']
    )
    comp_phif2_kk_BW_BW = component_BW_BW(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['phif2_kk_BW_BW_mass'], params['phif2_kk_BW_BW_width'], data_f_kk,
        data_phif2_kk, params['phif2_kk_BW_BW_const'], params['phif2_kk_BW_BW_theta']
    )
    comp_phif2_kk_BW_flatte1270 = component_BW_flatte1270(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['phif2_kk_BW_flatte1270_mass'], params['phif2_kk_BW_flatte1270_width'], data_f_kk,
        data_phif2_kk, params['phif2_kk_BW_flatte1270_const'], params['phif2_kk_BW_flatte1270_theta']
    )
    total_wt = np.sum(dplex_dabs(
        np.einsum("mljk->mjk", comp_phif0_kk_BW_BW) +
        np.einsum("mljk->mjk", comp_phif2_kk_BW_BW) +
        np.einsum("mljk->mjk", comp_phif2_kk_BW_flatte1270)
    ), axis=1)
    wt_list = [
        total_wt,
        np.einsum("ljk->lj", dplex_dabs(comp_phif0_kk_BW_BW)),
        np.einsum("ljk->lj", dplex_dabs(comp_phif2_kk_BW_BW)),
        np.einsum("ljk->lj", dplex_dabs(comp_phif2_kk_BW_flatte1270)),
    ]
    return wt_list

def weight_truth_kk(args):
    params = extract_parameters(args)
    comp_phif0_kk_BW_BW = component_BW_BW(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['phif0_kk_BW_BW_mass'], params['phif0_kk_BW_BW_width'], truth_f_kk,
        truth_phif0_kk, params['phif0_kk_BW_BW_const'], params['phif0_kk_BW_BW_theta']
    )
    comp_phif2_kk_BW_BW = component_BW_BW(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['phif2_kk_BW_BW_mass'], params['phif2_kk_BW_BW_width'], truth_f_kk,
        truth_phif2_kk, params['phif2_kk_BW_BW_const'], params['phif2_kk_BW_BW_theta']
    )
    comp_phif2_kk_BW_flatte1270 = component_BW_flatte1270(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['phif2_kk_BW_flatte1270_mass'], params['phif2_kk_BW_flatte1270_width'], truth_f_kk,
        truth_phif2_kk, params['phif2_kk_BW_flatte1270_const'], params['phif2_kk_BW_flatte1270_theta']
    )
    total_wt = np.sum(dplex_dabs(
        np.einsum("mljk->mjk", comp_phif0_kk_BW_BW) +
        np.einsum("mljk->mjk", comp_phif2_kk_BW_BW) +
        np.einsum("mljk->mjk", comp_phif2_kk_BW_flatte1270)
    ), axis=1)
    wt_list = [
        total_wt,
        np.einsum("ljk->lj", dplex_dabs(comp_phif0_kk_BW_BW)),
        np.einsum("ljk->lj", dplex_dabs(comp_phif2_kk_BW_BW)),
        np.einsum("ljk->lj", dplex_dabs(comp_phif2_kk_BW_flatte1270)),
    ]
    return wt_list

def run_weight(args_list, mode="pass"):
    args = np.array(args_list)
    jit_weight = jit(weight_kk)
    jit_weight_truth = jit(weight_truth_kk)
    if mode == "pass":
        wt_list = jit_weight(args)
    elif mode == "truth":
        wt_list = jit_weight_truth(args)
    sum_wt = onp.sum(wt_list[0])
    total_weight = {"all_mods_wt": wt_list[0]}
    total_fit_frac = 0.0
    resonance_names = ["phif0_kk_BW_BW", "phif2_kk_BW_BW", "phif2_kk_BW_flatte1270"]
    for i, name in enumerate(resonance_names):
        wt = wt_list[i + 1]
        for j in range(wt.shape[0]):
            key = f"{name}_{j}"
            total_weight[key] = wt[j]
            frac = onp.sum(wt[j]) / sum_wt
            print(f"{key} frac: {frac}")
            total_fit_frac += frac
    total_weight["fit_value"] = onp.array(args_list)
    total_weight["sum_wt"] = sum_wt
    print(f"total fit fraction: {total_fit_frac}")
    os.makedirs("output/draw", exist_ok=True)
    if mode == "pass":
        onp.savez("output/draw/weight.npz", **total_weight)
    elif mode == "truth":
        onp.savez("output/draw/weight_truth.npz", **total_weight)
    return total_fit_frac


# SECTION: draw_load_data_section


data = load_data()
data = normalize_data(data)
jax_data = prepare_data_for_jax(data)

# 实验数据
data_phi_kk = jax_data['data_phi_kk']
data_f_kk = jax_data['data_f_kk']
data_phif0_kk = jax_data['data_phif0_kk']
data_phif2_kk = jax_data['data_phif2_kk']
data_b123_kk = jax_data['data_b123_kk']
data_b124_kk = jax_data['data_b124_kk']

# MC数据
mc_phi_kk = jax_data['mc_phi_kk']
mc_f_kk = jax_data['mc_f_kk']
mc_phif0_kk = jax_data['mc_phif0_kk']
mc_phif2_kk = jax_data['mc_phif2_kk']
mc_b123_kk = jax_data['mc_b123_kk']
mc_b124_kk = jax_data['mc_b124_kk']

# Truth数据
truth_phi_kk = jax_data['truth_phi_kk']
truth_f_kk = jax_data['truth_f_kk']
truth_phif0_kk = jax_data['truth_phif0_kk']
truth_phif2_kk = jax_data['truth_phif2_kk']
truth_b123_kk = jax_data['truth_b123_kk']
truth_b124_kk = jax_data['truth_b124_kk']


# SECTION: draw_main_section

if __name__ == "__main__":
    logger = setup_logging()
    config.update("jax_enable_x64", True)

    args_list = onp.load("output/fit/fit_result_values.npy")

    logger.info("计算数据权重 (mode=pass)...")
    run_weight(args_list, mode="pass")

    logger.info("计算 truth MC 权重 (mode=truth)...")
    run_weight(args_list, mode="truth")

    logger.info("权重计算完成，结果已保存至 output/draw/")

#==============================================================================
