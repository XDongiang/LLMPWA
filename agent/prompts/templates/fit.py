# SECTION: LOGGING_CONFIG
def setup_logging():
    """Setup logging configuration"""
    with open("config/logconfig_fit.json", "r") as config_file:
        LOGGING_CONFIG = json.load(config_file)
        logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger("fit")


# SECTION: calculate_functions
def calculate_BW_BW(A_mass, A_width, phi_kk, B_mass, B_width, f_kk, Amplitude_param_AMP, Amplitude_param_const, Amplitude_param_theta):
    A_propagator = BW(A_mass, A_width, phi_kk)
    B_propagator = np.moveaxis(
        vmap(partial(BW, Sbc=f_kk))(B_mass, B_width), 1, 0
    )
    propagator_combined = dplex_deinsum("j, ij->ij", A_propagator, B_propagator)
    const_ph = dplex_dconstruct(Amplitude_param_const, Amplitude_param_theta)
    result = dplex_deinsum_ord("ijk,li->ljk", Amplitude_param_AMP, const_ph)
    result = dplex_deinsum("ljk,lj->jk", result, propagator_combined)
    return result

def component_BW_BW(A_mass, A_width, phi_kk, B_mass, B_width, f_kk, Amplitude_param_AMP, Amplitude_param_const, Amplitude_param_theta):
    A_propagator = BW(A_mass, A_width, phi_kk)
    B_propagator = np.moveaxis(
        vmap(partial(BW, Sbc=f_kk))(B_mass, B_width), 1, 0
    )
    propagator_combined = dplex_deinsum("j, ij->ij", A_propagator, B_propagator)
    const_ph = dplex_dconstruct(Amplitude_param_const, Amplitude_param_theta)
    result = dplex_deinsum_ord("ijk,li->ljk", Amplitude_param_AMP, const_ph)
    result = dplex_deinsum("ljk,lj->ljk", result, propagator_combined)
    return result

def calculate_BW_flatte980(A_mass, A_width, phi_kk, B_mass, B_g_kk, B_rg, f_kk, Amplitude_param_AMP, Amplitude_param_const, Amplitude_param_theta):
    A_propagator = BW(A_mass, A_width, phi_kk)
    B_propagator = flatte980(B_mass, B_g_kk, B_rg, f_kk)
    propagator_combined = dplex_deinsum("j, j->j", A_propagator, B_propagator)
    const_ph = dplex_dconstruct(Amplitude_param_const, Amplitude_param_theta)
    result = dplex_deinsum_ord("ijk,li->ljk", Amplitude_param_AMP, const_ph)
    result = dplex_deinsum("ljk,j->jk", result, propagator_combined)
    return result

def component_BW_flatte980(A_mass, A_width, phi_kk, B_mass, B_g_kk, B_rg, f_kk, Amplitude_param_AMP, Amplitude_param_const, Amplitude_param_theta):
    A_propagator = BW(A_mass, A_width, phi_kk)
    B_propagator = flatte980(B_mass, B_g_kk, B_rg, f_kk)
    propagator_combined = dplex_deinsum("j, j->j", A_propagator, B_propagator)
    const_ph = dplex_dconstruct(Amplitude_param_const, Amplitude_param_theta)
    result = dplex_deinsum_ord("ijk,li->ljk", Amplitude_param_AMP, const_ph)
    result = dplex_deinsum("ljk,j->ljk", result, propagator_combined)
    return result
# SECTION: likelihood_functions
def data_likelihood_kk(args):
    params = extract_parameters(args)
    data_phif0_kk_BW_BW = calculate_BW_BW(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['phif0_kk_BW_BW_mass'], params['phif0_kk_BW_BW_width'], data_f_kk,
        data_phif0_kk, params['phif0_kk_BW_BW_const'], params['phif0_kk_BW_BW_theta']
    )
    data_phif2_kk_BW_BW = calculate_BW_BW(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['phif2_kk_BW_BW_mass'], params['phif2_kk_BW_BW_width'], data_f_kk,
        data_phif2_kk, params['phif2_kk_BW_BW_const'], params['phif2_kk_BW_BW_theta']
    )
    data_phif2_kk_BW_flatte1270 = calculate_BW_flatte1270(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['phif2_kk_BW_flatte1270_mass'], params['phif2_kk_BW_flatte1270_width'], data_f_kk,
        data_phif2_kk, params['phif2_kk_BW_flatte1270_const'], params['phif2_kk_BW_flatte1270_theta']
    )
    component_data_phif0_kk_BW_BW = component_BW_BW(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['phif0_kk_BW_BW_mass'], params['phif0_kk_BW_BW_width'], truth_f_kk,
        truth_phif0_kk, params['phif0_kk_BW_BW_const'], params['phif0_kk_BW_BW_theta']
    )
    component_data_phif2_kk_BW_BW = component_BW_BW(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['phif2_kk_BW_BW_mass'], params['phif2_kk_BW_BW_width'], truth_f_kk,
        truth_phif2_kk, params['phif2_kk_BW_BW_const'], params['phif2_kk_BW_BW_theta']
    )
    component_data_phif2_kk_BW_flatte1270 = component_BW_flatte1270(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['phif2_kk_BW_flatte1270_mass'], params['phif2_kk_BW_flatte1270_width'], truth_f_kk,
        truth_phif2_kk, params['phif2_kk_BW_flatte1270_const'], params['phif2_kk_BW_flatte1270_theta']
    )
    sum_frac = np.sum(dplex_dabs(
        np.einsum("mljk->mjk", component_data_phif0_kk_BW_BW) +
        np.einsum("mljk->mjk", component_data_phif2_kk_BW_BW) +
        np.einsum("mljk->mjk", component_data_phif2_kk_BW_flatte1270)
    ))
    frac_f0_BW = np.sum(np.einsum("ljk->l", dplex_dabs(component_data_phif0_kk_BW_BW)) / sum_frac)
    frac_f2_BW = np.sum(np.einsum("ljk->l", dplex_dabs(component_data_phif2_kk_BW_BW)) / sum_frac)
    frac_f2_flatte = np.sum(np.einsum("ljk->l", dplex_dabs(component_data_phif2_kk_BW_flatte1270)) / sum_frac)
    total_frac = frac_f0_BW + frac_f2_BW + frac_f2_flatte
    step_function = np.power(total_frac - 1.03, 2.0) * constraint_strength
    total_amplitude = data_phif0_kk_BW_BW
    total_amplitude = total_amplitude + data_phif2_kk_BW_BW
    total_amplitude = total_amplitude + data_phif2_kk_BW_flatte1270
    likelihood = -np.sum(np.log(np.sum(dplex_dabs(total_amplitude), axis=1))) + step_function
    return likelihood

def mc_likelihood_kk(args):
    params = extract_parameters(args)
    total_mc = calculate_BW_flatte980(
        params['phi_mass'], params['phi_width'], mc_phi_kk,
        params['phif0_kk_BW_flatte980_mass'], params['phif0_kk_BW_flatte980_g_kk'], params['phif0_kk_BW_flatte980_rg'], mc_f_kk,
        mc_phif0_kk, params['phif0_kk_BW_flatte980_amplitude_consts'], params['phif0_kk_BW_flatte980_amplitude_thetas']
    )
    total_mc = total_mc + calculate_BW_BW(
        params['phi_mass'], params['phi_width'], mc_phi_kk,
        params['phif0_kk_BW_BW_mass'], params['phif0_kk_BW_BW_width'], mc_f_kk,
        mc_phif0_kk, params['phif0_kk_BW_BW_amplitude_consts'], params['phif0_kk_BW_BW_amplitude_thetas']
    )
    total_mc = total_mc + calculate_BW_flatte1270(
        params['phi_mass'], params['phi_width'], mc_phi_kk,
        params['phif2_kk_BW_flatte1270_mass'], params['phif2_kk_BW_flatte1270_width'], mc_f_kk,
        mc_phif2_kk, params['phif2_kk_BW_flatte1270_amplitude_consts'], params['phif2_kk_BW_flatte1270_amplitude_thetas']
    )
    total_mc = total_mc + calculate_BW_BW(
        params['phi_mass'], params['phi_width'], mc_phi_kk,
        params['phif2_kk_BW_BW_mass'], params['phif2_kk_BW_BW_width'], mc_f_kk,
        mc_phif2_kk, params['phif2_kk_BW_BW_amplitude_consts'], params['phif2_kk_BW_BW_amplitude_thetas']
    )
    return np.mean(np.sum(dplex_dabs(total_mc), axis=1))

# SECTION: combined_likelihood_function

def combined_likelihood(args):
    """组合似然函数: data_likelihood + datasize * log(mc_likelihood)"""
    data_lh = data_likelihood_kk(args)
    mc_lh = mc_likelihood_kk(args)
    combined_lh = data_lh + data_size * np.log(mc_lh)
    return combined_lh

def hvp_combined_likelihood(args, vector):
    """计算Hessian-vector product for combined likelihood"""
    return jvp(grad(combined_likelihood), [args], [vector])[1]

# SECTION: prepare_data_parameters
args_list = onp.array([
    0.9794812115574156, 0.10678616326827592, 8.570187550432664,
    0.1065182971468388, 0.03807025376236671, 1.6761965590304995,
    0.16270071108440043, 0.02201375198386279, 0.007433204877151768,
    0.008302300118288414, -0.018544178045626723, 1.2896149318644679,
    0.1959796900380022, -0.013533706721054747, -0.01650921272412254,
    0.015339403283337268, 0.029798824827026338, 0.020982478502465995,
    0.05458389002821978, 0.016032334798079934, 0.017179622009723918,
    -0.050143087437086675, -0.008718872479379924, 1.5222602842746435,
    2.1619576785269476, 2.547889297662712, 0.08576525399315078,
    0.15906049251159413, 0.324001266488012, -0.005345214125595505,
    0.0031770703345810553, -0.0036743677603351056, 0.004813366936315499,
    0.010000673114238258, 0.004643720949518616, -0.0009682564746806725,
    -0.0029804674908844213, 0.008315390493289363, 0.0005819695034846763,
    -0.15266341362240637, -0.05030214288962155, -0.01856511044577769,
    0.0908719088071242, 0.029544599934778714, -0.010529063356530807,
    0.003910028530572422, 0.0031787953173277725, 0.0333658928584713,
    -0.006053516058970728, 0.00545992748088964, -0.012498776031677225,
    0.002196563552197887, -0.016814307435916394, -0.007232946300343621,
    0.06277085590848677, -0.0031366601030189344, 0.13756083806604205,
    -0.0033564526519642953, 0.06849361819185686
])

def extract_parameters(args):
    """提取所有物理参数的公共函数"""
    return {
        'phi_mass': np.array([1.02]),
        'phi_width': np.array([0.004]),
        'f980_mass': np.array([args[0]]),
        'f980_g_kk': np.array([args[1]]),
        'f980_rg': np.array([args[2]]),
        'f980_const': np.array([0.1, args[3]]).reshape(-1, 2),
        'f980_theta': np.array([0.1, args[4]]).reshape(-1, 2),
        'f0_mass': np.array([args[5]]),
        'f0_width': np.array([args[6]]),
        'f0_const': np.array([args[7], args[8]]).reshape(-1, 2),
        'f0_theta': np.array([args[9], args[10]]).reshape(-1, 2),
        'f1270_mass': np.array([args[11]]),
        'f1270_width': np.array([args[12]]),
        'f1270_const': np.array([args[13], args[14], args[15], args[16], args[17]]).reshape(-1, 5),
        'f1270_theta': np.array([args[18], args[19], args[20], args[21], args[22]]).reshape(-1, 5),
        'f2_mass': np.array([args[23], args[24], args[25]]),
        'f2_width': np.array([args[26], args[27], args[28]]),
        'f2_const': np.array([args[29], args[30], args[31], args[32], args[33],
                               args[34], args[35], args[36], args[37], args[38],
                               args[39], args[40], args[41], args[42], args[43]]).reshape(-1, 5),
        'f2_theta': np.array([args[44], args[45], args[46], args[47], args[48],
                               args[49], args[50], args[51], args[52], args[53],
                               args[54], args[55], args[56], args[57], args[58]]).reshape(-1, 5)
    }

# SECTION: load_data_section

data = load_data()
data = normalize_data(data)
jax_data = prepare_data_for_jax(data)

# 实验数据
data_phi_kk = jax_data['data_phi_kk']
data_f_kk = jax_data['data_f_kk']
data_phif0_kk = jax_data['data_phif0_kk']
data_phif2_kk = jax_data['data_phif2_kk']

# MC数据
mc_phi_kk = jax_data['mc_phi_kk']
mc_f_kk = jax_data['mc_f_kk']
mc_phif0_kk = jax_data['mc_phif0_kk']
mc_phif2_kk = jax_data['mc_phif2_kk']

# Truth数据（用于约束）
truth_phi_kk = jax_data['truth_phi_kk']
truth_f_kk = jax_data['truth_f_kk']
truth_phif0_kk = jax_data['truth_phif0_kk']
truth_phif2_kk = jax_data['truth_phif2_kk']

# 数据大小（用于似然函数计算）
data_size = len(data_phi_kk)

# SECTION: main_section

if __name__  == "__main__":
    """使用Newton-CG方法和HVP的拟合函数"""
    logger = setup_logging()
    logger.info("开始HVP优化版PWA拟合（Newton-CG方法）")

    # 设置JAX
    config.update("jax_enable_x64", True)

    # 约束强度（可调节）
    constraint_strength = 0.0

    # 编译JAX函数（使用HVP版本）
    logger.info("编译JAX函数（HVP版本）...")
    jit_likelihood = jit(combined_likelihood)
    jit_grad = jit(grad(combined_likelihood))
    jit_hvp = jit(hvp_combined_likelihood)

    # 测试编译
    test_result = jit_likelihood(args_list)
    logger.info(f"初始似然值: {test_result}")

    # 测试HVP编译
    test_vector = onp.ones_like(args_list)
    test_hvp = jit_hvp(args_list, test_vector)
    logger.info(f"HVP测试完成，结果形状: {test_hvp.shape}")

    # 定义callback函数
    def my_callback(x):
        current_likelihood = jit_likelihood(x)
        logger.info(f"当前似然值: {current_likelihood}")

    # 定义HVP函数（适应scipy.optimize接口）
    def hessp(x, p):
        return onp.array(jit_hvp(x, p))

    # 优化（使用Newton-CG方法）
    logger.info("开始优化（Newton-CG + HVP）...")
    start_time = time.time()

    result = minimize(
        fun=lambda x: float(jit_likelihood(x)),
        x0=args_list,
        jac=lambda x: onp.array(jit_grad(x)),
        hessp=hessp,
        method="Newton-CG",
        callback=my_callback,
        options={"disp": False, "xtol": 1e-8}
    )

    end_time = time.time()

    # 结果报告
    logger.info("="*50)
    logger.info(f"HVP优化完成!")
    logger.info(f"成功: {result.success}")
    logger.info(f"最终似然值: {result.fun}")
    logger.info(f"迭代次数: {result.nit}")
    logger.info(f"函数调用次数: {result.nfev}")
    logger.info(f"梯度调用次数: {result.njev}")
    logger.info(f"Hessian调用次数: {result.nhev}")
    logger.info(f"优化时间: {end_time - start_time:.2f} 秒")
    logger.info(f"优化信息: {result.message}")
    logger.info("="*50)

    # 计算参数误差（Hessian逆矩阵对角线）
    logger.info("计算参数误差（Hessian逆矩阵）...")
    args_size = args_list.shape[0]
    hessian_matrix = onp.zeros([args_size, args_size])
    for i in range(args_size):
        v = onp.zeros(args_size)
        v[i] = 1.0
        hessian_matrix[:, i] = onp.array(jit_hvp(result.x, v))
    ferror = onp.sqrt(onp.diag(onp.linalg.inv(hessian_matrix)))
    logger.info(f"误差计算完成")

    # 保存结果
    os.makedirs("output/fit", exist_ok=True)
    onp.save("output/fit/fit_result_values.npy", result.x)
    onp.save("output/fit/fit_result_errors.npy", ferror)
    logger.info("参数已保存至 output/fit/fit_result_values.npy")

    result_config = build_config(result.x, ferror)
    with open("output/fit/fit_result_config.json", "w", encoding="utf-8") as f:
        json.dump(result_config, f, indent=4)
    logger.info("配置已保存至 output/fit/fit_result_config.json")
