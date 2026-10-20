"""kk_dis —— 原生 JAX 多进程拟合（由旧的 Python multiprocessing 版原地改造）。

======================================================================
移植说明
======================================================================

旧版（本文件的历史版本）
    用 ``multiprocessing.Process`` + ``Manager``/``Thread`` 在**每张卡上跑一个
    独立的拟合**：``Process_Initializer_Generator`` 把数据按事例轴切成
    ``data_slices`` 片，``Control.run_multiprocess`` 每次启动 ``max_processes``
    个进程，每个进程拿一片数据 + 一张卡，各自 ``scipy.optimize.minimize``，
    结果经 ``Manager.dict`` 回传。并行度靠"多进程 × 多线程"堆出来。

新版（本文件）
    用仓库里已经实现好的**原生 JAX 多进程**方案（同 ``run/fit_script_dist.py``）：

      * ``jax.distributed.initialize(coordinator_address, num_processes,
        process_id)`` —— 每个进程一张卡，靠 JAX 自己的集合通信互联；
      * ``Mesh(jax.devices(), ("event",))`` + ``NamedSharding(mesh, P("event"))``
        —— 把事例轴**真正分片**到各设备上；对事例轴的归约（``np.sum``）
        自动变成跨设备 all-reduce；
      * **chief-only SciPy**：只在 process 0 跑 ``scipy.optimize.minimize``，
        其余进程进入 ``serve_optimizer_requests()``，用
        ``multihost_utils.broadcast_one_to_all`` 接收 chief 广播的
        objective / gradient / hessp 请求并执行对应 JAX 计算。这样即使
        Newton-CG 的 CG 子迭代路径分叉，两端 collectives 也严格一一对应，
        不会死锁；
      * Hessian 逐列求 → 协方差 → 误差，与原版一致。

    **语义变化（重要）**：旧版是"每卡一个独立拟合、各拟合自己那片数据"；
    新版是"**一个拟合用满所有卡**、全局数据在事例轴上分片"。目标函数因此是
    对**全量事例**的似然，不再是逐片似然之和。

物理模型：完整保留旧版
    * 22 条振幅道（3686 / 3773 两组 × 11 条）
    * 219 槽参数向量 + 92 个自由参数（``FLOAT_LIST``）
    * ``weight_kk`` 事例权重
    * lasso / step-function 份额约束
    * 全部传播子（BW / Flatté / PBW / GS 等）

    ``PWAFunc`` 里 ``data_likelihood_kk`` / ``mc_likelihood_kk`` 的**表达式逐字
    未动**，只把数据访问从 ``self.data_*`` 改成显式传入的 ``data[...]``。
    这一步是必须的：闭包捕获的 jax 数组会被当成 jaxpr 常量（在 mesh 下会退化为
    复制），只有把数据作为 jit 的**显式参数**，分片才真正生效。

组合似然
    旧版 ``Control.thread_likelihood``::

        result = sum(data_lh) + 21181.60837312616 * log(sum(mc_lh) / 925000.0)

    新版用同一个式子（见 ``combined_likelihood``）。旧版把 grad / HVP 手工拆成
    data / mc 两块再拼（``thread_grad_likelihood`` / ``thread_hvp``）；新版直接
    对这一个标量做 autodiff —— 数学上恒等，且与 ``run/fit_script_dist.py`` 一致。

运行
    单进程::

        python run/fit_object_kk.py

    多进程（每卡一个进程/容器，env 与 ``run_fit_multinode.sh`` 一致）::

        JAX_COORDINATOR_ADDRESS=host:12345 JAX_NUM_PROCESSES=2 JAX_PROCESS_ID=0 \
            python run/fit_object_kk.py &
        JAX_COORDINATOR_ADDRESS=host:12345 JAX_NUM_PROCESSES=2 JAX_PROCESS_ID=1 \
            python run/fit_object_kk.py &

环境变量
    ======================  ==================================================
    ``JAX_COORDINATOR_ADDRESS`` / ``JAX_NUM_PROCESSES`` / ``JAX_PROCESS_ID``
                            分布式拓扑（``JAX_NUM_PROCESSES=1`` 时退化为本地拟合）
    ``KK_DATA_ROOT``        数据根目录（默认 ``data``）
    ``KK_REAL_DIR``         实验数据目录（默认 ``<root>/real_data``）
    ``KK_MC_DIR``           MC 积分样本目录（默认 ``<root>/mc_int``，缺失时退回
                            ``<root>/mc_truth``）
    ``KK_TRUTH_DIR``        truth-MC 目录（默认 ``<root>/mc_truth``）
    ``KK_WEIGHT_FILE``      权重文件（默认 ``<root>/weight/weight_kk.npy``）
    ``KKDIS_X0_NPY``        用 ``.npy`` 覆盖 92 维初值
    ``KKDIS_OUTDIR``        ``output/`` 下的输出子目录（默认 ``fit``）
    ``KKDIS_MAXITER``       Newton-CG 最大迭代（0 = 不限）
    ``KKDIS_XTOL``          Newton-CG xtol（默认 1e-6，与原版一致）
    ``KK_TRUTH_MC_CAP``     truth-MC 截断事例数（默认 150000，与原版一致）
    ``KK_MC_NORM_CONST``    组合似然 MC 归一化常数（默认 21181.60837312616）
    ``KK_MC_NORM_DENOM``    MC 归一化分母（默认 925000.0）
    ``KK_WEIGHT_FALLBACK``  权重长度不匹配时改用 1.0（默认 0 = 直接报错）
    ======================  ==================================================

数据布局（与旧版一致，缺失的 ``mc_int/`` 自动退回 ``mc_truth/``）
    ``<root>/real_data/<name>.npy``   实验数据
    ``<root>/mc_int/<name>.npy``      MC 积分样本（缺失时用 ``mc_truth/``）
    ``<root>/mc_truth/<name>.npy``    truth-MC（用于 lasso 约束）
    ``<root>/weight/weight_kk.npy``   事例权重

    ``<name>`` ∈ {phif0_kk, phif2_kk, u_rho_r_kk, u_rho_l_kk, u_b_r_kk, u_b_l_kk,
    phi_kk, f_kk, b124_kk, b123_kk}。

    注意：旧版权重 ``weight_kk.npy`` 是 5865 个事例，而 ``data/real_data`` 里
    的 ``phi_kk`` 等是 10000 个事例 —— 两者长度不一致。旧版之所以能跑，是因为
    它的数据来自 ``data/draw_mc/real_data``（5865 事例、含 ``u_*`` 张量）。要
    复现旧版，请把 ``KK_DATA_ROOT`` 指向该目录布局::

        KK_DATA_ROOT=data/draw_mc \\
        KK_TRUTH_DIR=data/mc_truth \\
        KK_WEIGHT_FILE=data/weight/weight_kk.npy \\
        python run/fit_object_kk.py

    加载器会在开跑前用 ``_validate_event_axes`` 校验事例轴一致性，长度对不上
    时直接报错并指出具体文件。
"""

import json
import logging
import logging.config
import os
import sys
import time
from functools import partial

import numpy as onp

import jax
import jax.numpy as np
from jax import config, device_put, grad, jit, jvp, vmap
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from scipy.optimize import minimize


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
os.chdir(_ROOT)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 旧版在 Control.run 里设的；放在模块级更安全（必须在建任何 jax 数组之前）
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# dplex：复数按 (实部, 虚部) 纵向堆叠的 dplex 表示
# ---------------------------------------------------------------------------
# 生产环境用 dlib.dplex（与旧版同一实现）。没有 dlib 时退回仓库里等价的
# 实现（run/base_functions.py 的 dplex_* 一族），公式逐条对齐：
#   dabs(aa)          = aa[0]**2 + aa[1]**2          （|z|^2）
#   dconstruct(a, b)  = stack([a, b], axis=0)
#   ddivide(a, bb)    = a * conj(bb) / |bb|^2
#   deinsum           = 复数 einsum（两个 dplex 相乘）
#   deinsum_ord       = 实数 × 复数（dplex 第二操作数）
#   dtomine           = 复数组 -> dplex
try:
    from dlib import dplex
except ModuleNotFoundError:  # 本地 / 无 dlib 环境

    class _DplexShim(object):
        """与 ``dlib.dplex`` 同接口的最小实现（公式同 run/base_functions.py）。"""

        @staticmethod
        def dconstruct(aa, bb):
            return np.stack([aa, bb], axis=0)

        @staticmethod
        def dabs(aa):
            return aa[0] ** 2 + aa[1] ** 2

        @staticmethod
        def ddivide(a, bb):
            denom = bb[0] ** 2 + bb[1] ** 2
            return np.stack([a * bb[0] / denom, -a * bb[1] / denom], axis=0)

        @staticmethod
        def deinsum(subscript, aa, bb):
            real = np.einsum(subscript, aa[0], bb[0]) - np.einsum(subscript, aa[1], bb[1])
            imag = np.einsum(subscript, aa[0], bb[1]) + np.einsum(subscript, aa[1], bb[0])
            return np.stack([real, imag], axis=0)

        @staticmethod
        def deinsum_ord(subscript, aa, bb):
            return np.stack(
                [np.einsum(subscript, aa, bb[0]), np.einsum(subscript, aa, bb[1])],
                axis=0,
            )

        @staticmethod
        def dtomine(aa):
            return np.stack([np.real(aa), np.imag(aa)], axis=0)

    dplex = _DplexShim()


def setup_logging():
    """与旧版 / run/base_functions.py 一致：读 config/logconfig_fit.json。

    多进程时每个 rank 写自己的日志文件（``logs/fit_p<id>.log``）：多个 rank
    共享同一个挂载目录时，都写 ``logs/fit.log`` 会互相交错。
    """
    with open("config/logconfig_fit.json", "r") as config_file:
        LOGGING_CONFIG = json.load(config_file)
    process_id = int(os.environ.get("JAX_PROCESS_ID", 0))
    num_processes = int(os.environ.get("JAX_NUM_PROCESSES", 1))
    if num_processes > 1:
        filename = "logs/fit_p%d.log" % process_id
        for handler in LOGGING_CONFIG.get("handlers", {}).values():
            if "filename" in handler:
                handler["filename"] = filename
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger("fit")


logger = logging.getLogger("fit")


# ---------------------------------------------------------------------------
# 模型常量（旧版硬编码值，原样保留；需要时可用环境变量覆盖）
# ---------------------------------------------------------------------------
# truth-MC 截断（旧版 truth_npz 里的 0:150000）
TRUTH_MC_CAP = 150000

# 组合似然的 MC 归一化：旧版 Control.thread_likelihood 里的字面量
MC_NORM_CONST = float(os.environ.get("KK_MC_NORM_CONST", "21181.60837312616"))
MC_NORM_DENOM = float(os.environ.get("KK_MC_NORM_DENOM", "925000.0"))

# 旧版 Control.__init__ 里的参数向量（219 槽）与自由参数下标（92 个）
ARGS_LIST = onp.array([-1.0, 0.0, 1.02, 0.004, 0.9672168652830319, 0.1253740669877132, 2.7005052659229323, 0.1, 0.02528579544349515, 0.1, 0.08996768704897123, 1.278468123947484, 0.18253178603942197, -0.00394557414604787, -0.031456917490503056, -0.006453975683670302, -0.000751259137593863, 0.005700617518362637, 0.0019064455025013442, -0.016674047180805302, 0.00042242238223968634, -0.03844830111424663, 0.005482331962888952, 1.5150133016307599, 0.09287088061436292, 0.013989586500962098, 6.536487778784994e-05, -0.003039878487010079, -0.02290051721070652, -0.005149185577791717, -0.002271692391140229, 0.0014266102701488276, -0.0002464838712524822, 0.027487458758986043, -0.021217704602433694, 2.4753883485342745, 2.3212816589917145, 1.6998114389127226, 0.09005646310357743, 0.08237684207631378, 0.15477458920519632, 0.020345057172526142, -0.019615646872485935, -0.011694357209471467, -0.011750000184762972, 0.01586061786100111, -0.012419424802564446, -0.044498982664504055, -0.04475789168359524, -0.007019105704104814, -0.010339726069162758, 0.023800660913466182, -0.009236629506304373, 1.0, 100000.0, 0.018412616428206293, -0.014362887948466602, 0.010468017555146498, 0.08366133891030483, 2.064921722894072, 0.30238434570737827, -0.008061498374653295, -0.009561621302788686, 0.025185456013695852, 0.020110419942420712, 1.9435205040750276, 2.1783628905892916, 2.3736695101913585, 0.46553324520367734, 0.171365802944736, 0.35232188312114704, 0.010712196314346355, 0.00347347287437411, -0.001123202599978466, 0.003038246648574247, 0.006169230483688115, -0.008715046306530293, -0.0037697755182356366, -0.0021439357115616313, 0.009843889611037352, -0.002311207291324084, -0.027166615697145177, -0.03686780441569233, -0.030199614610973834, -0.014486397366395492, -0.03600045818702371, 0.019906238124680783, -0.02259436401747235, 0.004982846050260414, -0.03157580467041962, -0.0029303473657870892, 0.011827388845592392, 0.003308659428709971, 0.0031953094204852034, -0.00042027920310028563, 0.007031918963374428, -0.0608692440971674, -0.011852304444291448, -0.004489335843117047, 0.044159240796422666, 0.014583440328481448, 2.397312161851163, 0.08930983775105905, 0.0022778257328804684, -0.0033688144994624505, 1.8383688963975429, 2.21695086990315, 0.2886776944601449, 0.30820571811928366, -0.004938490390296392, 0.016593974473936707, 0.003986752368883177, 0.012164816898769098, 0.005186646362675694, 0.0029039740223936127, 0.007183982423137571, 0.007637037419706918, -0.0016244356428979153, 0.0010019867956887615, 0.004619734538692254, 0.01106119138947309, 0.012480170130002325, 0.0025652685920836142, 0.0005478969786869395, 0.0012203694483755145, 0.0, 0.2, 0.1, 0.0812758802808357, 0.1, 0.04219055453438748, -0.001908795026008432, -0.002198862452744889, 9.263590492855337e-05, 0.0038573660399420315, 0.004216085267475821, 0.01328748133437328, 0.019267847356889318, 0.00823480715160893, -0.0017014797381603949, 0.013688984939122168, 0.00836160494789744, 0.0022731471899687674, -0.0046768086923496405, -0.030976560693756445, 0.011016487927278483, 0.0007651896888136746, -0.0021162876037446768, -0.005601756967141078, -0.0013397178875563952, -0.00916615679988274, 0.006776209842051121, -0.005211775708005117, 0.013982507928474695, 0.007221463747811338, 0.016393747705371626, 0.0026018914837070166, -0.019419455060917547, -0.02162941877883244, -0.016406491062264772, -0.011508544965435973, -0.02996999596857378, -0.016574134291907173, 0.17112767897801573, 0.05688549960596512, -0.11230387384356917, -0.06356583331362733, -0.01907412396118519, -0.014324317992303416, -0.026948437967964086, -0.009458867738581251, -0.009557649920876458, -0.0031727092103564524, 0.014100451734034392, 0.010479172150098524, 0.0003692045644304492, 0.0018597973868881273, 0.00457547644999926, 0.003597272060697062, 0.0062908372146082804, 0.0035507238700523296, -0.017485005393528725, 0.015226995460400562, 0.0013420411659886469, 0.03677091206682049, -0.0004435335025267973, -0.017692773889318524, -0.014879688649882716, 0.00157904926827019, 0.00635591994629108, 0.006381438521152356, -0.01207283493112012, -0.0029108265764609606, 0.0016613195856758618, 0.011585574654273742, 0.005166091307635481, -0.0645311336569186, 0.003900999450326149, -0.006707506973748528, 0.044428897338296275, 0.005620130634310312, -0.0017169985093901598, -0.003072954777635661, 0.003136380638161863, 0.0020679454615716384, -0.0024864348773066037, 0.005679669182781839, 0.007687456697620569, -0.0014652125102298027, 0.004360881343635086, 0.004918464768283009, -0.00832920259713428, -0.001168027729720911, -0.011101976414934987, -0.003775730069307858, 0.009518783705592684, 0.011551460555613131, -0.0002691609960762099, 0.0002696843264585296])
FLOAT_LIST = onp.array([8, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56, 57, 58, 61, 62, 63, 64, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 103, 104, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126])


# ---------------------------------------------------------------------------
# 数据布局
# ---------------------------------------------------------------------------
# 12 个逻辑张量名（3686 / 3773 是命名约定，读同一个文件）
_TENSOR_NAMES = (
    "phif0_kk_3686", "phif2_kk_3686",
    "u_rho_r_kk_3686", "u_rho_l_kk_3686", "u_b_r_kk_3686", "u_b_l_kk_3686",
    "phif0_kk_3773", "phif2_kk_3773",
    "u_rho_r_kk_3773", "u_rho_l_kk_3773", "u_b_r_kk_3773", "u_b_l_kk_3773",
)
_SCALAR_NAMES = ("phi_kk", "f_kk", "b124_kk", "b123_kk")

# 逻辑名 -> 文件名（旧版 data_npz / mc_npz / truth_npz 的映射）
_TENSOR_FILES = {
    "phif0_kk_3686": "phif0_kk", "phif2_kk_3686": "phif2_kk",
    "u_rho_r_kk_3686": "u_rho_r_kk", "u_rho_l_kk_3686": "u_rho_l_kk",
    "u_b_r_kk_3686": "u_b_r_kk", "u_b_l_kk_3686": "u_b_l_kk",
    "phif0_kk_3773": "phif0_kk", "phif2_kk_3773": "phif2_kk",
    "u_rho_r_kk_3773": "u_rho_r_kk", "u_rho_l_kk_3773": "u_rho_l_kk",
    "u_b_r_kk_3773": "u_b_r_kk", "u_b_l_kk_3773": "u_b_l_kk",
}

# 全部 data / mc / truth 键（供显式数据传参的替换使用）
DATA_KEYS = tuple(
    prefix + name
    for prefix in ("data_", "mc_", "truth_")
    for name in (_TENSOR_NAMES + _SCALAR_NAMES)
) + ("wt_data_kk",)


class FitConfig(object):
    """运行期配置。

    替代旧版 ``args`` 类里的进程/线程拓扑字段（``total_gpu_id``、
    ``processes_gpus``、``max_processes``、``thread_gpus``、
    ``threads_in_one_gpu`` 等）——原生 JAX 下这些由 ``jax.distributed``
    与 mesh 接管，不再需要手工编排。
    """

    def __init__(self):
        self.data_root = os.environ.get("KK_DATA_ROOT", "data")
        self.outdir = os.environ.get("KKDIS_OUTDIR", "fit")
        self.x0_npy = os.environ.get("KKDIS_X0_NPY", "")
        self.maxiter = int(os.environ.get("KKDIS_MAXITER", "0"))
        self.xtol = float(os.environ.get("KKDIS_XTOL", "1e-6"))
        self.truth_mc_cap = int(os.environ.get("KK_TRUTH_MC_CAP", str(TRUTH_MC_CAP)))
        self.weight_fallback = os.environ.get("KK_WEIGHT_FALLBACK", "0") == "1"
        # 数据分散在多个目录时的显式覆盖（留空则都挂在 data_root 下）
        self.real_dir = os.environ.get("KK_REAL_DIR", "")
        self.mc_dir = os.environ.get("KK_MC_DIR", "")
        self.truth_dir = os.environ.get("KK_TRUTH_DIR", "")
        self.weight_file = os.environ.get("KK_WEIGHT_FILE", "")


def _resolve_mc_dir(root, config):
    """MC 积分样本目录。

    旧版读 ``<root>/mc_int/``。本仓库把 MC 样本合并进了 ``mc_truth/``，
    因此 ``mc_int/`` 缺失（或为空）时自动退回 ``mc_truth/``，并打日志说明。
    数值语义不变：归一化因子与 MC 似然用的是同一份样本。

    可用 ``KK_MC_DIR`` 显式指定（例如数据分散在多个目录时）。
    """
    if config.mc_dir:
        return config.mc_dir
    mc_int = os.path.join(root, "mc_int")
    if os.path.isdir(mc_int):
        npy = [f for f in os.listdir(mc_int) if f.endswith(".npy")]
        if npy:
            return mc_int
    logger.warning(
        "mc_int/ not found or empty under %r -> falling back to mc_truth/ for the "
        "MC-integration sample (normalization factor and mc likelihood)",
        root,
    )
    return os.path.join(root, "mc_truth")


def _load_cached(path, cache):
    """按路径缓存，避免 3686 / 3773 两份逻辑名把同一文件读两遍。"""
    if path not in cache:
        cache[path] = onp.load(path)
    return cache[path]


def _require_files(paths):
    """开跑前一次性报出所有缺失文件（而不是在某个 onp.load 里逐个炸）。"""
    missing = sorted({p for p in paths if not os.path.exists(p)})
    if missing:
        raise FileNotFoundError(
            "missing %d data file(s):\n  %s\n"
            "Set KK_DATA_ROOT (and/or KK_REAL_DIR / KK_MC_DIR / KK_TRUTH_DIR / "
            "KK_WEIGHT_FILE) to point at the dataset."
            % (len(missing), "\n  ".join(missing))
        )


def _event_axis(array, rank):
    """事例轴的位置：1 维 -> 0；3 维 (n_ch, n_ev, 2) -> 1。"""
    return 0 if rank == 1 else 1


def _validate_event_axes(data):
    """开跑前检查各张量的事例轴是否自洽。

    旧版按事例轴 ``array_split`` 后每个进程各拿一片，形状不一致时只会在很深的
    jit 里报错。这里提前检查，把问题指到具体文件和实际形状上。

    注意：``data_*`` 与 ``mc_*`` / ``truth_*`` 的事例数**本来就不同**（实验
    事例 vs MC 样本），所以只要求各组内部一致。
    """
    groups = {"data_": {}, "mc_": {}, "truth_": {}}
    for key, value in data.items():
        for prefix in groups:
            if key.startswith(prefix):
                arr = onp.asarray(value)
                groups[prefix][key] = arr.shape[_event_axis(arr, arr.ndim)]

    n_data = None
    for prefix, sizes in groups.items():
        distinct = sorted(set(sizes.values()))
        if len(distinct) > 1:
            detail = ", ".join(
                "%s=%d" % (k, sizes[k]) for k in sorted(sizes)
                if sizes[k] != distinct[0]
            )
            raise ValueError(
                "event-axis mismatch within %r group: sizes %s; offending: %s"
                % (prefix, distinct, detail)
            )
        if prefix == "data_":
            n_data = distinct[0]

    # 权重必须与实验事例一一对应（旧版直接相乘）
    wt = onp.asarray(data["wt_data_kk"])
    if wt.shape[0] != n_data:
        raise ValueError(
            "wt_data_kk has %d entries but data has %d events" % (wt.shape[0], n_data)
        )
    logger.info(
        "data sizes: n_data=%d n_mc=%d n_truth=%d",
        n_data,
        sorted(set(groups["mc_"].values()))[0],
        sorted(set(groups["truth_"].values()))[0],
    )


def load_data(config=None):
    """加载全量数据（不再按事例轴切片——切片交给 mesh 分片）。

    与旧版 ``Process_Initializer_Generator.data_npz/mc_npz/truth_npz`` 一一对应，
    只去掉 ``array_split``：``data_*``/``mc_*`` 保留全量，``truth_*`` 按旧版
    截断到 ``truth_mc_cap``。

    目录可用 ``KK_REAL_DIR`` / ``KK_MC_DIR`` / ``KK_TRUTH_DIR`` /
    ``KK_WEIGHT_FILE`` 单独覆盖（默认都挂在 ``KK_DATA_ROOT`` 下）。
    """
    config = config or FitConfig()
    root = config.data_root
    real = config.real_dir or os.path.join(root, "real_data")
    mc = _resolve_mc_dir(root, config)
    truth = config.truth_dir or os.path.join(root, "mc_truth")
    cap = config.truth_mc_cap
    cache = {}
    data = {}

    # 先把所有需要的文件列出来，缺哪个一次性报清楚
    needed = []
    for name in _TENSOR_NAMES:
        fname = _TENSOR_FILES[name] + ".npy"
        needed += [os.path.join(real, fname), os.path.join(mc, fname),
                   os.path.join(truth, fname)]
    for name in _SCALAR_NAMES:
        fname = name + ".npy"
        needed += [os.path.join(real, fname), os.path.join(mc, fname),
                   os.path.join(truth, fname)]
    needed.append(config.weight_file or os.path.join(root, "weight", "weight_kk.npy"))
    # 权重允许缺失（缺失时退化为 1.0），其余必须存在
    _require_files(needed[:-1])

    for name in _TENSOR_NAMES:
        fname = _TENSOR_FILES[name] + ".npy"
        data["data_" + name] = _load_cached(os.path.join(real, fname), cache)
        data["mc_" + name] = _load_cached(os.path.join(mc, fname), cache)
        data["truth_" + name] = _load_cached(os.path.join(truth, fname), cache)[:, 0:cap]

    for name in _SCALAR_NAMES:
        fname = name + ".npy"
        data["data_" + name] = _load_cached(os.path.join(real, fname), cache)
        data["mc_" + name] = _load_cached(os.path.join(mc, fname), cache)
        data["truth_" + name] = _load_cached(os.path.join(truth, fname), cache)[0:cap]

    # 事例权重（旧版 data_npz 里的 data/weight/weight_kk.npy）
    n_events = int(onp.asarray(data["data_phi_kk"]).shape[0])
    wpath = config.weight_file or os.path.join(root, "weight", "weight_kk.npy")
    if os.path.exists(wpath):
        weight = onp.load(wpath)
        if weight.shape[0] == n_events:
            data["wt_data_kk"] = weight
        elif config.weight_fallback:
            logger.warning(
                "weight_kk length %d != n_events %d -> using ones (KK_WEIGHT_FALLBACK=1)",
                weight.shape[0], n_events,
            )
            data["wt_data_kk"] = onp.ones(n_events)
        else:
            raise ValueError(
                "weight_kk.npy has %d entries but data has %d events; the weighted "
                "log-likelihood needs matching lengths. Point KK_WEIGHT_FILE at the "
                "matching weight file, or set KK_WEIGHT_FALLBACK=1 to run unweighted."
                % (weight.shape[0], n_events)
            )
    else:
        logger.warning("%s not found -> using unweighted (ones) log-likelihood", wpath)
        data["wt_data_kk"] = onp.ones(n_events)

    _validate_event_axes(data)
    return data



def normalize_data(data):
    """逐道归一化，与旧版 ``regular()`` 完全一致。

    旧版：``regular = 1/mean_over_events(|tensor|)``，其中 |tensor| 由
    **MC 样本**算出，再对所有 data/mc/truth 张量按道轴缩放。
    """
    for name in _TENSOR_NAMES:
        ref = onp.asarray(data["mc_" + name])
        regular = 1.0 / onp.mean(onp.sqrt(onp.sum(ref ** 2, axis=2)), axis=1)
        for prefix in ("data_", "mc_", "truth_"):
            key = prefix + name
            arr = onp.asarray(data[key])
            # (n_ch, n_events, 2) 按道轴 (c) 缩放
            data[key] = onp.einsum("cek,c->cek", arr, regular)
    return data


def shard_data_distributed(data, mesh):
    """把数据放到设备上，事例轴分片；不能整除时退化为复制。

    ``NamedSharding(mesh, P("event"))`` 要求事例轴能被设备数整除。生产数据
    ``real_data`` 是 5865（奇数），3 卡时不能整除，``device_put`` 会抛
    ``IndivisibleError``。分片只是省显存的优化，不能整除时整份复制到每张卡，
    数值结果与单进程完全一致（复制不会重复计数）。
    """
    n_dev = mesh.devices.size
    jax_data = {}
    for key, value in data.items():
        array = np.asarray(value)
        if array.ndim == 1:            # (n_events,)
            axis, spec = 0, P("event")
        elif array.ndim == 3:          # (n_ch, n_events, 2)
            axis, spec = 1, P(None, "event", None)
        else:
            raise ValueError("Unsupported data rank for %s: %d" % (key, array.ndim))
        if n_dev > 1 and array.shape[axis] % n_dev != 0:
            spec = P()
        jax_data[key] = device_put(array, NamedSharding(mesh, spec))
    return jax_data


def init_distributed():
    """原生 JAX 多进程初始化（与 run/fit_script_dist.py 同一套 env 约定）。"""
    coordinator = os.environ.get("JAX_COORDINATOR_ADDRESS", "")
    num_processes = int(os.environ.get("JAX_NUM_PROCESSES", 1))
    process_id = int(os.environ.get("JAX_PROCESS_ID", 0))
    if num_processes > 1:
        jax.distributed.initialize(
            coordinator_address=coordinator,
            num_processes=num_processes,
            process_id=process_id,
        )
    return num_processes, process_id


def build_mesh():
    return Mesh(jax.devices(), axis_names=("event",))


def _mesh_context(mesh):
    """``jax.set_mesh`` 存在时用它；老版本 jax 退化为 nullcontext。"""
    if hasattr(jax, "set_mesh"):
        return jax.set_mesh(mesh)
    import contextlib
    return contextlib.nullcontext()




# ---------------------------------------------------------------------------
# PWAFunc —— 振幅 / 似然内核
# ---------------------------------------------------------------------------
# 下面 22 条道的表达式、传播子、lasso 份额约束**逐字保留旧版**，只把数据访问
# 从 ``self.data_*`` 改成显式参数 ``data["data_*"]``。
#
# 为什么必须显式传参：jit 的闭包常量会被编进 jaxpr，在 mesh 下退化为"每张卡一份
# 副本"；只有作为 jit 的**参数**，``NamedSharding(P("event"))`` 的分片才真正生效，
# 对事例轴的 ``np.sum`` 才会变成跨设备 all-reduce。
class PWAFunc(object):
    """旧版 PWAFunc 的物理部分（无设备/进程状态）。"""

    def __init__(self):
        pass

    def data_likelihood_kk(self, args, data):
        
        kk_3686_amp_inta = np.array([-1.0])
        
        kk_3686_tha_inta = np.array([0.0])
        
        phi_mass = np.array([1.02])
        
        phi_width = np.array([0.004])
        
        kk_f980_mass = np.array([0.9672168652830319])
        
        kk_f980_g_kk = np.array([0.1253740669877132])
        
        kk_f980_rg = np.array([2.7005052659229323])
        
        kk_f980_const_3686 = np.array([0.1,args[0]]).reshape(-1,2)
        
        kk_f980_theta_3686 = np.array([0.1,args[1]]).reshape(-1,2)
        
        kk_f1270_mass = np.array([1.278468123947484])
        
        kk_f1270_width = np.array([0.18253178603942197])
        
        kk_f1270_const_3686 = np.array([args[2],args[3],args[4],args[5],args[6]]).reshape(-1,5)
        
        kk_f1270_theta_3686 = np.array([args[7],args[8],args[9],args[10],args[11]]).reshape(-1,5)
        
        kk_f2_mass = np.array([1.5150133016307599])
        
        kk_f2_width = np.array([0.09287088061436292])
        
        kk_f2_const_3686 = np.array([args[12],args[13],args[14],args[15],args[16]]).reshape(-1,5)
        
        kk_f2_theta_3686 = np.array([args[17],args[18],args[19],args[20],args[21]]).reshape(-1,5)
        
        kk_f0_mass = np.array([2.4753883485342745,2.3212816589917145,1.6998114389127226])
        
        kk_f0_width = np.array([0.09005646310357743,0.08237684207631378,0.15477458920519632])
        
        kk_f0_const_3686 = np.array([args[22],args[23],args[24],args[25],args[26],args[27]]).reshape(-1,2)
        
        kk_f0_theta_3686 = np.array([args[28],args[29],args[30],args[31],args[32],args[33]]).reshape(-1,2)
        
        kk_f0_long_mass = np.array([1.0])
        
        kk_f0_long_width = np.array([100000.0])
        
        kk_f0_long_const_3686 = np.array([args[34],args[35]]).reshape(-1,2)
        
        kk_f0_long_theta_3686 = np.array([args[36],args[37]]).reshape(-1,2)
        
        kk_f0l_mass = np.array([2.064921722894072])
        
        kk_f0l_width = np.array([0.30238434570737827])
        
        kk_f0l_const_3686 = np.array([args[38],args[39]]).reshape(-1,2)
        
        kk_f0l_theta_3686 = np.array([args[40],args[41]]).reshape(-1,2)
        
        kk_f2l_mass = np.array([1.9435205040750276,2.1783628905892916,2.3736695101913585])
        
        kk_f2l_width = np.array([0.46553324520367734,0.171365802944736,0.35232188312114704])
        
        kk_f2l_const_3686 = np.array([args[42],args[43],args[44],args[45],args[46],args[47],args[48],args[49],args[50],args[51],args[52],args[53],args[54],args[55],args[56]]).reshape(-1,5)
        
        kk_f2l_theta_3686 = np.array([args[57],args[58],args[59],args[60],args[61],args[62],args[63],args[64],args[65],args[66],args[67],args[68],args[69],args[70],args[71]]).reshape(-1,5)
        
        kk_u_rho_mass = np.array([2.397312161851163])
        
        kk_u_rho_width = np.array([0.08930983775105905])
        
        kk_u_rho_const_3686 = np.array([args[72]]).reshape(-1,1)
        
        kk_u_rho_theta_3686 = np.array([args[73]]).reshape(-1,1)
        
        kk_u_b_mass = np.array([1.8383688963975429,2.21695086990315])
        
        kk_u_b_width = np.array([0.2886776944601449,0.30820571811928366])
        
        kk_u_b_const_3686 = np.array([args[74],args[75],args[76],args[77],args[78],args[79],args[80],args[81]]).reshape(-1,4)
        
        kk_u_b_theta_3686 = np.array([args[82],args[83],args[84],args[85],args[86],args[87],args[88],args[89]]).reshape(-1,4)
        
        kk_3773_amp_inta = np.array([args[90]])
        
        kk_3773_tha_inta = np.array([args[91]])
        
        kk_f980_const_3773 = np.array([0.1,0.0812758802808357]).reshape(-1,2)
        
        kk_f980_theta_3773 = np.array([0.1,0.04219055453438748]).reshape(-1,2)
        
        kk_f1270_const_3773 = np.array([-0.001908795026008432,-0.002198862452744889,9.263590492855337e-05,0.0038573660399420315,0.004216085267475821]).reshape(-1,5)
        
        kk_f1270_theta_3773 = np.array([0.01328748133437328,0.019267847356889318,0.00823480715160893,-0.0017014797381603949,0.013688984939122168]).reshape(-1,5)
        
        kk_f2_const_3773 = np.array([0.00836160494789744,0.0022731471899687674,-0.0046768086923496405,-0.030976560693756445,0.011016487927278483]).reshape(-1,5)
        
        kk_f2_theta_3773 = np.array([0.0007651896888136746,-0.0021162876037446768,-0.005601756967141078,-0.0013397178875563952,-0.00916615679988274]).reshape(-1,5)
        
        kk_f0_const_3773 = np.array([0.006776209842051121,-0.005211775708005117,0.013982507928474695,0.007221463747811338,0.016393747705371626,0.0026018914837070166]).reshape(-1,2)
        
        kk_f0_theta_3773 = np.array([-0.019419455060917547,-0.02162941877883244,-0.016406491062264772,-0.011508544965435973,-0.02996999596857378,-0.016574134291907173]).reshape(-1,2)
        
        kk_f0_long_const_3773 = np.array([0.17112767897801573,0.05688549960596512]).reshape(-1,2)
        
        kk_f0_long_theta_3773 = np.array([-0.11230387384356917,-0.06356583331362733]).reshape(-1,2)
        
        kk_f0l_const_3773 = np.array([-0.01907412396118519,-0.014324317992303416]).reshape(-1,2)
        
        kk_f0l_theta_3773 = np.array([-0.026948437967964086,-0.009458867738581251]).reshape(-1,2)
        
        kk_f2l_const_3773 = np.array([-0.009557649920876458,-0.0031727092103564524,0.014100451734034392,0.010479172150098524,0.0003692045644304492,0.0018597973868881273,0.00457547644999926,0.003597272060697062,0.0062908372146082804,0.0035507238700523296,-0.017485005393528725,0.015226995460400562,0.0013420411659886469,0.03677091206682049,-0.0004435335025267973]).reshape(-1,5)
        
        kk_f2l_theta_3773 = np.array([-0.017692773889318524,-0.014879688649882716,0.00157904926827019,0.00635591994629108,0.006381438521152356,-0.01207283493112012,-0.0029108265764609606,0.0016613195856758618,0.011585574654273742,0.005166091307635481,-0.0645311336569186,0.003900999450326149,-0.006707506973748528,0.044428897338296275,0.005620130634310312]).reshape(-1,5)
        
        kk_u_rho_const_3773 = np.array([-0.0017169985093901598]).reshape(-1,1)
        
        kk_u_rho_theta_3773 = np.array([-0.003072954777635661]).reshape(-1,1)
        
        kk_u_b_const_3773 = np.array([0.003136380638161863,0.0020679454615716384,-0.0024864348773066037,0.005679669182781839,0.007687456697620569,-0.0014652125102298027,0.004360881343635086,0.004918464768283009]).reshape(-1,4)
        
        kk_u_b_theta_3773 = np.array([-0.00832920259713428,-0.001168027729720911,-0.011101976414934987,-0.003775730069307858,0.009518783705592684,0.011551460555613131,-0.0002691609960762099,0.0002696843264585296]).reshape(-1,4)
        
        
        data_phif0_kk_3686_BW_flatte980 = self.calculate_BW_flatte980(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const_3686,kk_f980_theta_3686,data["data_phi_kk"],data["data_f_kk"],data["data_phif0_kk_3686"])
        
        data_phif2_kk_3686_BW_flatte1270 = self.calculate_BW_flatte1270(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const_3686,kk_f1270_theta_3686,data["data_phi_kk"],data["data_f_kk"],data["data_phif2_kk_3686"])
        
        data_phif2_kk_3686_BW_BW = self.calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const_3686,kk_f2_theta_3686,data["data_phi_kk"],data["data_f_kk"],data["data_phif2_kk_3686"])
        
        data_phif0_kk_3686_BW_BW = self.calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0_mass,kk_f0_width,kk_f0_const_3686,kk_f0_theta_3686,data["data_phi_kk"],data["data_f_kk"],data["data_phif0_kk_3686"])
        
        data_phif0_kk_3686_BW_BW_long = self.calculate_BW_BW_long(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0_long_mass,kk_f0_long_width,kk_f0_long_const_3686,kk_f0_long_theta_3686,data["data_phi_kk"],data["data_f_kk"],data["data_phif0_kk_3686"])
        
        data_phif0_kk_3686_BW_BW_f0_kk = self.calculate_BW_BW_f0_kk(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0l_mass,kk_f0l_width,kk_f0l_const_3686,kk_f0l_theta_3686,data["data_phi_kk"],data["data_f_kk"],data["data_phif0_kk_3686"])
        
        data_phif2_kk_3686_BW_BW_f2_kk = self.calculate_BW_BW_f2_kk(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f2l_mass,kk_f2l_width,kk_f2l_const_3686,kk_f2l_theta_3686,data["data_phi_kk"],data["data_f_kk"],data["data_phif2_kk_3686"])
        
        data_u_rho_r_kk_3686_BW_BW = self.calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3686,kk_u_rho_theta_3686,data["data_phi_kk"],data["data_b124_kk"],data["data_u_rho_r_kk_3686"])
        
        data_u_rho_l_kk_3686_BW_BW = self.calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3686,kk_u_rho_theta_3686,data["data_phi_kk"],data["data_b123_kk"],data["data_u_rho_l_kk_3686"])
        
        data_u_b_r_kk_3686_BW_BWb = self.calculate_BW_BWb(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3686,kk_u_b_theta_3686,data["data_phi_kk"],data["data_b124_kk"],data["data_u_b_r_kk_3686"])
        
        data_u_b_l_kk_3686_BW_BWb = self.calculate_BW_BWb(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3686,kk_u_b_theta_3686,data["data_phi_kk"],data["data_b123_kk"],data["data_u_b_l_kk_3686"])
        
        data_phif0_kk_3773_BW_flatte980 = self.calculate_BW_flatte980(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const_3773,kk_f980_theta_3773,data["data_phi_kk"],data["data_f_kk"],data["data_phif0_kk_3773"])
        
        data_phif2_kk_3773_BW_flatte1270 = self.calculate_BW_flatte1270(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const_3773,kk_f1270_theta_3773,data["data_phi_kk"],data["data_f_kk"],data["data_phif2_kk_3773"])
        
        data_phif2_kk_3773_BW_BW = self.calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const_3773,kk_f2_theta_3773,data["data_phi_kk"],data["data_f_kk"],data["data_phif2_kk_3773"])
        
        data_phif0_kk_3773_BW_BW = self.calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f0_mass,kk_f0_width,kk_f0_const_3773,kk_f0_theta_3773,data["data_phi_kk"],data["data_f_kk"],data["data_phif0_kk_3773"])
        
        data_phif0_kk_3773_BW_BW_long = self.calculate_BW_BW_long(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f0_long_mass,kk_f0_long_width,kk_f0_long_const_3773,kk_f0_long_theta_3773,data["data_phi_kk"],data["data_f_kk"],data["data_phif0_kk_3773"])
        
        data_phif0_kk_3773_BW_BW_f0_kk = self.calculate_BW_BW_f0_kk(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f0l_mass,kk_f0l_width,kk_f0l_const_3773,kk_f0l_theta_3773,data["data_phi_kk"],data["data_f_kk"],data["data_phif0_kk_3773"])
        
        data_phif2_kk_3773_BW_BW_f2_kk = self.calculate_BW_BW_f2_kk(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f2l_mass,kk_f2l_width,kk_f2l_const_3773,kk_f2l_theta_3773,data["data_phi_kk"],data["data_f_kk"],data["data_phif2_kk_3773"])
        
        data_u_rho_r_kk_3773_BW_BW = self.calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3773,kk_u_rho_theta_3773,data["data_phi_kk"],data["data_b124_kk"],data["data_u_rho_r_kk_3773"])
        
        data_u_rho_l_kk_3773_BW_BW = self.calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3773,kk_u_rho_theta_3773,data["data_phi_kk"],data["data_b123_kk"],data["data_u_rho_l_kk_3773"])
        
        data_u_b_r_kk_3773_BW_BWb = self.calculate_BW_BWb(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3773,kk_u_b_theta_3773,data["data_phi_kk"],data["data_b124_kk"],data["data_u_b_r_kk_3773"])
        
        data_u_b_l_kk_3773_BW_BWb = self.calculate_BW_BWb(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3773,kk_u_b_theta_3773,data["data_phi_kk"],data["data_b123_kk"],data["data_u_b_l_kk_3773"])
        
        
        lasso_data_phif0_kk_3686_BW_flatte980 = self.lasso_calculate_BW_flatte980(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const_3686,kk_f980_theta_3686,data["truth_phi_kk"],data["truth_f_kk"],data["truth_phif0_kk_3686"])
        
        lasso_data_phif2_kk_3686_BW_flatte1270 = self.lasso_calculate_BW_flatte1270(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const_3686,kk_f1270_theta_3686,data["truth_phi_kk"],data["truth_f_kk"],data["truth_phif2_kk_3686"])
        
        lasso_data_phif2_kk_3686_BW_BW = self.lasso_calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const_3686,kk_f2_theta_3686,data["truth_phi_kk"],data["truth_f_kk"],data["truth_phif2_kk_3686"])
        
        lasso_data_phif0_kk_3686_BW_BW = self.lasso_calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0_mass,kk_f0_width,kk_f0_const_3686,kk_f0_theta_3686,data["truth_phi_kk"],data["truth_f_kk"],data["truth_phif0_kk_3686"])
        
        lasso_data_phif0_kk_3686_BW_BW_long = self.lasso_calculate_BW_BW_long(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0_long_mass,kk_f0_long_width,kk_f0_long_const_3686,kk_f0_long_theta_3686,data["truth_phi_kk"],data["truth_f_kk"],data["truth_phif0_kk_3686"])
        
        lasso_data_phif0_kk_3686_BW_BW_f0_kk = self.lasso_calculate_BW_BW_f0_kk(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0l_mass,kk_f0l_width,kk_f0l_const_3686,kk_f0l_theta_3686,data["truth_phi_kk"],data["truth_f_kk"],data["truth_phif0_kk_3686"])
        
        lasso_data_phif2_kk_3686_BW_BW_f2_kk = self.lasso_calculate_BW_BW_f2_kk(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f2l_mass,kk_f2l_width,kk_f2l_const_3686,kk_f2l_theta_3686,data["truth_phi_kk"],data["truth_f_kk"],data["truth_phif2_kk_3686"])
        
        lasso_data_u_rho_r_kk_3686_BW_BW = self.lasso_calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3686,kk_u_rho_theta_3686,data["truth_phi_kk"],data["truth_b124_kk"],data["truth_u_rho_r_kk_3686"])
        
        lasso_data_u_rho_l_kk_3686_BW_BW = self.lasso_calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3686,kk_u_rho_theta_3686,data["truth_phi_kk"],data["truth_b123_kk"],data["truth_u_rho_l_kk_3686"])
        
        lasso_data_u_b_r_kk_3686_BW_BWb = self.lasso_calculate_BW_BWb(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3686,kk_u_b_theta_3686,data["truth_phi_kk"],data["truth_b124_kk"],data["truth_u_b_r_kk_3686"])
        
        lasso_data_u_b_l_kk_3686_BW_BWb = self.lasso_calculate_BW_BWb(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3686,kk_u_b_theta_3686,data["truth_phi_kk"],data["truth_b123_kk"],data["truth_u_b_l_kk_3686"])
        
        lasso_data_phif0_kk_3773_BW_flatte980 = self.lasso_calculate_BW_flatte980(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const_3773,kk_f980_theta_3773,data["truth_phi_kk"],data["truth_f_kk"],data["truth_phif0_kk_3773"])
        
        lasso_data_phif2_kk_3773_BW_flatte1270 = self.lasso_calculate_BW_flatte1270(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const_3773,kk_f1270_theta_3773,data["truth_phi_kk"],data["truth_f_kk"],data["truth_phif2_kk_3773"])
        
        lasso_data_phif2_kk_3773_BW_BW = self.lasso_calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const_3773,kk_f2_theta_3773,data["truth_phi_kk"],data["truth_f_kk"],data["truth_phif2_kk_3773"])
        
        lasso_data_phif0_kk_3773_BW_BW = self.lasso_calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f0_mass,kk_f0_width,kk_f0_const_3773,kk_f0_theta_3773,data["truth_phi_kk"],data["truth_f_kk"],data["truth_phif0_kk_3773"])
        
        lasso_data_phif0_kk_3773_BW_BW_long = self.lasso_calculate_BW_BW_long(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f0_long_mass,kk_f0_long_width,kk_f0_long_const_3773,kk_f0_long_theta_3773,data["truth_phi_kk"],data["truth_f_kk"],data["truth_phif0_kk_3773"])
        
        lasso_data_phif0_kk_3773_BW_BW_f0_kk = self.lasso_calculate_BW_BW_f0_kk(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f0l_mass,kk_f0l_width,kk_f0l_const_3773,kk_f0l_theta_3773,data["truth_phi_kk"],data["truth_f_kk"],data["truth_phif0_kk_3773"])
        
        lasso_data_phif2_kk_3773_BW_BW_f2_kk = self.lasso_calculate_BW_BW_f2_kk(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f2l_mass,kk_f2l_width,kk_f2l_const_3773,kk_f2l_theta_3773,data["truth_phi_kk"],data["truth_f_kk"],data["truth_phif2_kk_3773"])
        
        lasso_data_u_rho_r_kk_3773_BW_BW = self.lasso_calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3773,kk_u_rho_theta_3773,data["truth_phi_kk"],data["truth_b124_kk"],data["truth_u_rho_r_kk_3773"])
        
        lasso_data_u_rho_l_kk_3773_BW_BW = self.lasso_calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3773,kk_u_rho_theta_3773,data["truth_phi_kk"],data["truth_b123_kk"],data["truth_u_rho_l_kk_3773"])
        
        lasso_data_u_b_r_kk_3773_BW_BWb = self.lasso_calculate_BW_BWb(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3773,kk_u_b_theta_3773,data["truth_phi_kk"],data["truth_b124_kk"],data["truth_u_b_r_kk_3773"])
        
        lasso_data_u_b_l_kk_3773_BW_BWb = self.lasso_calculate_BW_BWb(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3773,kk_u_b_theta_3773,data["truth_phi_kk"],data["truth_b123_kk"],data["truth_u_b_l_kk_3773"])
        
        sum_frac = np.sum(dplex.dabs(np.einsum("mljk->mjk", lasso_data_phif0_kk_3686_BW_flatte980)+np.einsum("mljk->mjk", lasso_data_phif2_kk_3686_BW_flatte1270)+np.einsum("mljk->mjk", lasso_data_phif2_kk_3686_BW_BW)+np.einsum("mljk->mjk", lasso_data_phif0_kk_3686_BW_BW)+np.einsum("mljk->mjk", lasso_data_phif0_kk_3686_BW_BW_long)+np.einsum("mljk->mjk", lasso_data_phif0_kk_3686_BW_BW_f0_kk)+np.einsum("mljk->mjk", lasso_data_phif2_kk_3686_BW_BW_f2_kk)+np.einsum("mljk->mjk", lasso_data_u_rho_r_kk_3686_BW_BW)+np.einsum("mljk->mjk", lasso_data_u_rho_l_kk_3686_BW_BW)+np.einsum("mljk->mjk", lasso_data_u_b_r_kk_3686_BW_BWb)+np.einsum("mljk->mjk", lasso_data_u_b_l_kk_3686_BW_BWb)+np.einsum("mljk->mjk", lasso_data_phif0_kk_3773_BW_flatte980)+np.einsum("mljk->mjk", lasso_data_phif2_kk_3773_BW_flatte1270)+np.einsum("mljk->mjk", lasso_data_phif2_kk_3773_BW_BW)+np.einsum("mljk->mjk", lasso_data_phif0_kk_3773_BW_BW)+np.einsum("mljk->mjk", lasso_data_phif0_kk_3773_BW_BW_long)+np.einsum("mljk->mjk", lasso_data_phif0_kk_3773_BW_BW_f0_kk)+np.einsum("mljk->mjk", lasso_data_phif2_kk_3773_BW_BW_f2_kk)+np.einsum("mljk->mjk", lasso_data_u_rho_r_kk_3773_BW_BW)+np.einsum("mljk->mjk", lasso_data_u_rho_l_kk_3773_BW_BW)+np.einsum("mljk->mjk", lasso_data_u_b_r_kk_3773_BW_BWb)+np.einsum("mljk->mjk", lasso_data_u_b_l_kk_3773_BW_BWb))) 

        sum_strong = np.sum(dplex.dabs(np.einsum("mljk->mjk", lasso_data_phif0_kk_3686_BW_flatte980)+np.einsum("mljk->mjk", lasso_data_phif2_kk_3686_BW_flatte1270)+np.einsum("mljk->mjk", lasso_data_phif2_kk_3686_BW_BW)+np.einsum("mljk->mjk", lasso_data_phif0_kk_3686_BW_BW)+np.einsum("mljk->mjk", lasso_data_phif0_kk_3686_BW_BW_long)+np.einsum("mljk->mjk", lasso_data_phif0_kk_3686_BW_BW_f0_kk)+np.einsum("mljk->mjk", lasso_data_phif2_kk_3686_BW_BW_f2_kk)+np.einsum("mljk->mjk", lasso_data_u_rho_r_kk_3686_BW_BW)+np.einsum("mljk->mjk", lasso_data_u_rho_l_kk_3686_BW_BW)+np.einsum("mljk->mjk", lasso_data_u_b_r_kk_3686_BW_BWb)+np.einsum("mljk->mjk", lasso_data_u_b_l_kk_3686_BW_BWb)))
        total_ff = 1.589
        step_function = (np.power(np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3686_BW_flatte980))/sum_strong)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3686_BW_flatte1270))/sum_strong)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3686_BW_BW))/sum_strong)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3686_BW_BW))/sum_strong)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3686_BW_BW_long))/sum_strong)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3686_BW_BW_f0_kk))/sum_strong)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3686_BW_BW_f2_kk))/sum_strong)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_u_rho_l_kk_3686_BW_BW+lasso_data_u_rho_r_kk_3686_BW_BW))/sum_strong)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_u_b_l_kk_3686_BW_BWb+lasso_data_u_b_r_kk_3686_BW_BWb))/sum_strong)-1.195,2.0))*10000.0  + 0.0+(np.power(np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3773_BW_flatte980))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3773_BW_flatte1270))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3773_BW_BW))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3773_BW_BW))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3773_BW_BW_long))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3773_BW_BW_f0_kk))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3773_BW_BW_f2_kk))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_u_rho_l_kk_3773_BW_BW+lasso_data_u_rho_r_kk_3773_BW_BW))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_u_b_l_kk_3773_BW_BWb+lasso_data_u_b_r_kk_3773_BW_BWb))/sum_frac)-0.188,2.0))*10000.0 + (np.power(np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3686_BW_flatte980))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3686_BW_flatte1270))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3686_BW_BW))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3686_BW_BW))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3686_BW_BW_long))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3686_BW_BW_f0_kk))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3686_BW_BW_f2_kk))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_u_rho_l_kk_3686_BW_BW+lasso_data_u_rho_r_kk_3686_BW_BW))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_u_b_l_kk_3686_BW_BWb+lasso_data_u_b_r_kk_3686_BW_BWb))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3773_BW_flatte980))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3773_BW_flatte1270))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3773_BW_BW))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3773_BW_BW))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3773_BW_BW_long))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3773_BW_BW_f0_kk))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3773_BW_BW_f2_kk))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_u_rho_l_kk_3773_BW_BW+lasso_data_u_rho_r_kk_3773_BW_BW))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_u_b_l_kk_3773_BW_BWb+lasso_data_u_b_r_kk_3773_BW_BWb))/sum_frac)-total_ff,2.0))*10000.0
        # jax_exper.id_print(step_function)
        
        return -np.sum(data["wt_data_kk"]*np.log(np.sum(dplex.dabs(data_phif0_kk_3686_BW_flatte980 + data_phif2_kk_3686_BW_flatte1270 + data_phif2_kk_3686_BW_BW + data_phif0_kk_3686_BW_BW + data_phif0_kk_3686_BW_BW_long + data_phif0_kk_3686_BW_BW_f0_kk + data_phif2_kk_3686_BW_BW_f2_kk + data_u_rho_r_kk_3686_BW_BW + data_u_rho_l_kk_3686_BW_BW + data_u_b_r_kk_3686_BW_BWb + data_u_b_l_kk_3686_BW_BWb + data_phif0_kk_3773_BW_flatte980 + data_phif2_kk_3773_BW_flatte1270 + data_phif2_kk_3773_BW_BW + data_phif0_kk_3773_BW_BW + data_phif0_kk_3773_BW_BW_long + data_phif0_kk_3773_BW_BW_f0_kk + data_phif2_kk_3773_BW_BW_f2_kk + data_u_rho_r_kk_3773_BW_BW + data_u_rho_l_kk_3773_BW_BW + data_u_b_r_kk_3773_BW_BWb + data_u_b_l_kk_3773_BW_BWb),axis=1))) + step_function

    def mc_likelihood_kk(self, args, data):
        
        kk_3686_amp_inta = np.array([-1.0])
        
        kk_3686_tha_inta = np.array([0.0])
        
        phi_mass = np.array([1.02])
        
        phi_width = np.array([0.004])
        
        kk_f980_mass = np.array([0.9672168652830319])
        
        kk_f980_g_kk = np.array([0.1253740669877132])
        
        kk_f980_rg = np.array([2.7005052659229323])
        
        kk_f980_const_3686 = np.array([0.1,args[0]]).reshape(-1,2)
        
        kk_f980_theta_3686 = np.array([0.1,args[1]]).reshape(-1,2)
        
        kk_f1270_mass = np.array([1.278468123947484])
        
        kk_f1270_width = np.array([0.18253178603942197])
        
        kk_f1270_const_3686 = np.array([args[2],args[3],args[4],args[5],args[6]]).reshape(-1,5)
        
        kk_f1270_theta_3686 = np.array([args[7],args[8],args[9],args[10],args[11]]).reshape(-1,5)
        
        kk_f2_mass = np.array([1.5150133016307599])
        
        kk_f2_width = np.array([0.09287088061436292])
        
        kk_f2_const_3686 = np.array([args[12],args[13],args[14],args[15],args[16]]).reshape(-1,5)
        
        kk_f2_theta_3686 = np.array([args[17],args[18],args[19],args[20],args[21]]).reshape(-1,5)
        
        kk_f0_mass = np.array([2.4753883485342745,2.3212816589917145,1.6998114389127226])
        
        kk_f0_width = np.array([0.09005646310357743,0.08237684207631378,0.15477458920519632])
        
        kk_f0_const_3686 = np.array([args[22],args[23],args[24],args[25],args[26],args[27]]).reshape(-1,2)
        
        kk_f0_theta_3686 = np.array([args[28],args[29],args[30],args[31],args[32],args[33]]).reshape(-1,2)
        
        kk_f0_long_mass = np.array([1.0])
        
        kk_f0_long_width = np.array([100000.0])
        
        kk_f0_long_const_3686 = np.array([args[34],args[35]]).reshape(-1,2)
        
        kk_f0_long_theta_3686 = np.array([args[36],args[37]]).reshape(-1,2)
        
        kk_f0l_mass = np.array([2.064921722894072])
        
        kk_f0l_width = np.array([0.30238434570737827])
        
        kk_f0l_const_3686 = np.array([args[38],args[39]]).reshape(-1,2)
        
        kk_f0l_theta_3686 = np.array([args[40],args[41]]).reshape(-1,2)
        
        kk_f2l_mass = np.array([1.9435205040750276,2.1783628905892916,2.3736695101913585])
        
        kk_f2l_width = np.array([0.46553324520367734,0.171365802944736,0.35232188312114704])
        
        kk_f2l_const_3686 = np.array([args[42],args[43],args[44],args[45],args[46],args[47],args[48],args[49],args[50],args[51],args[52],args[53],args[54],args[55],args[56]]).reshape(-1,5)
        
        kk_f2l_theta_3686 = np.array([args[57],args[58],args[59],args[60],args[61],args[62],args[63],args[64],args[65],args[66],args[67],args[68],args[69],args[70],args[71]]).reshape(-1,5)
        
        kk_u_rho_mass = np.array([2.397312161851163])
        
        kk_u_rho_width = np.array([0.08930983775105905])
        
        kk_u_rho_const_3686 = np.array([args[72]]).reshape(-1,1)
        
        kk_u_rho_theta_3686 = np.array([args[73]]).reshape(-1,1)
        
        kk_u_b_mass = np.array([1.8383688963975429,2.21695086990315])
        
        kk_u_b_width = np.array([0.2886776944601449,0.30820571811928366])
        
        kk_u_b_const_3686 = np.array([args[74],args[75],args[76],args[77],args[78],args[79],args[80],args[81]]).reshape(-1,4)
        
        kk_u_b_theta_3686 = np.array([args[82],args[83],args[84],args[85],args[86],args[87],args[88],args[89]]).reshape(-1,4)
        
        kk_3773_amp_inta = np.array([args[90]])
        
        kk_3773_tha_inta = np.array([args[91]])
        
        kk_f980_const_3773 = np.array([0.1,0.0812758802808357]).reshape(-1,2)
        
        kk_f980_theta_3773 = np.array([0.1,0.04219055453438748]).reshape(-1,2)
        
        kk_f1270_const_3773 = np.array([-0.001908795026008432,-0.002198862452744889,9.263590492855337e-05,0.0038573660399420315,0.004216085267475821]).reshape(-1,5)
        
        kk_f1270_theta_3773 = np.array([0.01328748133437328,0.019267847356889318,0.00823480715160893,-0.0017014797381603949,0.013688984939122168]).reshape(-1,5)
        
        kk_f2_const_3773 = np.array([0.00836160494789744,0.0022731471899687674,-0.0046768086923496405,-0.030976560693756445,0.011016487927278483]).reshape(-1,5)
        
        kk_f2_theta_3773 = np.array([0.0007651896888136746,-0.0021162876037446768,-0.005601756967141078,-0.0013397178875563952,-0.00916615679988274]).reshape(-1,5)
        
        kk_f0_const_3773 = np.array([0.006776209842051121,-0.005211775708005117,0.013982507928474695,0.007221463747811338,0.016393747705371626,0.0026018914837070166]).reshape(-1,2)
        
        kk_f0_theta_3773 = np.array([-0.019419455060917547,-0.02162941877883244,-0.016406491062264772,-0.011508544965435973,-0.02996999596857378,-0.016574134291907173]).reshape(-1,2)
        
        kk_f0_long_const_3773 = np.array([0.17112767897801573,0.05688549960596512]).reshape(-1,2)
        
        kk_f0_long_theta_3773 = np.array([-0.11230387384356917,-0.06356583331362733]).reshape(-1,2)
        
        kk_f0l_const_3773 = np.array([-0.01907412396118519,-0.014324317992303416]).reshape(-1,2)
        
        kk_f0l_theta_3773 = np.array([-0.026948437967964086,-0.009458867738581251]).reshape(-1,2)
        
        kk_f2l_const_3773 = np.array([-0.009557649920876458,-0.0031727092103564524,0.014100451734034392,0.010479172150098524,0.0003692045644304492,0.0018597973868881273,0.00457547644999926,0.003597272060697062,0.0062908372146082804,0.0035507238700523296,-0.017485005393528725,0.015226995460400562,0.0013420411659886469,0.03677091206682049,-0.0004435335025267973]).reshape(-1,5)
        
        kk_f2l_theta_3773 = np.array([-0.017692773889318524,-0.014879688649882716,0.00157904926827019,0.00635591994629108,0.006381438521152356,-0.01207283493112012,-0.0029108265764609606,0.0016613195856758618,0.011585574654273742,0.005166091307635481,-0.0645311336569186,0.003900999450326149,-0.006707506973748528,0.044428897338296275,0.005620130634310312]).reshape(-1,5)
        
        kk_u_rho_const_3773 = np.array([-0.0017169985093901598]).reshape(-1,1)
        
        kk_u_rho_theta_3773 = np.array([-0.003072954777635661]).reshape(-1,1)
        
        kk_u_b_const_3773 = np.array([0.003136380638161863,0.0020679454615716384,-0.0024864348773066037,0.005679669182781839,0.007687456697620569,-0.0014652125102298027,0.004360881343635086,0.004918464768283009]).reshape(-1,4)
        
        kk_u_b_theta_3773 = np.array([-0.00832920259713428,-0.001168027729720911,-0.011101976414934987,-0.003775730069307858,0.009518783705592684,0.011551460555613131,-0.0002691609960762099,0.0002696843264585296]).reshape(-1,4)
        
        
        mc_phif0_kk_3686_BW_flatte980 = self.calculate_BW_flatte980(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const_3686,kk_f980_theta_3686,data["mc_phi_kk"],data["mc_f_kk"],data["mc_phif0_kk_3686"])
        
        mc_phif2_kk_3686_BW_flatte1270 = self.calculate_BW_flatte1270(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const_3686,kk_f1270_theta_3686,data["mc_phi_kk"],data["mc_f_kk"],data["mc_phif2_kk_3686"])
        
        mc_phif2_kk_3686_BW_BW = self.calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const_3686,kk_f2_theta_3686,data["mc_phi_kk"],data["mc_f_kk"],data["mc_phif2_kk_3686"])
        
        mc_phif0_kk_3686_BW_BW = self.calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0_mass,kk_f0_width,kk_f0_const_3686,kk_f0_theta_3686,data["mc_phi_kk"],data["mc_f_kk"],data["mc_phif0_kk_3686"])
        
        mc_phif0_kk_3686_BW_BW_long = self.calculate_BW_BW_long(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0_long_mass,kk_f0_long_width,kk_f0_long_const_3686,kk_f0_long_theta_3686,data["mc_phi_kk"],data["mc_f_kk"],data["mc_phif0_kk_3686"])
        
        mc_phif0_kk_3686_BW_BW_f0_kk = self.calculate_BW_BW_f0_kk(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0l_mass,kk_f0l_width,kk_f0l_const_3686,kk_f0l_theta_3686,data["mc_phi_kk"],data["mc_f_kk"],data["mc_phif0_kk_3686"])
        
        mc_phif2_kk_3686_BW_BW_f2_kk = self.calculate_BW_BW_f2_kk(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f2l_mass,kk_f2l_width,kk_f2l_const_3686,kk_f2l_theta_3686,data["mc_phi_kk"],data["mc_f_kk"],data["mc_phif2_kk_3686"])
        
        mc_u_rho_r_kk_3686_BW_BW = self.calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3686,kk_u_rho_theta_3686,data["mc_phi_kk"],data["mc_b124_kk"],data["mc_u_rho_r_kk_3686"])
        
        mc_u_rho_l_kk_3686_BW_BW = self.calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3686,kk_u_rho_theta_3686,data["mc_phi_kk"],data["mc_b123_kk"],data["mc_u_rho_l_kk_3686"])
        
        mc_u_b_r_kk_3686_BW_BWb = self.calculate_BW_BWb(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3686,kk_u_b_theta_3686,data["mc_phi_kk"],data["mc_b124_kk"],data["mc_u_b_r_kk_3686"])
        
        mc_u_b_l_kk_3686_BW_BWb = self.calculate_BW_BWb(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3686,kk_u_b_theta_3686,data["mc_phi_kk"],data["mc_b123_kk"],data["mc_u_b_l_kk_3686"])
        
        mc_phif0_kk_3773_BW_flatte980 = self.calculate_BW_flatte980(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const_3773,kk_f980_theta_3773,data["mc_phi_kk"],data["mc_f_kk"],data["mc_phif0_kk_3773"])
        
        mc_phif2_kk_3773_BW_flatte1270 = self.calculate_BW_flatte1270(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const_3773,kk_f1270_theta_3773,data["mc_phi_kk"],data["mc_f_kk"],data["mc_phif2_kk_3773"])
        
        mc_phif2_kk_3773_BW_BW = self.calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const_3773,kk_f2_theta_3773,data["mc_phi_kk"],data["mc_f_kk"],data["mc_phif2_kk_3773"])
        
        mc_phif0_kk_3773_BW_BW = self.calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f0_mass,kk_f0_width,kk_f0_const_3773,kk_f0_theta_3773,data["mc_phi_kk"],data["mc_f_kk"],data["mc_phif0_kk_3773"])
        
        mc_phif0_kk_3773_BW_BW_long = self.calculate_BW_BW_long(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f0_long_mass,kk_f0_long_width,kk_f0_long_const_3773,kk_f0_long_theta_3773,data["mc_phi_kk"],data["mc_f_kk"],data["mc_phif0_kk_3773"])
        
        mc_phif0_kk_3773_BW_BW_f0_kk = self.calculate_BW_BW_f0_kk(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f0l_mass,kk_f0l_width,kk_f0l_const_3773,kk_f0l_theta_3773,data["mc_phi_kk"],data["mc_f_kk"],data["mc_phif0_kk_3773"])
        
        mc_phif2_kk_3773_BW_BW_f2_kk = self.calculate_BW_BW_f2_kk(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f2l_mass,kk_f2l_width,kk_f2l_const_3773,kk_f2l_theta_3773,data["mc_phi_kk"],data["mc_f_kk"],data["mc_phif2_kk_3773"])
        
        mc_u_rho_r_kk_3773_BW_BW = self.calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3773,kk_u_rho_theta_3773,data["mc_phi_kk"],data["mc_b124_kk"],data["mc_u_rho_r_kk_3773"])
        
        mc_u_rho_l_kk_3773_BW_BW = self.calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3773,kk_u_rho_theta_3773,data["mc_phi_kk"],data["mc_b123_kk"],data["mc_u_rho_l_kk_3773"])
        
        mc_u_b_r_kk_3773_BW_BWb = self.calculate_BW_BWb(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3773,kk_u_b_theta_3773,data["mc_phi_kk"],data["mc_b124_kk"],data["mc_u_b_r_kk_3773"])
        
        mc_u_b_l_kk_3773_BW_BWb = self.calculate_BW_BWb(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3773,kk_u_b_theta_3773,data["mc_phi_kk"],data["mc_b123_kk"],data["mc_u_b_l_kk_3773"])
        
        return np.sum(dplex.dabs(mc_phif0_kk_3686_BW_flatte980 + mc_phif2_kk_3686_BW_flatte1270 + mc_phif2_kk_3686_BW_BW + mc_phif0_kk_3686_BW_BW + mc_phif0_kk_3686_BW_BW_long + mc_phif0_kk_3686_BW_BW_f0_kk + mc_phif2_kk_3686_BW_BW_f2_kk + mc_u_rho_r_kk_3686_BW_BW + mc_u_rho_l_kk_3686_BW_BW + mc_u_b_r_kk_3686_BW_BWb + mc_u_b_l_kk_3686_BW_BWb + mc_phif0_kk_3773_BW_flatte980 + mc_phif2_kk_3773_BW_flatte1270 + mc_phif2_kk_3773_BW_BW + mc_phif0_kk_3773_BW_BW + mc_phif0_kk_3773_BW_BW_long + mc_phif0_kk_3773_BW_BW_f0_kk + mc_phif2_kk_3773_BW_BW_f2_kk + mc_u_rho_r_kk_3773_BW_BW + mc_u_rho_l_kk_3773_BW_BW + mc_u_b_r_kk_3773_BW_BWb + mc_u_b_l_kk_3773_BW_BWb))

    def calculate_BW_flatte980(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const_3686,kk_f980_theta_3686,phi_kk,f_kk,phif0_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f980_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_flatte980(phi_mass,phi_width,phi_kk,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif0_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f980_const_3686, kk_f980_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f980_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        # print("inta_ph",inta_ph.shape)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)#给每个相位因子进行相位修正
        phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk_3686, const_ph)#给共振态的每个振幅乘上相位因子并把该共振态的全部振幅求和
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        
        return phif
    def BW_flatte980(self, phi_mass,phi_width,phi_kk,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,f_kk):
        a = self.BW(phi_mass,phi_width,phi_kk)
        # a = self.BW_phi(phi_mass,phi_width,phi_kk)
        b = np.moveaxis(vmap(partial(self.flatte980,Sbc=f_kk))(kk_f980_mass,kk_f980_g_kk,kk_f980_rg),1,0)

        return dplex.deinsum("j, ij->ij",a,b)

    
    def calculate_BW_flatte1270(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const_3686,kk_f1270_theta_3686,phi_kk,f_kk,phif2_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f1270_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_flatte1270(phi_mass,phi_width,phi_kk,kk_f1270_mass,kk_f1270_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif2_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f1270_const_3686, kk_f1270_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f1270_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        # print("inta_ph",inta_ph.shape)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)#给每个相位因子进行相位修正
        phif = dplex.deinsum_ord("ijk,li->ljk", phif2_kk_3686, const_ph)#给共振态的每个振幅乘上相位因子并把该共振态的全部振幅求和
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        
        return phif
    def BW_flatte1270(self, phi_mass,phi_width,phi_kk,kk_f1270_mass,kk_f1270_width,f_kk):
        a = self.BW(phi_mass,phi_width,phi_kk)
        # a = self.BW_phi(phi_mass,phi_width,phi_kk)
        b = np.moveaxis(vmap(partial(self.flatte1270,Sbc=f_kk))(kk_f1270_mass,kk_f1270_width),1,0)

        return dplex.deinsum("j, ij->ij",a,b)

    
    def calculate_BW_BW(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const_3686,kk_f2_theta_3686,phi_kk,f_kk,phif2_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f2_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_BW(phi_mass,phi_width,phi_kk,kk_f2_mass,kk_f2_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif2_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f2_const_3686, kk_f2_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f2_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        # print("inta_ph",inta_ph.shape)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)#给每个相位因子进行相位修正
        phif = dplex.deinsum_ord("ijk,li->ljk", phif2_kk_3686, const_ph)#给共振态的每个振幅乘上相位因子并把该共振态的全部振幅求和
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        
        return phif
    def BW_BW(self, phi_mass,phi_width,phi_kk,kk_f2_mass,kk_f2_width,f_kk):
        a = self.BW(phi_mass,phi_width,phi_kk)
        # a = self.BW_phi(phi_mass,phi_width,phi_kk)
        b = np.moveaxis(vmap(partial(self.BW,Sbc=f_kk))(kk_f2_mass,kk_f2_width),1,0)

        return dplex.deinsum("j, ij->ij",a,b)

    
    def calculate_BW_BW_long(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0_long_mass,kk_f0_long_width,kk_f0_long_const_3686,kk_f0_long_theta_3686,phi_kk,f_kk,phif0_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f0_long_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_BW_long(phi_mass,phi_width,phi_kk,kk_f0_long_mass,kk_f0_long_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif0_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f0_long_const_3686, kk_f0_long_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f0_long_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        # print("inta_ph",inta_ph.shape)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)#给每个相位因子进行相位修正
        phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk_3686, const_ph)#给共振态的每个振幅乘上相位因子并把该共振态的全部振幅求和
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        
        return phif
    def BW_BW_long(self, phi_mass,phi_width,phi_kk,kk_f0_long_mass,kk_f0_long_width,f_kk):
        a = self.BW(phi_mass,phi_width,phi_kk)
        # a = self.BW_phi(phi_mass,phi_width,phi_kk)
        b = np.moveaxis(vmap(partial(self.BW_long,Sbc=f_kk))(kk_f0_long_mass,kk_f0_long_width),1,0)

        return dplex.deinsum("j, ij->ij",a,b)

    
    def calculate_BW_BW_f0_kk(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0l_mass,kk_f0l_width,kk_f0l_const_3686,kk_f0l_theta_3686,phi_kk,f_kk,phif0_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f0l_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_BW_f0_kk(phi_mass,phi_width,phi_kk,kk_f0l_mass,kk_f0l_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif0_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f0l_const_3686, kk_f0l_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f0l_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        # print("inta_ph",inta_ph.shape)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)#给每个相位因子进行相位修正
        phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk_3686, const_ph)#给共振态的每个振幅乘上相位因子并把该共振态的全部振幅求和
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        
        return phif
    def BW_BW_f0_kk(self, phi_mass,phi_width,phi_kk,kk_f0l_mass,kk_f0l_width,f_kk):
        a = self.BW(phi_mass,phi_width,phi_kk)
        # a = self.BW_phi(phi_mass,phi_width,phi_kk)
        b = np.moveaxis(vmap(partial(self.BW_f0_kk,Sbc=f_kk))(kk_f0l_mass,kk_f0l_width),1,0)

        return dplex.deinsum("j, ij->ij",a,b)

    
    def calculate_BW_BW_f2_kk(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f2l_mass,kk_f2l_width,kk_f2l_const_3686,kk_f2l_theta_3686,phi_kk,f_kk,phif2_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f2l_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_BW_f2_kk(phi_mass,phi_width,phi_kk,kk_f2l_mass,kk_f2l_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif2_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f2l_const_3686, kk_f2l_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f2l_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        # print("inta_ph",inta_ph.shape)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)#给每个相位因子进行相位修正
        phif = dplex.deinsum_ord("ijk,li->ljk", phif2_kk_3686, const_ph)#给共振态的每个振幅乘上相位因子并把该共振态的全部振幅求和
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        
        return phif
    def BW_BW_f2_kk(self, phi_mass,phi_width,phi_kk,kk_f2l_mass,kk_f2l_width,f_kk):
        a = self.BW(phi_mass,phi_width,phi_kk)
        # a = self.BW_phi(phi_mass,phi_width,phi_kk)
        b = np.moveaxis(vmap(partial(self.BW_f2_kk,Sbc=f_kk))(kk_f2l_mass,kk_f2l_width),1,0)

        return dplex.deinsum("j, ij->ij",a,b)

    
    def calculate_BW_BWb(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3686,kk_u_b_theta_3686,phi_kk,b124_kk,u_b_r_kk_3686):

        
        bw1 = self.BW_BW_f0_k0(phi_mass,phi_width,phi_kk,kk_u_b_mass,kk_u_b_width,b124_kk)
        bw2 = self.BW_BW_f2_k0(phi_mass,phi_width,phi_kk,kk_u_b_mass,kk_u_b_width,b124_kk)

        amp_caled_1 = u_b_r_kk_3686[[0, 2], :, :]  # SS and DS
        amp_caled_2 = u_b_r_kk_3686[[1, 3], :, :]  # DS and DD

        const_ph = dplex.dconstruct(kk_u_b_const_3686, kk_u_b_theta_3686)
        
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)
        const_ph1 = const_ph[:, :, [0, 2]]
        const_ph2 = const_ph[:, :, [1, 3]]

        amp_caled_wphase1 = dplex.deinsum_ord("ijk,li->ljk", amp_caled_1, const_ph1)
        amp_caled_wphase2 = dplex.deinsum_ord("ijk,li->ljk", amp_caled_2, const_ph2)

        phif1 = dplex.deinsum("ljk,lj->jk", amp_caled_wphase1, bw1)
        phif2 = dplex.deinsum("ljk,lj->jk", amp_caled_wphase2, bw2)

        phif = phif1 + phif2
        
        return phif 
    def BW_BW_f0_k0(self, phi_mass,phi_width,phi_kk,kk_u_b_mass,kk_u_b_width,b124_kk):
        a = self.BW(phi_mass,phi_width,phi_kk)
        b = np.moveaxis(vmap(partial(self.BW_f0_k0,Sbc=b124_kk))(kk_u_b_mass,kk_u_b_width),1,0)
        return dplex.deinsum("j, ij->ij",a,b)
    def BW_BW_f2_k0(self, phi_mass,phi_width,phi_kk,kk_u_b_mass,kk_u_b_width,b124_kk):
        a = self.BW(phi_mass,phi_width,phi_kk)
        b = np.moveaxis(vmap(partial(self.BW_f2_k0,Sbc=b124_kk))(kk_u_b_mass,kk_u_b_width),1,0)
        return dplex.deinsum("j, ij->ij",a,b)
    
    def lasso_calculate_BW_flatte980(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const_3686,kk_f980_theta_3686,phi_kk,f_kk,phif0_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f980_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_flatte980(phi_mass,phi_width,phi_kk,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif0_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f980_const_3686, kk_f980_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f980_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)
        phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk_3686, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        # lasso_phif = np.einsum("ljk->l",dplex.dabs(phif))
        # return lasso_phif
        
        return phif
    
    def lasso_calculate_BW_flatte1270(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const_3686,kk_f1270_theta_3686,phi_kk,f_kk,phif2_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f1270_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_flatte1270(phi_mass,phi_width,phi_kk,kk_f1270_mass,kk_f1270_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif2_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f1270_const_3686, kk_f1270_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f1270_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)
        phif = dplex.deinsum_ord("ijk,li->ljk", phif2_kk_3686, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        # lasso_phif = np.einsum("ljk->l",dplex.dabs(phif))
        # return lasso_phif
        
        return phif
    
    def lasso_calculate_BW_BW(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const_3686,kk_f2_theta_3686,phi_kk,f_kk,phif2_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f2_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_BW(phi_mass,phi_width,phi_kk,kk_f2_mass,kk_f2_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif2_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f2_const_3686, kk_f2_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f2_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)
        phif = dplex.deinsum_ord("ijk,li->ljk", phif2_kk_3686, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        # lasso_phif = np.einsum("ljk->l",dplex.dabs(phif))
        # return lasso_phif
        
        return phif
    
    def lasso_calculate_BW_BW_long(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0_long_mass,kk_f0_long_width,kk_f0_long_const_3686,kk_f0_long_theta_3686,phi_kk,f_kk,phif0_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f0_long_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_BW_long(phi_mass,phi_width,phi_kk,kk_f0_long_mass,kk_f0_long_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif0_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f0_long_const_3686, kk_f0_long_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f0_long_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)
        phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk_3686, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        # lasso_phif = np.einsum("ljk->l",dplex.dabs(phif))
        # return lasso_phif
        
        return phif
    
    def lasso_calculate_BW_BW_f0_kk(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0l_mass,kk_f0l_width,kk_f0l_const_3686,kk_f0l_theta_3686,phi_kk,f_kk,phif0_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f0l_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_BW_f0_kk(phi_mass,phi_width,phi_kk,kk_f0l_mass,kk_f0l_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif0_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f0l_const_3686, kk_f0l_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f0l_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)
        phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk_3686, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        # lasso_phif = np.einsum("ljk->l",dplex.dabs(phif))
        # return lasso_phif
        
        return phif
    
    def lasso_calculate_BW_BW_f2_kk(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f2l_mass,kk_f2l_width,kk_f2l_const_3686,kk_f2l_theta_3686,phi_kk,f_kk,phif2_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f2l_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_BW_f2_kk(phi_mass,phi_width,phi_kk,kk_f2l_mass,kk_f2l_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif2_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f2l_const_3686, kk_f2l_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f2l_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)
        phif = dplex.deinsum_ord("ijk,li->ljk", phif2_kk_3686, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        # lasso_phif = np.einsum("ljk->l",dplex.dabs(phif))
        # return lasso_phif
        
        return phif
    
    def lasso_calculate_BW_BWb(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3686,kk_u_b_theta_3686,phi_kk,b124_kk,u_b_r_kk_3686):

        
        bw1 = self.BW_BW_f0_k0(phi_mass,phi_width,phi_kk,kk_u_b_mass,kk_u_b_width,b124_kk)
        bw2 = self.BW_BW_f2_k0(phi_mass,phi_width,phi_kk,kk_u_b_mass,kk_u_b_width,b124_kk)

        amp_caled_1 = u_b_r_kk_3686[[0, 2], :, :]  # SS and DS
        amp_caled_2 = u_b_r_kk_3686[[1, 3], :, :]  # DS and DD

        const_ph = dplex.dconstruct(kk_u_b_const_3686, kk_u_b_theta_3686)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        # const_ph = dplex.deinsum("lij,l->lij", const_ph, inta_ph)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)
        const_ph1 = const_ph[:, :, [0, 2]]
        const_ph2 = const_ph[:, :, [1, 3]]

        amp_caled_wphase1 = dplex.deinsum_ord("ijk,li->ljk", amp_caled_1, const_ph1)
        amp_caled_wphase2 = dplex.deinsum_ord("ijk,li->ljk", amp_caled_2, const_ph2)

        phif1 = dplex.deinsum("ljk,lj->ljk", amp_caled_wphase1, bw1)
        phif2 = dplex.deinsum("ljk,lj->ljk", amp_caled_wphase2, bw2)

        phif = phif1 + phif2
        
        return phif

    def phase(self, theta):
        return vmap(self._phase)(theta)

    def _phase(self, theta):
        return dplex.dconstruct(np.cos(theta), np.sin(theta))
    
    
    def BW(self, m_,w_,Sbc):
        l = (Sbc.shape)[0]
        temp = dplex.dconstruct(m_*m_ - Sbc,  -m_*w_*np.ones(l))
        return dplex.ddivide(1.0, temp)

    def BW_relativity(self, m_,w_,Sbc):
        gamma=np.sqrt(m_*m_*(m_*m_+w_*w_))
        k = np.sqrt(2*np.sqrt(2)*m_*np.abs(w_)*gamma/np.pi/np.sqrt(m_*m_+gamma))
        l = (Sbc.shape)[0]
        temp = dplex.dconstruct(m_*m_ - Sbc,  -m_*w_*np.ones(l))
        return dplex.ddivide(k, temp)

    def BW_phi(self,m_,w_,Sbc):
        l = (Sbc.shape)[0]
        m_k = 0.493677
        q_0 = 1/3
        q_m = np.sqrt(Sbc-4*m_k*m_k)/2
        q_phi = np.sqrt(m_*m_-4*m_k*m_k)/2
        b_qm = 1 / np.sqrt(1+(q_m/q_0)*(q_m/q_0))
        b_qphi = 1 / np.sqrt(1+(q_phi/q_0)*(q_phi/q_0))

        k = q_m / q_0 *b_qm
        Gamma_phi = w_ * m_ / np.sqrt(Sbc)*(q_m/q_phi)*(q_m/q_phi)*b_qm*b_qm/b_qphi/b_qphi

        temp = dplex.dconstruct(m_*m_ - Sbc, -m_*Gamma_phi*np.ones(l))
        return dplex.ddivide(k, temp)

    def flatte980(self,m_,g_pipi,rg,Sbc):
        # 对于phipipi的过程的rho_kk的计算，不能单纯取绝对值，因为有一部分pipi在kk的质量阈值之下
        g_kk = rg * g_pipi
        m_k = 0.493677
        m_pi = 0.13957061

        tmp_kk = 1 - 4*m_k*m_k / Sbc
        sign_kk = np.sign(tmp_kk)
        tmp_pipi = 1 - 4*m_pi*m_pi / Sbc
        sign_pipi = np.sign(tmp_pipi)

        # 若这一项大于零，应该留在虚部，如果小于零，应该*加*到实部

        real_kk_factor = ( tmp_kk - sign_kk * tmp_kk ) / (tmp_kk * 2)
        img_kk_factor = ( tmp_kk + sign_kk * tmp_kk ) / (tmp_kk * 2)

        rho_kk = np.sqrt(np.abs(1 - 4*m_k*m_k / Sbc))
        rho_pipi = np.sqrt(np.abs(1 - 4*m_pi*m_pi / Sbc))
        tmp_A = dplex.dconstruct(m_**2 - Sbc + g_kk*rho_kk*real_kk_factor, -1*m_*(g_pipi*rho_pipi + g_kk*rho_kk*img_kk_factor))
        return dplex.ddivide(1.0, tmp_A)

    def flatte1270(self,m_,w_,Sbc):
        rm = m_ * m_
        gr = m_ * w_
        q2r = 0.25 * rm - 0.0194792#这实际上是一个参数化宽度的BW(PBW)，其中scale factor这里取1fm
        b2r = q2r * (q2r + 0.1825) + 0.033306
        g11270 = gr * b2r / np.power(q2r,2.5)
        q2 = 0.25 * Sbc - 0.0194792
        b2 = q2 * (q2 + 0.1825) + 0.033306
        g1 = g11270 * np.power(q2,2.5) / b2
        tmp = dplex.dconstruct(Sbc - rm, g1)
        return dplex.ddivide(gr, tmp)
    
    def flatte500(self,m_,b1,b2,A,Sbc):
        m_pi = 0.139556995
        m_k = 0.493677
        m_eta = 0.547862
        S_a = 0.41*m_pi*m_pi
        m2 = m_*m_
        alpha = 1.3
        g4 = 0.011 # 参考文献里有很多取值

        g1_sqr = m_*(b1+b2*Sbc)*np.exp(-(Sbc-m_*m_)/A)
        rho_pipi = np.sqrt(np.abs(1 - 4*m_pi*m_pi / Sbc))
        rho_M = np.sqrt(np.abs(1 - 4*m_pi*m_pi / (m_*m_)))

        tmp_kk = 1 - 4*m_k*m_k / Sbc
        rho_kk = np.sqrt(np.abs(1 - 4*m_k*m_k / Sbc))
        tmp_nn = 1 - 4*m_eta*m_eta / Sbc
        rho_nn = np.sqrt(np.abs(1 - 4*m_eta*m_eta / Sbc))

        rho_kk_sign = np.sign(1 - 4*m_k*m_k / Sbc) 
        rho_nn_sign = np.sign(1 - 4*m_eta*m_eta / Sbc)

        kk_real_factor = ( tmp_kk - rho_kk_sign * tmp_kk ) / (tmp_kk * 2)
        kk_img_factor = ( tmp_kk + rho_kk_sign * tmp_kk ) / (tmp_kk * 2)
        nn_real_factor = ( tmp_nn - rho_nn_sign * tmp_nn ) / (tmp_nn * 2)
        nn_img_factor = ( tmp_nn + rho_nn_sign * tmp_nn ) / (tmp_nn * 2)
        # pipi 不变质量会出现在 kk 和 etaeta 质量阈值 之下
        
        rho_4pis = 1.0/(1+np.exp(7.082-2.845*Sbc))
        rho_4pim = 1.0/(1+np.exp(7.082-2.845*(m_*m_)))

        mgam1 = g1_sqr*(Sbc-S_a)/(m_*m_-S_a)*rho_pipi
        mgam2 = 0.6*g1_sqr*(Sbc/m2)*np.exp(-1*alpha*np.abs(Sbc-4*m_k*m_k))*rho_kk 
        mgam3 = 0.2*g1_sqr*(Sbc/m2)*np.exp(-1*alpha*np.abs(Sbc-4*m_eta*m_eta))*rho_nn 
        mgam4 = m_*g4*rho_4pis/rho_4pim

        j1_s = (2 + rho_pipi * np.log((1-rho_pipi)/(1+rho_pipi))) / np.pi #log本身就是以e为底
        j1_m = (2 + rho_M * np.log((1-rho_M )/(1+rho_M ))) / np.pi
        zs = j1_s - j1_m 

        # mgam = mgam1 + mgam2 + mgam3 + mgam4 
        T_pipi = dplex.dconstruct(m_*m_ - Sbc - g1_sqr*(Sbc-S_a)/(m_*m_-S_a)*zs + kk_real_factor*mgam2 + nn_real_factor*mgam3 ,mgam1+mgam4 + kk_img_factor*mgam2 +nn_img_factor *mgam3 )

        return dplex.ddivide(1.0,T_pipi)

    def BW_long(self, m_,w_,Sbc):
        # LHCb note BW=1 for non resonnace
        l = (Sbc.shape)[0]
        temp = dplex.dconstruct(np.ones(l),  np.zeros(l))
        return dplex.ddivide(1.0, temp)

    def BW_f0_kk(self,m_,w_,Sbc):# paramtered width l=0
        S_b = 0.24371698
        S_c = 0.24371698
        m02 = m_*m_
        q2 = np.abs(0.25*(Sbc + S_b - S_c)**2/Sbc - S_b)
        q = np.sqrt(q2)
        q02 = np.abs(0.25*(m02 + S_b - S_c)**2/m02 - S_b)
        q0 = np.sqrt(q02)
        t = q/q0

        gam = w_*t*m_/Sbc # barrier_factor == 1
        tmp = dplex.dconstruct(m_*m_ - Sbc,-m_*gam)
        return dplex.ddivide(1.0, tmp)

    def BW_f2_kk(self,m_,w_,Sbc):# paramtered width l=2
        S_b = 0.24371698
        S_c = 0.24371698
        m02 = m_*m_
        q2 = np.abs(0.25*(Sbc + S_b - S_c)**2/Sbc - S_b)
        q = np.sqrt(q2)
        q02 = np.abs(0.25*(m02 + S_b - S_c)**2/m02 - S_b)
        q0 = np.sqrt(q02)
        t = (q/q0)**5#not sure who is more accuracy
        bf = (9+3*q02+q02*q02)/(9+3*q2+q2*q2)

        gam = w_*t*m_/Sbc*bf
        tmp = dplex.dconstruct(m_*m_ - Sbc,-m_*gam)
        return dplex.ddivide(1.0, tmp)

    def BW_f0_pipi(self,m_,w_,Sbc):# paramtered width l=0
        S_b = 0.0194797849
        S_c = 0.0194797849
        m02 = m_*m_
        q2 = np.abs(0.25*(Sbc + S_b - S_c)**2/Sbc - S_b)
        q = np.sqrt(q2)
        q02 = np.abs(0.25*(m02 + S_b - S_c)**2/m02 - S_b)
        q0 = np.sqrt(q02)
        t = q/q0

        gam = w_*t*m_/Sbc # barrier_factor == 1
        tmp = dplex.dconstruct(m_*m_ - Sbc,-m_*gam)
        return dplex.ddivide(1.0, tmp)

    def BW_f2_pipi(self,m_,w_,Sbc):# paramtered width l=2
        S_b = 0.0194797849
        S_c = 0.0194797849
        m02 = m_*m_
        q2 = np.abs(0.25*(Sbc + S_b - S_c)**2/Sbc - S_b)
        q = np.sqrt(q2)
        q02 = np.abs(0.25*(m02 + S_b - S_c)**2/m02 - S_b)
        q0 = np.sqrt(q02)
        t = (q/q0)**5#not sure who is more accuracy
        bf = (9+3*q02+q02*q02)/(9+3*q2+q2*q2)

        gam = w_*t*m_/Sbc*bf
        tmp = dplex.dconstruct(m_*m_ - Sbc,-m_*gam)
        return dplex.ddivide(1.0, tmp)

    def BW_rho770(self,m_,w_,Sbc):# GS_lineshape
        S_b = 0.0194797849
        S_c = 0.0194797849
        GS1 = 0.636619783 # 2/ma.pi
        GS2 = 0.01860182466 # 3*mpion2/ma.pi
        GS3 = 0.1591549458; # 1/(2*ma.pi)
        GS4 = 0.00620060822; # mass_Pion2/ma.pi
        m = np.sqrt(Sbc)
        m02 = m_*m_ # m0 = m_ 是我们要拟合的参数
        q2 = 0.25*(Sbc + S_b - S_c)**2/Sbc - S_b
        q = np.sqrt(q2)
        q02 = 0.25*(m02 + S_b - S_c)**2/m02 - S_b
        q0 = np.sqrt(q02)
        q03 = q0*q02
        t = q/q0
        #按照zbs的文章和程序其他的部分，R被设置为1
        bf = (1+q02)/(1+q2)# rho2pipi l=1
        #log(mass_2Pion) = 1.2760418309
        h  = GS1*q/m*(np.log(m+2*q)+1.2760418309)
        h0 = GS1*q0/m_*(np.log(m_+2*q0)+1.2760418309)
        dh = h0*(0.125/q02-0.5/m_)+GS3/m02
        d  = GS2/q02*(np.log(m_+2*q0)+1.2760418309)+GS3*m_/q0-GS4*m_/q03

        gam = w_*t*t*t*m_/m*bf 
        f = w_*m02/q03*(q2*(h-h0)+(m02-Sbc)*q02*dh)

        a = 1.0 + d*w_/m_
        b = dplex.dconstruct(m_*m_ - Sbc + f, -m_*gam)
        return dplex.ddivide(a, b)
    def BW_f0_b1(self,m_,w_,Sbc):# paramtered width l=0 b1 state SS and DS
        S_b = 1.0392986916 #b1=m_phi+m_pi
        S_c = 0.0194797849
        m02 = m_*m_
        q2 = np.abs(0.25*(Sbc + S_b - S_c)**2/Sbc - S_b)
        q = np.sqrt(q2)
        q02 = np.abs(0.25*(m02 + S_b - S_c)**2/m02 - S_b)
        q0 = np.sqrt(q02)
        t = q/q0
        gam = w_*t*m_/Sbc # barrier_factor == 1
        tmp = dplex.dconstruct(m_*m_ - Sbc,-m_*gam)
        return dplex.ddivide(1.0, tmp)
    def BW_f2_b1(self,m_,w_,Sbc):# paramtered width l=0 b1 state SS and DS
        S_b = 1.0392986916 #b1=m_phi+m_pi
        S_c = 0.0194797849
        m02 = m_*m_
        q2 = np.abs(0.25*(Sbc + S_b - S_c)**2/Sbc - S_b)
        q = np.sqrt(q2)
        q02 = np.abs(0.25*(m02 + S_b - S_c)**2/m02 - S_b)
        q0 = np.sqrt(q02)
        t = (q/q0)**5#not sure who is more accuracy
        bf = (9+3*q02+q02*q02)/(9+3*q2+q2*q2)
        gam = w_*t*m_/Sbc*bf
        tmp = dplex.dconstruct(m_*m_ - Sbc,-m_*gam)
        return dplex.ddivide(1.0, tmp)
    def BW_f0_k0(self,m_,w_,Sbc):# paramtered width l=0 b1 state SS and DS
        S_b = 1.2627 #kst=m_phi+m_k
        S_c = 0.24372
        m02 = m_*m_
        q2 = np.abs(0.25*(Sbc + S_b - S_c)**2/Sbc - S_b)
        q = np.sqrt(q2)
        q02 = np.abs(0.25*(m02 + S_b - S_c)**2/m02 - S_b)
        q0 = np.sqrt(q02)
        t = q/q0
        gam = w_*t*m_/Sbc # barrier_factor == 1
        tmp = dplex.dconstruct(m_*m_ - Sbc,-m_*gam)
        return dplex.ddivide(1.0, tmp)
    def BW_f2_k0(self,m_,w_,Sbc):# paramtered width l=0 b1 state SS and DS
        S_b = 1.2627 #kst=m_phi+m_k
        S_c = 0.24372
        m02 = m_*m_
        q2 = np.abs(0.25*(Sbc + S_b - S_c)**2/Sbc - S_b)
        q = np.sqrt(q2)
        q02 = np.abs(0.25*(m02 + S_b - S_c)**2/m02 - S_b)
        q0 = np.sqrt(q02)
        t = (q/q0)**5#not sure who is more accuracy
        bf = (9+3*q02+q02*q02)/(9+3*q2+q2*q2)
        gam = w_*t*m_/Sbc*bf
        tmp = dplex.dconstruct(m_*m_ - Sbc,-m_*gam)
        return dplex.ddivide(1.0, tmp)

    def hvp_data_fwdrev_kk(self, args_float, any_vector, data):
        return jvp(grad(self.data_likelihood_kk), [args_float, data], [any_vector, _zero_like(data)])[1]

    def hvp_mc_fwdrev_kk(self, args_float, any_vector, data):
        return jvp(grad(self.mc_likelihood_kk), [args_float, data], [any_vector, _zero_like(data)])[1]



def _zero_like(data):
    """与 data 同结构、同分片的零树（给 jvp 的切向用）。"""
    return jax.tree_util.tree_map(np.zeros_like, data)


# ---------------------------------------------------------------------------
# 组合似然（旧版 Control.thread_likelihood / thread_grad_likelihood / thread_hvp）
# ---------------------------------------------------------------------------
# 旧版：
#     result_kk = sum(data_lh) + 21181.60837312616 * log(sum(mc_lh) / 925000.0)
#     grad      = sum(data_grad) + 21181.60837312616 * sum(mc_grad) / sum(mc_lh)
#     hvp       = sum(data_hvp)  + 21181.60837312616 * (sum(mc_hvp)/sum(mc_lh)
#                   - <mc_grad, mc_hvp_g, v> / sum(mc_lh)^2)
#
# 新版直接对同一个标量做 autodiff —— 数学上恒等（链式法则给出的正是上面三式），
# 且与 run/fit_script_dist.py 的做法一致。旧版手工拆分只是为了跨进程拼装中间量，
# 原生 JAX 下由 XLA 的 all-reduce 完成，不再需要。
def combined_likelihood(pwaf, args, data):
    data_lh = pwaf.data_likelihood_kk(args, data)
    mc_lh = pwaf.mc_likelihood_kk(args, data)
    return data_lh + MC_NORM_CONST * np.log(mc_lh / MC_NORM_DENOM)


def make_objective(pwaf):
    """返回 (nll, grad, hvp)，三者都是 jit 过的 (args, data) -> ...。"""

    def nll(args, data):
        return combined_likelihood(pwaf, args, data)

    grad_nll = grad(nll, argnums=0)

    def hvp(args, vector, data):
        return jvp(
            lambda a: grad_nll(a, data),
            (args,),
            (vector,),
        )[1]

    return jit(nll), jit(grad_nll), jit(hvp)


# ---------------------------------------------------------------------------
# 原生 JAX 分布式驱动（chief-only SciPy）
# ---------------------------------------------------------------------------
OP_STOP = 0
OP_OBJECTIVE = 1
OP_GRADIENT = 2
OP_HESSIAN_PRODUCT = 3


class DistributedFit(object):
    """一个拟合用满所有卡。

    与旧版 ``Control`` 的对应关系：

    ==========================  ==============================================
    旧版 Control                新版 DistributedFit
    ==========================  ==============================================
    ``run_multiprocess()``      进程由 ``jax.distributed.initialize`` 拉起
    ``pwaf_list[i][j]``         单个 ``PWAFunc``（无设备副本）
    多进程各自 minimize         **只在 chief** 跑 minimize，worker 服务请求
    ``thread_likelihood``       ``objective``（全局分片上的标量 NLL）
    ``thread_grad_likelihood``  ``gradient``（autodiff，恒等于旧式）
    ``thread_hvp``              ``hessian_vector``（autodiff，恒等于旧式）
    ``save_in_npz`` / ``save_in_json``  ``save_result``
    ==========================  ==============================================
    """

    def __init__(self, config=None):
        self.config = config or FitConfig()
        self.num_processes = 1
        self.process_id = 0
        self.mesh = None
        self.pwaf = PWAFunc()
        self.jit_nll = None
        self.jit_grad = None
        self.jit_hvp = None
        self.iteration = 0
        self.start_time = 0.0

    # -- 初始化 ------------------------------------------------------------
    def setup(self):
        self.num_processes, self.process_id = init_distributed()
        self.is_chief = self.process_id == 0
        if self.is_chief:
            logger.info(
                "native jax fit: processes=%d devices=%d",
                self.num_processes, len(jax.devices()),
            )
        self.mesh = build_mesh()
        return self

    def _replicated(self, value):
        return device_put(np.asarray(value), NamedSharding(self.mesh, P()))

    # -- 优化器请求同步 ----------------------------------------------------
    def _broadcast_request(self, op, x, vector):
        packet = {
            "op": onp.asarray(op, dtype=onp.int32),
            "x": onp.asarray(x, dtype=onp.float64),
            "vector": onp.asarray(vector, dtype=onp.float64),
        }
        return multihost_utils.broadcast_one_to_all(packet)

    def _serve_optimizer_requests(self, jax_data, zero_vector):
        """worker：循环接收 chief 的请求并执行（collectives 与 chief 一一对应）。"""
        while True:
            packet = self._broadcast_request(OP_STOP, zero_vector, zero_vector)
            op = int(onp.asarray(packet["op"]))
            if op == OP_STOP:
                return
            x = onp.asarray(packet["x"])
            vector = onp.asarray(packet["vector"])
            if op == OP_OBJECTIVE:
                self.jit_nll(self._replicated(x), jax_data)
            elif op == OP_GRADIENT:
                self.jit_grad(self._replicated(x), jax_data)
            elif op == OP_HESSIAN_PRODUCT:
                self.jit_hvp(self._replicated(x), self._replicated(vector), jax_data)
            else:
                raise ValueError("Unknown optimizer request op: %s" % (op,))

    def _dispatch(self, op, x, vector, zero_vector):
        if self.num_processes > 1:
            self._broadcast_request(
                op, x, zero_vector if vector is None else vector
            )

    # -- 主流程 ------------------------------------------------------------
    def run(self):
        config = self.config
        data = normalize_data(load_data(config))
        n_events = int(onp.asarray(data["data_phi_kk"]).shape[0])
        args_list = ARGS_LIST.copy()
        if config.x0_npy:
            args_list = onp.asarray(onp.load(config.x0_npy), dtype=onp.float64)
            if self.is_chief:
                logger.info(
                    "initial args overridden from %s (n=%d)",
                    config.x0_npy, args_list.size,
                )
        args_float = args_list[FLOAT_LIST]

        with _mesh_context(self.mesh):
            jax_data = shard_data_distributed(data, self.mesh)
            self.jit_nll, self.jit_grad, self.jit_hvp = make_objective(self.pwaf)

            zero_vector = onp.zeros_like(args_float)
            initial = self._replicated(args_float)

            if self.is_chief:
                logger.info("compiling nll / grad / hvp ...")
            smoke = float(self.jit_nll(initial, jax_data))
            smoke_hvp = onp.asarray(
                self.jit_hvp(initial, self._replicated(onp.ones_like(args_float)), jax_data)
            )
            if self.is_chief:
                logger.info(
                    "smoke: nll=%.12g hvp_shape=%s n_events=%d",
                    smoke, smoke_hvp.shape, n_events,
                )

            if self.num_processes > 1:
                multihost_utils.sync_global_devices("fit-smoke-complete")
                if not self.is_chief:
                    self._serve_optimizer_requests(jax_data, zero_vector)
                    return 0

            self.start_time = time.time()

            def raw_objective(x):
                return float(self.jit_nll(self._replicated(x), jax_data))

            def raw_gradient(x):
                return onp.asarray(
                    self.jit_grad(self._replicated(x), jax_data), dtype=onp.float64
                )

            def raw_hvp(x, vector):
                return onp.asarray(
                    self.jit_hvp(self._replicated(x), self._replicated(vector), jax_data),
                    dtype=onp.float64,
                )

            def objective(x):
                self._dispatch(OP_OBJECTIVE, x, None, zero_vector)
                return raw_objective(x)

            def gradient(x):
                self._dispatch(OP_GRADIENT, x, None, zero_vector)
                return raw_gradient(x)

            def hessian_vector(x, vector):
                self._dispatch(OP_HESSIAN_PRODUCT, x, vector, zero_vector)
                return raw_hvp(x, vector)

            def callback(xk):
                self.iteration += 1
                logger.info(
                    "iteration %d: nll=%.12g elapsed=%.1fs",
                    self.iteration, objective(xk), time.time() - self.start_time,
                )

            options = {"disp": False, "xtol": config.xtol}
            if config.maxiter > 0:
                options["maxiter"] = config.maxiter

            try:
                result = minimize(
                    fun=objective,
                    x0=onp.asarray(args_float, dtype=onp.float64),
                    jac=gradient,
                    hessp=hessian_vector,
                    method="Newton-CG",
                    callback=callback,
                    options=options,
                )

                fitted = onp.asarray(result.x)
                n_par = fitted.size
                basis = onp.eye(n_par, dtype=onp.float64)
                hessian = onp.column_stack(
                    [hessian_vector(fitted, basis[:, i]) for i in range(n_par)]
                )
                hessian = 0.5 * (hessian + hessian.T)
                try:
                    covariance = onp.linalg.inv(hessian)
                except onp.linalg.LinAlgError:
                    covariance = onp.linalg.pinv(hessian)
                ferror = onp.sqrt(onp.maximum(onp.real(onp.diag(covariance)), 0.0))
            finally:
                if self.num_processes > 1:
                    self._broadcast_request(OP_STOP, zero_vector, zero_vector)

            self.save_result(fitted, ferror, result)
        return 0

    # -- 结果保存（旧版 save_in_npz / save_in_json）-------------------------
    def save_result(self, fitted, ferror, result=None):
        outdir = os.path.join("output", self.config.outdir)
        os.makedirs(outdir, exist_ok=True)

        onp.save(os.path.join(outdir, "fit_result_values.npy"), fitted)
        onp.save(os.path.join(outdir, "fit_result_errors.npy"), ferror)

        # 旧版 save_in_npz: output/fit/fit_result_kk.npz(fvalue=, ferror=)
        onp.savez(
            os.path.join(outdir, "fit_result_kk.npz"),
            fvalue=fitted, ferror=ferror,
        )

        # 把自由参数写回 219 槽的参数向量（旧版 args_list 语义）
        full = ARGS_LIST.copy()
        full[FLOAT_LIST] = fitted
        onp.save(os.path.join(outdir, "fit_result_full_args.npy"), full)

        # 旧版 save_in_json: pwa_info_kk.json + result_info.json
        mod_info = []
        try:
            with open("gen/fragments/classification.json", "r") as fh:
                mod_info = json.load(fh)
        except (IOError, ValueError):
            mod_info = {}
        with open(os.path.join(outdir, "pwa_info_kk.json"), "w") as fh:
            json.dump({"mod_info": mod_info, "free_args": fitted.tolist()}, fh, indent=4)
        if result is not None:
            info = {
                "success": bool(result.success),
                "status": int(result.status),
                "nll": float(result.fun),
                "nit": int(getattr(result, "nit", -1)),
                "message": str(result.message),
            }
            with open(os.path.join(outdir, "result_info.json"), "w") as fh:
                json.dump(info, fh, indent=4)
            logger.info(
                "fit complete: success=%s nll=%.12g nit=%s message=%s",
                result.success, float(result.fun),
                getattr(result, "nit", "?"), result.message,
            )
        logger.info("results written to %s", outdir)


def main():
    os.makedirs("logs", exist_ok=True)
    setup_logging()
    return DistributedFit(FitConfig()).setup().run()


if __name__ == "__main__":
    raise SystemExit(main())
