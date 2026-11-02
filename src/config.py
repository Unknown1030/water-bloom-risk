# 项目：基于无人机多源遥感数据的三峡库区典型库湾水华风险预警与模拟
#
# 本文件是项目的“总开关”，所有模块都从这里读取参数与路径。
# 设计原则：默认值可直接跑通 demo；换成真实数据时只改这里，不动模块代码。

from pathlib import Path  # 路径处理，统一用 pathlib 避免 Windows/Linux 斜杠差异

# ---------- 1. 项目根目录 ----------
ROOT = Path(__file__).resolve().parent.parent  # src/config.py 的上一级即项目根目录

# ---------- 2. 数据与输出目录 ----------
DATA_RAW = ROOT / "data" / "raw"            # 原始数据：GEE 导出、无人机原图
DATA_INTERIM = ROOT / "data" / "interim"    # 中间数据：掩膜后、重采样后的影像
DATA_PROCESSED = ROOT / "data" / "processed"  # 最终数据：dataset.csv、预测结果
OUT_FIG = ROOT / "outputs" / "figures"      # 图件输出目录
OUT_TAB = ROOT / "outputs" / "tables"       # 表格输出目录
LOG_DIR = ROOT / "logs"                     # 日志目录

# ---------- 3. 空间参考与网格 ----------
CRS = "EPSG:32649"      # WGS84 / UTM 49N，三峡库区统一投影
GRID_SIZE = 10          # 网格边长，单位：米（与开题报告的 10 m 网格一致）

# ---------- 4. 研究区范围（香溪河库湾粗略外接矩形） ----------
# 顺序：西、南、东、北（经度、纬度），用于 GEE 裁剪
ROI_BOUNDS = (110.60, 30.95, 110.85, 31.10)

# ---------- 5. NDCI 与标签参数 ----------
QUANTILES = [70, 75, 80, 85]   # 标签切点候选，做敏感性分析
DEFAULT_QUANTILE = 80          # 默认主方案：80% 分位以上记为高风险

# ---------- 6. 气象累积窗口 ----------
LAG_WINDOWS = [3, 10]          # 前 3 日、前 10 日累积降水/均温窗口

# ---------- 7. 模型与验证参数 ----------
RANDOM_SEED = 42               # 全局随机种子，保证可复现
RF_PARAMS = {                  # 随机森林超参数
    "n_estimators": 500,       # 树的数量
    "min_samples_leaf": 5,     # 叶节点最少样本，防止过拟合
    "class_weight": "balanced",# 正负样本不均衡时自动加权
    "random_state": RANDOM_SEED,
    "n_jobs": -1,              # 使用全部 CPU 核
}
BOOTSTRAP_N = 1000             # paired bootstrap 重采样次数
SPATIAL_BLOCKS = 5             # 空间块验证的段数

# ---------- 8. 绘图参数 ----------
FIG_DPI = 300                  # 出图分辨率，满足论文要求
FONT_FAMILY = "SimHei"         # 中文字体，Windows 自带
PROB_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 风险概率分级

# ---------- 9. 目录自检 ----------
def ensure_dirs() -> None:
    """确保所有输出目录存在；不存在就创建，避免写文件时报错。"""
    for p in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, OUT_FIG, OUT_TAB, LOG_DIR):
        p.mkdir(parents=True, exist_ok=True)  # parents=True 允许递归创建


# ---------- 10. 参数快照（写进日志，保证可追溯） ----------
def snapshot() -> dict:
    """把关键参数打包成字典，跑实验前打印并写入日志，方便复现。"""
    return {
        "crs": CRS,
        "grid_size": GRID_SIZE,
        "roi": ROI_BOUNDS,
        "default_quantile": DEFAULT_QUANTILE,
        "lag_windows": LAG_WINDOWS,
        "random_seed": RANDOM_SEED,
        "rf_params": RF_PARAMS,
        "bootstrap_n": BOOTSTRAP_N,
    }
