# 模块 4.2：风险概率制图
# 作用：把网格级预测概率写回空间（10 m 网格），输出概率栅格与分级图。
#      只负责“把概率放回地图”，不重新训练模型。

import numpy as np   # 数值计算
import pandas as pd  # 表格处理

from . import config, plot_utils
from .logger import get_logger

log = get_logger(__name__)  # 模块日志器

try:
    import rasterio               # 写 GeoTIFF 用
    from rasterio.transform import from_origin  # 构造仿射变换
    _HAS_RIO = True
except Exception:                 # 没装 rasterio 时只出 PNG
    rasterio = None
    from_origin = None
    _HAS_RIO = False


def attach_probability(df: pd.DataFrame, prob: np.ndarray,
                       col: str = "prob") -> pd.DataFrame:
    """把预测概率按行序贴回样本表。"""
    out = df.copy()                                    # 不改原表
    out[col] = np.asarray(prob, dtype=float)           # 写入概率列
    log.info("已附加概率列 %s，均值 %.3f", col, out[col].mean())
    return out


def to_raster(df: pd.DataFrame, value_col: str = "prob",
              out_name: str = "risk_prob.tif") -> str | None:
    """把网格点还原成 10 m 栅格并保存 GeoTIFF。

    前提：df 含 row、col 两列（网格行列号）。缺列或无 rasterio 时返回 None。
    """
    if not _HAS_RIO:                                   # 缺依赖直接跳过
        log.warning("未安装 rasterio，跳过栅格输出")
        return None
    if not {"row", "col"}.issubset(df.columns):        # 缺行列号无法还原
        log.warning("缺少 row/col 列，跳过栅格输出")
        return None

    rows = int(df["row"].max()) + 1                    # 栅格行数
    cols = int(df["col"].max()) + 1                    # 栅格列数
    arr = np.full((rows, cols), np.nan, dtype="float32")  # 初始化空栅格
    for r, c, v in zip(df["row"], df["col"], df[value_col]):  # 逐点填值
        arr[int(r), int(c)] = float(v)

    transform = from_origin(0, 0, config.GRID_SIZE, config.GRID_SIZE)  # 临时原点
    out_path = config.OUT_FIG / out_name               # 输出路径
    profile = {
        "driver": "GTiff", "height": rows, "width": cols, "count": 1,
        "dtype": "float32", "crs": config.CRS, "transform": transform,
        "nodata": np.nan,
    }
    with rasterio.open(out_path, "w", **profile) as dst:  # 写出
        dst.write(arr, 1)
    log.info("概率栅格已保存：%s", out_path)
    return str(out_path)


def draw_map(df: pd.DataFrame, value_col: str = "prob",
             filename: str = "risk_map.png") -> str:
    """画风险概率空间散点图（轻量版制图，论文示意够用）。"""
    plot_utils.setup_style()                           # 统一样式
    out = plot_utils.plot_prob_map(df, value_col=value_col, filename=filename)
    log.info("风险图已保存：%s", out)
    return str(out)


def classify(prob: np.ndarray, levels: list[float] = config.PROB_LEVELS) -> np.ndarray:
    """把连续概率分成 5 级，返回级别编号（0—4），用于分级统计。"""
    return np.digitize(prob, levels[1:-1])             # 去掉首尾断点后分级
