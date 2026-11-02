# 模块 2.3：多源数据时空对齐
# 作用：把气象（ERA5-Land 日表）、地形（DEM）、光谱特征按“网格 + 期次”对齐，
#      并生成前 3 日、前 10 日的累积特征，输出一张可用于建模的大表 dataset.csv。

import numpy as np   # 数值计算
import pandas as pd  # 表格处理

from . import config
from .logger import get_logger

log = get_logger(__name__)  # 模块日志器


def add_lag_features(weather: pd.DataFrame, windows: list[int] = config.LAG_WINDOWS) -> pd.DataFrame:
    """给逐日气象表加累积特征：前 N 日累积降水、前 N 日均温。

    参数：
        weather: 至少含 date、precip、temp 三列，按日期升序
        windows: 累积窗口，如 [3, 10]
    返回：
        新增 precip_3d、precip_10d、temp_3d、temp_10d 四列的表
    """
    df = weather.sort_values("date").copy()            # 按日期排序，滚动才有意义
    for w in windows:                                  # 遍历窗口
        df[f"precip_{w}d"] = df["precip"].rolling(w, min_periods=1).sum()  # 累积降水
        df[f"temp_{w}d"] = df["temp"].rolling(w, min_periods=1).mean()     # 滑动均温
    log.info("已生成累积特征：%s", [f"precip_{w}d" for w in windows])
    return df


def merge_weather(samples: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """按日期把气象特征并到样本表（样本表需有 date 列，格式 YYYY-MM-DD）。"""
    left = samples.copy()                              # 不改原表
    left["date"] = pd.to_datetime(left["date"]).dt.strftime("%Y-%m-%d")  # 统一日期格式
    right = weather.copy()                             # 气象表
    right["date"] = pd.to_datetime(right["date"]).dt.strftime("%Y-%m-%d")
    merged = left.merge(right, on="date", how="left")  # 左连接，保留全部样本
    missing = merged["temp"].isna().sum()              # 统计没匹配上的行
    if missing:                                        # 有缺失就提醒
        log.warning("有 %d 行未匹配到气象数据", int(missing))
    log.info("气象合并完成，共 %d 行 %d 列", *merged.shape)
    return merged


def merge_dem(samples: pd.DataFrame, dem: pd.DataFrame) -> pd.DataFrame:
    """按 grid_id 把地形特征（高程、坡度）并到样本表。"""
    dem_cols = ["grid_id", "elevation", "slope"]       # DEM 提供的列
    keep = [c for c in dem_cols if c in dem.columns]   # 只取存在的列，避免报错
    merged = samples.merge(dem[keep], on="grid_id", how="left")  # 按网格连接
    log.info("地形合并完成，共 %d 行", len(merged))
    return merged


def build_dataset(samples: pd.DataFrame, weather: pd.DataFrame,
                  dem: pd.DataFrame) -> pd.DataFrame:
    """完整对齐流程：气象累积 → 合并气象 → 合并地形 → 清理缺失。

    返回的 DataFrame 即“Excel 大表”，每行是一个网格在某期次的观测。
    """
    weather_lag = add_lag_features(weather)            # 先算累积特征
    df = merge_weather(samples, weather_lag)           # 合并气象
    df = merge_dem(df, dem)                            # 合并地形
    before = len(df)                                   # 记录合并前行数
    df = df.dropna(subset=["ndci_mean", "temp", "elevation"])  # 关键列缺失就删
    log.info("数据集构建完成：%d → %d 行", before, len(df))
    return df.reset_index(drop=True)                   # 重置索引，便于后续切片


def save_dataset(df: pd.DataFrame, filename: str = "dataset.csv") -> str:
    """把大表保存到 data/processed，返回路径。"""
    config.ensure_dirs()                               # 确保目录存在
    out = config.DATA_PROCESSED / filename             # 输出路径
    df.to_csv(out, index=False, encoding="utf-8-sig")  # utf-8-sig 让 Excel 不乱码
    log.info("已保存数据集：%s", out)
    return str(out)


def feature_columns(df: pd.DataFrame, label_col: str = "y80") -> list[str]:
    """自动挑出用于建模的特征列：排除 ID、坐标、日期、标签列。"""
    exclude = {"grid_id", "period", "date", "x", "y", "row", "col", "block", "pred_base", "prob"} | {
        c for c in df.columns if c.startswith("y") and (c[1:].isdigit() or c[1:2] == "_")  # 所有标签列（y80、y_next）
    }
    cols = [c for c in df.columns if c not in exclude and df[c].dtype != object]
    log.info("特征列 %d 个：%s", len(cols), cols)
    return cols
