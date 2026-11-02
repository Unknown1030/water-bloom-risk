# 模块 2.1：光谱特征计算
# 作用：计算 NDCI，并按网格统计均值、方差、分位数；必要时做文献二次多项式
#      叶绿素 a 映射（仅作敏感性分析，不作标签的唯一来源）。

import numpy as np   # 数值计算
import pandas as pd  # 表格处理

from . import config
from .logger import get_logger

log = get_logger(__name__)  # 模块日志器


def compute_ndci(red_edge: np.ndarray, red: np.ndarray) -> np.ndarray:
    """按像元计算 NDCI = (红边 - 红光) / (红边 + 红光)。

    参数：
        red_edge: 红边波段反射率（P4M 约 730 nm）
        red     : 红光波段反射率（约 650 nm）
    返回：
        NDCI 数组，范围约 -1 到 1
    """
    ndci = (red_edge - red) / (red_edge + red + 1e-6)  # 加 1e-6 防止除零
    ndci = np.clip(ndci, -1, 1)                        # 截断到合理区间
    return ndci.astype("float32")                      # 统一类型


def chla_from_ndci(ndci: np.ndarray, coef=(14.039, 86.115, 194.325)) -> np.ndarray:
    """用 Mishra & Mishra 二次多项式估算叶绿素 a（仅敏感性分析用）。

    Chl-a = a0 + a1*NDCI + a2*NDCI^2
    注意：P4M 红边中心 730 nm，与原文 708 nm 有偏移，结果仅作参考。
    """
    a0, a1, a2 = coef                                   # 解包系数
    chla = a0 + a1 * ndci + a2 * np.square(ndci)        # 二次多项式
    return np.clip(chla, 0, None).astype("float32")     # 浓度不为负


def grid_statistics(ndci: np.ndarray, mask: np.ndarray, factor: int) -> dict:
    """按网格统计 NDCI 的均值、标准差与 80% 分位。

    参数：
        ndci  : NDCI 影像（二维）
        mask  : 水体掩膜（布尔）
        factor: 一个网格包含的像元数
    返回：
        含三个二维数组的字典
    """
    masked = np.where(mask, ndci, np.nan)               # 非水体置为 NoData
    h, w = masked.shape                                 # 原尺寸
    h2, w2 = h // factor, w // factor                   # 网格数
    trimmed = masked[: h2 * factor, : w2 * factor]      # 裁边
    blocks = trimmed.reshape(h2, factor, w2, factor)    # 四维分块

    mean = np.nanmean(blocks, axis=(1, 3))              # 网格均值
    std = np.nanstd(blocks, axis=(1, 3))                # 网格标准差
    p80 = np.nanpercentile(blocks, 80, axis=(1, 3))     # 网格 80% 分位

    log.info("网格统计完成：%d 个网格", h2 * w2)
    return {"ndci_mean": mean, "ndci_std": std, "ndci_p80": p80}


def build_spectral_table(df: pd.DataFrame) -> pd.DataFrame:
    """从“网格—期次”明细表计算光谱特征列，返回新表。

    输入 df 需包含：grid_id、period、ndci（像元级或网格级均值）
    输出新增：ndci_mean、ndci_std、ndci_p80（缺列时按已有值补齐）
    """
    out = df.copy()                                     # 不改原表
    if "ndci_mean" not in out.columns and "ndci" in out.columns:
        # 若只有单值 ndci，则视为均值，方差记 0、分位等于自身
        out["ndci_mean"] = out["ndci"]
    if "ndci_std" not in out.columns:
        out["ndci_std"] = 0.0                           # 占位：真实数据由 grid_statistics 提供
    if "ndci_p80" not in out.columns:
        out["ndci_p80"] = out["ndci_mean"]              # 占位：同上
    log.info("光谱特征列就绪：%s", list(out.columns))
    return out


def add_sensitivity_chla(df: pd.DataFrame) -> pd.DataFrame:
    """给表加上文献系数的 Chl-a 估算列，用于敏感性分析。"""
    out = df.copy()                                     # 不改原表
    out["chla_mishra"] = chla_from_ndci(out["ndci_mean"].to_numpy())  # 批量估算
    log.info("已添加 chla_mishra 列（敏感性分析用）")
    return out
