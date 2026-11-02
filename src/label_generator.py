# 模块 2.2：标签生成
# 作用：按 NDCI 分位数把网格分成高/低风险两类，生成不同切点下的标签列，
#      供后续敏感性分析使用（70/75/80/85 分位）。

import numpy as np   # 数值计算
import pandas as pd  # 表格处理

from . import config
from .logger import get_logger

log = get_logger(__name__)  # 模块日志器


def quantile_threshold(values: np.ndarray, q: float) -> float:
    """计算给定分位数的阈值（忽略 NaN）。"""
    return float(np.nanpercentile(values, q))  # q 为 0—100 的百分位


def binary_label(values: np.ndarray, q: float = config.DEFAULT_QUANTILE) -> np.ndarray:
    """按分位数阈值把连续值转成 0/1 标签：超过阈值记 1（高风险）。"""
    thr = quantile_threshold(values, q)                 # 先算阈值
    labels = (values >= thr).astype(int)                # 大于等于阈值记 1
    log.info("分位 %d%% 阈值=%.4f，高风险占比 %.1f%%", q, thr, 100 * labels.mean())
    return labels


def add_labels(df: pd.DataFrame, value_col: str = "ndci_mean",
               quantiles: list[int] = config.QUANTILES) -> pd.DataFrame:
    """按期次分别计算分位数标签，新增 y70/y75/y80/y85 列。

    关键：必须按 period 分组算阈值，不能全表一起算，
         否则不同期次的水华强度差异会被抹平。
    """
    out = df.copy()                                     # 不改原表
    for q in quantiles:                                 # 逐个切点
        col = f"y{q}"                                   # 标签列名，如 y80
        out[col] = 0                                    # 先初始化为 0
        for period, idx in out.groupby("period").groups.items():  # 按期次分组
            vals = out.loc[idx, value_col].to_numpy()   # 该期次的 NDCI 值
            out.loc[idx, col] = binary_label(vals, q)   # 写入该期标签
        log.info("已生成标签列 %s", col)
    return out


def check_balance(df: pd.DataFrame, label_col: str = "y80") -> pd.DataFrame:
    """检查每个期次的正负样本比例，返回统计表。"""
    stats = (
        df.groupby("period")[label_col]                 # 按期次分组
        .agg(["count", "sum", "mean"])                  # 计数、正样本数、正样本比例
        .rename(columns={"count": "样本数", "sum": "高风险数", "mean": "高风险占比"})
    )
    log.info("标签平衡情况：\n%s", stats)                # 打印，便于判断是否失衡
    return stats


def label_summary_by_quantile(df: pd.DataFrame,
                              quantiles: list[int] = config.QUANTILES) -> pd.DataFrame:
    """汇总不同切点下的高风险占比，用于敏感性分析表。"""
    rows = []
    for q in quantiles:                                 # 遍历切点
        col = f"y{q}"                                   # 对应标签列
        rows.append({
            "切点": f"{q}%",
            "高风险占比": float(df[col].mean()),         # 全表比例
            "正样本数": int(df[col].sum()),              # 正样本数
            "总样本数": int(len(df)),                    # 总样本数
        })
    return pd.DataFrame(rows)                           # 返回汇总表
