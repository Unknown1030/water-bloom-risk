# 模块 3.2：基准模型（朴素基线）
# 作用：实现“状态持续”基线——下一期状态等于当前期状态。
#      这是全项目的“及格线”：随机森林必须稳定超过它，才谈得上“预警价值”。

import numpy as np   # 数值计算
import pandas as pd  # 表格处理

from .logger import get_logger

log = get_logger(__name__)  # 模块日志器


def persistence_predict(df: pd.DataFrame, period_col: str = "period",
                        grid_col: str = "grid_id",
                        label_col: str = "y80") -> pd.Series:
    """对每一行预测“下一期状态”，预测值 = 同一网格上一期的标签。

    参数：
        df       : 含网格、期次、标签的长表
        period_col/grid_col/label_col: 对应列名
    返回：
        与 df 同索引的预测序列；上一期不存在的行返回 NaN。
    """
    data = df[[grid_col, period_col, label_col]].copy()  # 只取必要列
    periods = sorted(data[period_col].unique())          # 期次排序
    prev_map = {p: periods[i - 1] for i, p in enumerate(periods) if i > 0}  # 上一期映射
    data["pred"] = np.nan                                # 初始化预测列

    for p, prev in prev_map.items():                     # 逐期回填
        now = data[period_col] == p                      # 当前期掩码
        last = data[data[period_col] == prev].set_index(grid_col)[label_col]  # 上一期标签
        data.loc[now, "pred"] = data.loc[now, grid_col].map(last)             # 按网格映射

    valid = data["pred"].notna().sum()                   # 有预测值的行数
    log.info("状态持续基线：%d 行有预测（共 %d 行）", int(valid), len(data))
    return data["pred"]                                  # 返回预测序列


def baseline_name() -> str:
    """返回基线名称，写报告时统一用这个词，避免“朴素/持续/naive”混用。"""
    return "状态持续基线（persistence）"
