# 模块 3.1：数据划分
# 作用：提供两种“反作弊”划分方式——留一期交叉验证（LOPO）与空间块验证，
#      防止同一期次/相邻网格同时出现在训练集和验证集，导致精度虚高。

import numpy as np   # 数值计算
import pandas as pd  # 表格处理

from . import config
from .logger import get_logger

log = get_logger(__name__)  # 模块日志器


def lopo_folds(df: pd.DataFrame, period_col: str = "period") -> list[tuple[np.ndarray, np.ndarray]]:
    """留一期交叉验证：每次用 1 期做验证，其余期次做训练。

    返回：
        [(train_idx, test_idx), ...]，期次不足 2 时返回空列表。
    """
    periods = sorted(df[period_col].unique())          # 全部期次
    if len(periods) < 2:                               # 只有 1 期无法交叉验证
        log.warning("仅 %d 期数据，无法做留一期验证", len(periods))
        return []
    folds = []
    for p in periods:                                  # 逐期留出
        test_idx = np.where(df[period_col] == p)[0]    # 该期做验证
        train_idx = np.where(df[period_col] != p)[0]   # 其余做训练
        folds.append((train_idx, test_idx))
        log.info("留一期：验证期=%s，训练 %d 条，验证 %d 条", p, len(train_idx), len(test_idx))
    return folds


def spatial_block_folds(df: pd.DataFrame, n_blocks: int = config.SPATIAL_BLOCKS,
                        period_col: str = "period",
                        block_col: str = "block") -> list[tuple[np.ndarray, np.ndarray]]:
    """空间块验证：在同一期次内部，按纵向连续段切分，整段留出做验证。

    前提：df 已有 block 列（由“沿库湾纵向分段”得到，1..n）。
    返回：
        [(train_idx, test_idx), ...]
    """
    folds = []
    for p in sorted(df[period_col].unique()):          # 逐期处理
        sub = df[df[period_col] == p]                  # 该期数据
        if block_col not in sub.columns:               # 没有分块列就跳过
            log.warning("缺少 %s 列，跳过空间块验证", block_col)
            return []
        for b in range(1, n_blocks + 1):               # 逐块留出
            test_idx = np.where((df[period_col] == p) & (df[block_col] == b))[0]
            train_idx = np.where(~((df[period_col] == p) & (df[block_col] == b)))[0]
            if len(test_idx) == 0:                     # 空块跳过
                continue
            folds.append((train_idx, test_idx))
    log.info("空间块验证共 %d 折", len(folds))
    return folds


def check_split_leakage(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
                        period_col: str = "period") -> bool:
    """检查训练集与验证集是否存在同期次重叠（有重叠即为泄漏）。"""
    train_periods = set(df.iloc[train_idx][period_col].unique())  # 训练期次
    test_periods = set(df.iloc[test_idx][period_col].unique())    # 验证期次
    overlap = train_periods & test_periods                        # 交集
    if overlap:                                                   # 有交集就报警
        log.error("发现期次泄漏：%s", overlap)
        return False
    return True                                                   # 无交集说明划分干净


def stratified_sample(df: pd.DataFrame, label_col: str = "y80",
                      frac: float = 0.3, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """按标签分层抽样，用于快速试验时缩小数据量而保持正负比例。"""
    parts = []
    for _, group in df.groupby(label_col):             # 按标签分组
        parts.append(group.sample(frac=frac, random_state=seed))  # 每组抽同样比例
    out = pd.concat(parts).sample(frac=1, random_state=seed)      # 合并后打乱
    log.info("分层抽样：%d → %d 行", len(df), len(out))
    return out.reset_index(drop=True)
