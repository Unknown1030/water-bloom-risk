# 模块 3.4：评价指标与统计检验
# 作用：计算分类指标、ΔAUC 与 paired bootstrap 置信区间。
#      核心问题：随机森林相对“状态持续”基线，增量是否稳定大于 0。

import numpy as np   # 数值计算
import pandas as pd  # 表格处理
from sklearn.metrics import (accuracy_score, f1_score,  # 常用分类指标
                             precision_score, recall_score, roc_auc_score)

from . import config
from .logger import get_logger

log = get_logger(__name__)  # 模块日志器


def classification_metrics(y: np.ndarray, prob: np.ndarray,
                           threshold: float = 0.5) -> dict:
    """计算 AUC、准确率、精确率、召回率、F1。

    参数：
        y        : 真实标签 0/1
        prob     : 预测为高风险的概率
        threshold: 概率转类别的阈值
    返回：
        指标字典
    """
    pred = (prob >= threshold).astype(int)             # 概率转 0/1
    auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else np.nan  # 只有一类时 AUC 无意义
    return {
        "AUC": float(auc),                             # 排序能力
        "Accuracy": float(accuracy_score(y, pred)),    # 准确率
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
    }


def delta_auc(y: np.ndarray, prob_rf: np.ndarray, prob_base: np.ndarray) -> float:
    """计算 ΔAUC = AUC_RF − AUC_基线。"""
    auc_rf = roc_auc_score(y, prob_rf)                 # 主模型 AUC
    auc_base = roc_auc_score(y, prob_base)             # 基线 AUC
    return float(auc_rf - auc_base)                    # 差值


def paired_bootstrap(y: np.ndarray, prob_rf: np.ndarray, prob_base: np.ndarray,
                     n_boot: int = config.BOOTSTRAP_N,
                     seed: int = config.RANDOM_SEED) -> dict:
    """配对 bootstrap：同一组重采样索引同时用于两个模型，估计 ΔAUC 的抽样分布。

    返回：
        {"delta": 点估计, "ci_low": 下界, "ci_high": 上界, "n_valid": 有效次数}
    """
    rng = np.random.default_rng(seed)                  # 固定种子，保证可复现
    n = len(y)                                         # 样本量
    point = delta_auc(y, prob_rf, prob_base)           # 点估计
    deltas = []                                        # 存放每次重采样的 ΔAUC

    for _ in range(n_boot):                            # 重复 n_boot 次
        idx = rng.integers(0, n, n)                    # 有放回抽样
        y_b = y[idx]                                   # 同一索引取标签
        if len(np.unique(y_b)) < 2:                    # 只有一类时跳过
            continue
        deltas.append(roc_auc_score(y_b, prob_rf[idx]) - roc_auc_score(y_b, prob_base[idx]))

    deltas = np.asarray(deltas)                        # 转数组
    if deltas.size == 0:                               # 全部退化时返回缺失
        return {"delta": point, "ci_low": float("nan"), "ci_high": float("nan"), "n_valid": 0}
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])  # 95% 区间
    log.info("ΔAUC=%.4f，95%% 区间 [%.4f, %.4f]（有效 %d 次）",
             point, ci_low, ci_high, len(deltas))
    return {"delta": point, "ci_low": float(ci_low), "ci_high": float(ci_high),
            "n_valid": int(len(deltas))}


def judge_increment(result: dict) -> str:
    """按开题报告的操作规则给结论定性，避免措辞夸大。

    规则：区间下限 > 0 → “稳定增量”；区间跨 0 → “未见稳定增量”；上限 < 0 → “劣于基线”。
    """
    if result["ci_low"] > 0:                            # 整个区间在 0 以上
        return "稳定增量"
    if result["ci_high"] < 0:                           # 整个区间在 0 以下
        return "劣于基线"
    return "未见稳定增量"                                # 区间跨 0


def summarize_folds(rows: list[dict]) -> pd.DataFrame:
    """把各折结果整理成表，便于直接贴进论文。"""
    df = pd.DataFrame(rows)                             # 列表转表
    log.info("各折结果：\n%s", df)                       # 打印
    return df
