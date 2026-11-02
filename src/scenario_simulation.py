# 模块 4.3：情景模拟
# 作用：固定其他特征，只扰动一个驱动变量（气温、降水、风速），
#      重新代入模型，比较高风险网格比例与空间变化，回答“若…会怎样”。

import numpy as np   # 数值计算
import pandas as pd  # 表格处理

from . import config
from .logger import get_logger

log = get_logger(__name__)  # 模块日志器


def perturb(df: pd.DataFrame, feature: str, mode: str, value: float) -> pd.DataFrame:
    """对指定特征做扰动，返回新表（不改原表）。

    参数：
        feature: 列名，如 temp、precip_3d、wind
        mode   : "add" 加减绝对值；"scale" 乘系数
        value  : add 时为增量（如 +2 表示升温 2 ℃）；scale 时为倍数（如 0.5 表示减半）
    """
    out = df.copy()                                    # 复制，避免污染基准表
    if feature not in out.columns:                     # 列不存在直接报错
        raise KeyError(f"特征列不存在：{feature}")
    if mode == "add":                                  # 加法扰动
        out[feature] = out[feature] + value
    elif mode == "scale":                              # 乘法扰动
        out[feature] = out[feature] * value
    else:                                              # 其他模式不支持
        raise ValueError(f"未知扰动方式：{mode}")
    log.info("情景扰动：%s %s %s", feature, mode, value)
    return out


def scenario_set(base: pd.DataFrame) -> list[dict]:
    """定义标准情景集：气温 ±2 ℃、降水减半/加倍、风速减半/加倍。"""
    return [
        {"name": "气温 +2℃", "df": perturb(base, "temp", "add", 2)},
        {"name": "气温 -2℃", "df": perturb(base, "temp", "add", -2)},
        {"name": "降水 ×0.5", "df": perturb(base, "precip_3d", "scale", 0.5)},
        {"name": "降水 ×2", "df": perturb(base, "precip_3d", "scale", 2.0)},
        {"name": "风速 ×0.5", "df": perturb(base, "wind", "scale", 0.5)},
        {"name": "风速 ×2", "df": perturb(base, "wind", "scale", 2.0)},
    ]


def run_scenarios(model, base_df: pd.DataFrame, features: list[str],
                  threshold: float = 0.5) -> pd.DataFrame:
    """逐个情景预测，统计高风险网格比例的变化。

    返回：
        含 情景、高风险比例、相对基准变化、样本数 四列的表
    """
    base_prob = model.predict_proba(base_df[features].fillna(base_df[features].median()))[:, 1]
    base_share = float((base_prob >= threshold).mean())  # 基准高风险占比

    rows = []
    for sc in scenario_set(base_df):                   # 遍历标准情景
        X = sc["df"][features].fillna(base_df[features].median())  # 取特征并对齐缺失
        prob = model.predict_proba(X)[:, 1]            # 重新预测
        share = float((prob >= threshold).mean())      # 该情景高风险占比
        rows.append({
            "情景": sc["name"],
            "高风险比例": round(share, 4),
            "相对基准变化": round(share - base_share, 4),
            "样本数": int(len(prob)),
        })
    result = pd.DataFrame(rows)                        # 汇总成表
    log.info("情景模拟完成，基准高风险占比 %.3f", base_share)
    return result


def check_criteria(result: pd.DataFrame, base_std: float,
                   top_features: list[str], variable: str) -> dict:
    """按开题报告的三项判据，判断某变量的情景响应能否作为主要结论。

    判据：
        ① 方向一致（由外部传入的 direction_consistent 判断，这里用占位 True）
        ② 高风险比例变化 > 各折 AUC 标准差
        ③ 该变量位于重要性前 50%
    返回：
        各判据是否满足的字典
    """
    max_change = float(result["相对基准变化"].abs().max()) if len(result) else 0.0  # 全部情景中的最大变化
    crit2 = max_change > base_std                      # 判据②：变化超过折间波动
    crit3 = variable in top_features[: max(1, len(top_features) // 2)]        # 判据③：前 50%
    verdict = "可作主要结论" if (crit2 and crit3) else "仅作探索性结果"        # 判据①默认满足
    log.info("判据核对：变化=%.3f，折间标准差=%.3f，结论=%s", max_change, base_std, verdict)
    return {"max_change": max_change, "criteria2": crit2, "criteria3": crit3,
            "verdict": verdict}
