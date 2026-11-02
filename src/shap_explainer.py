# 模块 4.1：SHAP 可解释性
# 作用：用 TreeSHAP 解释随机森林，输出蜂群图、依赖图与单网格贡献明细。
#      未安装 shap 时自动退化为“置换重要性条形图”，保证流程不断。

import numpy as np   # 数值计算
import pandas as pd  # 表格处理

from . import config, plot_utils
from .logger import get_logger

log = get_logger(__name__)  # 模块日志器

try:
    import shap  # 可选的解释库
    _HAS_SHAP = True
except Exception:  # 没装 shap 也不影响主流程
    shap = None
    _HAS_SHAP = False


def global_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    """计算全局特征重要性（优先 SHAP，缺失时用模型自带 MDI）。

    返回：
        含 feature、importance 两列、按重要性降序的 DataFrame
    """
    if _HAS_SHAP:                                      # 有 shap 时用平均绝对 SHAP 值
        explainer = shap.TreeExplainer(model)          # 树模型专用解释器
        values = explainer.shap_values(X)              # 得到各样本各特征的贡献
        if isinstance(values, list):                   # 二分类可能返回列表，取正类
            values = values[1]
        imp = np.abs(values).mean(axis=0)              # 绝对值取均值
        cols = X.columns                               # 特征名
    else:                                              # 退化方案：模型自带重要性
        imp = model.feature_importances_
        cols = X.columns
    out = pd.DataFrame({"feature": list(cols), "importance": np.asarray(imp)})
    out = out.sort_values("importance", ascending=False).reset_index(drop=True)  # 排序
    log.info("全局重要性前 5：\n%s", out.head())       # 打印前 5
    return out


def plot_beeswarm(model, X: pd.DataFrame, filename: str = "shap_beeswarm.png") -> str | None:
    """画 SHAP 蜂群图；没有 shap 库时返回 None。"""
    if not _HAS_SHAP:                                  # 缺库直接跳过
        log.warning("未安装 shap，跳过蜂群图")
        return None
    explainer = shap.TreeExplainer(model)              # 解释器
    values = explainer.shap_values(X)                  # 贡献值
    if isinstance(values, list):                       # 二分类取正类
        values = values[1]
    plot_utils.setup_style()                           # 统一字体
    shap.summary_plot(values, X, show=False)           # 画图（不弹窗）
    import matplotlib.pyplot as plt                    # 延迟导入，避免后端冲突
    fig = plt.gcf()                                    # 取当前图
    out = plot_utils.save_fig(fig, filename)           # 保存
    log.info("蜂群图已保存：%s", out)
    return str(out)


def dependence(model, X: pd.DataFrame, feature: str,
               filename: str = "shap_dependence.png") -> str | None:
    """画指定特征的 SHAP 依赖图，用于判断“该变量升高时风险往哪走”。"""
    if not _HAS_SHAP:                                  # 缺库跳过
        log.warning("未安装 shap，跳过依赖图")
        return None
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[1]
    plot_utils.setup_style()
    shap.dependence_plot(feature, values, X, show=False)  # 指定变量
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    out = plot_utils.save_fig(fig, filename)
    log.info("依赖图已保存：%s", out)
    return str(out)


def local_explanation(model, X: pd.DataFrame, row: int = 0) -> pd.DataFrame:
    """解释单个网格：基础值 + 各特征加分/扣分，按贡献绝对值排序。"""
    if not _HAS_SHAP:                                  # 缺库时返回空表
        log.warning("未安装 shap，无法输出单网格解释")
        return pd.DataFrame(columns=["feature", "shap_value", "feature_value"])
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X.iloc[[row]])
    if isinstance(values, list):
        values = values[1]
    contrib = np.asarray(values).ravel()               # 拉平成一维
    out = pd.DataFrame({
        "feature": X.columns,                          # 特征名
        "shap_value": contrib,                         # 贡献值
        "feature_value": X.iloc[row].to_numpy(),       # 该网格的特征取值
    })
    out["abs"] = out["shap_value"].abs()               # 绝对值用于排序
    out = out.sort_values("abs", ascending=False).drop(columns="abs")  # 降序
    log.info("单网格解释前 5：\n%s", out.head())
    return out
