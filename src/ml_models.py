# 模块 3.3：机器学习模型
# 作用：训练随机森林（主模型）与逻辑回归（简单对照），输出模型、概率与特征重要性。
#      统一封装，保证每次训练都用同一套随机种子与参数。

import numpy as np   # 数值计算
import pandas as pd  # 表格处理

from sklearn.ensemble import RandomForestClassifier   # 主模型
from sklearn.linear_model import LogisticRegression    # 对照模型
from sklearn.pipeline import Pipeline                  # 组合标准化与分类器
from sklearn.preprocessing import StandardScaler       # 逻辑回归需要标准化
from sklearn.impute import SimpleImputer               # 缺失值填中位数

from . import config
from .logger import get_logger

log = get_logger(__name__)  # 模块日志器


def prepare_xy(df: pd.DataFrame, features: list[str],
               label_col: str = "y80") -> tuple[pd.DataFrame, np.ndarray]:
    """从大表里取特征矩阵 X 与标签 y，并做缺失值填充。

    返回：
        X（DataFrame）、y（一维数组）
    """
    X = df[features].copy()                            # 只取特征列
    X = X.fillna(X.median(numeric_only=True))          # 缺失值用中位数填，简单可靠
    y = df[label_col].astype(int).to_numpy()           # 标签转 0/1 整数
    log.info("X 形状 %s，正样本比例 %.3f", X.shape, y.mean())
    return X, y


def train_rf(X: pd.DataFrame, y: np.ndarray,
             params: dict = config.RF_PARAMS) -> RandomForestClassifier:
    """训练随机森林并返回模型对象。"""
    model = RandomForestClassifier(**params)           # 按配置建模型
    model.fit(X, y)                                    # 训练
    log.info("RF 训练完成：%d 棵树，%d 个特征", params["n_estimators"], X.shape[1])
    return model


def train_lr(X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    """训练逻辑回归（对照模型）：标准化 + LR，防止量纲影响。"""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # 兜底填缺失
        ("scaler", StandardScaler()),                   # 标准化
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X, y)                                      # 训练
    log.info("LR 对照模型训练完成")
    return pipe


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """统一取正类（高风险）概率，RF 与 Pipeline 都适用。"""
    prob = model.predict_proba(X)[:, 1]                # 第二列是正类概率
    return np.asarray(prob, dtype=float)               # 统一为 numpy 数组


def rf_importance(model: RandomForestClassifier, features: list[str]) -> dict:
    """提取随机森林基于不纯度减少（MDI）的特征重要性。"""
    imp = model.feature_importances_                   # 重要性数组
    result = {name: float(v) for name, v in zip(features, imp)}  # 配对成字典
    top = sorted(result.items(), key=lambda kv: kv[1], reverse=True)[:5]  # 前 5 个
    log.info("特征重要性前 5：%s", top)
    return result


def permutation_importance_dict(model, X: pd.DataFrame, y: np.ndarray,
                                features: list[str], n_repeats: int = 10) -> dict:
    """置换重要性：打乱某特征后看 AUC 掉多少，比 MDI 更稳健。"""
    from sklearn.inspection import permutation_importance  # 延迟导入，减少启动开销

    result = permutation_importance(model, X, y, n_repeats=n_repeats,
                                    random_state=config.RANDOM_SEED, scoring="roc_auc")
    imp = {name: float(v) for name, v in zip(features, result.importances_mean)}  # 取均值
    log.info("置换重要性计算完成")
    return imp
