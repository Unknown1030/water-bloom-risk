# 项目包初始化文件
# 作用：把 src 目录标记为 Python 包，使 `from src import xxx` 可用。
# 这里不做任何导入，避免包初始化时触发重型依赖（rasterio、gee、shap）。

__all__ = [
    "config",            # 全局配置
    "logger",            # 日志
    "plot_utils",        # 绘图工具
    "gee_extract",       # 1.1 GEE 取数
    "uav_preprocess",    # 1.2 无人机预处理
    "grid_system",       # 1.3 网格系统
    "spectral_features", # 2.1 光谱特征
    "label_generator",   # 2.2 标签生成
    "data_fusion",       # 2.3 多源对齐
    "data_splitter",     # 3.1 数据划分
    "baseline_model",    # 3.2 朴素基线
    "ml_models",         # 3.3 机器学习模型
    "evaluation",        # 3.4 指标与检验
    "shap_explainer",    # 4.1 SHAP 解释
    "risk_mapping",      # 4.2 概率制图
    "scenario_simulation",  # 4.3 情景模拟
]

__version__ = "1.0.0"   # 项目版本号，改代码时同步更新
