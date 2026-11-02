# 绘图工具：统一字体、配色、分辨率，避免每张图各写一套设置
# 原则：绘图函数只负责“画”，不负责“算”，数据从外部传入

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # 无界面后端，服务器/命令行也能出图
import matplotlib.pyplot as plt  # 主绘图库
import numpy as np               # 数值处理

from . import config  # 读取分辨率、字体等配置


def setup_style() -> None:
    """统一全局绘图样式：中文字体、负号显示、网格与边框。"""
    plt.rcParams["font.sans-serif"] = [config.FONT_FAMILY]  # 指定中文字体
    plt.rcParams["axes.unicode_minus"] = False              # 正常显示负号
    plt.rcParams["figure.dpi"] = 100                        # 屏幕显示分辨率
    plt.rcParams["savefig.dpi"] = config.FIG_DPI            # 保存分辨率
    plt.rcParams["axes.grid"] = True                        # 打开网格
    plt.rcParams["grid.alpha"] = 0.3                        # 网格淡一点


def save_fig(fig: plt.Figure, filename: str) -> Path:
    """保存图片到 outputs/figures，返回完整路径。"""
    config.ensure_dirs()                         # 确保目录存在
    out = config.OUT_FIG / filename              # 拼接输出路径
    fig.tight_layout()                           # 自动调整边距，防止文字被裁
    fig.savefig(out, bbox_inches="tight")        # 保存，tight 去掉多余白边
    plt.close(fig)                               # 关闭图，释放内存
    return out


def plot_hist(values: np.ndarray, filename: str = "ndci_hist.png",
              title: str = "NDCI 分布直方图") -> Path:
    """画一维数据直方图，用于检查 NDCI 分布是否合理。"""
    fig, ax = plt.subplots(figsize=(6, 4))       # 画布尺寸 6x4 英寸
    ax.hist(values, bins=60, color="#4C72B0", alpha=0.85)  # 60 个分箱
    ax.set_xlabel("NDCI")                        # 横轴
    ax.set_ylabel("像元数")                       # 纵轴
    ax.set_title(title)                          # 标题
    return save_fig(fig, filename)               # 保存并返回路径


def plot_prob_map(df, value_col: str = "prob", filename: str = "risk_map.png",
                  title: str = "水华风险概率图") -> Path:
    """用网格中心点坐标画风险概率散点图，作为空间制图的轻量替代。

    参数：
        df         : 含 x、y、prob 三列的 DataFrame
        value_col  : 概率列名
        filename   : 输出文件名
    """
    fig, ax = plt.subplots(figsize=(8, 5))  # 画布
    sc = ax.scatter(  # 用散点表示每个 10 m 网格
        df["x"], df["y"], c=df[value_col], cmap="YlOrRd",
        s=8, vmin=0, vmax=1, edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax)  # 颜色条
    cbar.set_label("高风险概率")     # 颜色条含义
    ax.set_xlabel("东向坐标 / m")    # 横轴
    ax.set_ylabel("北向坐标 / m")    # 纵轴
    ax.set_title(title)             # 标题
    ax.set_aspect("equal")          # 等比例，避免空间变形
    return save_fig(fig, filename)


def plot_importance(importances: dict, filename: str = "feature_importance.png",
                    top_n: int = 10) -> Path:
    """画特征重要性水平条形图，取前 top_n 个特征。"""
    items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    names = [k for k, _ in items][::-1]   # 反转，使最重要的在最上方
    vals = [v for _, v in items][::-1]    # 对应数值
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(names, vals, color="#55A868")  # 水平条形
    ax.set_xlabel("重要性（MDI 或置换重要性）")
    ax.set_title("特征重要性排序")
    return save_fig(fig, filename)
