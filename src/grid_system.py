# 模块 1.3：空间网格系统
# 作用：建立 10 m × 10 m 基准网格，把栅格值聚合到网格中心点，
#      保证不同期次、不同传感器最终落在同一套网格上（空间可比）。

import numpy as np  # 数组运算

from . import config
from .logger import get_logger

log = get_logger(__name__)  # 模块日志器


def cell_pixels(resolution: float, grid_size: int = config.GRID_SIZE) -> int:
    """计算一个网格对应多少像元：网格边长 / 像元大小。

    例：10 m 网格 + 3 cm 影像 → 333 像元（向下取整）。
    """
    return max(1, int(round(grid_size / resolution)))  # 至少 1 个像元


def block_reduce(arr: np.ndarray, factor: int, reducer: str = "mean") -> np.ndarray:
    """把二维数组按 factor×factor 分块聚合（均值或最大值）。

    参数：
        arr     : 二维数组（允许含 np.nan）
        factor  : 每块边长（像元数）
        reducer : "mean" 取均值，"max" 取最大值
    返回：
        聚合后的二维数组
    """
    h, w = arr.shape                                  # 原尺寸
    h2, w2 = h // factor, w // factor                 # 聚合后尺寸（截断）
    trimmed = arr[: h2 * factor, : w2 * factor]       # 裁掉不能整除的边缘
    blocks = trimmed.reshape(h2, factor, w2, factor)  # 拆成四维块
    if reducer == "mean":
        out = np.nanmean(blocks, axis=(1, 3))         # 忽略 NoData 求均值
    else:
        out = np.nanmax(blocks, axis=(1, 3))          # 取最大值
    log.info("聚合 %s → %s（factor=%d）", arr.shape, out.shape, factor)
    return out


def grid_index(shape: tuple[int, int]) -> np.ndarray:
    """生成网格编号矩阵：每个网格一个唯一 ID，按行优先编号。"""
    h, w = shape                                     # 网格行列数
    ids = np.arange(h * w).reshape(h, w)             # 顺序编号后变形
    return ids


def grid_centroids(transform, shape: tuple[int, int], grid_size: int = config.GRID_SIZE):
    """根据仿射变换计算每个网格中心点的地图坐标。

    参数：
        transform : rasterio 仿射变换（左上角原点、像元大小）
        shape     : 网格行列数 (rows, cols)
        grid_size : 网格边长（米）
    返回：
        xs, ys 两个二维数组，分别对应每个网格中心的 X、Y
    """
    rows, cols = shape                               # 网格行列数
    xs = transform.c + (np.arange(cols) + 0.5) * grid_size  # 每个网格中心的 X
    ys = transform.f - (np.arange(rows) + 0.5) * grid_size  # 每个网格中心的 Y
    xx, yy = np.meshgrid(xs, ys)                     # 展成二维网格
    return xx, yy


def flatten_grid(values: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> list[dict]:
    """把网格值矩阵拍平成“一行一个网格”的记录列表，便于存 CSV。"""
    records = []
    rows, cols = values.shape                        # 网格尺寸
    for r in range(rows):                            # 遍历行
        for c in range(cols):                        # 遍历列
            v = values[r, c]                         # 当前网格值
            if np.isnan(v):                          # 跳过无效网格
                continue
            records.append({
                "grid_id": r * cols + c,             # 网格编号
                "row": r, "col": c,                  # 行列位置
                "x": float(xx[r, c]),                # 中心 X
                "y": float(yy[r, c]),                # 中心 Y
                "value": float(v),                   # 该网格的聚合值
            })
    log.info("网格记录 %d 条", len(records))
    return records
