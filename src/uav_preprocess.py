# 模块 1.2：无人机多光谱影像预处理
# 作用：读入大疆智图导出的 5 波段正射影像，做辐射校正、NDWI 水体掩膜、
#      与 JRC 水体图层交叉核对，输出“只保留水面像元”的反射率影像。
# 波段顺序（P4M 标准）：0 蓝、1 绿、2 红、3 红边、4 近红外

import numpy as np  # 数组运算

from . import config
from .logger import get_logger

log = get_logger(__name__)  # 模块日志器

# rasterio 是可选依赖：没装时导入本模块不报错，调用时才提示
try:
    import rasterio  # 读写 GeoTIFF
    _HAS_RIO = True
except Exception:  # 捕获导入异常
    rasterio = None
    _HAS_RIO = False

BAND = {"blue": 0, "green": 1, "red": 2, "rededge": 3, "nir": 4}  # 波段索引常量


def read_stack(path: str) -> tuple[np.ndarray, dict]:
    """读取 5 波段正射影像。

    参数：
        path: GeoTIFF 路径
    返回：
        (array, profile)：array 形状为 (5, H, W)，profile 为元数据
    """
    if not _HAS_RIO:  # 缺少 rasterio 时给出明确提示
        raise ImportError("需要安装 rasterio：pip install rasterio")
    with rasterio.open(path) as ds:  # 打开影像
        arr = ds.read().astype("float32")  # 读成浮点数组，避免整数截断
        profile = ds.profile.copy()        # 复制元数据，后面写文件要用
    log.info("读入影像 %s，形状 %s", path, arr.shape)
    return arr, profile


def dn_to_reflectance(arr: np.ndarray, max_dn: float = 65535.0) -> np.ndarray:
    """把 DN 值线性缩放到 0—1 反射率。

    说明：若已在大疆智图里做过辐射校正（输出即反射率），此函数可跳过；
         否则按传感器位深做简单归一化，作为兜底方案。
    """
    refl = arr / max_dn                     # 简单线性缩放
    refl = np.clip(refl, 0, 1)              # 截断到 0—1，防止异常值
    return refl.astype("float32")           # 统一为 float32，省内存


def gray_board_calibration(arr: np.ndarray, board_dn: float,
                           board_reflectance: float = 0.2) -> np.ndarray:
    """用标准灰板做单点辐射定标：反射率 = DN / 灰板DN × 灰板已知反射率。

    参数：
        arr              : 影像数组
        board_dn         : 灰板区域的平均 DN
        board_reflectance: 灰板标称反射率（常见 0.2 或 0.5）
    """
    if board_dn <= 0:                        # 灰板 DN 异常时不处理
        log.warning("灰板 DN 异常（%s），跳过定标", board_dn)
        return arr
    return (arr / board_dn * board_reflectance).astype("float32")  # 线性定标


def ndwi_mask(arr: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """计算 NDWI 并返回布尔水体掩膜。

    NDWI = (绿 - 近红外) / (绿 + 近红外)，阈值默认 0，可按影像微调。
    """
    green = arr[BAND["green"]]               # 绿波段
    nir = arr[BAND["nir"]]                   # 近红外波段
    ndwi = (green - nir) / (green + nir + 1e-6)  # 加 1e-6 防止除零
    mask = ndwi > threshold                  # 大于阈值判为水体
    log.info("NDWI 掩膜水体占比 %.1f%%", 100 * mask.mean())  # 打印占比
    return mask


def jrc_crosscheck(mask: np.ndarray, jrc_occurrence: np.ndarray,
                   occurrence_thr: float = 50.0) -> np.ndarray:
    """与 JRC 常年水体图层交叉核对，剔除“NDWI 判为水、JRC 判为陆”的像元。

    参数：
        mask          : NDWI 水体掩膜（布尔）
        jrc_occurrence: JRC occurrence 百分比（0—100）
        occurrence_thr: 超过该值视为常年水体
    返回：
        修正后的掩膜
    """
    jrc_water = jrc_occurrence >= occurrence_thr      # JRC 判为水体的像元
    diff = mask & (~jrc_water)                         # NDWI 说水、JRC 说陆：可疑
    fixed = mask & (~diff)                             # 先直接剔除可疑像元
    log.info("JRC 交叉核对剔除 %d 个可疑像元", int(diff.sum()))
    return fixed


def process_uav(in_path: str, out_path: str, threshold: float = 0.0) -> str:
    """完整预处理流程：读入 → 定标 → NDWI 掩膜 → 掩膜外置 NoData → 另存。

    参数：
        in_path : 原始 5 波段 GeoTIFF
        out_path: 输出路径
        threshold: NDWI 阈值
    返回：
        输出文件路径
    """
    arr, profile = read_stack(in_path)          # 读影像与元数据
    refl = dn_to_reflectance(arr)               # DN 转反射率（已定标则可跳过）
    mask = ndwi_mask(refl, threshold)           # 水体掩膜
    refl[:, ~mask] = np.nan                     # 非水体像元置为 NoData
    profile.update(dtype="float32", count=refl.shape[0], nodata=np.nan)  # 更新元数据
    with rasterio.open(out_path, "w", **profile) as dst:  # 写出新影像
        dst.write(refl)                         # 写入 5 个波段
    log.info("预处理完成：%s", out_path)
    return out_path


def check_reflectance_range(arr: np.ndarray) -> dict:
    """检查反射率量纲是否正常（应为 0—1），返回各波段均值供人工核对。"""
    stats = {}
    for name, idx in BAND.items():              # 逐个波段统计
        band = arr[idx]                         # 取该波段
        stats[name] = float(np.nanmean(band))   # 忽略 NoData 求均值
    log.info("各波段均值：%s", stats)            # 打印，正常应在 0.01—0.3
    return stats
