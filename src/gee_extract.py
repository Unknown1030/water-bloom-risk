# 模块 1.1：GEE 数据提取
# 作用：从 Google Earth Engine 取 Sentinel-2、ERA5-Land、SRTM、JRC 水体，
#      按研究区裁剪后导出 GeoTIFF 或 CSV，供本地流程使用。
# 注意：本模块需要安装 earthengine-api 并完成账号认证，未认证时仅能查看导出代码。

import csv
from pathlib import Path

from . import config
from .logger import get_logger

log = get_logger(__name__)  # 模块级日志器

# GEE 是可选依赖：没装也能 import 本模块，只是调用时报错提示
try:
    import ee  # noqa: F401  # Google Earth Engine Python API
    _HAS_EE = True
except Exception:  # 捕获所有导入异常，避免整个项目起不来
    ee = None
    _HAS_EE = False


def init_ee(project: str | None = None) -> bool:
    """初始化并认证 GEE。

    参数：
        project: GEE Cloud 项目 ID，None 表示使用默认项目
    返回：
        True 表示初始化成功
    """
    if not _HAS_EE:  # 没装依赖直接返回 False
        log.error("未安装 earthengine-api，请先 pip install earthengine-api")
        return False
    try:
        ee.Initialize(project=project)  # 使用已保存的凭据初始化
        log.info("GEE 初始化成功")
        return True
    except Exception as exc:  # 首次使用通常需要 ee.Authenticate()
        log.warning("GEE 初始化失败：%s；请先运行 python -m src.gee_extract --auth", exc)
        return False


def roi_geometry():
    """把配置中的经纬度矩形转成 GEE 几何对象。"""
    west, south, east, north = config.ROI_BOUNDS  # 解包范围
    return ee.Geometry.Rectangle([west, south, east, north])  # GEE 矩形


def sentinel2_composite(start: str, end: str, cloud_pct: int = 20):
    """获取研究区 Sentinel-2 去云中值合成（仅用于阶段一代码调试）。"""
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")  # 大气校正后的 L2A 产品
        .filterBounds(roi_geometry())                      # 空间筛选
        .filterDate(start, end)                            # 时间筛选
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))  # 云量筛选
        .median()                                          # 中值合成去云
    )
    return s2.clip(roi_geometry())  # 按研究区裁剪


def compute_ndci(image):
    """在 GEE 端计算 NDCI = (B5 - B4) / (B5 + B4)，B5 为红边、B4 为红光。"""
    return image.normalizedDifference(["B5", "B4"]).rename("NDCI")


def compute_ndwi(image):
    """在 GEE 端计算 NDWI = (B3 - B8) / (B3 + B8)，用于水体掩膜。"""
    return image.normalizedDifference(["B3", "B8"]).rename("NDWI")


def export_image(image, description: str, scale: int = 10) -> None:
    """把 GEE 影像导出到 Google Drive 的 waterbloom 文件夹。

    导出是异步任务，需要在 GEE Tasks 面板手动点 Run，或在网页端确认。
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,          # 任务名，也是文件名前缀
        folder="waterbloom",              # Drive 目标文件夹
        scale=scale,                      # 空间分辨率（米）
        region=roi_geometry(),            # 导出范围
        maxPixels=1e13,                   # 放宽像元上限，防止大区域报错
    )
    task.start()                          # 提交任务
    log.info("已提交导出任务：%s", description)


def era5_daily_table(start: str, end: str) -> list[dict]:
    """取 ERA5-Land 日尺度气象数据并整理成字典列表（气温、降水、风速）。

    返回的列表可直接写 CSV，字段与本地特征表对齐。
    """
    era = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start, end)
        .select(["temperature_2m", "total_precipitation", "u_component_of_wind_10m",
                 "v_component_of_wind_10m"])
    )
    rows: list[dict] = []
    for img in era.toList(era.size()).getInfo():  # 逐日取回元数据
        date = img["properties"]["system:time_start"]  # 毫秒时间戳
        rows.append({"date_ms": date, "image_id": img["id"]})  # 先记录日期与影像 ID
    log.info("ERA5-Land 共 %d 天", len(rows))
    return rows


def srtm_dem():
    """取 SRTM 30 m DEM，并计算坡度（单位：度）。"""
    dem = ee.Image("USGS/SRTMGL1_003").clip(roi_geometry())  # 高程
    slope = ee.Terrain.slope(dem).rename("slope")             # 坡度
    return dem.rename("elevation").addBands(slope)            # 两个波段一起返回


def jrc_water():
    """取 JRC 全球地表水 occurrence 图层，用于掩膜交叉核对。"""
    return ee.Image("JRC/GSW1_0/GlobalSurfaceWater").select("occurrence")


def write_csv(rows: list[dict], filename: str) -> Path:
    """把字典列表写成 CSV，存到 data/raw 下。"""
    config.ensure_dirs()                       # 确保目录存在
    out = config.DATA_RAW / filename           # 输出路径
    with open(out, "w", newline="", encoding="utf-8") as fh:  # 打开文件
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))  # 表头取第一行键
        writer.writeheader()                   # 写表头
        writer.writerows(rows)                 # 写数据
    log.info("已写出 %s", out)
    return out


def test_pipeline() -> None:
    """阶段一自检：不导出大数据，只确认 GEE 能连通、能取到影像。"""
    if not init_ee():  # 初始化失败直接返回
        return
    img = sentinel2_composite("2022-01-01", "2022-12-31")  # 取一年合成
    ndci = compute_ndci(img)                               # 算 NDCI
    stats = ndci.reduceRegion(
        reducer=ee.Reducer.percentile([10, 50, 80, 90]),
        geometry=roi_geometry(),
        scale=30,
        maxPixels=1e9,
    ).getInfo()                                            # 取分位数，验证数据可用
    log.info("NDCI 分位数：%s", stats)                      # 打印，作为阶段一验收依据


if __name__ == "__main__":  # 直接运行本文件时执行自检
    test_pipeline()
