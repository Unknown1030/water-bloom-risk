# 项目：基于无人机多源遥感数据的三峡库区典型库湾水华风险预警与模拟

本项目是《基于无人机多源遥感数据的三峡库区典型库湾水华风险预警与模拟》的**代码实现骨架**，
按“数据流驱动”拆成 5 个模块、16 个 Python 文件。每个阶段只做一件事，上一步的输出是下一步的输入；
**计算与绘图严格分离**，改图不用重跑模型。

---

## 一、目录结构

```
14.program/
├── main.py                  # 主入口：按阶段调用各模块
├── requirements.txt         # 依赖清单
├── config.yaml              # 参数参考文件（实际以 src/config.py 为准）
├── src/                     # 全部模块代码
│   ├── config.py            # 路径、参数、随机种子（总开关）
│   ├── logger.py            # 日志：每次运行生成独立日志文件
│   ├── plot_utils.py        # 统一绘图样式与保存
│   ├── gee_extract.py       # 1.1 GEE 取数（S2/ERA5/SRTM/JRC）
│   ├── uav_preprocess.py    # 1.2 无人机影像：定标、NDWI 掩膜、JRC 核对
│   ├── grid_system.py       # 1.3 10 m 网格与聚合
│   ├── spectral_features.py # 2.1 NDCI 与网格统计
│   ├── label_generator.py   # 2.2 分位数标签（70/75/80/85）
│   ├── data_fusion.py       # 2.3 气象/地形/光谱对齐成 dataset.csv
│   ├── data_splitter.py     # 3.1 留一期 + 空间块划分
│   ├── baseline_model.py    # 3.2 状态持续基线
│   ├── ml_models.py         # 3.3 随机森林 / 逻辑回归
│   ├── evaluation.py        # 3.4 指标、ΔAUC、paired bootstrap
│   ├── shap_explainer.py    # 4.1 SHAP 解释（无 shap 时自动退化）
│   ├── risk_mapping.py      # 4.2 概率制图
│   └── scenario_simulation.py # 4.3 情景扰动与判据核对
├── notebooks/               # 调试用 Jupyter（可选）
├── data/raw|interim|processed  # 原始 / 中间 / 成品数据
├── outputs/figures|tables      # 图与表
└── logs/                    # 运行日志
```

---

## 二、快速开始（无需任何数据）

```bash
# 1. 创建虚拟环境（推荐 Python 3.10）
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# 2. 安装依赖（最少只需前四个）
pip install numpy pandas scikit-learn matplotlib

# 3. 跑通演示流程
python main.py --stage demo
```

运行后会在 `outputs/` 下得到：

| 文件 | 内容 |
|---|---|
| `tables/validation_folds.csv` | 各折 AUC、ΔAUC 及 95% 区间、结论定性 |
| `tables/feature_importance.csv` | 特征重要性排序 |
| `tables/test_probabilities.csv` | 验证集每个网格的风险概率 |
| `tables/scenario_results.csv` | 情景扰动下高风险比例变化 |
| `figures/feature_importance.png` | 重要性条形图 |
| `figures/risk_map.png` | 风险概率空间图 |
| `processed/dataset.csv` | 合成数据大表（真实数据的同款格式） |

---

## 三、换成真实数据的四步

1. **取公开数据**：`python main.py --stage gee`（需先 `earthengine authenticate` 并填写项目 ID）。
2. **预处理航拍影像**：`python main.py --stage uav --uav D:\data\flight1.tif`
   - 输入需为 5 波段正射影像，波段顺序：蓝、绿、红、红边、近红外。
3. **拼特征表**：按 `data/processed/dataset.csv` 的列结构，把光谱、气象、地形对齐；
   字段要求见 `data_fusion.py` 顶部注释。
4. **训练与出图**：把 `main.py` 的 demo 数据源替换为读取 `dataset.csv`，其余模块不用改。

---

## 四、关键约定（改代码前先读）

| 约定 | 值 | 位置 |
|---|---|---|
| 坐标系 | EPSG:32649 | `config.CRS` |
| 网格尺寸 | 10 m | `config.GRID_SIZE` |
| 标签切点 | 默认 80% 分位，另算 70/75/85 | `config.DEFAULT_QUANTILE` |
| 随机种子 | 42 | `config.RANDOM_SEED` |
| 训练/验证 | 按航拍期次留一（LOPO），绝不随机拆 | `data_splitter.lopo_folds` |
| 基线 | 下一期状态 = 当前期状态 | `baseline_model.persistence_predict` |
| 核心指标 | ΔAUC 及 paired bootstrap 95% 区间 | `evaluation.paired_bootstrap` |

---

## 五、代码风格

- 每个函数都有中文 docstring，关键行有行尾注释；
- 模块之间只通过 DataFrame 和文件传递数据，不共享全局变量；
- 可选依赖（rasterio / ee / shap）全部用 try-import 包裹，缺库时给出明确提示而不是崩溃；
- 所有随机过程固定种子，保证同一份数据跑出同一结果。

---

## 六、常见问题

**Q：`python` 命令没反应？**
Windows 应用商店的占位符。请从 python.org 安装 3.10 并勾选 “Add Python to PATH”。

**Q：demo 报 `ModuleNotFoundError: sklearn`？**
只装了 numpy/pandas。执行 `pip install scikit-learn matplotlib`。

**Q：没有 shap 会怎样？**
`shap_explainer` 自动退化为使用模型自带的 MDI 重要性，蜂群图会跳过，不影响主流程。

**Q：真实影像读入报错？**
先确认波段数为 5、位深与 NoData 设置正确；不确定时用 QGIS 打开检查，再跑 `uav_preprocess.read_stack`。
