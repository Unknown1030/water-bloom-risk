# notebooks 使用说明

这个文件夹放调试用的 Jupyter Notebook，**不要**把正式流程写在这里。
正式代码一律进 `src/`，Notebook 只用来做探索性验证。

建议的 Notebook 顺序：

| 文件 | 用途 |
|---|---|
| `01_check_gee.ipynb` | 验证 GEE 能取到影像、NDCI 范围是否合理 |
| `02_check_uav.ipynb` | 打开一期航拍影像，看掩膜效果与反射率量纲 |
| `03_check_features.ipynb` | 检查 dataset.csv 的缺失、分布、相关性 |
| `04_check_model.ipynb` | 试不同超参数，结论写回 src/config.py |

约定：

1. Notebook 里只调用 `src/` 的函数，不复制粘贴实现；
2. 每个 Notebook 顶部写清“输入文件、输出结论、日期”；
3. 结论确定后，把最终参数写进 `src/config.py`，并在 README 的参数表中更新。
