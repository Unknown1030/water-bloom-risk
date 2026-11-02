# 在 PyCharm 里跑起来（5 步）

> 本项目已经预先配好 `.idea`，正常情况下打开文件夹就能直接运行。

## 第 1 步：打开“文件夹”，不要只打开 main.py

PyCharm 菜单：**File → Open**，选择文件夹：

```
E:\MyProgram\601Program\Python\14.program
```

> 只双击 main.py 打开，PyCharm 不会加载项目配置，`from src import ...` 会报错。

## 第 2 步：确认右下角解释器

打开后看窗口**右下角**，应显示 `Python 3.10`。

- 如果显示 `Python 3.10 (PythonProgram)` 并带红色下划线 → 那是另一个项目的旧环境，已失效；
- 如果显示 `No interpreter` / `<No interpreter>` → 见第 2 步的修复。

**修复（30 秒）**：`File → Settings → Project: 14.program → Python Interpreter → 齿轮/Add Interpreter → Add Local Interpreter → System Interpreter → 选择`

```
D:\MyProgram\Python\python.exe
```

确定后等待 1—2 分钟建立索引。

## 第 3 步：运行

右上角运行配置下拉框选 **“demo 演示流程”**，点绿色三角 ▶。

或者：在左侧项目树右键 `main.py → Run 'main'`（工作目录需为项目根，PyCharm 默认就是）。

## 第 4 步：看结果

运行成功（exit code 0）后，`outputs` 下会出现：

| 路径 | 内容 |
|---|---|
| `outputs/tables/validation_folds.csv` | 各折 AUC、ΔAUC、95% 区间、结论 |
| `outputs/tables/feature_importance.csv` | 特征重要性排序 |
| `outputs/tables/scenario_results.csv` | 情景模拟结果 |
| `outputs/tables/test_probabilities.csv` | 每个网格的风险概率 |
| `outputs/figures/feature_importance.png` | 重要性图 |
| `outputs/figures/risk_map.png` | 风险概率图 |

## 第 5 步：命令行方式（备用）

```bat
cd /d E:\MyProgram\601Program\Python\14.program
D:\MyProgram\Python\python.exe main.py --stage demo
```

---

## 常见报错对照表

| 报错 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: No module named 'sklearn'` | 解释器选错（用了没装库的环境） | 切到 `D:\MyProgram\Python\python.exe` |
| `ModuleNotFoundError: No module named 'src'` | 只打开了文件，没打开项目 | File → Open 选文件夹 |
| `ModuleNotFoundError: No module named 'numpy'` | 解释器是另一个空 venv | 同上 |
| 中文乱码 | 控制台编码 | 运行配置里已设 `PYTHONIOENCODING=utf-8`；命令行可先 `chcp 65001` |
| `PermissionError: E:\...` | 文件被占用 | 关掉 Excel/资源管理器预览后重试 |

---

## 说明

- 项目内已建好虚拟环境 `.venv`（装了 numpy/pandas/scikit-learn/matplotlib），
  想用它就把解释器切到 `E:\MyProgram\601Program\Python\14.program\.venv\Scripts\python.exe`；
- 用系统 Python（`D:\MyProgram\Python\python.exe`）也可以，scikit-learn 已装好；
- 两者任选其一，不要混用（否则又会出现“这个库有、那个库没有”）。
