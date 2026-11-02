"""项目主入口：按“数据流”顺序调用各模块。

用法：
    python main.py --stage demo      # 用合成数据跑通全流程（无需任何外部数据）
    python main.py --stage gee       # 检查 GEE 连通性并打印导出代码
    python main.py --stage uav       # 预处理一张无人机影像（需 rasterio）
    python main.py --stage all       # 依次执行以上步骤

设计原则：每个阶段只做一件事，上一步输出是下一步输入；所有结果落盘，
         改图不用重跑模型（绘图与计算分离）。
"""

import argparse  # 命令行参数解析库，用来接收你在终端输入的 --stage 等指令
from pathlib import Path # 路径处理库，用于跨平台管理文件夹

import numpy as np   # 科学计算库，用来算矩阵、随机数、公式
import pandas as pd  # 数据处理库，用来生成和处理类似Excel的表格数据（大表）

# 从 src 文件夹下的各个模块中导入功能
# 这就相当于把之前规划的“零件”全部拿过来组装
from src import config, data_fusion, data_splitter, evaluation, label_generator
from src import ml_models, plot_utils, risk_mapping, scenario_simulation, spectral_features
from src.logger import get_logger, log_config

log = get_logger("main")  # 启动日志记录器，用来在后台记录运行状态和报错信息


# ==========================================================================
# 一、合成数据：没有真实数据时也能跑通全流程
# ==========================================================================
# 定义函数，接收三个参数，并声明返回值是 Pandas 的 DataFrame 类型
# 参数含义：
# n_grids: 每一期航拍有多少个 10m x 10m 的网格（默认 1500 个）
# periods: 总共模拟几期航拍数据（默认 4 期）
# seed: 随机种子。设定一个固定的数字，保证每次跑代码生成的假数据都一模一样，方便你复现结果。
def make_demo_dataset(n_grids: int = 1500, periods: int = 4,
                      seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """生成 3 期、每期 n_grids 个网格的合成数据，用于验证代码链路。"""

    # 调用 numpy 库 (np) 的随机数生成器。
    # default_rng 是 numpy 推荐的现代随机数生成方式，传入 seed 保证随机结果可重复。
    rng = np.random.default_rng(seed)

    frames = []  # 准备一个空列表，用来收集每一期的数据表（相当于准备几个空筐）

    # 开始循环，从第 1 期到第 periods 期（比如 1, 2, 3, 4）
    for p in range(1, periods + 1):
        # 模拟 NDCI 值（归一化叶绿素指数）。
        # 调用 rng.beta：生成符合 Beta 分布的随机数。参数 2.2 和 3.0 决定了分布的形状（像一个小山坡），
        # 乘 0.6 减 0.05，是为了把数值拉伸平移到大致 [-0.05, 0.55] 的合理区间内，看起来更像真实水体的 NDCI。
        ndci = rng.beta(2.2, 3.0, n_grids) * 0.6 - 0.05

        # 创建这一期的数据大表（DataFrame，类似 Excel 表）
        df = pd.DataFrame({
            # 用 np.arange 生成从 0 到 1499 的连续整数，作为网格的唯一编号
            "grid_id": np.arange(n_grids),
            "period": p,  # 当前是第几期（就是当前的循环变量 p）
            "date": f"2027-0{p + 3}-15",  # 生成假日期，比如 p=1 时，日期是 2027-04-15
            "row": np.arange(n_grids) // 50,  # 整除 50，模拟空间上每行 50 个格子，计算行号
            "col": np.arange(n_grids) % 50,  # 取模 50，模拟空间上每列 50 个格子，计算列号
            "block": (np.arange(n_grids) // 300) + 1,  # 每 300 个格子分一个区，用于后面的“空间块验证”，防止作弊

            # rng.uniform(下限, 上限, 数量)：生成均匀分布的随机数，模拟空间坐标 X 和 Y
            "x": rng.uniform(500000, 505000, n_grids),
            "y": rng.uniform(3420000, 3425000, n_grids),

            "ndci_mean": ndci,  # 把上面算好的 ndci 放进来作为核心特征
            # rng.uniform：生成均匀分布的随机数。模拟 NDCI 的波动范围（标准差）
            "ndci_std": rng.uniform(0.005, 0.03, n_grids),
            # rng.normal(平均值, 标准差, 数量)：生成正态分布（钟形曲线）的随机数，用于占位分位数
            "ndci_p80": ndci + rng.normal(0, 0.01, n_grids),

            "red_mean": rng.uniform(0.02, 0.06, n_grids),  # 模拟红光反射率
            "re_mean": rng.uniform(0.05, 0.12, n_grids),  # 模拟红边反射率

            # 模拟气温。22 + p 表示随着期数增加，整体温度稍微变暖。2.0 是波动的标准差。
            "temp": rng.normal(22 + p, 2.0, n_grids),
            # rng.gamma(形状参数, 尺度参数, 数量)：生成伽马分布随机数。常用于模拟降水、风速（因为不可能为负数）
            "precip": rng.gamma(1.5, 2.0, n_grids),  # 模拟当日降水
            "wind": rng.gamma(2.0, 0.8, n_grids),  # 模拟当日风速
            "elevation": rng.normal(180, 15, n_grids),  # 模拟高程（正态分布）
            "slope": rng.uniform(0, 8, n_grids),  # 模拟坡度（均匀分布）
        })

        frames.append(df)  # 把这一期造好的表放进“筐”里

    # 调用 pandas 库 (pd) 的 concat 函数。
    # 把列表 frames 里的 4 张表上下拼接到一起，变成一张超级大表。
    # ignore_index=True 表示重新从 0 开始排列行号，避免原表行号乱掉。
    df = pd.concat(frames, ignore_index=True)

    # 🚨 人为制造物理规律（让 AI 有规律可学）：
    # 公式设定：基础 NDCI 为 0.22。温度每升高 1 度，NDCI 增加 0.02；降水每增加 1，NDCI 减少 0.004。
    # rng.normal(0, 0.03, len(df)) 是在这个完美的公式上，撒上一把“随机噪音”，防止数据太完美。
    df["ndci_mean"] = (0.22 + 0.02 * (df["temp"] - 22) - 0.004 * df["precip"]
                       + rng.normal(0, 0.03, len(df)))

    # 重新计算 ndci_p80，确保它和新算出来的 ndci_mean 对应。
    df["ndci_p80"] = df["ndci_mean"] + rng.normal(0, 0.01, len(df))

    # 模拟前 3 天和 10 天的累积降水、气温。
    # 真实情况是用历史数据滚动累加，这里为了演示直接乘一个倍数假装算好了。
    df["precip_3d"] = df["precip"] * 2.6
    df["precip_10d"] = df["precip"] * 8.5
    df["temp_3d"] = df["temp"].rolling(1).mean()  # 占位：单日均温
    df["temp_10d"] = df["temp"].rolling(1).mean()  # 占位

    # 🚨 排序：因为要用时间序列做预测，所以必须把同一个网格按时间顺序排好。
    # 先按 grid_id 排，同一个 grid 里再按 period 排。reset_index 重置行号。
    df = df.sort_values(["grid_id", "period"]).reset_index(drop=True)

    # 🚨 核心关键：引入“时间惯性（自相关）”。
    # 现实中，昨天水华多，今天大概率也多。为了让“笨办法（基线）”有效，我们必须模拟这种惯性。
    # groupby("grid_id")：把同一个网格的数据分为一组。
    # transform：对这组数据应用函数，但保持原表结构不变。
    # ewm(alpha=0.6).mean()：指数加权移动平均。相当于让第 2 期吸收了 40% 的第 1 期特征，让状态有延续性。
    df["ndci_mean"] = df.groupby("grid_id")["ndci_mean"].transform(
        lambda s: s.ewm(alpha=0.6).mean())

    # 再对齐一次 ndci_p80
    df["ndci_p80"] = df["ndci_mean"] + rng.normal(0, 0.01, len(df))

    # 调用外部模块 label_generator.py 里的 add_labels 函数。
    # 作用：根据 ndci_mean 的数值大小，按分位数（比如 80% 分位）切片，打上 0 或 1 的标签。
    # 结果会在表里新增 y70, y75, y80 等几列。
    df = label_generator.add_labels(df, "ndci_mean")

    # 🚨 画龙点睛的一笔：制作标准答案（预测目标）。
    # groupby("grid_id")：按网格分组。
    # shift(-1)：把数据向上挪一行（把下一期第 4 期的状态，挪到第 3 期这一行）。
    # 这样，"y_next" 这一列装的就是“明天（下一期）”的真实状态，用来给今天的模型当预测目标。
    df["y_next"] = df.groupby("grid_id")["y80"].shift(-1)

    # 因为最后一期（比如第 4 期）执行 shift(-1) 后没有下一期的数据了，会变成空值（NaN）。
    # dropna：把所有 y_next 为 NaN 的行删掉，因为模型不能学习没有标准答案的数据。
    df = df.dropna(subset=["y_next"]).reset_index(drop=True)

    # 打印日志，告诉你造了多少行、多少列的数据。
    log.info("合成数据就绪：%d 行 × %d 列", *df.shape)

    # 把这张完美的“假大表”返回给 main.py
    return df


# ==========================================================================
# 二、训练与验证：留一期交叉验证 + 基线对照 + bootstrap
# ==========================================================================
def run_validation(df: pd.DataFrame, label_col: str = "y_next",
                   base_col: str = "y80") -> tuple[pd.DataFrame, dict]:
    """执行留一期验证，返回各折结果表与最后一折的训练产物。"""

    # 自动挑出用来做预测的列（比如气温、降水、NDCI等），剔除 ID、日期等无关列
    features = data_fusion.feature_columns(df, label_col)

    # 执行“留一期交叉验证”：比如 用 1,2,3 期训练预测第 4 期；用 1,2,4 期训练预测第 3 期...
    folds = data_splitter.lopo_folds(df)
    if not folds:
        log.error("期次不足，无法验证")
        return pd.DataFrame(), {}

    rows, last = [], {}                                    # rows 存每次考试的分数表，last 存最后一次训练的模型
    for k, (tr, te) in enumerate(folds, start=1):          # 逐折开始“考试”
        train_df, test_df = df.iloc[tr], df.iloc[te]       # 切分训练集和测试集

        # 把表格拆成 X（特征大表）和 y（标准答案）
        X_tr, y_tr = ml_models.prepare_xy(train_df, features, label_col)
        X_te, y_te = ml_models.prepare_xy(test_df, features, label_col)

        rf = ml_models.train_rf(X_tr, y_tr)                # 训练随机森林（AI去学习）
        p_rf = ml_models.predict_proba(rf, X_te)           # AI交卷，输出每个格子的风险概率（0~1）

        # ⚠️ 笨办法（基线）：直接抄上一期的答案（y80），作为预测结果
        p_base = df[base_col].iloc[te].to_numpy()
        mask = ~np.isnan(p_base.astype(float))             # 剔除空值，只保留有基准数据的行

        row = {"fold": k, "valid_period": int(test_df["period"].iloc[0]),
               "n_train": len(train_df), "n_test": len(test_df)}  # 记录基本考试信息

        # 只有测试样本够多（>=30），且有正负两类标签时，才计算 AUC
        if mask.sum() >= 30 and len(np.unique(y_te[mask])) > 1:
            # 算 AI 的考试成绩（AUC等）
            met_rf = evaluation.classification_metrics(y_te[mask], p_rf[mask])
            # ⚠️ 重点：用 Bootstrap 重采样算 AI 和笨办法的分数差值（ΔAUC）的 95% 置信区间
            boot = evaluation.paired_bootstrap(y_te[mask], p_rf[mask], p_base[mask])

            row["AUC_RF"] = met_rf["AUC"]                   # 记录 AI 的 AUC
            row["AUC_base"] = evaluation.classification_metrics(
                y_te[mask], p_base[mask])["AUC"]            # 记录笨办法的 AUC
            row["delta_auc"] = boot["delta"]                # 记录两者的差值
            row["ci_low"] = boot["ci_low"]                  # 记录置信区间下界（＞0说明AI真的有用）
            row["ci_high"] = boot["ci_high"]                # 记录置信区间上界
            row["verdict"] = evaluation.judge_increment(boot)  # 自动判定“AI有用”还是“AI没用”
        else:                                               # 数据不够时打标记
            row["note"] = "基线样本不足，未计算 ΔAUC"

        rows.append(row)                                    # 收集这次考试的成绩

        # 把最后一次训练的模型和测试数据保存下来，后面画图要用
        last = {"model": rf, "X_test": X_te, "y_test": y_te, "p_rf": p_rf,
                "features": features, "test_df": test_df}

    result = evaluation.summarize_folds(rows)               # 把所有折的成绩汇总成一张表
    return result, last


# ==========================================================================
# 三、主流程调度器
# ==========================================================================
def run_demo() -> None:
    """完整演示：造数 → 验证 → 出表出图 → 情景模拟。"""
    config.ensure_dirs()                                     # 自动检查并创建所需要的文件夹（比如 output）
    log_config(log)                                          # 记录这次跑的参数

    df = make_demo_dataset()                                 # 1. 造数据
    # 把大表存成 CSV，方便你用 Excel 打开检查
    df.to_csv(config.DATA_PROCESSED / "dataset.csv", index=False, encoding="utf-8-sig")

    # 2. 跑模型验证（留一期 + 基线）
    folds_df, last = run_validation(df, "y_next")
    if len(folds_df):                                        # 如果有结果，就存下成绩单
        folds_df.to_csv(config.OUT_TAB / "validation_folds.csv", index=False, encoding="utf-8-sig")
        log.info("各折结果：\n%s", folds_df)

    if last:                                                 # 3. 只要有模型产物，就开始画图和模拟
        # 3.1 算特征重要性（看谁排第一）
        imp = ml_models.rf_importance(last["model"], last["features"])
        pd.DataFrame({"feature": list(imp), "importance": list(imp.values())}) \
            .sort_values("importance", ascending=False) \
            .to_csv(config.OUT_TAB / "feature_importance.csv", index=False, encoding="utf-8-sig")

        # 3.2 画图（统一样式，画重要性图）
        plot_utils.setup_style()
        plot_utils.plot_importance(imp, "feature_importance.png")

        # 3.3 制作风险概率空间分布图（把概率画到空间坐标上）
        test_df = last["test_df"].copy()
        test_df["prob"] = last["p_rf"]                       # 把概率附加到大表里
        risk_mapping.draw_map(test_df, "prob", "risk_map.png")  # 生成红绿渐变的图
        # 把预测概率表存成 CSV
        test_df[["grid_id", "period", "x", "y", "prob"]].to_csv(
            config.OUT_TAB / "test_probabilities.csv", index=False, encoding="utf-8-sig")

        # 3.4 情景模拟（假如明天升温2度，降雨减半会怎样？）
        sc = scenario_simulation.run_scenarios(last["model"], last["X_test"], last["features"])
        sc.to_csv(config.OUT_TAB / "scenario_results.csv", index=False, encoding="utf-8-sig")
        log.info("情景模拟结果：\n%s", sc)

    log.info("demo 流程结束，结果见 outputs/")               # 全部跑完


def run_gee() -> None:
    """阶段一：检查 GEE 是否能正常取数（不导出）。"""
    from src import gee_extract
    gee_extract.test_pipeline()                              # 调用 GEE 自检函数


def run_uav(path: str) -> None:
    """阶段二：预处理一张无人机五波段影像。"""
    from src import uav_preprocess
    out = config.DATA_INTERIM / "uav_L1_mask.tif"             # 设定输出路径
    uav_preprocess.process_uav(path, str(out))               # 执行无人机影像裁剪、去陆地、算反射率
    log.info("预处理完成：%s", out)


# ==========================================================================
# 四、总开关：决定执行哪个阶段
# ==========================================================================
def main() -> None:
    """解析命令行参数并分派任务。"""
    # 这一块是用来接收你在终端里敲的命令的。比如 `python main.py --stage demo`
    parser = argparse.ArgumentParser(description="无人机水华风险预警项目主程序")
    parser.add_argument("--stage", default="demo",
                        choices=["demo", "gee", "uav", "all"], help="要执行的阶段")
    parser.add_argument("--uav", default="", help="uav 阶段输入的 5 波段影像路径")
    args = parser.parse_args()                               # 解析你的命令

    config.ensure_dirs()                                     # 统一建好文件夹
    log_config(log)                                          # 记录参数

    # 根据你敲的命令，决定运行哪部分
    if args.stage in ("demo", "all"):                        # 如果输入了 demo
        run_demo()                                           # 跑全流程演示
    if args.stage in ("gee", "all"):                         # 如果输入了 gee
        run_gee()                                            # 去连 Google 服务器
    if args.stage in ("uav", "all") and args.uav:            # 如果输入了 uav 并且给了图片路径
        run_uav(args.uav)                                    # 处理无人机照片


if __name__ == "__main__":                                   # 整个程序的总入口
    main()