@echo off
chcp 65001 >nul
cd /d %~dp0
echo ============================================
echo  无人机水华预警项目 - 一键演示
echo ============================================

if not exist .venv (
    echo [1/3] 正在创建虚拟环境...
    python -m venv .venv
)

echo [2/3] 安装最小依赖...
call .venv\Scripts\activate
python -m pip install --upgrade pip >nul
pip install numpy pandas scikit-learn matplotlib

echo [3/3] 运行演示流程...
python main.py --stage demo

echo.
echo 运行结束，结果见 outputs\figures 与 outputs\tables
pause
