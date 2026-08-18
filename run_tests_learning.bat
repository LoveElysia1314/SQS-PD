@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"

where uv.exe >nul 2>nul
if errorlevel 1 (
    echo ERROR: uv was not found in PATH.
    echo Install it with: winget install --id astral-sh.uv --exact
    pause
    exit /b 1
)

REM 教学模式: 直接运行Python脚本，输出详细说明
echo ========================================
echo   教学模式 - 详细输出
echo ========================================
echo.

REM 设置教学模式环境变量
set LEARNING_MODE=1

echo [1/5] 测试01: 超胞基础与LCMM投影一致性
uv run python tests\test_01_supercell_basics.py
echo.

echo [2/5] 测试02: sqsgenerator集成与关键语义
uv run python tests\test_02_sqsgenerator_api.py
echo.

echo [3/5] 测试03: 项目API工作流与自动超胞策略
uv run python tests\test_03_critical_semantics.py
echo.

echo [4/5] 测试04: CIF无序分析与dry-run优化
uv run python tests\test_04_project_api.py
echo.

echo [5/5] 测试05: 批量入口与pipeline路径一致性
uv run python tests\test_05_cif_disorder_analysis.py
echo.

echo ========================================
echo   所有教学测试完成！
echo ========================================
pause
