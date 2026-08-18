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

REM pytest模式: 标准测试运行
echo ========================================
echo   pytest标准测试模式
echo ========================================
echo.

REM 清除环境变量
set LEARNING_MODE=0

REM 运行pytest
uv run pytest tests\ -v --tb=short

echo.
echo 测试完成！
pause
