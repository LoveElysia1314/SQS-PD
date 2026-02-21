@echo off
REM pytest模式: 标准测试运行
echo ========================================
echo   pytest标准测试模式
echo ========================================
echo.

REM 清除环境变量
set LEARNING_MODE=0

REM 运行pytest
pytest tests\ -v --tb=short

echo.
echo 测试完成！
pause
