@echo off
REM 运行单个测试文件
if "%1"=="" (
    echo 用法: run_single_test.bat ^<test_number^> [mode]
    echo.
    echo 示例:
    echo   run_single_test.bat 01          - pytest模式运行测试01
    echo   run_single_test.bat 01 learn    - 教学模式运行测试01
    echo.
    echo 可用测试:
    echo   01 - 超胞基础与LCMM投影一致性
    echo   02 - sqsgenerator集成与关键语义
    echo   03 - 项目API工作流与自动超胞策略
    echo   04 - CIF无序分析与dry-run优化
    echo   05 - 批量入口与pipeline路径一致性
    pause
    exit /b
)

set TEST_NUM=%1
set MODE=%2

if "%MODE%"=="learn" (
    set LEARNING_MODE=1
    python tests\test_%TEST_NUM%*.py
) else (
    set LEARNING_MODE=0
    pytest tests\test_%TEST_NUM%*.py -v -s
)

pause
