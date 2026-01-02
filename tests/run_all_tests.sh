#!/bin/bash

# 定义日志目录
LOG_DIR="test_logs"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "SNNTorch 批量测试脚本"
echo "环境: conda env 'SNN'"
echo "日志目录: $(pwd)/$LOG_DIR"
echo "========================================"

# 查找所有 test_*.py 和 verify_*.py 文件
TEST_FILES=$(ls test_*.py verify_*.py 2>/dev/null)

if [ -z "$TEST_FILES" ]; then
    echo "未找到测试文件 (test_*.py 或 verify_*.py)"
    exit 1
fi

COUNT=0
PASS_COUNT=0
FAIL_COUNT=0

for file in $TEST_FILES; do
    echo -n "正在运行 $file ... "
    
    # 构造日志文件名
    LOG_FILE="$LOG_DIR/${file}.log"
    
    # 使用 conda run -n SNN 运行 python 脚本
    # 既保证环境正确，又捕获所有输出
    conda run -n SNN python "$file" > "$LOG_FILE" 2>&1
    
    RET_CODE=$?
    
    if [ $RET_CODE -eq 0 ]; then
        echo "PASS"
        ((PASS_COUNT++))
    else
        echo "FAIL (查看日志: $LOG_FILE)"
        ((FAIL_COUNT++))
    fi
    
    ((COUNT++))
done

echo "========================================"
echo "测试完成."
echo "总计: $COUNT"
echo "通过: $PASS_COUNT"
echo "失败: $FAIL_COUNT"
echo "详细日志已保存在 $LOG_DIR 目录下"
echo "========================================"
