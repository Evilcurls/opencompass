#!/bin/bash
# ================================================
#  一键运行评测 + 飞书通知
#  用法: bash tools/run_with_notify.sh eval_qwen35_2b_dpo.py
# ================================================
set -e

CONFIG=$1
if [ -z "$CONFIG" ]; then
    echo "用法: bash tools/run_with_notify.sh <config_file>"
    echo "示例: bash tools/run_with_notify.sh eval_qwen35_2b_dpo.py"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOL="$SCRIPT_DIR/report_to_feishu.py"

echo "========================================="
echo "  🚀 Starting Evaluation with Notify"
echo "  Config: $CONFIG"
echo "========================================="

# 1. 发送开始通知
echo "[1/3] Sending start notification..."
python "$TOOL" --mode start --config "$CONFIG"

# 2. 运行评测
echo "[2/3] Running evaluation..."
START_TIME=$(date +%s)
python run.py "$CONFIG"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "[INFO] Evaluation took ${ELAPSED}s"

# 3. 发送结束报告
echo "[3/3] Sending end report..."
python "$TOOL" --mode end --auto-scan

echo "========================================="
echo "  ✅ All Done! (${ELAPSED}s)"
echo "========================================="
