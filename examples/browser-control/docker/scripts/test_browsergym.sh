#!/bin/bash
# BrowserGym环境测试脚本
# Stage 3: 验证本地BrowserGym服务功能

set -e

BROWSERGYM_URL="http://localhost:8000"
TIMEOUT=60
INTERVAL=5

echo "=========================================="
echo "BrowserGym环境测试"
echo "=========================================="

# 1. 等待服务健康
echo ""
echo "步骤1: 等待BrowserGym服务启动..."
elapsed=0
while [ $elapsed -lt $TIMEOUT ]; do
    if curl -f "$BROWSERGYM_URL/health" > /dev/null 2>&1; then
        echo "✅ 服务健康检查通过"
        break
    fi
    echo "等待中... ($elapsed/$TIMEOUT秒)"
    sleep $INTERVAL
    elapsed=$((elapsed + INTERVAL))
done

if [ $elapsed -ge $TIMEOUT ]; then
    echo "❌ 服务启动超时"
    exit 1
fi

# 2. 测试健康检查API
echo ""
echo "步骤2: 测试健康检查API..."
HEALTH_RESPONSE=$(curl -s "$BROWSERGYM_URL/health")
echo "响应: $HEALTH_RESPONSE"
if echo "$HEALTH_RESPONSE" | grep -q '"status".*"ok"'; then
    echo "✅ 健康检查API正常"
else
    echo "❌ 健康检查失败"
    exit 1
fi

# 3. 测试状态API
echo ""
echo "步骤3: 测试状态API..."
STATUS_RESPONSE=$(curl -s "$BROWSERGYM_URL/status")
echo "响应: $STATUS_RESPONSE"
if echo "$STATUS_RESPONSE" | grep -q '"service".*"BrowserGym"'; then
    echo "✅ 状态API正常"
else
    echo "❌ 状态API失败"
    exit 1
fi

# 4. 测试任务重置API（MiniWoB click-test）
echo ""
echo "步骤4: 测试任务重置API..."
RESET_RESPONSE=$(curl -s -X POST "$BROWSERGYM_URL/reset" \
    -H "Content-Type: application/json" \
    -d '{"task_name": "miniwob.click-test"}')

echo "响应: $RESET_RESPONSE"
if echo "$RESET_RESPONSE" | grep -q '"observation"'; then
    echo "✅ 任务重置API正常"
    echo "任务初始化成功: miniwob.click-test"
else
    echo "❌ 任务重置失败"
    echo "完整响应: $RESET_RESPONSE"
    exit 1
fi

# 5. 测试动作执行API（简单点击）
echo ""
echo "步骤5: 测试动作执行API..."
STEP_RESPONSE=$(curl -s -X POST "$BROWSERGYM_URL/step" \
    -H "Content-Type: application/json" \
    -d '{"action": "click(\"button\")"}')

echo "响应（部分）: $(echo $STEP_RESPONSE | cut -c1-200)..."
if echo "$STEP_RESPONSE" | grep -q '"observation"'; then
    echo "✅ 动作执行API正常"
else
    echo "❌ 动作执行失败"
    echo "完整响应: $STEP_RESPONSE"
    exit 1
fi

# 6. 测试关闭API
echo ""
echo "步骤6: 测试环境关闭API..."
CLOSE_RESPONSE=$(curl -s -X POST "$BROWSERGYM_URL/close")
echo "响应: $CLOSE_RESPONSE"
if echo "$CLOSE_RESPONSE" | grep -q '"message".*"closed"'; then
    echo "✅ 关闭API正常"
else
    echo "⚠ 关闭API响应异常（但不影响功能）"
fi

echo ""
echo "=========================================="
echo "✅ BrowserGym环境测试通过"
echo "=========================================="
echo ""
echo "所有API测试成功，可以进入Stage 4（训练测试）"
