#!/bin/bash
# 监控Docker构建进度并自动进入下一步

BUILD_LOG="/home/tony/project/cookbook/examples/browser-control/docker/build.log"
CHECK_INTERVAL=30  # 每30秒检查一次
MAX_WAIT=1200      # 最多等待20分钟

echo "=========================================="
echo "Docker构建进度监控"
echo "=========================================="
echo ""

elapsed=0
while [ $elapsed -lt $MAX_WAIT ]; do
    # 检查进程是否还在运行
    if ! pgrep -f "docker compose.*build" > /dev/null; then
        echo ""
        echo "构建进程已完成，检查结果..."
        break
    fi
    
    # 显示最新日志
    echo "[$elapsed秒] 构建进行中..."
    tail -n 3 "$BUILD_LOG" 2>/dev/null | grep -v "^$" | head -n 1
    
    sleep $CHECK_INTERVAL
    elapsed=$((elapsed + CHECK_INTERVAL))
done

# 检查镜像是否成功构建
echo ""
echo "=========================================="
echo "检查构建结果"
echo "=========================================="

if docker images | grep -q "docker-browsergym"; then
    echo "✅ BrowserGym镜像构建成功"
    BROWSERGYM_OK=1
else
    echo "❌ BrowserGym镜像构建失败"
    BROWSERGYM_OK=0
fi

if docker images | grep -q "docker-training"; then
    echo "✅ Training镜像构建成功"
    TRAINING_OK=1
else
    echo "❌ Training镜像构建失败"
    TRAINING_OK=0
fi

echo ""

if [ $BROWSERGYM_OK -eq 1 ] && [ $TRAINING_OK -eq 1 ]; then
    echo "=========================================="
    echo "✅ Stage 2 完成 - Docker镜像构建成功"
    echo "=========================================="
    echo ""
    echo "下一步: Stage 3 - BrowserGym环境测试"
    echo "执行命令: make docker-up-browsergym && make test-browsergym"
    exit 0
else
    echo "=========================================="
    echo "❌ 构建失败，请检查日志"
    echo "=========================================="
    echo "日志位置: $BUILD_LOG"
    exit 1
fi
