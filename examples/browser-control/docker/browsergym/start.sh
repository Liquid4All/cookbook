#!/bin/bash
# BrowserGym环境启动脚本（使用OpenEnv服务器）

set -e

echo "Starting BrowserGym Local Server (OpenEnv)..."

# 启动虚拟显示（Chromium无头模式需要）
# 使用 || true 避免 Xvfb 已存在时导致 set -e 退出
echo "Starting Xvfb virtual display..."
rm -f /tmp/.X99-lock 2>/dev/null || true
Xvfb :99 -screen 0 1280x1024x24 &
export DISPLAY=:99

# 等待Xvfb启动
sleep 2

# 决定 MiniWoB HTML 来源：
# - 若环境变量 MINIWOB_URL 已指向外部服务（非 localhost:9000），直接使用
# - 否则启动本地端口 9000 静态服务器作为后备
if [[ "${MINIWOB_URL}" == *"localhost:9000"* ]] || [[ -z "${MINIWOB_URL}" ]]; then
    echo "Starting MiniWoB static file server on http://0.0.0.0:9000"
    cd /app/miniwob-plusplus/miniwob/html
    nohup python3 -m http.server 9000 >/tmp/miniwob_http.log 2>&1 &
    MINIWOB_HTTP_PID=$!
    echo "MiniWoB HTTP server PID: $MINIWOB_HTTP_PID"

    # 等待静态服务器启动并验证
    sleep 3
    if curl -sf http://localhost:9000/ >/dev/null 2>&1; then
        echo "MiniWoB HTTP server is ready on port 9000"
    else
        echo "WARNING: MiniWoB HTTP server may not be ready, continuing anyway..."
        cat /tmp/miniwob_http.log || true
    fi

    # 使用本地服务器
    export MINIWOB_URL="http://localhost:9000/miniwob/"
else
    echo "Using external MiniWoB server: MINIWOB_URL=${MINIWOB_URL}"
fi

# 设置Python路径包含OpenEnv
export PYTHONPATH=/app/OpenEnv/src:/app/OpenEnv/envs:$PYTHONPATH

# 启动OpenEnv BrowserGym服务器（继承上面 export 的 MINIWOB_URL）
echo "Starting OpenEnv BrowserGym server on http://0.0.0.0:8000"
echo "MINIWOB_URL=${MINIWOB_URL}"
cd /app
exec python -m browsergym_env.server.app
