#!/bin/bash
# Docker 训练环境构建和运行脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Browser Control 本地训练环境 ==="
echo "项目目录: $PROJECT_DIR"

# 检查 OpenEnv 是否已克隆
check_openenv() {
    local dir=$1
    if [ ! -d "$dir/OpenEnv" ]; then
        echo "克隆 OpenEnv 到 $dir..."
        git clone https://github.com/meta-pytorch/OpenEnv.git "$dir/OpenEnv"
    else
        echo "OpenEnv 已存在: $dir/OpenEnv"
    fi
}

case "${1:-help}" in
    setup)
        echo "准备 OpenEnv 依赖..."
        check_openenv "$PROJECT_DIR/docker/training"
        check_openenv "$PROJECT_DIR/docker/browsergym"
        echo "完成！现在可以运行: $0 build"
        ;;

    build)
        echo "构建 Docker 镜像..."
        docker compose -f "$PROJECT_DIR/docker/docker-compose.training.yml" build
        echo "构建完成！"
        ;;

    start)
        echo "启动 BrowserGym 服务..."
        docker compose -f "$PROJECT_DIR/docker/docker-compose.training.yml" up browsergym -d
        echo "等待 BrowserGym 就绪..."
        sleep 10
        curl -s http://localhost:8080/health && echo "BrowserGym 已就绪" || echo "BrowserGym 尚未就绪，请稍候"
        ;;

    train)
        CONFIG="${2:-lfm2_350m_local_lora.yaml}"
        echo "开始训练（配置: $CONFIG）..."
        docker compose -f "$PROJECT_DIR/docker/docker-compose.training.yml" run --rm training \
            python3 -m src.browser_control.fine_tune_local \
            --config-file-name "$CONFIG"
        ;;

    stop)
        docker compose -f "$PROJECT_DIR/docker/docker-compose.training.yml" down
        echo "已停止所有容器"
        ;;

    logs)
        docker compose -f "$PROJECT_DIR/docker/docker-compose.training.yml" logs -f
        ;;

    tensorboard)
        echo "启动 TensorBoard: http://localhost:6006"
        tensorboard --logdir "$PROJECT_DIR/docker/logs" --port 6006
        ;;

    *)
        echo "用法: $0 {setup|build|start|train [config]|stop|logs|tensorboard}"
        echo ""
        echo "  setup       - 克隆 OpenEnv 依赖"
        echo "  build       - 构建 Docker 镜像"
        echo "  start       - 启动 BrowserGym 服务"
        echo "  train       - 运行训练（默认配置: lfm2_350m_local_lora.yaml）"
        echo "  stop        - 停止所有容器"
        echo "  logs        - 查看容器日志"
        echo "  tensorboard - 启动 TensorBoard"
        ;;
esac
