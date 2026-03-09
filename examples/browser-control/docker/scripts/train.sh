#!/bin/bash
# 训练启动脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Browser Control本地训练启动脚本 ===${NC}"

# 配置文件（可通过环境变量传入）
CONFIG_FILE=${CONFIG_FILE:-lfm2_350m_local.yaml}

# 1. 检查NVIDIA GPU和Docker环境
echo -e "${YELLOW}[1/5] 检查NVIDIA GPU和Docker环境...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}错误: nvidia-smi未找到，请确认已安装NVIDIA驱动${NC}"
    exit 1
fi

# 2. 检查docker compose
if ! docker compose version &> /dev/null; then
    echo -e "${RED}错误: docker compose未找到${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 环境检查通过${NC}"

# 3. 启动BrowserGym环境
echo -e "${YELLOW}[2/5] 启动BrowserGym环境...${NC}"
docker compose -f docker-compose.training.yml up -d browsergym

# 等待健康检查
echo "等待BrowserGym就绪..."
for i in {1..30}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ BrowserGym已就绪${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}错误: BrowserGym启动超时${NC}"
        docker compose -f docker-compose.training.yml logs browsergym
        exit 1
    fi
    sleep 2
done

# 4. 启动训练（前台运行）
echo -e "${YELLOW}[3/5] 启动GRPO训练...${NC}"
echo "配置文件: ${CONFIG_FILE}"
echo "日志目录: ./logs"
echo "检查点目录: docker volume 'browser-control-checkpoints'"
echo ""

# 修改docker-compose配置中的config文件名
export CONFIG_FILE
docker compose -f docker-compose.training.yml up training

# 5. 训练完成提示
echo -e "${YELLOW}[4/5] 训练完成${NC}"
echo "查看检查点: docker exec -it lfm2-grpo-training ls -lh /model_checkpoints"
echo "启动TensorBoard: make tensorboard"
echo "清理环境: make docker-down"
