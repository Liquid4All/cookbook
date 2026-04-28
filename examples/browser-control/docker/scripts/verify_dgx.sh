#!/bin/bash
# DGX SPARK B10环境验证脚本

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  DGX SPARK B10 环境检查${NC}"
echo -e "${BLUE}========================================${NC}"

# GPU信息
echo -e "\n${YELLOW}[1/7] GPU配置${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ nvidia-smi 可用${NC}"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv
    
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
    echo -e "\n${GREEN}检测到 ${GPU_COUNT} 个GPU${NC}"
else
    echo -e "${RED}✗ nvidia-smi 未找到，请确认已安装NVIDIA驱动${NC}"
    exit 1
fi

# CUDA版本
echo -e "\n${YELLOW}[2/7] CUDA版本${NC}"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
else
    echo -e "${YELLOW}⚠ nvcc未安装（仅运行时不影响，CUDA工具包可选）${NC}"
fi

# 驱动版本
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
echo -e "NVIDIA Driver: ${GREEN}${DRIVER_VERSION}${NC}"

if [ "$(echo "$DRIVER_VERSION >= 530" | bc -l 2>/dev/null || echo "0")" == "1" ]; then
    echo -e "${GREEN}✓ 驱动版本满足要求 (≥530)${NC}"
else
    echo -e "${RED}✗ 驱动版本过低，需要 ≥530 (CUDA 13.0要求)${NC}"
fi

# Docker版本
echo -e "\n${YELLOW}[3/7] Docker版本${NC}"
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker已安装${NC}"
    docker --version
    docker compose version 2>/dev/null || docker-compose --version 2>/dev/null || echo -e "${RED}✗ Docker Compose未找到${NC}"
else
    echo -e "${RED}✗ Docker未安装${NC}"
    echo "安装方法: https://docs.docker.com/engine/install/"
    exit 1
fi

# NVIDIA Container Toolkit
echo -e "\n${YELLOW}[4/7] NVIDIA Container Toolkit${NC}"
if docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA Container Toolkit 正常工作${NC}"
    echo "测试命令: docker run --rm --gpus all nvidia/cuda:13.0.0-base nvidia-smi"
else
    echo -e "${RED}✗ NVIDIA Container Toolkit 未正确配置${NC}"
    echo "安装方法: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo ""
    echo "快速安装命令:"
    echo "  distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
    echo "  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -"
    echo "  curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list"
    echo "  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
    echo "  sudo systemctl restart docker"
    exit 1
fi

# 磁盘空间（需要至少100GB）
echo -e "\n${YELLOW}[5/7] 磁盘空间${NC}"
DISK_AVAIL=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
echo "当前目录可用空间: ${DISK_AVAIL}GB"

if [ "$DISK_AVAIL" -ge 100 ]; then
    echo -e "${GREEN}✓ 磁盘空间充足 (≥100GB)${NC}"
else
    echo -e "${RED}✗ 磁盘空间不足，建议至少100GB可用空间${NC}"
fi

# 系统内存
echo -e "\n${YELLOW}[6/7] 系统内存${NC}"
TOTAL_MEM=$(free -g | grep Mem | awk '{print $2}')
AVAIL_MEM=$(free -g | grep Mem | awk '{print $7}')
echo "总内存: ${TOTAL_MEM}GB, 可用: ${AVAIL_MEM}GB"

if [ "$TOTAL_MEM" -ge 32 ]; then
    echo -e "${GREEN}✓ 系统内存充足 (≥32GB)${NC}"
else
    echo -e "${YELLOW}⚠ 系统内存偏低，推荐≥32GB${NC}"
fi

# Python环境
echo -e "\n${YELLOW}[7/7] Python环境${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ Python已安装: ${PYTHON_VERSION}${NC}"
else
    echo -e "${RED}✗ Python3未安装${NC}"
fi

# 总结和推荐配置
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  推荐配置建议${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "\n${GREEN}针对当前环境的配置建议:${NC}"
echo ""
echo "1. 如果有多GPU，修改 docker/docker-compose.training.yml:"
echo "   ${YELLOW}NVIDIA_VISIBLE_DEVICES: all${NC}  # 或指定GPU ID如 '0,1'"
echo ""
echo "2. vLLM GPU内存利用率调优 (根据GPU型号):"
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "   当前GPU: ${GREEN}${GPU_MODEL}${NC}"

if [[ $GPU_MODEL == *"A100"* ]]; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    if [ "$GPU_MEM" -ge 70000 ]; then
        echo "   推荐: ${YELLOW}vllm_gpu_memory_utilization: 0.05${NC} (A100 80GB)"
    else
        echo "   推荐: ${YELLOW}vllm_gpu_memory_utilization: 0.10${NC} (A100 40GB)"
    fi
elif [[ $GPU_MODEL == *"4090"* ]] || [[ $GPU_MODEL == *"RTX 4090"* ]]; then
    echo "   推荐: ${YELLOW}vllm_gpu_memory_utilization: 0.15 + use_peft: true${NC} (需LoRA)"
else
    echo "   推荐: ${YELLOW}vllm_gpu_memory_utilization: 0.10${NC} (通用配置)"
fi

echo ""
echo "3. 数据并行 (多GPU训练):"
echo "   需修改 fine_tune_local.py 添加 DistributedDataParallel"
echo ""

echo -e "\n${GREEN}环境验证完成！${NC}"
echo "下一步: make docker-build (构建Docker镜像)"
