#!/bin/bash
# 从Docker volume复制检查点到宿主机（替代Modal volume get）

CHECKPOINT_NAME=$1
OUTPUT_DIR=${2:-../checkpoints}

if [ -z "$CHECKPOINT_NAME" ]; then
    echo "用法: $0 <checkpoint-name> [output-dir]"
    echo "示例: $0 LFM2-350M-browsergym-20260223-120000"
    echo ""
    echo "可用的检查点列表:"
    docker run --rm \
        -v browser-control-checkpoints:/model_checkpoints:ro \
        ubuntu:22.04 \
        ls -1 /model_checkpoints 2>/dev/null || echo "  (无可用检查点)"
    exit 1
fi

echo "正在下载检查点: $CHECKPOINT_NAME"
echo "输出目录: $OUTPUT_DIR"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 从Docker volume复制到宿主机
docker run --rm \
    -v browser-control-checkpoints:/model_checkpoints:ro \
    -v "$(cd "$OUTPUT_DIR" && pwd)":/output \
    ubuntu:22.04 \
    bash -c "
    if [ ! -d /model_checkpoints/$CHECKPOINT_NAME ]; then
        echo '错误: 检查点不存在: $CHECKPOINT_NAME'
        echo '可用的检查点:'
        ls -1 /model_checkpoints
        exit 1
    fi
    cp -r /model_checkpoints/$CHECKPOINT_NAME /output/
    echo '复制完成'
    "

if [ $? -eq 0 ]; then
    echo "✓ 检查点已复制到: $OUTPUT_DIR/$CHECKPOINT_NAME"
    echo ""
    echo "文件列表:"
    ls -lh "$OUTPUT_DIR/$CHECKPOINT_NAME/"
else
    echo "✗ 下载失败"
    exit 1
fi
