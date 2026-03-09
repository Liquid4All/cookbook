"""
本地GGUF转换脚本 - 使用CUDA 13.0 (ARM64 + GB10)
基于DGX Spark论坛方案：https://forums.developer.nvidia.cn/t/dgx-spark-vllm-cuda13/28377
"""

import subprocess
import sys
from pathlib import Path


def check_cuda():
    """检查CUDA环境"""
    print("=== 检查CUDA环境 ===")
    try:
        import torch

        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"计算能力: {torch.cuda.get_device_capability(0)}")
            return True
        else:
            print("❌ CUDA不可用")
            return False
    except ImportError:
        print("❌ PyTorch未安装")
        return False


def convert_to_gguf():
    """使用llama.cpp转换HuggingFace checkpoint为GGUF"""

    # 配置路径
    checkpoint_dir = Path("checkpoints/LFM2-350M-browsergym-20260227-031350")
    output_dir = Path("gguf_models_local")
    llama_cpp_dir = Path("llama.cpp")

    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint不存在: {checkpoint_dir}")
        return False

    if not llama_cpp_dir.exists():
        print(f"❌ llama.cpp目录不存在: {llama_cpp_dir}")
        print(
            "请先运行: git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build -j"
        )
        return False

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== 开始转换 ===")
    print(f"输入: {checkpoint_dir}")
    print(f"输出: {output_dir}\n")

    # Step 1: HuggingFace → FP16 GGUF
    print("=== Step 1: 转换为FP16 GGUF ===")
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    fp16_output = output_dir / "lfm2-350m-browsergym-fp16.gguf"

    cmd = [
        sys.executable,
        str(convert_script),
        str(checkpoint_dir),
        "--outfile",
        str(fp16_output),
        "--outtype",
        "f16",
    ]

    print(f"运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
    print(result.stdout)
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        return False

    # Step 2-4: 量化
    quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"

    if not quantize_bin.exists():
        print(f"❌ llama-quantize不存在: {quantize_bin}")
        print(
            "请先编译llama.cpp: cd llama.cpp && cmake -B build && cmake --build build -j"
        )
        return False

    quantizations = [
        ("Q8_0", "q8_0"),
        ("Q5_K_M", "q5_k_m"),
        ("Q4_K_M", "q4_k_m"),
    ]

    for quant_method, output_suffix in quantizations:
        print(f"\n=== Step: 量化为 {quant_method} ===")
        quant_output = output_dir / f"lfm2-350m-browsergym-{output_suffix}.gguf"

        cmd = [str(quantize_bin), str(fp16_output), str(quant_output), quant_method]

        print(f"运行: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
        print(result.stdout)
        if result.returncode != 0:
            print(f"错误: {result.stderr}")
            return False

    # 显示结果
    print("\n=== 转换完成 ===")
    print("\n生成的文件:")
    for file in output_dir.glob("*.gguf"):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {file.name}: {size_mb:.2f} MB")

    return True


def main():
    print("=" * 60)
    print("本地GGUF转换 - DGX Spark (CUDA 13.0 + ARM64)")
    print("=" * 60)

    # 检查CUDA
    if not check_cuda():
        print("\n⚠️ 警告: CUDA不可用，使用CPU进行转换")

    print()

    # 转换
    if convert_to_gguf():
        print("\n✅ 所有GGUF文件已成功生成!")
        print(f"\n文件位置: gguf_models_local/")
        print("\n下一步:")
        print("1. 使用llama.cpp验证:")
        print(
            "   ./llama.cpp/build/bin/llama-cli -m gguf_models_local/lfm2-350m-browsergym-q8_0.gguf -p 'test' -n 20"
        )
        print("\n2. 部署到Android (参考 docs/browser-control-android-deploy.md)")
    else:
        print("\n❌ 转换失败，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()
