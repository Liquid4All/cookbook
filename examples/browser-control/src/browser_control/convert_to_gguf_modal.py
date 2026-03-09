"""
在Modal GPU上将HuggingFace checkpoint转换为GGUF格式
"""
import modal
from pathlib import Path

app = modal.App("convert-gguf")

# 使用与训练相同的镜像，并预装llama.cpp所需的系统依赖
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "libcurl4-openssl-dev", "cmake", "build-essential")
    .pip_install(
        "torch",
        "transformers",
        "gguf",
        "protobuf",
        "sentencepiece",
        "mistral_common",
        "unsloth",
        "unsloth_zoo",
    )
    .run_commands(
        # 克隆并编译llama.cpp
        "git clone https://github.com/ggerganov/llama.cpp /root/llama.cpp",
        "cd /root/llama.cpp && cmake -B build && cmake --build build --config Release -j4",
        # 确保可执行文件在正确位置
        "cd /root/llama.cpp && ls -la build/bin/",
    )
)

# 使用与训练相同的volume
volume = modal.Volume.from_name("browser-control-fine-tune-with-grpo", create_if_missing=False)

@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/model_checkpoints": volume},
)
def convert_to_gguf(
    checkpoint_name: str = "LFM2-350M-browsergym-20260220-182152",
    quantization_method: str = "q8_0"
):
    """在Modal GPU上转换模型为GGUF格式"""
    import os
    import sys
    from unsloth import FastLanguageModel
    
    # 确保当前目录是/root，让unsloth能找到llama.cpp子目录
    os.chdir("/root")
    
    # 验证llama.cpp可执行文件
    llama_cpp_dir = "/root/llama.cpp"
    build_dir = os.path.join(llama_cpp_dir, "build", "bin")
    
    print(f"当前工作目录: {os.getcwd()}")
    print(f"llama.cpp目录存在: {os.path.exists('llama.cpp')}")
    
    if os.path.exists(build_dir):
        print(f"✓ llama.cpp build目录存在: {build_dir}")
        print(f"可执行文件:")
        for f in os.listdir(build_dir)[:10]:  # 只显示前10个
            print(f"  - {f}")
    else:
        print(f"✗ 警告: build目录不存在: {build_dir}")
    
    checkpoint_path = f"/model_checkpoints/{checkpoint_name}"
    output_base = "/model_checkpoints/gguf"
    
    print(f"\n检查checkpoint路径: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint不存在: {checkpoint_path}")
    
    print(f"Checkpoint内容:")
    for item in os.listdir(checkpoint_path):
        print(f"  - {item}")
    
    print(f"\n加载模型: {checkpoint_path}")
    
    try:
        # 加载模型和tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,
        )
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        raise
    
    print(f"\n转换为 {quantization_method.upper()} 量化...")
    save_dir = f"{output_base}/{checkpoint_name}-{quantization_method}"
    
    try:
        # 创建输出目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 转换为GGUF格式
        model.save_pretrained_gguf(
            save_dir,
            tokenizer,
            quantization_method=quantization_method
        )
        
        print(f"✓ 保存到: {save_dir}")
        
        # 获取文件信息
        gguf_files = [f for f in os.listdir(save_dir) if f.endswith('.gguf')]
        if gguf_files:
            gguf_path = os.path.join(save_dir, gguf_files[0])
            size_mb = os.path.getsize(gguf_path) / (1024 * 1024)
            print(f"  文件名: {gguf_files[0]}")
            print(f"  文件大小: {size_mb:.1f} MB")
            
            result = {
                "quantization": quantization_method,
                "path": save_dir,
                "size_mb": size_mb,
                "filename": gguf_files[0]
            }
        else:
            print("⚠ 警告: 未找到GGUF文件")
            result = {
                "quantization": quantization_method,
                "path": save_dir,
                "size_mb": 0,
                "filename": None
            }
            
    except Exception as e:
        print(f"✗ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 提交volume更改
    volume.commit()
    
    print("\n✓ 转换完成！")
    return result


@app.local_entrypoint()
def main(quantization: str = "q8_0"):
    """本地入口点"""
    print(f"开始转换模型为 {quantization.upper()} 格式...")
    
    result = convert_to_gguf.remote(quantization_method=quantization)
    
    print("\n" + "="*60)
    print("转换完成！")
    print("="*60)
    print(f"量化方法: {result['quantization'].upper()}")
    print(f"文件大小: {result['size_mb']:.1f} MB")
    print(f"保存路径: {result['path']}")
    if result['filename']:
        print(f"文件名: {result['filename']}")
    
    print("\n下载命令:")
    print(f"  uv run modal volume get browser-control-fine-tune-with-grpo \\")
    print(f"    gguf/LFM2-350M-browsergym-20260220-182152-{quantization} \\")
    print(f"    ./gguf_models/")
    
    print("\n提示：如需转换其他量化级别，运行:")
    print("  uv run modal run src/browser_control/convert_to_gguf_modal.py --quantization q5_k_m")
    print("  uv run modal run src/browser_control/convert_to_gguf_modal.py --quantization q4_k_m")
