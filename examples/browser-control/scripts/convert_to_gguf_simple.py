"""
Simplified GGUF conversion using Modal with llama.cpp directly.
No unsloth dependency - pure llama.cpp conversion.
"""

import modal

# Create Modal app
app = modal.App("browser-control-gguf-conversion")

# Create volume for checkpoints
volume = modal.Volume.from_name("browser-control-fine-tune-with-grpo", create_if_missing=False)

# Simple image with llama.cpp and HuggingFace tools
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "cmake")
    .pip_install(
        "transformers>=4.57.0",
        "torch>=2.5.0",
        "accelerate>=0.27.0",
        "sentencepiece>=0.2.0",
        "protobuf>=5.29.0",
    )
    .run_commands(
        # Clone and build llama.cpp
        "cd /root && git clone https://github.com/ggerganov/llama.cpp.git",
        "cd /root/llama.cpp && cmake -B build && cmake --build build --config Release -j$(nproc)",
    )
)


@app.function(
    image=image,
    gpu="A10G",  # Use A10G GPU for conversion
    volumes={"/checkpoints": volume},
    timeout=3600,  # 1 hour timeout
)
def convert_checkpoint_to_gguf():
    """Convert HuggingFace checkpoint to GGUF format."""
    import subprocess
    import os
    from pathlib import Path
    
    checkpoint_dir = "/checkpoints/LFM2-350M-browsergym-20260220-182152"
    output_dir = "/checkpoints/gguf_output"
    
    print(f"Converting checkpoint from: {checkpoint_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Convert HuggingFace to GGUF (FP16)
    print("\n=== Step 1: Converting to FP16 GGUF ===")
    convert_script = "/root/llama.cpp/convert_hf_to_gguf.py"
    fp16_output = f"{output_dir}/lfm2-350m-browsergym-fp16.gguf"
    
    cmd = [
        "python3",
        convert_script,
        checkpoint_dir,
        "--outfile", fp16_output,
        "--outtype", "f16",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, errors='ignore')
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError(f"Conversion failed with code {result.returncode}")
    
    # Step 2: Quantize to Q8_0
    print("\n=== Step 2: Quantizing to Q8_0 ===")
    quantize_bin = "/root/llama.cpp/build/bin/llama-quantize"
    q8_output = f"{output_dir}/lfm2-350m-browsergym-q8_0.gguf"
    
    cmd = [quantize_bin, fp16_output, q8_output, "Q8_0"]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, errors='ignore')
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError(f"Quantization Q8_0 failed")
    
    # Step 3: Quantize to Q5_K_M
    print("\n=== Step 3: Quantizing to Q5_K_M ===")
    q5_output = f"{output_dir}/lfm2-350m-browsergym-q5_k_m.gguf"
    
    cmd = [quantize_bin, fp16_output, q5_output, "Q5_K_M"]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, errors='ignore')
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError(f"Quantization Q5_K_M failed")
    
    # Step 4: Quantize to Q4_K_M
    print("\n=== Step 4: Quantizing to Q4_K_M ===")
    q4_output = f"{output_dir}/lfm2-350m-browsergym-q4_k_m.gguf"
    
    cmd = [quantize_bin, fp16_output, q4_output, "Q4_K_M"]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, errors='ignore')
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError(f"Quantization Q4_K_M failed")
    
    # List output files with sizes
    print("\n=== Conversion Complete ===")
    print("\nGenerated files:")
    for file in Path(output_dir).glob("*.gguf"):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name}: {size_mb:.2f} MB")
    
    # Commit volume changes
    volume.commit()
    
    return {
        "fp16": fp16_output,
        "q8_0": q8_output,
        "q5_k_m": q5_output,
        "q4_k_m": q4_output,
    }


@app.local_entrypoint()
def main():
    """Run the conversion."""
    print("Starting GGUF conversion on Modal...")
    result = convert_checkpoint_to_gguf.remote()
    print(f"\n✅ Conversion successful!")
    print(f"Output files: {result}")
    print("\nFiles are stored in Modal volume: browser-control-fine-tune-with-grpo")
    print("Location: /checkpoints/gguf_output/")
