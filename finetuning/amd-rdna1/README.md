# Fine-tuning LFM3-1.7B on AMD RDNA1 (RX 5700 XT)

Verified on Windows 11 with an AMD Radeon RX 5700 XT 8GB (gfx1010, RDNA1),
Adrenalin 32.0.21030.2001, Python 3.12.10. All scripts below were run end-to-end
on real hardware on 2026-11-21.

## Results

| Configuration | grad_norm | Peak VRAM | Status |
|---|---|---|---|
| fp16 LoRA (official smoke test unchanged) | **NaN** | 3.41 GiB | Broken — gradients are corrupted while loss looks finite |
| fp32 LoRA (float32 weights, AMP disabled) | 12.36 → 7.73 | 6.63 GiB | Works, little headroom |
| **QLoRA NF4 4bit (fp32 compute)** | **10.62 → 7.56** | **1.42 GiB** | **Recommended** |

Root cause: RDNA1 lacks bf16 and dot-product instructions, and the MIOpen CK
grouped-conv library is not shipped for gfx1010
(`CK grouped conv library not found for device gfx1010`). The fallback kernel
produces NaN gradients under fp16 AMP. Disabling AMP (or using fp32 compute with
4bit weights) fixes it.

## Real-data validation

`train_real_data.py` fine-tunes on 64 examples from `HuggingFaceTB/smoltalk`
for 20 steps (seq=512, batch=2, grad_accum=2, LoRA r=16):

- Loss: 1.697 → 1.158 (final train_loss 1.561), no divergence
- `grad_norm`: finite at every step (peak 4.031, final 0.6094)
- Peak VRAM: **2.47 GiB**
- Saved adapter reloads successfully via `PeftModel.from_pretrained`

## Setup

```powershell
python -m venv C:\path\to\.venv-lfm3

# AMD's multi-arch wheel is required: pytorch.org ROCm wheels only target gfx1030+
.\.venv-lfm3\Scripts\python.exe -m pip install `
  "torch[device-gfx1010]==2.11.0+rocm7.14.1" `
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ `
  --extra-index-url https://pypi.org/simple/

.\.venv-lfm3\Scripts\python.exe -m pip install `
  "numpy<2.3" "transformers==5.23.0" "trl==1.18.1" "peft==0.21.3" `
  "accelerate>=1.10" datasets safetensors huggingface_hub bitsandbytes
```

Verify the GPU:

```powershell
$env:AMDGCN_USE_BUFFER_OPS="0"
.\.venv-lfm3\Scripts\python.exe -c "import torch; print(torch.cuda.get_device_name(0)); print((torch.randn(1024,1024,device='cuda')@torch.randn(1024,1024,device='cuda')).sum())"
```

## Scripts

- `smoke_test_qlora.py` — recommended: NF4 4bit + fp32 compute, ~1.42 GiB peak
- `smoke_test_fp32.py` — fp32 LoRA control run, ~6.63 GiB peak
- `smoke_test_lora_fp16.py` — fp16 control run that reproduces the NaN gradient issue
- `train_real_data.py` — 20-step run on real smoltalk data with loss/grad_norm logging

Replace the two synthetic examples with your own `messages`-format instruction
dataset. With only 1.42 GiB peak, `max_length` and batch size can be raised
substantially on an 8GB card. LFM3 enables thinking by default; for structured
output tasks set `enable_thinking=False` in the chat template.

## Gotchas

1. **Do not use WSL2.** The Windows driver of RX 5700 XT fails the GPU virtual
   address reservation required by `librocdxg` (`dxgkio_reserve_gpu_va: EOVERFLOW`).
   Run natively on Windows.
2. **Set `AMDGCN_USE_BUFFER_OPS=0`.** Triton's default buffer ops on gfx1010
   report `hipSuccess` but write nothing, so training silently learns garbage.
3. **Disable AMP.** Use `fp16=False, bf16=False`; keep weights in float32 or use
   4bit quantization with `bnb_4bit_compute_dtype=torch.float32`.
4. **`trust_remote_code=True` is required** for LFM3's custom architecture
   (`Lfm3Config` and gated short convolutions). transformers 5.23.0 works.
5. **Use `target_modules="all-linear"`** for LoRA. Llama/Qwen module names do not
   apply to LFM3.
6. The bitsandbytes warning `Could not detect ROCm GPU architecture` can be
   ignored; NF4 and 8-bit AdamW work correctly.
7. The MIOpen `CK grouped conv library not found for gfx1010` warning can be
   ignored when compute is fp32.
8. Never set `bf16=True` — RDNA1 has no bf16 instructions.
