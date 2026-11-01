"""Mac tuning for LTX-2.5 in ComfyUI, on Apple Silicon (Metal) or Intel (CPU).

Apple Silicon: ComfyUI puts MPS in SHARED mode, which copies the whole model into Metal memory. A 16 GB GGUF
can't fit on a small Mac, so switch to LOW_VRAM: weights stay memory-mapped from the GGUF file on disk and
each layer is streamed to the GPU only while it runs.

Intel: everything runs on the CPU. Use one thread per physical core (hyper-threads only add contention
in big matrix multiplies).
"""
import logging
import os
import subprocess
import torch
import comfy.model_management as mm

if mm.cpu_state == mm.CPUState.MPS and mm.vram_state == mm.VRAMState.SHARED:
    mm.vram_state = mm.VRAMState.LOW_VRAM
    # LOW_VRAM would also move the 12B text encoder to the CPU, which is several times slower
    # than streaming it through the GPU the same way as the video model.
    mm.text_encoder_device = mm.get_torch_device
    logging.info("[mac_lowvram] MPS: SHARED -> LOW_VRAM (stream weights from mmapped GGUF), text encoder on MPS")

    # The LTX-2.5 diffusion video decoder uses neighborhood attention, which off CUDA runs as masked
    # attention over query tiles. Smaller tiles mean less wasted attention and much less memory.
    try:
        import comfy_kitchen.backends.eager.na as na
        na.NA_SCORE_BUDGET = 2 ** 23
    except ImportError:
        pass

if mm.cpu_state == mm.CPUState.CPU:
    try:
        cores = int(subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True).stdout)
    except ValueError:
        cores = os.cpu_count() or 4
    torch.set_num_threads(int(os.environ.get("LTX_THREADS", cores)))
    logging.info(f"[mac_lowvram] CPU mode, {torch.get_num_threads()} threads")

# The LTX Gemma tokenizer left-pads every prompt to 1024 tokens. Padding is masked out of attention and
# ComfyUI trims the padded positions off the output, so it only costs compute: encoding the bare prompt
# gives the same conditioning (measured cosine 1.000000 vs padded) in a fraction of the time.
import comfy.text_encoders.lt as lt


def ltxav_gemma4_tokenizer_unpadded(tokenizer):
    class LTXAVGemma4Tokenizer(tokenizer):
        pass
    return LTXAVGemma4Tokenizer


lt.ltxav_gemma4_tokenizer = ltxav_gemma4_tokenizer_unpadded

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
