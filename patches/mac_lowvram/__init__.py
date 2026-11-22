"""Mac tuning for LTX-2.5 in ComfyUI, on Apple Silicon (Metal) or Intel (CPU).

Apple Silicon: ComfyUI puts MPS in SHARED mode, which copies the whole model into Metal memory. A 16 GB GGUF
can't fit on a small Mac, so switch to LOW_VRAM: weights stay memory-mapped from the GGUF file on disk and
are streamed to the GPU a transformer block at a time (see ComfyUI-GGUF/stream.py). Together with the
changes below and the split into parts in ltx.py, a whole 10 s clip runs in under 4 GB of RAM.

Intel: everything runs on the CPU. Use one thread per physical core (hyper-threads only add contention
in big matrix multiplies).
"""
import contextlib
import ctypes
import logging
import os
import subprocess
import threading
import time

import safetensors
import torch

import comfy.ldm.lightricks.latent_upsampler as latent_upsampler
import comfy.ldm.lightricks.vae.causal_video_autoencoder as video_vae
import comfy.model_management as mm
import comfy.sd
import comfy.text_encoders.lt as lt
import folder_paths

if mm.cpu_state == mm.CPUState.MPS and mm.vram_state == mm.VRAMState.SHARED:
    mm.vram_state = mm.VRAMState.LOW_VRAM
    # LOW_VRAM would also move the 12B text encoder to the CPU, which is several times slower
    # than streaming it through the GPU the same way as the video model.
    mm.text_encoder_device = mm.get_torch_device
    logging.info("[mac_lowvram] MPS: SHARED -> LOW_VRAM (stream weights from mmapped GGUF), text encoder on MPS")

    # ComfyUI sizes what it keeps on the GPU from free system RAM. With other apps open that is often near
    # zero, so it ran even the 1.4 GB video decoder in "partial" mode, re-streaming its weights for every
    # tile (a 10 s decode took 34 min instead of 6). The GGUF models always stream (ComfyUI-GGUF gives them
    # no budget), so this fixed budget only decides that the decoders and upscalers load fully.
    _budget = int(os.environ.get("LTX_GPU_BUDGET_MB", "5000")) * 2**20
    _get_free_memory = mm.get_free_memory

    def get_free_memory(dev=None, torch_free_too=False):
        if dev is None:
            dev = mm.get_torch_device()
        if getattr(dev, "type", None) != "mps":
            return _get_free_memory(dev, torch_free_too)
        free = max(_budget - torch.mps.current_allocated_memory(), 0)
        return (free, free) if torch_free_too else free

    mm.get_free_memory = get_free_memory

    # The video VAE and the latent upscalers are built in float32, randomly initialised, then cast to
    # bfloat16 and overwritten by the checkpoint: 3.2 GB of short-lived RAM for the 1.45 GB VAE. Building
    # them straight in bfloat16 without the random init gives the same weights (checked bit for bit).
    @contextlib.contextmanager
    def lean_build(device=None):
        dtype = torch.get_default_dtype()
        inits = {n: getattr(torch.nn.init, n) for n in dir(torch.nn.init) if n.endswith("_") and not n.startswith("_")}
        torch.set_default_dtype(torch.bfloat16)
        if device is not None:  # build straight on the GPU: no CPU copy that would stay behind
            torch.set_default_device(device)
        for n in inits:
            setattr(torch.nn.init, n, lambda t, *args, **kwargs: t)
        try:
            yield
        finally:
            torch.set_default_dtype(dtype)
            torch.set_default_device(None)
            for n, f in inits.items():
                setattr(torch.nn.init, n, f)

    _vae_init = comfy.sd.VAE.__init__

    def vae_init(self, sd=None, *args, **kwargs):
        # only the video VAE: the audio VAE computes float32 tables at build time
        video = sd is not None and "per_channel_statistics.mean-of-means" in sd
        with lean_build(mm.get_torch_device()) if video else contextlib.nullcontext():
            _vae_init(self, sd, *args, **kwargs)

    comfy.sd.VAE.__init__ = vae_init
    _upsampler_init = latent_upsampler.LatentUpsampler.__init__

    def upsampler_init(self, *args, **kwargs):
        with lean_build():
            _upsampler_init(self, *args, **kwargs)

    latent_upsampler.LatentUpsampler.__init__ = upsampler_init

    # Decoding a 10 s clip: keep the 241 decoded frames, and the tiler's full-clip output and blend buffers,
    # in float16 (half the RAM; plenty for 8-bit video), and give Metal's cache back after every tile.
    _vae_output_dtype = comfy.sd.VAE.vae_output_dtype

    def vae_output_dtype(self):
        if isinstance(self.first_stage_model, video_vae.VideoVAE):
            return torch.float16
        return _vae_output_dtype(self)

    comfy.sd.VAE.vae_output_dtype = vae_output_dtype
    _decode_tiled_3d = comfy.sd.VAE.decode_tiled_3d

    def decode_tiled_3d(self, *args, **kwargs):
        if not isinstance(self.first_stage_model, video_vae.VideoVAE):
            return _decode_tiled_3d(self, *args, **kwargs)
        dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)  # the tiler allocates its buffers with the default dtype
        # A tight Metal allocator cap only here: it keeps the decode under 4 GB, but during sampling it
        # makes the allocator free and re-allocate all the time (refine went from 11 to 15 min).
        torch.mps.set_per_process_memory_fraction(float(os.environ.get("LTX_DECODE_MPS_CAP", "0.4")))
        try:
            return _decode_tiled_3d(self, *args, **kwargs)
        finally:
            torch.set_default_dtype(dtype)
            torch.mps.set_per_process_memory_fraction(float(os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")))

    comfy.sd.VAE.decode_tiled_3d = decode_tiled_3d
    _decoder_forward = video_vae.Decoder.forward

    def decoder_forward(self, *args, **kwargs):
        out = _decoder_forward(self, *args, **kwargs)
        torch.mps.empty_cache()
        return out

    video_vae.Decoder.forward = decoder_forward

    # The LTX-2.5 diffusion video decoder uses neighborhood attention, which off CUDA runs as masked
    # attention over query tiles. Smaller tiles mean less wasted attention and much less memory.
    try:
        import comfy_kitchen.backends.eager.na as na
        na.NA_SCORE_BUDGET = 2 ** 23
    except ImportError:
        pass

    if os.environ.get("LTX_MEMLOG"):  # debug: log every change of more than 100 MB in RAM use
        def _memlog():
            buf = (ctypes.c_uint64 * 64)()
            lib = ctypes.CDLL("/usr/lib/libproc.dylib")
            last = 0
            while True:
                lib.proc_pid_rusage(os.getpid(), 4, buf)
                if abs(buf[9] - last) > 100 * 2**20:
                    last = buf[9]
                    logging.info("[memlog] footprint %.0f MB | mps tensors %.0f MB | mps driver %.0f MB",
                                 buf[9] / 2**20, torch.mps.current_allocated_memory() / 2**20,
                                 torch.mps.driver_allocated_memory() / 2**20)
                time.sleep(0.2)

        threading.Thread(target=_memlog, daemon=True).start()

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
def ltxav_gemma4_tokenizer_unpadded(tokenizer):
    class LTXAVGemma4Tokenizer(tokenizer):
        pass
    return LTXAVGemma4Tokenizer


lt.ltxav_gemma4_tokenizer = ltxav_gemma4_tokenizer_unpadded

# ltx.py runs a clip as several prompts (draft, upscale, refine, sound, decode) and frees every model in
# between, so only one stage's models are ever in memory. These pass the intermediate latents,
# conditioning and sound from one prompt to the next inside the server process.
_stash = {}


class LTXStash:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"key": ("STRING", {"default": "latent"})},
                "optional": {"latent": ("LATENT",), "conditioning": ("CONDITIONING",), "audio": ("AUDIO",)}}
    RETURN_TYPES = ()
    FUNCTION = "stash"
    OUTPUT_NODE = True
    CATEGORY = "latent"

    def stash(self, key, latent=None, conditioning=None, audio=None):
        _stash[key] = next(v for v in (latent, conditioning, audio) if v is not None)
        return ()


class LTXUnstashLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"key": ("STRING", {"default": "latent"})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "latent"

    @classmethod
    def IS_CHANGED(cls, key):
        return float("nan")

    def load(self, key):
        return (_stash[key],)


class LTXUnstashConditioning(LTXUnstashLatent):
    RETURN_TYPES = ("CONDITIONING",)


class LTXUnstashAudio(LTXUnstashLatent):
    RETURN_TYPES = ("AUDIO",)


class LTXLatentUpsamplerStats:
    """LTXVLatentUpsampler, but it reads the two per-channel latent statistics it needs straight from the
    video VAE file instead of loading the whole 1.4 GB VAE."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "upscale_model": ("LATENT_UPSCALE_MODEL",),
                             "vae_name": (folder_paths.get_filename_list("vae"),)}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upsample"
    CATEGORY = "latent"

    def upsample(self, samples, upscale_model, vae_name):
        with safetensors.safe_open(folder_paths.get_full_path_or_raise("vae", vae_name), "pt") as f:
            mean = f.get_tensor("per_channel_statistics.mean-of-means").view(1, -1, 1, 1, 1)
            std = f.get_tensor("per_channel_statistics.std-of-means").view(1, -1, 1, 1, 1)
        latents = samples["samples"]
        mm.load_models_gpu([upscale_model], memory_required=latents.numel() * 3000.0)
        x = latents.to(dtype=upscale_model.model_dtype(), device=upscale_model.load_device)
        x = upscale_model.model(x * std.to(x) + mean.to(x))
        out = samples.copy()
        out["samples"] = ((x - mean.to(x)) / std.to(x)).to(dtype=latents.dtype, device=mm.intermediate_device())
        out.pop("noise_mask", None)
        return (out,)


NODE_CLASS_MAPPINGS = {"LTXStash": LTXStash, "LTXUnstashLatent": LTXUnstashLatent,
                       "LTXUnstashConditioning": LTXUnstashConditioning, "LTXUnstashAudio": LTXUnstashAudio,
                       "LTXLatentUpsamplerStats": LTXLatentUpsamplerStats}
NODE_DISPLAY_NAME_MAPPINGS = {}
