"""Make a video (with audio) from a text prompt using LTX-2.5 on this Mac (Apple Silicon or Intel).

Starts the local ComfyUI server if it isn't running, submits the LTX-2.5 distilled
graph, waits, and prints/records timing, peak RAM and swap.

  ltx "your prompt" [options]
"""
import argparse, json, os, platform, random, re, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMFY = ROOT / "ComfyUI"
OUT = ROOT / "outputs"
URL = "http://127.0.0.1:8188"

TRANSFORMER = "ltx-2.5-22b-distilled-UD-Q5_K_XL.gguf"
TEXT_ENCODER = "gemma4-12b-with-proj-ltx-2.5-Q5_1-mac.gguf"
VIDEO_VAE = "ltx-2.5-video-vae-conv-bf16.safetensors"
AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

# Official LTX-2.5 distilled schedules (from ComfyUI's LTX-2.5 template)
STAGE1_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
STAGE2_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"

# Appended to every prompt unless --raw: LTX follows cinematography language closely, and these terms pull it
# toward live-action footage (real skin/fur texture, natural light, real lens) instead of a CG or painted look.
REALISM = (" Photorealistic live-action footage shot on a full-frame cinema camera with a 35mm lens, natural light, "
           "true-to-life colors, fine natural skin and surface texture, realistic motion, subtle film grain, sharp focus.")
# --style adds a look tuned for marketing footage on top of REALISM.
STYLES = {
    "real": "",
    "product": (" High-end commercial product film: the product stays in crisp focus with accurate shape, materials, labels "
                "and reflections, soft studio key light with a gentle rim light, clean uncluttered set, smooth slow camera "
                "push-in, shallow depth of field, premium advertising look."),
    "app": (" Lifestyle app commercial: relaxed natural hands holding a modern smartphone, realistic fingers tapping and "
            "scrolling, the bright screen glowing, genuine reactions and expressions, soft window light, clean modern "
            "interior, smooth gimbal camera movement, shallow depth of field, premium advertising look."),
    "people": (" Natural people: realistic faces with skin pores and fine detail, believable expressions and eye contact, "
               "anatomically correct hands, natural body movement, flattering soft key light, commercial lifestyle look."),
    "animal": (" Real animals: accurate anatomy, detailed fur or feathers that move naturally, lifelike eyes, natural "
               "animal behavior and weight, warm natural light, nature-documentary realism."),
}


def api(path, data=None):
    req = urllib.request.Request(URL + path, data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def server_up():
    try:
        api("/system_stats")
        return True
    except Exception:
        return False


APPLE_SILICON = platform.machine() == "arm64"


def start_server(gpu=False):
    (ROOT / "logs").mkdir(exist_ok=True)
    log = open(ROOT / "logs" / "server.log", "a")
    args = [str(ROOT / "env/bin/python"), "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--disable-dynamic-vram", "--preview-method", "none", "--output-directory", str(OUT)]
    if APPLE_SILICON or gpu:
        # Cap PyTorch's Metal allocator at the GPU working set and have it release cached buffers early.
        # Metal memory is wired: on 8 GB a bigger cache pushed everything else into swap.
        env = dict(os.environ, PYTORCH_MPS_HIGH_WATERMARK_RATIO="1.0", PYTORCH_MPS_LOW_WATERMARK_RATIO="0.4")
        if APPLE_SILICON:
            args.append("--bf16-text-enc")
        else:  # Intel Mac GPUs (AMD Radeon) have no bf16; compute in fp16
            args += ["--fp16-unet", "--fp16-vae", "--fp16-text-enc"]
    else:
        # Intel Macs: PyTorch has no GPU backend for their AMD/Intel graphics by default, so run on the CPU in
        # fp32 (these CPUs have no fp16/bf16 math units; ComfyUI would otherwise pick fp16 for the text encoder).
        env = dict(os.environ)
        args += ["--cpu", "--fp32-text-enc"]
    subprocess.Popen(args, cwd=COMFY, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(300):
        if server_up():
            return
        time.sleep(1)
    sys.exit(f"ComfyUI did not start; see {ROOT / 'logs/server.log'}")


def server_pid():
    out = subprocess.run(["pgrep", "-f", "main.py --listen 127.0.0.1 --port 8188"], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def swap_used_mb():
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", s)
    return float(m.group(1)) if m else 0.0


def rss_mb(pid):
    s = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(s) / 1024 if s else 0.0


def build(a):
    frames = a.frames
    fps = a.fps
    g = {}
    def n(i, cls, **inputs):
        g[str(i)] = {"class_type": cls, "inputs": inputs}
        return str(i)

    unet = n(1, "UnetLoaderGGUF", unet_name=TRANSFORMER)
    clip = n(2, "CLIPLoaderGGUF", clip_name=TEXT_ENCODER, type="ltxv")
    vvae = n(3, "VAELoader", vae_name=VIDEO_VAE)
    avae = n(4, "VAELoader", vae_name=AUDIO_VAE)
    pos = n(5, "CLIPTextEncode", text=a.full_prompt, clip=[clip, 0])
    if a.image:  # official LTX-2.5 image-to-video recipe: resize, mild JPEG-like preprocessing, lock the first frame
        img = n(40, "LoadImage", image=a.image)
        img = n(41, "ResizeImageMaskNode", input=[img, 0], **{"resize_type": "scale longer dimension",
                "resize_type.longer_size": 1536, "scale_method": "lanczos"})
        img = n(42, "LTXVPreprocess", image=[img, 0], img_compression=18)
    if a.negative:
        neg = n(6, "CLIPTextEncode", text=a.negative, clip=[clip, 0])
    else:  # CFG is 1 for the distilled model, so the negative is unused; skip a second Gemma pass
        neg = n(6, "ConditioningZeroOut", conditioning=[pos, 0])
    cond = n(7, "LTXVConditioning", positive=[pos, 0], negative=[neg, 0], frame_rate=float(fps))
    guider = n(8, "LTXVDualCFGGuider", model=[unet, 0], positive=[cond, 0], negative=[cond, 1], video_cfg=1.0, audio_cfg=1.0)
    sampler = n(9, "KSamplerSelect", sampler_name="euler_ancestral")

    w1, h1 = (a.width // 2, a.height // 2) if a.two_stage else (a.width, a.height)
    vlat = n(10, "EmptyLTXVLatentVideo", width=w1, height=h1, length=frames, batch_size=1)
    if a.image:
        vlat = n(43, "LTXVImgToVideoInplace", vae=[vvae, 0], image=[img, 0], latent=[vlat, 0],
                 strength=0.7 if a.two_stage else 1.0, bypass=False)
    alat = n(11, "LTXVEmptyLatentAudio", frames_number=frames, frame_rate=int(fps), batch_size=1, audio_vae=[avae, 0])
    av = n(12, "LTXVConcatAVLatent", video_latent=[vlat, 0], audio_latent=[alat, 0])
    s1 = n(13, "SamplerCustomAdvanced", noise=[n(14, "RandomNoise", noise_seed=a.seed), 0], guider=[guider, 0],
           sampler=[sampler, 0], sigmas=[n(15, "ManualSigmas", sigmas=STAGE1_SIGMAS), 0], latent_image=[av, 0])
    sep = n(16, "LTXVSeparateAVLatent", av_latent=[s1, 0])
    if a.two_stage:
        up = n(17, "LTXVLatentUpsampler", samples=[sep, 0], upscale_model=[n(18, "LatentUpscaleModelLoader", model_name=UPSCALER), 0], vae=[vvae, 0])
        if a.image:
            up = n(44, "LTXVImgToVideoInplace", vae=[vvae, 0], image=[img, 0], latent=[up, 0], strength=1.0, bypass=False)
        av2 = n(19, "LTXVConcatAVLatent", video_latent=[up, 0], audio_latent=[sep, 1])
        s2 = n(20, "SamplerCustomAdvanced", noise=[n(21, "RandomNoise", noise_seed=a.seed + 1), 0], guider=[guider, 0],
               sampler=[sampler, 0], sigmas=[n(22, "ManualSigmas", sigmas=STAGE2_SIGMAS), 0], latent_image=[av2, 0])
        sep = n(23, "LTXVSeparateAVLatent", av_latent=[s2, 0])
    images = n(30, "VAEDecodeTiled", samples=[sep, 0], vae=[vvae, 0], tile_size=a.tile, overlap=64, temporal_size=64, temporal_overlap=16)
    audio = n(31, "LTXVAudioVAEDecode", samples=[sep, 1], audio_vae=[avae, 0])
    video = n(32, "CreateVideo", images=[images, 0], audio=[audio, 0], fps=float(fps))
    n(33, "SaveVideo", video=[video, 0], filename_prefix=f"ltx25/{a.name}", format="mp4",
      **{"format.codec": "h264", "format.codec.encoding": "re-encode", "format.codec.encoding.crf": 16.0})
    return g


def main():
    ap = argparse.ArgumentParser(description="LTX-2.5 text-to-video (+audio) on this Mac")
    ap.add_argument("prompt")
    ap.add_argument("--width", type=int, default=768, help="multiple of 32 (64 for two-stage)")
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--frames", type=int, default=49, help="must be 8*n+1, e.g. 25, 49, 97, 121")
    ap.add_argument("--seconds", type=float, default=None, help="clip length in seconds (sets --frames; up to ~20)")
    ap.add_argument("--style", default="real", choices=sorted(STYLES), help="look tuned for: product, app, people, animal")
    ap.add_argument("--image", default=None, help="start from this photo (product shot, app screenshot on a phone, a person...)")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fast", action="store_true", help="quick draft: 512x288 single pass (default is the sharper two-stage 768x448)")
    ap.add_argument("--two-stage", action="store_true", help="(default unless --fast) half-res pass + 2x latent upscale + refine")
    ap.add_argument("--raw", action="store_true", help="use the prompt exactly as written, without the photoreal style suffix")
    ap.add_argument("--negative", default="", help="optional; distilled model runs at CFG 1 so it has little effect")
    ap.add_argument("--tile", type=int, default=256, help="VAE decode tile size (smaller = less RAM)")
    ap.add_argument("--name", default=None, help="output file prefix")
    ap.add_argument("--open", action="store_true", help="open the video when it's done")
    ap.add_argument("--gpu", action="store_true", help="Intel Macs: EXPERIMENTAL, use the AMD graphics card via Metal instead of the CPU "
                    "(run 'ltx --stop' first if the server is already running)")
    ap.add_argument("--stop", action="store_true", help="stop the background server and exit")
    if "--stop" in sys.argv:
        subprocess.run(["pkill", "-f", "main.py --listen 127.0.0.1 --port 8188"])
        sys.exit("Stopped the LTX server.")
    a = ap.parse_args()
    if a.fast:
        a.two_stage = False
        if a.width == 768 and a.height == 448:
            a.width, a.height = 512, 288
    else:
        a.two_stage = True
    if a.seconds:
        a.frames = round(a.seconds * a.fps / 8) * 8 + 1
    a.full_prompt = a.prompt.rstrip() + ("" if a.raw else REALISM + STYLES[a.style])
    if a.image:
        src = Path(a.image).expanduser().resolve()
        if not src.exists():
            sys.exit(f"--image not found: {src}")
        (COMFY / "input").mkdir(exist_ok=True)
        dst = COMFY / "input" / src.name
        dst.write_bytes(src.read_bytes())
        a.image = src.name
    if (a.frames - 1) % 8:
        sys.exit("--frames must be 8*n+1 (25, 33, 49, 97, 121 ...)")
    mult = 64 if a.two_stage else 32
    if a.width % mult or a.height % mult:
        sys.exit(f"--width/--height must be multiples of {mult}")
    a.seed = random.randint(0, 2**31) if a.seed is None else a.seed
    a.name = a.name or time.strftime("ltx25_%Y%m%d_%H%M%S")

    if not server_up():
        print("Starting ComfyUI ...", flush=True)
        start_server(a.gpu)
    pid = server_pid()
    # Keep the Mac awake (display may sleep) for as long as the server runs this job.
    awake = subprocess.Popen(["caffeinate", "-i", "-w", str(pid)]) if pid else None

    swap0 = swap_used_mb()
    pid_resp = api("/prompt", {"prompt": build(a)})
    if pid_resp.get("node_errors"):
        sys.exit(json.dumps(pid_resp, indent=1))
    prompt_id = pid_resp["prompt_id"]
    print(f"Queued {prompt_id}: {a.width}x{a.height}, {a.frames} frames @ {a.fps} fps, seed {a.seed}, "
          f"{'two-stage' if a.two_stage else 'single-stage'}", flush=True)

    t0 = time.time(); peak_rss = peak_swap = 0.0; last = 0
    log_path = ROOT / "logs" / "server.log"
    log_pos = log_path.stat().st_size if log_path.exists() else 0
    stages = {}  # first time (s since queue) each stage marker shows up in the server log
    while True:
        time.sleep(3)
        if log_path.exists():
            with open(log_path, "rb") as lf:
                lf.seek(log_pos)
                chunk = lf.read().decode("utf-8", "replace")
                log_pos = lf.tell()
            for marker, stage in (("Requested to load LTXAVTEModel", "text_encode"), ("Requested to load LTXAV\n", "sampling"),
                                  ("Requested to load AudioVAE", "decode"), ("Prompt executed", "done")):
                if marker in chunk and stage not in stages:
                    stages[stage] = round(time.time() - t0)
                    print(f"  [{stages[stage]}s] {stage}", flush=True)
        if pid:
            peak_rss = max(peak_rss, rss_mb(pid))
        peak_swap = max(peak_swap, swap_used_mb())
        try:
            h = api(f"/history/{prompt_id}")
        except Exception:
            h = {}
        if prompt_id in h:
            st = h[prompt_id].get("status", {})
            if st.get("completed") or st.get("status_str") in ("success", "error"):
                break
        if not server_up():
            sys.exit("ComfyUI server died (likely out of memory); see logs/server.log")
        if time.time() - last > 60:
            print(f"  ... {time.time()-t0:.0f}s  server RSS {rss_mb(pid) if pid else 0:.0f} MB  swap {swap_used_mb():.0f} MB", flush=True)
            last = time.time()
    elapsed = time.time() - t0
    rec = h[prompt_id]
    if rec["status"].get("status_str") != "success":
        print(json.dumps(rec["status"], indent=1)[-4000:])
        sys.exit("Generation failed")
    files = [OUT / v["subfolder"] / v["filename"] for o in rec["outputs"].values() for k in ("images", "videos", "gifs")
             for v in o.get(k, []) if "filename" in v]
    files = files or sorted((OUT / "ltx25").glob(a.name + "*"))
    report = {"prompt": a.prompt, "full_prompt": a.full_prompt, "width": a.width, "height": a.height, "frames": a.frames, "fps": a.fps,
              "seconds": round(a.frames / a.fps, 2), "seed": a.seed, "two_stage": a.two_stage, "style": a.style, "image": a.image,
              "elapsed_s": round(elapsed), "stage_start_s": stages, "peak_server_rss_mb": round(peak_rss), "swap_before_mb": round(swap0),
              "peak_swap_mb": round(peak_swap), "files": [str(f) for f in files],
              "file_bytes": [f.stat().st_size for f in files if f.exists()]}
    for f in files:
        Path(str(f) + ".json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if awake:
        awake.terminate()
    if files:
        print(f"\nVideo: {files[0]}")
        subprocess.run(["osascript", "-e", f'display notification "Finished in {elapsed/60:.0f} min" with title "LTX video ready"'],
                       capture_output=True)
        if a.open:
            subprocess.run(["open", str(files[0])])


if __name__ == "__main__":
    main()
