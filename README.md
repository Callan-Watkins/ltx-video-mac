# LTX-2.5 video on a Mac: local and free

Type a sentence and get a realistic video clip with sound, generated entirely on your own Mac. There is no
cloud service or subscription, and nothing to pay for. It works on **Intel Macs** (CPU) and
**Apple Silicon Macs** (GPU).

## Install (paste into Terminal)

```bash
curl -fsSL https://raw.githubusercontent.com/Callan-Watkins/ltx-video-mac/main/install.sh | bash
```

The installer:
1. checks the Mac: chip, RAM (Intel needs 16 GB), free disk and drive type;
2. installs a private Python with PyTorch in `~/ltx-video/env` (nothing system-wide; no admin password);
3. installs a pinned, tested ComfyUI with the patched ComfyUI-GGUF loader;
4. asks you to sign in to Hugging Face once. This is free; LTX-2.5's files require it. If your account hasn't
   accepted the LTX-2.5 license yet, it gives you the link and waits;
5. downloads about 45 GB and keeps 29.4 GB of models. The 26 GB original text encoder is compressed on your Mac
   to 10.6 GB and then deleted;
6. adds an `ltx` command.

It's safe to run again. Finished steps are skipped and interrupted downloads resume.

Peak disk use is about **60 GB free**: 29.4 GB of models, plus the temporary 26 GB original encoder, plus 3 GB
for Python.

## Make a video

Open a **new** Terminal window after installing, then:

```bash
ltx "A golden retriever running through a sunny meadow, cinematic lighting" --open
```

| Option | What it does |
|---|---|
| *(none)* | Best quality: 768×448, 49 frames (2 s), 24 fps, with sound, two-stage (draft, then 2× upscale and refine) |
| `--fast` | Quick draft at 512×288 in a single pass, about half the time |
| `--seconds 10` | Clip length in seconds (default 2). Time grows a bit faster than length: 10 s takes several times longer than 2 s |
| `--style app` | A look tuned for marketing: `app` (hands using a phone), `product`, `people`, `animal`, or `real` (default) |
| `--image photo.jpg` | Start the video from your own picture, e.g. a real product shot or a person holding your app. Use this when the product or screen has to look exactly right |
| `--seed 42` | Repeat a result exactly; change the number for a different take |
| `--raw` | Skip the automatic photoreal camera and lighting description added to your prompt |
| `--open` | Open the video when it's done (you also get a notification) |
| `ltx --stop` | Stop the background server and free its memory |

Videos are saved in `~/ltx-video/outputs/ltx25/`. Each one has a `.json` file next to it with the prompt,
settings, time taken and memory use.

Write prompts like a shot description: who or what, the action, the place, the light and the sounds. The audio
is generated from the same prompt.

### Marketing videos

```bash
ltx "A woman on a sofa picks up her phone, taps the screen and smiles at her golden retriever" --style app --seconds 10 --open
ltx "The bottle slowly rotates on a wet black stone as water droplets catch the light" --style product --image ~/Desktop/bottle.jpg
```

AI video can't draw your real app's interface or logo text reliably. For app ads, either keep the screen out
of view or glowing, then put a real screen recording on it in an editor, or start from `--image` with a photo
of a phone already showing your app.

## How long it takes

| Mac | Best quality (768×448, 2 s) | `--fast` |
|---|---|---|
| Apple A18 Pro, 8 GB (measured) | **20 min** | 9–19 min |
| Intel i5 iMac, 16 GB (estimated from CPU speed) | about 1 hour | about 30 min |

Intel Macs run everything on the CPU. PyTorch has no GPU support for the AMD Radeon or Intel graphics in Intel
Macs, so the graphics card can't help. The Mac stays awake during a run, but the screen may still sleep.
Closing other apps makes generation faster. An SSD is much faster than a Fusion Drive or hard drive.

## Quality, honestly

- The model is **LTX-2.5 22B distilled**, quantized to about 5.3 bits per weight (unsloth UD-Q5_K_XL). This is
  not original bf16 quality, but it is the closest that fits in 30 GB.
- The text encoder is Gemma 4 12B, quantized to 5-bit layers with 8-bit heads.
- The video decoder is the official conv decoder. It is about 7× faster than LTX's default diffusion decoder,
  with slightly softer fine texture.
- Clips are short (2 s by default). Faces, skin, hair and lighting look photoreal, but the direction of motion
  doesn't always follow the prompt. If one take comes out wrong, try another `--seed`.

## What's patched (all in `patches/`)

- **`mac_lowvram`**:
  - Apple Silicon: streams the 16 GB model from disk one layer at a time instead of copying it into memory.
  - Intel: uses one CPU thread per core.
  - Everywhere: skips the tokenizer's 1,024-token padding. The output is identical (cosine 1.000000) and
    it is much faster.
- **ComfyUI-GGUF** (Apache-2.0, city96):
  - Metal kernels to unpack 5- and 8-bit weights on Apple GPUs, bit-identical and 15–35× faster.
  - Read-ahead file streaming.
  - Keeps weights memory-mapped instead of copying them into swap.
  - Row-gather embedding lookups and sliced huge layers, so the text encoder fits.
  - Recognizes the `gemma4` architecture.
- **`tools/quantize_encoder.py`**: builds the text-encoder GGUF from the official bf16 file.

## Uninstall

```bash
rm -rf ~/ltx-video
```

Then delete the line ending in `# ltx-video` from `~/.zshrc`.
