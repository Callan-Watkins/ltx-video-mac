#!/bin/bash
# LTX-2.5 text-to-video (+ sound), fully local and free, for Macs: Intel (CPU) or Apple Silicon (Metal GPU).
#
#   curl -fsSL https://raw.githubusercontent.com/Callan-Watkins/ltx-video-mac/main/install.sh | bash
#
# Installs into ~/ltx-video (override with LTX_HOME=...). Safe to re-run: finished steps are skipped and
# interrupted downloads resume. Needs a free Hugging Face account that has accepted the LTX-2.5 terms.
set -euo pipefail

REPO="Callan-Watkins/ltx-video-mac"
BRANCH="${LTX_BRANCH:-main}"
HOME_DIR="${LTX_HOME:-$HOME/ltx-video}"
COMFY_SHA="0dbfd493b4cc604db743c699f0f7ce01348e54a4"   # ComfyUI with LTX-2.5 support (tested)
GGUF_SHA="6ea2651e7df66d7585f6ffee804b20e92fb38b8a"    # city96/ComfyUI-GGUF (tested; patched below)
PY="$HOME_DIR/env/bin/python"

LTX_REPO="Lightricks/LTX-2.5"
GGUF_REPO="unsloth/LTX-2.5-GGUF"
F_TRANSFORMER="distilled/ltx-2.5-22b-distilled-UD-Q5_K_XL.gguf"
F_ENCODER_BF16="text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
F_VIDEO_VAE="vae/ltx-2.5-video-vae-conv-bf16.safetensors"
F_AUDIO_VAE="vae/ltx-2.5-audio-vae-bf16.safetensors"
F_UPSCALER="latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
ENCODER_GGUF="gemma4-12b-with-proj-ltx-2.5-Q5_1-mac.gguf"

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
say() { printf '  %s\n' "$*"; }
die() { printf '\n\033[31mStopped: %s\033[0m\n' "$*" >&2; exit 1; }
gb() { echo $(( $1 / 1000000000 )); }

# ---------------------------------------------------------------- 1. check the Mac
bold "1/6  Checking this Mac"
ARCH=$(uname -m)
CPU=$(sysctl -n machdep.cpu.brand_string)
CORES=$(sysctl -n hw.physicalcpu)
RAM=$(sysctl -n hw.memsize)
MACOS=$(sw_vers -productVersion)
mkdir -p "$HOME_DIR"
FREE=$(( $(df -k "$HOME_DIR" | awk 'NR==2 {print $4}') * 1024 ))
say "$CPU, $CORES cores, $(gb "$RAM") GB RAM, macOS $MACOS, $(gb "$FREE") GB free disk"

if [ "$ARCH" = "x86_64" ]; then
    # Rosetta reports x86_64 too; the installer must match the real chip so the right PyTorch is used.
    if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = "1" ] && [ -z "${LTX_ALLOW_ROSETTA:-}" ]; then
        die "this Terminal runs under Rosetta on an Apple Silicon Mac. Open a normal Terminal and paste again."
    fi
    [ "$(sysctl -n hw.optional.avx2_0 2>/dev/null || echo 1)" = "1" ] || [ -n "${LTX_ALLOW_ROSETTA:-}" ] || die "this CPU has no AVX2; PyTorch can't run the model on it."
    [ "$RAM" -ge 15000000000 ] || [ -n "${LTX_ALLOW_ROSETTA:-}" ] || die "Intel Macs need 16 GB RAM or more for this model ($(gb "$RAM") GB found)."
    MODE="Intel CPU"
    say "Mode: CPU. PyTorch can't use the Radeon/Intel graphics in Intel Macs, so all $CORES cores do the work."
else
    MODE="Apple GPU"
    say "Mode: Apple Silicon GPU (Metal)."
fi
DISK=$(diskutil info / 2>/dev/null | awk -F: '/Solid State/ {gsub(/ /,"",$2); print $2}')
if [ "$DISK" = "No" ]; then
    say "Note: the startup disk is a hard drive or Fusion Drive. Each step re-reads part of the 16 GB model,"
    say "so expect it to be noticeably slower than on an SSD."
fi

NEED=$(( 3 * 1000000000 ))  # Python environment + ComfyUI
[ -f "$HOME_DIR/models/diffusion_models/$(basename "$F_TRANSFORMER")" ] || NEED=$(( NEED + 16000033152 ))
[ -f "$HOME_DIR/models/text_encoders/$ENCODER_GGUF" ] || NEED=$(( NEED + 10621378272 + 26263858182 ))  # bf16 download is temporary
[ -f "$HOME_DIR/models/vae/$(basename "$F_VIDEO_VAE")" ] || NEED=$(( NEED + 1452269922 + 364866540 + 995778752 ))
[ -n "${LTX_MODELS_FROM:-}" ] && NEED=$(( 3 * 1000000000 ))
[ "$FREE" -ge "$NEED" ] || die "needs about $(gb "$NEED") GB free disk for the download (only $(gb "$FREE") GB free). Free some space and re-run."

# ---------------------------------------------------------------- 2. Python + PyTorch
bold "2/6  Python and PyTorch (private copy in $HOME_DIR/env, nothing system-wide)"
MAMBA="$HOME_DIR/bin/micromamba"
if [ ! -x "$PY" ]; then
    if [ ! -x "$MAMBA" ]; then
        PLAT=$([ "$ARCH" = "arm64" ] && echo osx-arm64 || echo osx-64)
        mkdir -p "$HOME_DIR/bin"
        curl -fsSL "https://micro.mamba.pm/api/micromamba/$PLAT/latest" | tar -xj -C "$HOME_DIR" bin/micromamba
    fi
    export MAMBA_ROOT_PREFIX="$HOME_DIR/mamba"
    if [ "$ARCH" = "x86_64" ]; then
        # PyPI stopped publishing PyTorch for Intel Macs at 2.2; conda-forge still builds current versions.
        "$MAMBA" create -y -p "$HOME_DIR/env" -c conda-forge python=3.12 pytorch torchvision
    else
        "$MAMBA" create -y -p "$HOME_DIR/env" -c conda-forge python=3.12
        "$PY" -m pip install -q torch torchvision
    fi
fi
"$PY" -c "import torch; print('  PyTorch', torch.__version__, '| Metal GPU available' if torch.backends.mps.is_available() else '| CPU only')"

# ---------------------------------------------------------------- 3. ComfyUI + Mac patches
bold "3/6  ComfyUI (pinned, tested version) and the Mac speed patches"
TMP="$HOME_DIR/tmp"; mkdir -p "$TMP"
if [ ! -f "$HOME_DIR/ComfyUI/.ltx_sha" ] || [ "$(cat "$HOME_DIR/ComfyUI/.ltx_sha")" != "$COMFY_SHA" ]; then
    rm -rf "$TMP/comfy" && mkdir -p "$TMP/comfy"
    curl -fsSL "https://codeload.github.com/comfyanonymous/ComfyUI/tar.gz/$COMFY_SHA" | tar -xz -C "$TMP/comfy" --strip-components 1
    rm -rf "$HOME_DIR/ComfyUI" && mv "$TMP/comfy" "$HOME_DIR/ComfyUI"
    echo "$COMFY_SHA" > "$HOME_DIR/ComfyUI/.ltx_sha"
fi
GG="$HOME_DIR/ComfyUI/custom_nodes/ComfyUI-GGUF"
if [ ! -f "$GG/.ltx_sha" ] || [ "$(cat "$GG/.ltx_sha")" != "$GGUF_SHA" ]; then
    rm -rf "$GG" && mkdir -p "$GG"
    curl -fsSL "https://codeload.github.com/city96/ComfyUI-GGUF/tar.gz/$GGUF_SHA" | tar -xz -C "$GG" --strip-components 1
    echo "$GGUF_SHA" > "$GG/.ltx_sha"
fi
# This harness itself (patches, ltx command, tools). LTX_SRC lets a local checkout be used instead.
if [ -n "${LTX_SRC:-}" ]; then
    SRC="$LTX_SRC"
else
    rm -rf "$TMP/src" && mkdir -p "$TMP/src"
    curl -fsSL "https://codeload.github.com/$REPO/tar.gz/$BRANCH" | tar -xz -C "$TMP/src" --strip-components 1
    SRC="$TMP/src"
fi
cp "$SRC"/patches/ComfyUI-GGUF/*.py "$GG/"
mkdir -p "$HOME_DIR/ComfyUI/custom_nodes/mac_lowvram" "$HOME_DIR/tools"
cp "$SRC/patches/mac_lowvram/__init__.py" "$HOME_DIR/ComfyUI/custom_nodes/mac_lowvram/"
cp "$SRC/ltx.py" "$HOME_DIR/ltx.py"
cp "$SRC"/tools/*.py "$HOME_DIR/tools/"
cat > "$HOME_DIR/ComfyUI/extra_model_paths.yaml" <<EOF
ltx:
    base_path: $HOME_DIR/models
    diffusion_models: diffusion_models
    text_encoders: text_encoders
    vae: vae
    latent_upscale_models: latent_upscale_models
EOF

# Python packages ComfyUI needs. Torch comes from step 2; comfy-angle (OpenGL, optional) has no Intel build.
grep -v -E '^(torch|torchvision|torchaudio|comfy-angle)([<>=~ ]|$)' "$HOME_DIR/ComfyUI/requirements.txt" > "$TMP/req.txt"
"$PY" -m pip install -q --disable-pip-version-check -r "$TMP/req.txt" -r "$GG/requirements.txt" "huggingface_hub>=1.0"
say "ComfyUI ready."

# ---------------------------------------------------------------- 4. Hugging Face sign-in
bold "4/6  Hugging Face sign-in (free; the LTX-2.5 files require it)"
HF="$HOME_DIR/env/bin/hf"
access() {
    "$PY" -I - <<'EOF' 2>/dev/null
import sys
from huggingface_hub import get_hf_file_metadata, hf_hub_url
try:
    get_hf_file_metadata(hf_hub_url("Lightricks/LTX-2.5", "vae/ltx-2.5-audio-vae-bf16.safetensors"))
    print("ok")
except Exception as e:
    print(type(e).__name__)
EOF
}
if [ -z "${LTX_MODELS_FROM:-}" ]; then
    if ! "$HF" auth whoami >/dev/null 2>&1; then
        say "Sign in with a Hugging Face access token (huggingface.co -> Settings -> Access Tokens, type 'Read')."
        say "The token is typed into Hugging Face's own login and stored on this Mac only."
        "$HF" auth login < /dev/tty
    fi
    STATE=$(access)
    while [ "$STATE" != "ok" ]; do
        say "This account can't download LTX-2.5 yet ($STATE)."
        say "Open https://huggingface.co/Lightricks/LTX-2.5 while signed in, accept the license, then press Return."
        read -r _ < /dev/tty
        STATE=$(access)
    done
    say "Signed in as $("$HF" auth whoami 2>/dev/null | head -1)."
else
    say "Skipped: using existing model files from $LTX_MODELS_FROM"
fi

# ---------------------------------------------------------------- 5. models (29.4 GB)
bold "5/6  Model files (~29.4 GB total; downloads resume if interrupted)"
M="$HOME_DIR/models"; DL="$HOME_DIR/downloads"
mkdir -p "$M/diffusion_models" "$M/text_encoders" "$M/vae" "$M/latent_upscale_models" "$DL"
get() {  # repo, path-in-repo, destination folder
    local dest="$3/$(basename "$2")"
    [ -f "$dest" ] && return
    if [ -n "${LTX_MODELS_FROM:-}" ]; then
        ln -s "$(find "$LTX_MODELS_FROM" -name "$(basename "$2")" | head -1)" "$dest"; return
    fi
    say "Downloading $(basename "$2") ..."
    "$HF" download "$1" "$2" --local-dir "$DL"
    mv "$DL/$2" "$dest"
}
get "$GGUF_REPO" "$F_TRANSFORMER" "$M/diffusion_models"
get "$LTX_REPO" "$F_VIDEO_VAE" "$M/vae"
get "$LTX_REPO" "$F_AUDIO_VAE" "$M/vae"
get "$LTX_REPO" "$F_UPSCALER" "$M/latent_upscale_models"
if [ ! -f "$M/text_encoders/$ENCODER_GGUF" ]; then
    if [ -n "${LTX_MODELS_FROM:-}" ]; then
        get "" "$ENCODER_GGUF" "$M/text_encoders"
    else
        # The text encoder (Gemma 4 12B) is published in bf16 only (26 GB). Shrink it to 10.6 GB here:
        # 5-bit for the 48 Gemma layers, 8-bit for embeddings and LTX projection heads, full precision norms.
        "$HF" download "$LTX_REPO" "$F_ENCODER_BF16" --local-dir "$DL"
        say "Compressing the text encoder to 10.6 GB (about 15-40 min, one-time) ..."
        "$PY" -I "$HOME_DIR/tools/quantize_encoder.py" "$DL/$F_ENCODER_BF16" "$M/text_encoders/$ENCODER_GGUF.part"
        mv "$M/text_encoders/$ENCODER_GGUF.part" "$M/text_encoders/$ENCODER_GGUF"
        rm -f "$DL/$F_ENCODER_BF16"   # temporary 26 GB original, no longer needed
    fi
fi
rm -rf "$DL/.cache" "$TMP"
say "Model set: $(du -sh -L "$M" | cut -f1)"

# ---------------------------------------------------------------- 6. the ltx command
bold "6/6  Installing the 'ltx' command"
mkdir -p "$HOME_DIR/bin"
cat > "$HOME_DIR/bin/ltx" <<EOF
#!/bin/bash
exec "$PY" "$HOME_DIR/ltx.py" "\$@"
EOF
chmod +x "$HOME_DIR/bin/ltx"
LINE="export PATH=\"$HOME_DIR/bin:\$PATH\"  # ltx-video"
for rc in "$HOME/.zshrc" "$HOME/.bash_profile"; do
    [ -f "$rc" ] || [ "$rc" = "$HOME/.zshrc" ] || continue
    grep -q "# ltx-video" "$rc" 2>/dev/null || echo "$LINE" >> "$rc"
done

bold "Done ($MODE). Open a new Terminal window, then:"
say 'ltx "A golden retriever running through a sunny meadow, cinematic lighting" --open'
say ""
say "Quick low-res draft: add --fast.   Stop the background server: ltx --stop"
if [ "$ARCH" = "x86_64" ]; then
    say "On an Intel Mac a clip takes roughly an hour (--fast: about half). Close other apps for speed."
fi
