"""Quantize the official LTX-2.5 Gemma 4 12B text encoder (bf16 safetensors) to GGUF.

Streams one tensor (in row chunks) at a time so it runs on an 8 GB Mac.
Keeps ComfyUI-native tensor names so ComfyUI-GGUF's CLIPLoaderGGUF can pass them
straight to ComfyUI's LTX-2.5 Gemma 4 text encoder.

Policy (default):
  Gemma decoder-layer linear weights ......... Q5_1  (--layer-type)
  LTX text_embedding_projection heads ........ Q8_0
  token embeddings, vision/audio projectors .. Q8_0
  1-D tensors (norms, biases, scalars) ....... F32
  3-D tensors ................................ F32
  U8 asset tensors (tokenizer, configs) ...... I16 (lossless byte values)

Usage: python -I quantize_encoder.py SRC.safetensors DST.gguf [--layer-type Q5_1]
"""
import argparse, json, sys, time
import numpy as np
import torch
import gguf
from gguf.quants import quantize
from safetensors import safe_open

Q = gguf.GGMLQuantizationType
CHUNK_ELEMS = 16 * 1024 * 1024  # floats per quantization chunk (~64 MB fp32)


def pick_type(name, shape, src_dtype, layer_type):
    if src_dtype == torch.uint8:
        return Q.I16
    if len(shape) != 2:
        return Q.F32
    if shape[1] % 32 != 0:
        return Q.F16
    if name.startswith("model.layers."):
        return layer_type
    return Q.Q8_0


def byte_shape(shape, qtype):
    if qtype in (Q.F32, Q.F16, Q.I16):
        return shape
    bs, ts = gguf.GGML_QUANT_SIZES[qtype]
    return (shape[0], shape[1] // bs * ts)


def np_dtype(qtype):
    return {Q.F32: np.float32, Q.F16: np.float16, Q.I16: np.int16}.get(qtype, np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--layer-type", default="Q5_1", choices=["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"])
    a = ap.parse_args()
    layer_type = Q[a.layer_type]

    f = safe_open(a.src, framework="pt")
    names = sorted(f.keys())
    plan = []
    for n in names:
        sl = f.get_slice(n)
        shape = tuple(sl.get_shape())
        dt = {"BF16": torch.bfloat16, "U8": torch.uint8, "F32": torch.float32, "F16": torch.float16}[sl.get_dtype()]
        qt = pick_type(n, shape, dt, layer_type)
        bshape = byte_shape(shape, qt)
        nbytes = int(np.prod(bshape)) * np.dtype(np_dtype(qt)).itemsize
        plan.append((n, shape, qt, bshape, nbytes))

    w = gguf.GGUFWriter(a.dst, arch="gemma4")
    w.add_name("gemma4-12b-with-proj-ltx-2.5")
    w.add_file_type(gguf.LlamaFileType.MOSTLY_Q5_1 if layer_type == Q.Q5_1 else gguf.LlamaFileType.MOSTLY_Q8_0)
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)
    w.add_string("ltx.text_encoder.version", "gemma4-12b-ltx-v1")
    w.add_string("comfy.gguf.tensor_name_layout", "comfy_state_dict")
    w.add_string("quantize.source", "Lightricks/LTX-2.5 text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
    w.add_string("quantize.policy", f"layers={a.layer_type}; projections/embeddings=Q8_0; 1D/3D=F32; assets=I16")
    meta = f.metadata() or {}
    if "gemma_config" in meta:
        w.add_string("ltx.gemma_config", meta["gemma_config"])
    for n, shape, qt, bshape, nbytes in plan:
        w.add_tensor_info(n, bshape, np.dtype(np_dtype(qt)), nbytes, raw_dtype=None if qt in (Q.F32, Q.F16, Q.I16) else qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    fout = w.fout[0]

    t0 = time.time(); total = 0
    counts = {}
    for i, (n, shape, qt, bshape, nbytes) in enumerate(plan):
        ti = w.tensors[0].pop(n)
        assert ti.nbytes == nbytes
        w.write_padding(fout, fout.tell())
        start = fout.tell()
        if qt == Q.I16:
            f.get_tensor(n).numpy().astype(np.int16).tofile(fout)
        elif qt in (Q.F32, Q.F16):
            f.get_tensor(n).to(torch.float32).numpy().astype(np_dtype(qt)).tofile(fout)
        else:
            sl = f.get_slice(n)
            rows_per = max(1, CHUNK_ELEMS // shape[1])
            for r in range(0, shape[0], rows_per):
                chunk = sl[r:r + rows_per].to(torch.float32).numpy()
                quantize(chunk, qt).tofile(fout)
        assert fout.tell() - start == nbytes, (n, fout.tell() - start, nbytes)
        w.write_padding(fout, nbytes)
        total += nbytes
        counts[qt.name] = counts.get(qt.name, 0) + 1
        if i % 20 == 0 or nbytes > 2e8:
            print(f"[{i+1}/{len(plan)}] {n} {shape} -> {qt.name}  {total/1e9:.2f} GB  {time.time()-t0:.0f}s", flush=True)
    w.state = gguf.gguf_writer.WriterState.WEIGHTS
    w.close()
    print("done", counts, f"{total/1e9:.2f} GB in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
