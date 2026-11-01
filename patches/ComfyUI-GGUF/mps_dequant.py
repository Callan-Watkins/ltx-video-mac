# Metal (Apple GPU) dequantization kernels for GGUF block formats.
# The generic torch path builds each weight from a dozen elementwise ops; on an Apple GPU that costs
# ~3x the matmul it feeds. These kernels unpack one weight per thread in a single pass, with the
# same block layouts and formulas as dequant.py (math is done in float32, then rounded once).
import gguf
import torch

_SRC = r"""
#include <metal_stdlib>
using namespace metal;

inline float f16(device const uchar* p) { return float(*(device const half*)p); }

inline void scale_min_k4(int j, device const uchar* q, thread float& sc, thread float& m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >>  4) | ((q[j]     >> 6) << 4);
    }
}

#define KERNEL(NAME, BS, TS) \
kernel void NAME##_bf16(device const uchar* src [[buffer(0)]], device bfloat* dst [[buffer(1)]], uint i [[thread_position_in_grid]]) { \
    device const uchar* b = src + (i / BS) * TS; uint j = i % BS; dst[i] = bfloat(NAME(b, j)); } \
kernel void NAME##_f16(device const uchar* src [[buffer(0)]], device half* dst [[buffer(1)]], uint i [[thread_position_in_grid]]) { \
    device const uchar* b = src + (i / BS) * TS; uint j = i % BS; dst[i] = half(NAME(b, j)); } \
kernel void NAME##_f32(device const uchar* src [[buffer(0)]], device float* dst [[buffer(1)]], uint i [[thread_position_in_grid]]) { \
    device const uchar* b = src + (i / BS) * TS; uint j = i % BS; dst[i] = NAME(b, j); }

inline float q8_0(device const uchar* b, uint j) { return f16(b) * float(((device const char*)(b + 2))[j]); }
KERNEL(q8_0, 32, 34)

inline float q5_1(device const uchar* b, uint j) {
    uint qh = uint(b[4]) | (uint(b[5]) << 8) | (uint(b[6]) << 16) | (uint(b[7]) << 24);
    uint ql = (b[8 + (j & 15)] >> ((j >> 4) * 4)) & 0xF;
    return f16(b) * float(ql | (((qh >> j) & 1) << 4)) + f16(b + 2);
}
KERNEL(q5_1, 32, 24)

inline float q2_k(device const uchar* b, uint j) {
    uint s = b[j / 16];
    uint q = (b[16 + (j / 128) * 32 + (j % 32)] >> (((j % 128) / 32) * 2)) & 3;
    return f16(b + 80) * float(s & 0xF) * float(q) - f16(b + 82) * float(s >> 4);
}
KERNEL(q2_k, 256, 84)

inline float q3_k(device const uchar* b, uint j) {
    device const uchar* hmask = b; device const uchar* qs = b + 32; device const uchar* sc = b + 96;
    uint si = j / 16;
    int ls = (sc[si % 8] >> ((si / 8) * 4)) & 0xF;
    int hs = (sc[8 + si % 4] >> ((si / 4) * 2)) & 3;
    int scale = (ls | (hs << 4)) - 32;
    int ql = (qs[(j / 128) * 32 + (j % 32)] >> (((j % 128) / 32) * 2)) & 3;
    int qh = ((hmask[j % 32] >> (j / 32)) & 1) ^ 1;
    return f16(b + 108) * float(scale) * float(ql - (qh << 2));
}
KERNEL(q3_k, 256, 110)

inline float q4_k(device const uchar* b, uint j) {
    float sc, m; scale_min_k4(j / 32, b + 4, sc, m);
    uint q = (b[16 + (j / 64) * 32 + (j % 32)] >> (((j % 64) / 32) * 4)) & 0xF;
    return f16(b) * sc * float(q) - f16(b + 2) * m;
}
KERNEL(q4_k, 256, 144)

inline float q5_k(device const uchar* b, uint j) {
    float sc, m; scale_min_k4(j / 32, b + 4, sc, m);
    uint ql = (b[48 + (j / 64) * 32 + (j % 32)] >> (((j % 64) / 32) * 4)) & 0xF;
    uint qh = (b[16 + (j % 32)] >> (j / 32)) & 1;
    return f16(b) * sc * float(ql | (qh << 4)) - f16(b + 2) * m;
}
KERNEL(q5_k, 256, 176)

inline float q6_k(device const uchar* b, uint j) {
    uint ql = (b[(j / 128) * 64 + (j % 64)] >> (((j % 128) / 64) * 4)) & 0xF;
    uint qh = (b[128 + (j / 128) * 32 + (j % 32)] >> (((j % 128) / 32) * 2)) & 3;
    int q = int(ql | (qh << 4)) - 32;
    return f16(b + 208) * float(((device const char*)(b + 192))[j / 16]) * float(q);
}
KERNEL(q6_k, 256, 210)
"""

Q = gguf.GGMLQuantizationType
_NAMES = {Q.Q8_0: "q8_0", Q.Q5_1: "q5_1", Q.Q2_K: "q2_k", Q.Q3_K: "q3_k",
          Q.Q4_K: "q4_k", Q.Q5_K: "q5_k", Q.Q6_K: "q6_k"}
_SUFFIX = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
_lib = None


def supported(data, qtype, dtype):
    return data.device.type == "mps" and qtype in _NAMES and (dtype or torch.float32) in _SUFFIX


def dequantize(data, qtype, oshape, dtype=None):
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(_SRC)
    dtype = dtype or torch.float32
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    src = data.view(torch.uint8).contiguous()
    n = src.numel() // type_size * block_size
    out = torch.empty(n, dtype=dtype, device=data.device)
    getattr(_lib, f"{_NAMES[qtype]}_{_SUFFIX[dtype]}")(src, out, threads=n)
    return out.reshape(oshape)
