# Read-ahead weight streaming for GGUF files on small-memory Macs.
# When a model is bigger than RAM every sampling step re-reads it from SSD. Paging it in through
# the mmap faults in small pieces (~0.5 GB/s); one large pread per tensor runs at ~1.3 GB/s.
# The streamer also learns the order the model asks for tensors during the first pass, and after
# that reads the next ones on a background thread while the GPU is busy with the current layer.
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch

MIN_BYTES = 1 << 20           # smaller tensors just use the normal mmap path
ENABLED = os.environ.get("GGUF_STREAM", "1") != "0"
# Kept small on purpose: on an 8 GB Mac anything much bigger gets pushed to swap before it is used.
READ_AHEAD_BYTES = 192 << 20
# Move each transformer block to the GPU in one piece (see BlockStreamer).
BLOCKS = ENABLED and os.environ.get("GGUF_STREAM_BLOCKS", "1") != "0"
STATS = bool(os.environ.get("LTX_MEMLOG"))
GROUP_RE = re.compile(r"(.*?\.\d+)\.")


def group_of(name):
    """Tensors that are used together: everything up to the first layer index, else the top module."""
    m = GROUP_RE.match(name)
    return m.group(1) if m else name.split(".")[0]


class FileStreamer:
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gguf-read")
        self.lock = threading.Lock()
        self.order = []        # (offset, nbytes) in the order they were first requested
        self.index = {}        # offset -> position in self.order
        self.pending = {}      # offset -> Future of a uint8 tensor
        self.calls = 0

    def _read(self, offset, nbytes):
        buf = torch.empty(nbytes, dtype=torch.uint8)
        view, done = memoryview(buf.numpy()), 0
        while done < nbytes:
            n = os.preadv(self.fd, [view[done:]], offset + done)
            if n <= 0:
                raise IOError(f"short read at {offset + done}")
            done += n
        return buf

    def get(self, offset, nbytes):
        self.calls += 1
        if self.calls % 32 == 0 and torch.backends.mps.is_available():  # see BlockStreamer: keep Metal's cache small
            torch.mps.empty_cache()
        with self.lock:
            fut = self.pending.pop(offset, None)
            pos = self.index.get(offset)
            if pos is None:
                self.index[offset] = len(self.order)
                self.order.append((offset, nbytes))
            else:
                window, ahead, i = set(), 0, pos + 1
                while ahead < READ_AHEAD_BYTES and i - pos < len(self.order):
                    o, n = self.order[i % len(self.order)]
                    window.add(o)
                    if o != offset and o not in self.pending:
                        self.pending[o] = self.pool.submit(self._read, o, n)
                    ahead += n
                    i += 1
                # the model skipped ahead (new pass, other stage): drop reads it will not ask for next
                for o in [o for o in self.pending if o not in window]:
                    self.pending.pop(o).cancel()
        return fut.result() if fut is not None else self._read(offset, nbytes)


class BlockStreamer:
    """Copy the diffusion model to the GPU one group of tensors (a transformer block, an AdaLN
    module, ...) at a time, through two reused GPU buffers.

    Moving tensors one by one allocates a new Metal buffer and waits for the GPU on every copy, about
    3,000 times per step; with a 16 GB model that cost more time than the matmuls. Here a background
    thread gathers the group the model will ask for next (learned from the previous step) into a CPU
    staging buffer while the GPU works, then a single non-blocking copy moves it, and weights are
    views into that GPU buffer. The GPU stream is in order, so a buffer is only overwritten after the
    group that used it before has run; the staging buffer is refilled once its last copy has finished.
    """
    def __init__(self, path, groups):
        self.fd = os.open(path, os.O_RDONLY)
        self.where = {}        # tensor offset -> (group, position in the group buffer, nbytes)
        self.used = {}         # group -> bytes used in its buffer
        for g, tensors in groups.items():
            pos = 0
            for offset, nbytes in sorted(tensors):
                self.where[offset] = (g, pos, nbytes)
                pos += -(-nbytes // 64) * 64
            self.used[g] = pos
        self.size = max(self.used.values())
        self.staging = None    # CPU buffer, allocated on first use
        self.copied = None     # event: the staging buffer's last copy to the GPU has finished
        self.slots = None      # the two GPU buffers, allocated on first use
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gguf-group")
        self.next_of = {}      # group -> the group that followed it last time
        self.first = None      # first group of a step
        self.pending = None    # (group, Future) of the read-ahead
        self.current = None
        self.k = 1             # which GPU buffer holds the current group
        self.switches = 0
        self.waited = 0.0

    def _gather(self, g):
        if self.copied is not None:
            self.copied.synchronize()
        view = memoryview(self.staging.numpy())
        for offset, (grp, pos, nbytes) in self.where.items():
            if grp != g:
                continue
            done = 0
            while done < nbytes:
                n = os.preadv(self.fd, [view[pos + done:pos + nbytes]], offset + done)
                if n <= 0:
                    raise IOError(f"short read at {offset + done}")
                done += n

    def get(self, offset, device):
        g, pos, nbytes = self.where[offset]
        if g != self.current:
            self._switch(g, device)
        return self.slots[self.k][pos:pos + nbytes]

    def _switch(self, g, device):
        if self.current is not None:
            self.next_of[self.current] = g
        if self.first is None:
            self.first = g
        if g == self.first and STATS and self.switches:
            logging.info("[stream] %d group switches, %.1fs waiting for reads", self.switches, self.waited)
            self.switches, self.waited = 0, 0.0
        self.switches += 1
        if self.switches % 2 == 0:  # hand Metal's cache of freed buffers back to the system
            torch.mps.empty_cache()
        t0 = time.time()
        if self.staging is None:
            self.staging = torch.empty(self.size, dtype=torch.uint8)
            self.slots = [torch.empty(self.size, dtype=torch.uint8, device=device) for _ in range(2)]
        if self.pending is not None:
            guess, fut = self.pending
            self.pending = None
            fut.result()
            if guess != g:
                self._gather(g)
        else:
            self._gather(g)
        self.waited += time.time() - t0
        self.k = 1 - self.k
        self.slots[self.k][:self.used[g]].copy_(self.staging[:self.used[g]], non_blocking=True)
        self.copied = torch.mps.Event()
        self.copied.record()
        self.current = g
        nxt = self.next_of.get(g)
        if nxt is not None:
            self.pending = (nxt, self.pool.submit(self._gather, nxt))

    def release(self):
        if self.pending is not None:
            self.pending[1].result()
            self.pending = None
        if self.copied is not None:
            torch.mps.synchronize()
        self.slots = self.staging = self.current = self.copied = None


_streamers = {}
_block_streamers = {}


def block_streamer(path, groups=None):
    """The block streamer for `path`; `groups` ({group: [(offset, nbytes)]}) creates it."""
    if groups is not None:
        _block_streamers[path] = BlockStreamer(path, groups)
    return _block_streamers.get(path)


def release_blocks():
    for s in _block_streamers.values():
        s.release()
    from . import mps_dequant
    mps_dequant.release()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def streamer(path):
    if path not in _streamers:
        _streamers[path] = FileStreamer(path)
    return _streamers[path]


def to_device(src, like, *args, **kwargs):
    """Copy the bytes behind mmap-backed tensor `like` (src = path, offset, nbytes) to a device."""
    path, offset, nbytes = src
    device = torch._C._nn._parse_to(*args, **kwargs)[0]
    blocks = _block_streamers.get(path)
    if blocks is not None and offset in blocks.where and device.type == "mps":
        raw = blocks.get(offset, device)
    else:
        raw = streamer(path).get(offset, nbytes)
    like = like.as_subclass(torch.Tensor)  # raw storage layout, not the logical (dequantized) shape
    return raw.view(like.dtype).reshape(like.shape).to(*args, **kwargs)
