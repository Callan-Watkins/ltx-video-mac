# Read-ahead weight streaming for GGUF files on small-memory Macs.
# When a model is bigger than RAM every sampling step re-reads it from SSD. Paging it in through
# the mmap faults in small pieces (~0.5 GB/s); one large pread per tensor runs at ~1.3 GB/s.
# The streamer also learns the order the model asks for tensors during the first pass, and after
# that reads the next ones on a background thread while the GPU is busy with the current layer.
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import torch

MIN_BYTES = 1 << 20           # smaller tensors just use the normal mmap path
ENABLED = os.environ.get("GGUF_STREAM", "1") != "0"
# Kept small on purpose: on an 8 GB Mac anything much bigger gets pushed to swap before it is used.
READ_AHEAD_BYTES = 192 << 20


class FileStreamer:
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gguf-read")
        self.lock = threading.Lock()
        self.order = []        # (offset, nbytes) in the order they were first requested
        self.index = {}        # offset -> position in self.order
        self.pending = {}      # offset -> Future of a uint8 tensor

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


_streamers = {}


def streamer(path):
    if path not in _streamers:
        _streamers[path] = FileStreamer(path)
    return _streamers[path]


def to_device(src, like, *args, **kwargs):
    """Copy the bytes behind mmap-backed tensor `like` (src = path, offset, nbytes) to a device."""
    path, offset, nbytes = src
    raw = streamer(path).get(offset, nbytes)
    like = like.as_subclass(torch.Tensor)  # raw storage layout, not the logical (dequantized) shape
    return raw.view(like.dtype).reshape(like.shape).to(*args, **kwargs)
