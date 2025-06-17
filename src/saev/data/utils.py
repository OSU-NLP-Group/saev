import atexit
import time

import beartype
import torch
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)


@beartype.beartype
class RingBuffer:
    """
    Lock-free single-producer / single-consumer circular buffer backed by torch shared memory. All slots must have the same tensor shape & dtype.

    Parameters
    ----------
    slots : int
        Capacity in number of batches (power of two recommended).
    shape : tuple[int]
        Shape of each sample tensor, e.g. (batch, d_vit).
    dtype : torch.dtype
        Data type stored.
    """

    def __init__(self, slots: int, shape: tuple[int], dtype: torch.dtype):
        self.slots = slots
        self.shape = shape
        self.dtype = dtype
        self._buffer = torch.empty(
            (slots, *shape), dtype=dtype, pin_memory=False, device="cpu"
        )
        self._buffer.share_memory_()
        self._head = mp.Value("L", 0)
        self._tail = mp.Value("L", 0)
        atexit.register(self.close)

    def put(self, tensor: object):
        """
        * Blocks until space is available.
        * Accepts a contiguous tensor with .shape == shape, .dtype == dtype.
        * Tensor storage is *copied* into next slot (no view); caller owns its buffer.
        """
        if tensor.shape != self.shape or tensor.dtype != self.dtype:
            raise ValueError(
                f"Tensor shape {tensor.shape} or dtype {tensor.dtype} does not match buffer "
                f"shape {self.shape} and dtype {self.dtype}"
            )
        # wait for space
        while True:
            with self._head.get_lock(), self._tail.get_lock():
                if self._head.value - self._tail.value < self.slots:
                    idx = self._head.value % self.slots
                    self._head.value += 1
                    break
            time.sleep(0)
        # copy data into buffer slot
        self._buffer[idx].copy_(tensor, non_blocking=True)

    def get(self) -> object:
        """
        * Blocks until an item is available.
        * Returns a tensor *view* into shared memory; caller must .clone() before mutating.
        """
        # wait for item
        while True:
            with self._head.get_lock(), self._tail.get_lock():
                if self._head.value > self._tail.value:
                    idx = self._tail.value % self.slots
                    self._tail.value += 1
                    break
            time.sleep(0)
        return self._buffer[idx]

    def close(self):
        """Frees shared memory, safe to call multiple times."""
        try:
            self._buffer.untyped_storage()._free_shared_mem()
        except Exception:
            pass

    def qsize(self) -> int:
        """Advisory, not exact under racing."""
        with self._head.get_lock(), self._tail.get_lock():
            return self._head.value - self._tail.value


"""
# Implementation guidance

* Allocate one `torch.empty((slots, *shape), dtype, pin_memory=False, device='cpu', shared_memory=True)`.
* Maintain two `mp.Value('L')` counters `head`, `tail`.
* `put` waits while `(head.value - tail.value) >= slots`.
* `get` waits while `head == tail`.
* Use `torch.Tensor.copy_(src, non_blocking=True)` to move user tensor into slot.
* `close` calls `.storage()._free_shared_mem()` then unlinks the `mp.SharedMemory`.

Other tips:

* **Start method**: `mp.set_start_method("spawn", force=True)`.
* **Use torch.shared\_memory**: `torch.tensor(..., shared_memory=True)`.
* **Avoid GIL in hot paths**: counters read/write only under `with head.get_lock():`.
* **Always `clone()` in consumer before in-place ops**.
* **Call `ring.close()` in `atexit` handler** to cover Ctrl-C.
* Document that only one producer & one consumer are supported; extending to MPSC needs atomic CAS.
"""
