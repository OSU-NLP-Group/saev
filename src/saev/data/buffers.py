# src/saev/data/buffers.py
import collections.abc
import itertools
import time

import beartype
import numpy as np
import torch
import torch.multiprocessing as mp
from jaxtyping import Shaped, jaxtyped
from torch import Tensor


@beartype.beartype
class RingBuffer:
    """
    Fixed-capacity, multiple-producer / multiple-consumer queue backed by a shared-memory tensor.

    Parameters
    ----------
    slots  : int           capacity in number of items (tensor rows)
    shape  : tuple[int]    shape of one item, e.g. (batch, dim)
    dtype  : torch.dtype   tensor dtype

    put(tensor)  : blocks if full
    get() -> tensor  : blocks if empty
    qsize() -> int        advisory size (approximate)
    close()               frees shared storage (call in the main process)
    """

    def __init__(self, slots: int, shape: tuple[int, ...], dtype: torch.dtype):
        assert slots > 0, "slots must be positive"
        self.slots = slots
        # 123456789 -> Should make you very worried.
        self.buf = torch.full((slots, *shape), 123456789, dtype=dtype)
        self.buf.share_memory_()

        ctx = mp.get_context()  # obeys the global start method ("spawn")

        # shared, lock-free counters
        self.head = ctx.Value("L", 0, lock=False)  # next free slot
        self.tail = ctx.Value("L", 0, lock=False)  # next occupied slot

        # semaphores for blocking semantics
        self.free = ctx.Semaphore(slots)  # initially all slots free
        self.fill = ctx.Semaphore(0)  # no filled slots yet

        # one mutex for pointer updates
        self.mutex = ctx.Lock()

    def put(self, tensor: torch.Tensor) -> None:
        """Copy `tensor` into the next free slot; blocks if the queue is full."""
        if tensor.shape != self.buf.shape[1:] or tensor.dtype != self.buf.dtype:
            raise ValueError("tensor shape / dtype mismatch")

        self.free.acquire()  # wait for a free slot
        with self.mutex:  # exclusive update of head
            idx = self.head.value % self.slots
            self.buf[idx].copy_(tensor)
            self.head.value += 1
        self.fill.release()  # signal there is data

    def get(self) -> torch.Tensor:
        """Return a view of the next item; blocks if the queue is empty."""
        self.fill.acquire()  # wait for data
        with self.mutex:  # exclusive update of tail
            idx = self.tail.value % self.slots
            out = self.buf[idx].clone()
            self.tail.value += 1
        self.free.release()  # signal one more free slot
        return out

    def qsize(self) -> int:
        """Approximate number of filled slots (race-safe enough for tests)."""
        return (self.head.value - self.tail.value) % (1 << 64)

    def fill(self) -> float:
        """Approximate proportion of filled slots (race-safe enough for tests)."""
        return self.qsize() / self.capacity

    def close(self) -> None:
        """Release the shared-memory backing store (call once in the parent)."""
        try:
            self.buf.untyped_storage()._free_shared_mem()
        except (AttributeError, FileNotFoundError):
            pass  # already freed or never allocated


@jaxtyped(typechecker=beartype.beartype)
class ReservoirBuffer:
    """
    Pool of (tensor, meta) pairs.
    Multiple producers call put(batch_x, batch_meta).
    Multiple consumers call get(batch_size) -> (x, meta).
    Random order, each sample delivered once, blocking semantics.
    """

    def __init__(
        self,
        capacity: int,
        shape: tuple[int, ...],
        *,
        dtype=torch.float32,
        seed: int = 0,
        collate_fn: collections.abc.Callable | None = None,
    ):
        self.capacity = capacity
        self.data = torch.full((capacity, *shape), 123456789, dtype=dtype)
        self.data.share_memory_()

        self.ctx = mp.get_context()
        mgr = self.ctx.Manager()
        self.meta = mgr.list([None] * capacity)

        self.size = self.ctx.Value("L", 0)  # current live items
        self.lock = self.ctx.Lock()  # guards size+swap
        self.free = self.ctx.Semaphore(capacity)
        self.full = self.ctx.Semaphore(0)
        # Each process has its own RNG.
        self.rng = np.random.default_rng(seed)

        self.collate_fn = collate_fn

    def put(
        self,
        xs: Shaped[Tensor, "n? ..."],
        metas: collections.abc.Sequence[object] | None = None,
    ):
        if xs.dtype != self.data.dtype:
            raise ValueError("tensor dtype mismatch")

        if xs.shape == self.data.shape[1:]:
            # No batch dim, add one
            xs = xs[None, ...]

        elif xs.shape[1:] == self.data.shape[1:]:
            # Already has a batch dim, we're good.
            pass
        else:
            raise ValueError("tensor shape mismatch")

        n, *_ = xs.shape
        if metas is None:
            metas_it = itertools.repeat(None, n)
        else:
            if len(metas) != n:
                raise ValueError(f"len(xs) = {len(xs)} != len(metas) = {len(metas)}")
            metas_it = iter(metas)

        for x, m in zip(xs, metas_it):
            self.free.acquire()  # block if full
            with self.lock:
                idx = self.size.value  # append at tail
                self.data[idx].copy_(x)
                self.meta[idx] = m
                self.size.value += 1
            self.full.release()

    def get(
        self, bsz: int, timeout: float | None = None
    ) -> tuple[Shaped[Tensor, "bsz ..."], collections.abc.Sequence[object]]:
        n_acquired = 0
        deadline = None if timeout is None else time.monotonic() + timeout

        try:
            for _ in range(bsz):
                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                if not self.full.acquire(timeout=remaining):
                    raise TimeoutError  # couldn’t get the whole batch in time
                n_acquired += 1
        except Exception:
            # Roll back any tokens we already grabbed.
            for _ in range(n_acquired):
                self.full.release()
            raise

        out_x = torch.empty((bsz, *self.data.shape[1:]), dtype=self.data.dtype)
        out_m = [None] * bsz
        with self.lock:
            for i in range(bsz):
                r = self.rng.integers(self.size.value)
                out_x[i].copy_(self.data[r])
                out_m[i] = self.meta[r]

                last = self.size.value - 1  # swap-with-tail
                if r != last:
                    self.data[r].copy_(self.data[last])
                    self.meta[r] = self.meta[last]
                self.size.value -= 1
                self.free.release()
        if self.collate_fn:
            out_m = self.collate_fn(out_m)
        return out_x, out_m

    def close(self) -> None:
        """Release the shared-memory backing store (call once in the parent)."""
        try:
            self.data.untyped_storage()._free_shared_mem()
        except (AttributeError, FileNotFoundError):
            pass  # already freed or never allocated

    def qsize(self) -> int:
        """Approximate number of filled slots (race-safe enough for tests)."""
        return self.size.value

    def fill(self) -> float:
        """Approximate proportion of filled slots (race-safe enough for tests)."""
        return self.qsize() / self.capacity
