import beartype
import torch
import torch.multiprocessing as mp


@beartype.beartype
class RingBuffer:
    """
    Fixed-capacity, multiple-producer / multiple-consumer queue
    backed by a shared-memory tensor.

    Parameters
    ----------
    slots  : int           capacity in number of items (tensor rows)
    shape  : tuple[int]    shape of one item, e.g. (batch, dim)
    dtype  : torch.dtype   tensor dtype

    put(tensor)  : blocks if full
    get() -> tensor view : blocks if empty
    qsize() -> int        advisory size (approximate)
    close()               frees shared storage (call in the main process)
    """

    def __init__(self, slots: int, shape: tuple[int, ...], dtype: torch.dtype):
        assert slots > 0, "slots must be positive"
        self.slots = slots
        self.buf = torch.empty((slots, *shape), dtype=dtype)
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

    # --------------------------------------------------------------------- #
    def get(self) -> torch.Tensor:
        """Return a view of the next item; blocks if the queue is empty."""
        self.fill.acquire()  # wait for data
        with self.mutex:  # exclusive update of tail
            idx = self.tail.value % self.slots
            out = self.buf[idx]
            self.tail.value += 1
        self.free.release()  # signal one more free slot
        return out

    # --------------------------------------------------------------------- #
    def qsize(self) -> int:
        """Approximate number of filled slots (race-safe enough for tests)."""
        return (self.head.value - self.tail.value) % (1 << 64)

    def close(self) -> None:
        """Release the shared-memory backing store (call once in the parent)."""
        try:
            self.buf.untyped_storage()._free_shared_mem()
        except (AttributeError, FileNotFoundError):
            pass  # already freed or never allocated
