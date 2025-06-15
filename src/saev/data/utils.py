import beartype
import torch


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

    def __init__(self, slots: int, shape: tuple[int], dtype: torch.dtype): ...

    def put(self, tensor: object):
        """
        * Blocks until space is available.
        * Accepts a contiguous tensor with .shape == shape, .dtype == dtype.
        * Tensor storage is *copied* into next slot (no view); caller owns its buffer.
        """

    def get(self) -> object:
        """
        * Blocks until an item is available.
        * Returns a tensor *view* into shared memory; caller must .clone() before mutating.
        """

    def close(self):
        """Frees shared memory, safe to call multiple times."""

    def qsize(self) -> int:
        """Advisory, not exact under racing."""
