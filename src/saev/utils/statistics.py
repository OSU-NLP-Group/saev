import beartype
import torch
from torch import Tensor


@beartype.beartype
class PercentileEstimator:
    def __init__(
        self,
        percentile: float | int,
        total: int,
        lr: float = 1e-3,
        shape: tuple[int, ...] = (),
    ):
        self.percentile = percentile
        self.total = total
        self.lr = lr

        self._estimate = torch.zeros(shape)
        self._step = 0

    def update(self, x):
        """
        Update the estimator with a new value.

        This method maintains the marker positions using the P2 algorithm rules. When a new value arrives, it's placed in the appropriate position relative to existing markers, and marker positions are adjusted to maintain their desired percentile positions.

        Arguments:
            x: The new value to incorporate into the estimation
        """
        self._step += 1

        step_size = self.lr * (self.total - self._step) / self.total

        # Is a no-op if it's already on the same device.
        if isinstance(x, Tensor):
            self._estimate = self._estimate.to(x.device)

        self._estimate += step_size * (
            torch.sign(x - self._estimate) + 2 * self.percentile / 100 - 1.0
        )

    @property
    def estimate(self):
        return self._estimate
