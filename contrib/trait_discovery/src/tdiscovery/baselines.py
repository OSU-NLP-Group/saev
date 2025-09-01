import io
import json
import logging
import os.path

import beartype
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

import saev.data
from saev import helpers

from . import cub200


@jaxtyped(typechecker=beartype.beartype)
class Scorer(torch.nn.Module):
    """ """

    def __init__(self):
        super().__init__()
        self._trained = False

    @property
    def n_prototypes(self) -> int:
        """ """
        raise NotImplementedError()

    @property
    def kwargs(self) -> dict[str, object]:
        """The constructor's kwargs."""
        raise NotImplementedError()

    def train(self, dataloader: saev.data.ShuffledDataLoader):
        """ """
        raise NotImplementedError()

    def forward(self, activations: Float[Tensor, "B D"]) -> Float[Tensor, "B K"]:
        """ """

        raise NotImplementedError()

    def load_state_dict(self, *args, **kwargs):
        self._trained = True
        super().load_state_dict(*args, **kwargs)


@beartype.beartype
def load(fpath: str) -> Scorer:
    """Loads a `Scorer` from disk."""
    with open(fpath, "rb") as fd:
        header = json.loads(fd.readline())
        buffer = io.BytesIO(fd.read())

    if header["schema"] == 1:
        cls = globals()[header["cls"]]
        scorer = cls(**header["kwargs"])
    else:
        raise ValueError(f"Unknown schema version: {header['schema']}")

    scorer.load_state_dict(torch.load(buffer, weights_only=True, map_location="cpu"))
    return scorer


@beartype.beartype
def dump(scorer: Scorer, fpath: str):
    """
    Save an Scorer checkpoint to disk along with configuration, using the [trick from equinox](https://docs.kidger.site/equinox/examples/serialisation).

    Arguments:
        scorer: checkpoint to save.
        fpath: filepath to save checkpoint to.
    """
    header = {
        "schema": 1,
        "kwargs": scorer.kwargs,
        "cls": scorer.__class__.__name__,
        "commit": helpers.current_git_commit() or "unknown",
    }

    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "wb") as fd:
        header_str = json.dumps(header)
        fd.write((header_str + "\n").encode("utf-8"))
        torch.save(scorer.state_dict(), fd)


@beartype.beartype
class RandomVectors(Scorer):
    def __init__(self, *, n_prototypes: int, d: int, seed: int):
        super().__init__()
        self._trained = False

        self._n_prototypes = n_prototypes
        self._d = d
        self._seed = seed

        self.register_buffer("prototypes_KD", torch.empty(self.n_prototypes, self._d))

    @property
    def n_prototypes(self) -> int:
        return self._n_prototypes

    @property
    def kwargs(self) -> dict[str, object]:
        """The constructor's kwargs."""
        return dict(n_prototypes=self.n_prototypes, d=self._d, seed=self._seed)

    def train(self, dataloader: saev.data.ShuffledDataLoader):
        """Uniformly sample n_prototypes vectors from a streaming DataLoader using reservoir sampling.

        Args:
            dataloader: DataLoader.
        """
        d_vit = saev.data.Metadata.load(dataloader.cfg.shard_root).d_vit
        assert d_vit == self._d, "mismatch in ViT dimension"

        n_seen = 0
        rng = torch.Generator().manual_seed(self._seed)

        for batch in helpers.progress(dataloader):
            x_BD = batch["act"]
            b, d = x_BD.shape

            # 1. Fill reservoir if not full
            need = max(self.n_prototypes - n_seen, 0)
            if need:
                take = min(need, b)
                self.prototypes_KD[n_seen : n_seen + take] = x_BD[:take]
                n_seen += take
                x_BD = x_BD[take:]
                b -= take
                assert b >= 0  # take is at most bsz, so bsz - take >= 0
                if b == 0:
                    continue

            # 2. Vectorized replacement for remaining items in batch
            idxs_B = torch.arange(n_seen, n_seen + b)  # global indices
            probs = (self.n_prototypes / (idxs_B + 1)).to(
                dtype=torch.float32
            )  # shape (B,)
            keep_mask = torch.rand(b, generator=rng) < probs  # Bernoulli draws
            n_keep = keep_mask.sum().item()
            if n_keep:
                replace_pos = torch.randint(
                    0, self.n_prototypes, (n_keep,), generator=rng
                )
                self.prototypes_KD[replace_pos] = x_BD[keep_mask]
            n_seen += b

        self._trained = True

    def forward(self, activations_BD: Float[Tensor, "B D"]) -> Float[Tensor, "B K"]:
        if not self._trained:
            raise RuntimeError("Call train() first.")

        return activations_BD @ self.prototypes_KD.T


@beartype.beartype
class KMeans(Scorer):
    def __init__(self, *, n_means: int, d: int, seed: int, device: str = "cpu"):
        super().__init__()

        self._n_means = n_means
        self._d = d
        self._seed = seed
        self._device = device

        self.register_buffer("means_KD", torch.empty(self._n_means, self._d))

    @property
    def n_prototypes(self) -> int:
        return self._n_means

    def train(self, dataloader: saev.data.ShuffledDataLoader):
        raise NotImplementedError()

    def forward(self, activations_BD: Float[Tensor, "B D"]) -> Float[Tensor, "B K"]:
        if not self._trained:
            raise RuntimeError("Call train() first.")

        return activations_BD @ self.means_KD.T


@beartype.beartype
class PCA(Scorer):
    def __init__(self, *, n_components: int, d: int, seed: int, device: str = "cpu"):
        super().__init__()

        self._n_components = n_components
        self._d = d
        self._seed = seed
        self._device = device

        self.register_buffer("components_KD", torch.empty(self._n_components, self._d))

    @property
    def n_prototypes(self) -> int:
        return self._n_components

    def train(self, dataloader: saev.data.ShuffledDataLoader):
        raise NotImplementedError()

    def forward(self, activations_BD: Float[Tensor, "B D"]) -> Float[Tensor, "B K"]:
        if not self._trained:
            raise RuntimeError("Call train() first.")

        return activations_BD @ self.components_KD.T


@beartype.beartype
class LinearClassifier(Scorer):
    def __init__(
        self,
        *,
        cub_root: str,
        d: int,
        seed: int,
        n_traits: int = 312,
        device: str = "cpu",
    ):
        super().__init__()

        self._d = d
        self._n_traits = n_traits
        self._seed = seed
        self._device = device
        self._trained = False
        self._cub_root = cub_root

        # Store all training hyperparameters
        self._lr = 1e-3
        self._patience = 20
        self._min_delta = 1e-3
        self._loss_window_size = 10
        self._max_steps = 10_000
        self._log_every = 10
        self._convergence_check_every = 10

        self.linear = torch.nn.Linear(d, n_traits, device=device)
        self._train_y_true_NT = cub200.load_attrs(cub_root, is_train=True)

        self.logger = logging.getLogger("linear-clf")

    @property
    def n_prototypes(self) -> int:
        return self._n_traits

    @property
    def kwargs(self) -> dict[str, object]:
        """The constructor's kwargs."""
        return dict(
            cub_root=self._cub_root,
            d=self._d,
            seed=self._seed,
            n_traits=self._n_traits,
            device=self._device,
        )

    def train(self, dataloader: saev.data.ShuffledDataLoader):
        """Fits a linear classifier using the image-level traits as supervision.

        Uses AdamW optimizer with early stopping based on validation loss.
        Automatically determines convergence without needing to set epochs.

        Args:
            dataloader: DataLoader.
        """
        d_vit = saev.data.Metadata.load(dataloader.cfg.shard_root).d_vit
        assert d_vit == self._d, "mismatch in ViT dimension"

        # Use AdamW for faster convergence
        optim = torch.optim.AdamW(self.linear.parameters(), lr=self._lr)

        # Early stopping state
        best_loss = float("inf")
        patience_counter = 0

        # Moving average for loss smoothing
        loss_history = []

        step = 0
        converged = False

        while not converged:
            for batch in dataloader:
                # Forward pass
                logits_BK = self.linear(batch["act"])
                y_true_BK = self._train_y_true_NT[batch["image_i"]].float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits_BK, y_true_BK
                )

                # Backward pass
                optim.zero_grad()
                loss.backward()

                optim.step()

                # Track loss
                current_loss = loss.item()
                loss_history.append(current_loss)

                n_losses = min(len(loss_history), self._loss_window_size)
                # Log progress periodically
                if step % self._log_every == 0:
                    avg_loss = sum(loss_history[-self._loss_window_size :]) / n_losses
                    self.logger.info(
                        "step: %d, loss: %.5f, avg_loss: %.5f",
                        step,
                        current_loss,
                        avg_loss,
                    )

                # Check for convergence periodically
                if step > 0 and step % self._convergence_check_every == 0:
                    avg_loss = sum(loss_history[-self._loss_window_size :]) / n_losses

                    # Check if loss improved
                    if avg_loss < best_loss - self._min_delta:
                        best_loss = avg_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Early stopping check
                    if patience_counter >= self._patience:
                        self.logger.info(
                            "Early stopping triggered at step %d with loss %.5f",
                            step,
                            avg_loss,
                        )
                        converged = True
                        break

                step += 1

                # Hard limit to prevent infinite training
                if step >= self._max_steps:
                    self.logger.info("Reached maximum steps limit")
                    converged = True
                    break

            # Break outer loop if converged
            if converged:
                break

        self._trained = True
        self.logger.info("Training completed in %d steps", step)

    def forward(self, activations_BD: Float[Tensor, "B D"]) -> Float[Tensor, "B K"]:
        if not self._trained:
            raise RuntimeError("Call train() first.")

        with torch.no_grad():
            return self.linear(activations_BD)
