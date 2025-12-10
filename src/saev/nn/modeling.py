"""Neural network architectures for sparse autoencoders."""

import dataclasses
import io
import json
import logging
import pathlib
import typing as tp

import beartype
import einops
import orjson
import torch
import tyro.conf
from jaxtyping import Float, Int64, jaxtyped
from torch import Tensor

from .. import __version__, helpers

SCHEMA_VERSION = 5


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class NoSparsity:
    """No explicit sparsity penalty (e.g. for TopK/BatchTopK where k controls sparsity)."""

    key: tyro.conf.Suppress[tp.Literal["no-sparsity"]] = "no-sparsity"

    def loss(self, f_x: Float[Tensor, "batch d_sae"]) -> Float[Tensor, ""]:
        return torch.tensor(0.0)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class L1Sparsity:
    key: tyro.conf.Suppress[tp.Literal["l1-sparsity"]] = "l1-sparsity"
    coeff: float = 1e-4

    def loss(self, f_x: Float[Tensor, "batch d_sae"]) -> Float[Tensor, ""]:
        l1 = f_x.abs().sum(axis=1).mean(axis=0)
        return l1 * self.coeff


Sparsity = NoSparsity | L1Sparsity


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class NoAux:
    """No auxiliary loss (e.g., for ReLU)."""

    key: tyro.conf.Suppress[tp.Literal["no-aux"]] = "no-aux"

    def loss(
        self,
        *,
        sae: "SparseAutoencoder",
        x: Float[Tensor, "batch d_model"],
        out: "SparseAutoencoder.Output",
        dead_mask: Float[Tensor, " d_sae"] | None,
    ) -> Float[Tensor, ""]:
        return torch.zeros((), device=x.device, dtype=x.dtype)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class AuxK:
    """AuxK auxiliary reconstruction loss for dead latents."""

    key: tyro.conf.Suppress[tp.Literal["auxk"]] = "auxk"
    k_aux: int = 512
    alpha: float = 1 / 32

    def loss(
        self,
        *,
        sae: "SparseAutoencoder",
        x: Float[Tensor, "batch d_model"],
        out: "SparseAutoencoder.Output",
        dead_mask: Float[Tensor, " d_sae"] | None,
    ) -> Float[Tensor, ""]:
        if sae.training:
            assert dead_mask is not None, "dead_mask required during training"
        else:
            assert dead_mask is None, "dead_mask must be None during eval"
            return x.new_zeros(())

        x_hat = out.x_hats[:, -1, :]
        residual = (x - x_hat).detach()
        masked = out.h_x.masked_fill(~dead_mask, float("-inf"))
        n_dead = int(dead_mask.sum().item())
        k_use = min(self.k_aux, n_dead)
        if k_use == 0:
            return residual.new_zeros(())

        _, top_i = masked.topk(k_use, dim=-1)
        aux_acts = torch.zeros_like(out.h_x)
        aux_acts.scatter_(-1, top_i, out.h_x.gather(-1, top_i))

        # Use full-prefix reconstruction (last prefix) to match main x_hat shape
        aux_recon = sae.decode(aux_acts)[:, -1, :]
        return self.alpha * (aux_recon - residual).pow(2).mean()


Aux = AuxK | NoAux


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Relu:
    """Vanilla ReLU"""

    key: tyro.conf.Suppress[tp.Literal["relu"]] = "relu"
    sparsity: Sparsity = L1Sparsity(coeff=4e-4)
    aux: Aux = NoAux()


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class TopK:
    key: tyro.conf.Suppress[tp.Literal["top-k"]] = "top-k"
    top_k: int = 32
    """How many values are allowed to be non-zero."""
    sparsity: Sparsity = NoSparsity()
    aux: Aux = AuxK()

    def __post_init__(self):
        assert self.top_k > 0, "top_k must be a positive integer."


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class BatchTopK:
    key: tyro.conf.Suppress[tp.Literal["batch-top-k"]] = "batch-top-k"
    top_k: int = 32
    """How many values are allowed to be non-zero per sample in the batch."""
    sparsity: Sparsity = NoSparsity()
    momentum: float = 0.1
    aux: AuxK = AuxK()

    def __post_init__(self):
        assert self.top_k > 0, "top_k must be a positive integer."


ActivationConfig = Relu | TopK | BatchTopK
Config = ActivationConfig
_ACTIVATION_REGISTRY = {
    cls.__name__: cls
    for cls in (Relu, TopK, BatchTopK, AuxK, NoAux, L1Sparsity, NoSparsity)
}


@jaxtyped(typechecker=beartype.beartype)
class ReluActivation(torch.nn.Module):
    def __init__(self, cfg: Relu):
        super().__init__()
        self.cfg = cfg

    def forward(self, x: Float[Tensor, "batch d_sae"]) -> Float[Tensor, "batch d_sae"]:
        return torch.nn.functional.relu(x)


@jaxtyped(typechecker=beartype.beartype)
class TopKActivation(torch.nn.Module):
    """
    Top-K activation function. For use as activation function of sparse encoder.
    """

    def __init__(self, cfg: TopK):
        super().__init__()
        self.cfg = cfg

    def forward(self, x: Float[Tensor, "batch d_sae"]) -> Float[Tensor, "batch d_sae"]:
        """
        Apply top-k activation to the input tensor.
        """

        bsz, d_sae = x.shape
        k = min(self.cfg.top_k, d_sae)
        _, idxs = torch.topk(x, k, dim=-1, sorted=False)
        mask = torch.zeros_like(x).scatter(-1, idxs, 1.0)

        return torch.mul(mask, x)


@jaxtyped(typechecker=beartype.beartype)
class BatchTopKActivation(torch.nn.Module):
    """
    BatchTopK activation and inference-time threshold for sparse autoencoders.

    This module implements a BatchTopK nonlinearity that enforces a fixed sparsity budget across a batch, together with an inference-time approximation that replaces the batch-coupled operation with a simple elementwise threshold.

    Training mode (model.train()):
        Given pre-activation codes x with shape [batch, d_sae], the BatchTopK activation flattens the batch to shape [batch * d_sae], selects the largest (batch * top_k) entries by value, and sets all other entries to zero. This enforces an average of exactly `top_k` active features per example while allowing the "activation budget" to move between examples in the batch.

        During training, we also estimate an inference threshold theta that approximates the effective cutoff induced by BatchTopK. For each batch, we compute the minimum positive activation that survives the BatchTopK mask and update an exponential moving average of this quantity. This running estimate plays the same role as BatchNorm running statistics: it is updated only in training mode and treated as fixed at inference.

    Eval mode (model.eval()):
        At inference time we do not apply a batch-coupled top-k, since that would make each example depend on the rest of the eval batch. Instead, we use the stored running threshold theta to define a JumpReLU nonlinearity:

            y = x if x > theta else 0

        applied elementwise and independently to each example. This preserves the approximate sparsity level learned during training, but makes the layer deterministic and sample-wise independent for evaluation, probing, and downstream use.

    Inputs:
        x: Tensor of shape [batch, d_sae] containing pre-activation codes.

    Outputs:
        Tensor of shape [batch, d_sae] with the same dtype and device as x, where either:
            - in training mode: exactly (batch * top_k) entries are non-zero across the batch due to the BatchTopK mask, or
            - in eval mode: entries are zeroed by an elementwise JumpReLU with the learned threshold theta.
    """

    def __init__(self, cfg: BatchTopK):
        super().__init__()
        self.cfg = cfg

        self.register_buffer("threshold", torch.tensor(0.0))

    def forward(self, x: Float[Tensor, "batch d_sae"]) -> Float[Tensor, "batch d_sae"]:
        """
        Apply top-k activation to each sample in the batch.
        """

        if not self.training:
            if self.threshold <= 0:
                return torch.where(x > 0, x, torch.zeros_like(x))

            return torch.where(x > self.threshold, x, torch.zeros_like(x))

        bsz, d_sae = x.shape
        x_flat = x.flatten()

        bsz, d_sae = x.shape
        k = min(self.cfg.top_k * bsz, d_sae * bsz)
        _, idxs = torch.topk(x_flat, k, sorted=False)
        mask = torch.zeros_like(x_flat).scatter(-1, idxs, 1.0).reshape(x.shape)

        x = torch.mul(mask, x)

        with torch.no_grad():
            pos = x[x > 0]
            if pos.numel() >= 0:
                self.threshold.mul_(1 - self.cfg.momentum).add_(
                    self.cfg.momentum * pos.min()
                )

        return x


@beartype.beartype
def get_activation(cfg: ActivationConfig) -> torch.nn.Module:
    if isinstance(cfg, Relu):
        return ReluActivation(cfg)
    elif isinstance(cfg, TopK):
        return TopKActivation(cfg)
    elif isinstance(cfg, BatchTopK):
        return BatchTopKActivation(cfg)
    else:
        tp.assert_never(cfg)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class SparseAutoencoderConfig:
    #########
    # MODEL #
    #########
    d_model: int = 1024
    """Size of x."""
    d_sae: int = 1024 * 16
    """Number of features in SAE latent space; size of f(x)."""
    activation: ActivationConfig = TopK()
    """Activation function."""
    ##################
    # INITIALIZATION #
    ##################
    reinit_blend: float = 0.8
    """"""
    reinit_enc_dec_tranpose: bool = True
    """"""
    ################
    # OPTIMIZATION #
    ################
    remove_parallel_grads: bool = True
    """Whether to remove gradients parallel to W_dec columns (which will be ignored because we force the columns to have unit norm). See [Towards Monosemanticity; Appendix "Advice for Training Sparse Autoencoders: Autoencoder Architecture"](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-optimization) for discussion by Anthropic."""
    normalize_w_dec: bool = True
    """Whether to make sure W_dec has unit norm columns. See [Towards Monosemanticity; Appendix "Advice for Training Sparse Autoencoders: Autoencoder Architecture"](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder)."""


@jaxtyped(typechecker=beartype.beartype)
class SparseAutoencoder(torch.nn.Module):
    """Sparse auto-encoder (SAE)"""

    @jaxtyped(typechecker=beartype.beartype)
    class EncodeOut(tp.NamedTuple):
        """Outputs of encode: pre-activations and activated latents."""

        h_x: Float[Tensor, "batch d_sae"]
        f_x: Float[Tensor, "batch d_sae"]

    @jaxtyped(typechecker=beartype.beartype)
    class Output(tp.NamedTuple):
        """Full SAE forward outputs for objectives and metrics."""

        h_x: Float[Tensor, "batch d_sae"]
        f_x: Float[Tensor, "batch d_sae"]
        x_hats: Float[Tensor, "batch n_prefixes d_model"]

    def __init__(self, cfg: SparseAutoencoderConfig):
        super().__init__()

        self.cfg = cfg
        self.logger = logging.getLogger("sae")

        self.W_dec = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_sae, cfg.d_model))
        )
        self.b_dec = torch.nn.Parameter(torch.zeros(cfg.d_model))

        self.normalize_w_dec()

        # Initialize W_enc to the transpose of W_dec.
        self.W_enc = torch.nn.Parameter(self.W_dec.data.T)
        self.b_enc = torch.nn.Parameter(torch.zeros(cfg.d_sae))

        self.activation = get_activation(cfg.activation)

    def forward(self, x: Float[Tensor, "batch d_model"]) -> Output:
        """
        Given x, calculates the reconstructed x_hat and the intermediate activations f_x.

        Arguments:
            x: a batch of transformer activations.
        """
        enc = self.encode(x)
        x_hats = self.decode(enc.f_x)

        return self.Output(h_x=enc.h_x, f_x=enc.f_x, x_hats=x_hats)

    def encode(self, x: Float[Tensor, "batch d_model"]) -> EncodeOut:
        h_x = (
            einops.einsum(x, self.W_enc, "... d_model, d_model d_sae -> ... d_sae")
            + self.b_enc
        )
        f_x = self.activation(h_x)
        return self.EncodeOut(h_x=h_x, f_x=f_x)

    def decode(
        self,
        f_x: Float[Tensor, "batch d_sae"],
        *,
        prefixes: Int64[Tensor, " n_prefixes"] | None = None,
    ) -> Float[Tensor, "batch n_prefixes d_model"]:
        """
        Decode latent features to reconstructions.

        Args:
            f_x: Latent features of shape (batch, d_sae)
            prefixes: Optional tensor of prefix lengths for Matryoshka decoding.

        Returns:
            Matryoshka reconstructions (batch, n_prefixes, d_model).
        """
        b, d_sae = f_x.shape

        # Matryoshka cumulative decode
        device = f_x.device
        if prefixes is None:
            prefixes = torch.tensor([d_sae], dtype=torch.int64)
        assert torch.all(prefixes[1:] > prefixes[:-1])
        assert 1 <= int(prefixes[0]) and int(prefixes[-1]) == d_sae
        prefixes = prefixes.to(device)

        # Build blocks from prefix cuts: [0, cut1), [cut1, cut2), ...
        block_indices = torch.cat([
            torch.tensor([0], dtype=prefixes.dtype, device=device),
            prefixes,
        ])
        blocks = list(zip(block_indices[:-1], block_indices[1:]))

        # Compute block outputs
        block_outputs = []
        for i, (start, end) in enumerate(blocks):
            # Each block uses its portion of f_x and W_dec
            block_f_x = f_x[:, start:end]
            block_W_dec = self.W_dec[start:end, :]

            # Compute block output: (batch, d_sae_block) @ (d_sae_block, d_model) -> (batch, d_model)
            # Note: W_dec is (d_sae, d_model), so block_W_dec is (block_size, d_model)
            block_output = einops.einsum(
                block_f_x,
                block_W_dec,
                "... d_sae_block, d_sae_block d_model -> ... d_model",
            )

            # Add bias only to the first block
            if i == 0:
                block_output = block_output + self.b_dec

            block_outputs.append(block_output)

        # Cumulative sum to get prefix reconstructions
        x_hats = torch.cumsum(torch.stack(block_outputs, dim=-2), dim=-2)

        # (sam) This is clearly wrong. Needs to be cleaned up.
        return x_hats

    @torch.no_grad()
    def normalize_w_dec(self):
        """
        Set W_dec to unit-norm columns.
        """
        if self.cfg.normalize_w_dec:
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_parallel_grads(self):
        """
        Update grads so that they remove the parallel component
        """
        if not self.cfg.remove_parallel_grads:
            return

        if self.W_dec.grad is None:
            return

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_model, d_sae d_model -> d_sae",
        )

        norm_sq = torch.sum(self.W_dec.data * self.W_dec.data, dim=1)
        scales = torch.zeros_like(parallel_component)
        nonzero = norm_sq > 0
        scales[nonzero] = parallel_component[nonzero] / norm_sq[nonzero]

        self.W_dec.grad -= einops.einsum(
            scales,
            self.W_dec.data,
            "d_sae, d_sae d_model -> d_sae d_model",
        )


@beartype.beartype
def _normalize_cfg_kwargs(cfg_dict: dict[str, tp.Any]) -> dict[str, tp.Any]:
    cfg = dict(cfg_dict)
    # Backwards compatibility
    cfg.pop("n_reinit_samples", None)
    if "exp_factor" in cfg and "d_sae" not in cfg:
        exp_factor = cfg.pop("exp_factor")
        d_model = cfg.get("d_model")
        if d_model is None:
            raise ValueError(
                "Cannot infer d_sae from exp_factor without d_model in checkpoint."
            )
        cfg["d_sae"] = d_model * exp_factor
    return cfg


@beartype.beartype
def _serialize_dataclass(obj: tp.Any) -> dict[str, tp.Any]:
    assert dataclasses.is_dataclass(obj), f"Cannot serialize non-dataclass: {type(obj)}"
    params: dict[str, tp.Any] = {}
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        params[field.name] = _serialize_value(value)
    return {"cls": obj.__class__.__name__, "params": params}


@beartype.beartype
def _serialize_value(value: tp.Any) -> tp.Any:
    if dataclasses.is_dataclass(value):
        return _serialize_dataclass(value)
    if isinstance(value, tuple):
        return [_serialize_value(v) for v in value]
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value


@beartype.beartype
def _deserialize_dataclass_payload(
    payload: dict[str, tp.Any], *, allow_legacy_nested: bool = False
):
    cls_name = payload["cls"]
    activation_cls = globals().get(cls_name)
    msg = f"Unknown activation class '{cls_name}' in payload."
    assert activation_cls is not None, msg
    params: dict[str, tp.Any] = {}
    for key, value in payload["params"].items():
        params[key] = _deserialize_value(
            value, field_name=key, allow_legacy_nested=allow_legacy_nested
        )
    return activation_cls(**params)


@beartype.beartype
def _deserialize_value(
    value: tp.Any, *, field_name: str, allow_legacy_nested: bool
) -> tp.Any:
    if isinstance(value, dict):
        if "cls" in value and "params" in value:
            return _deserialize_dataclass_payload(
                value, allow_legacy_nested=allow_legacy_nested
            )
        if allow_legacy_nested and field_name == "sparsity":
            legacy = _deserialize_legacy_sparsity(value)
            if legacy is not None:
                return legacy
        return {
            k: _deserialize_value(
                v, field_name=field_name, allow_legacy_nested=allow_legacy_nested
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            _deserialize_value(
                item, field_name=field_name, allow_legacy_nested=allow_legacy_nested
            )
            for item in value
        ]
    return value


@beartype.beartype
def _deserialize_legacy_sparsity(
    payload: dict[str, tp.Any],
) -> Sparsity | None:
    if not payload:
        return NoSparsity()
    if set(payload.keys()) <= {"coeff"}:
        return L1Sparsity(**payload)
    return None


@beartype.beartype
def dump(fpath: pathlib.Path | str, sae: SparseAutoencoder):
    """
    Save an SAE checkpoint to disk along with configuration, using the [trick from equinox](https://docs.kidger.site/equinox/examples/serialisation).

    Arguments:
        fpath: filepath to save checkpoint to.
        sae: sparse autoencoder checkpoint to save.
    """
    # Custom serialization to handle activation object
    cfg_dict = dataclasses.asdict(sae.cfg)
    # Replace activation dict with custom format
    activation = sae.cfg.activation
    cfg_dict["activation"] = _serialize_dataclass(activation)

    header = {
        "schema": SCHEMA_VERSION,
        "cfg": cfg_dict,
        "commit": helpers.current_git_commit() or "unknown",
        "lib": __version__,
    }

    fpath = pathlib.Path(fpath)
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "wb") as fd:
        helpers.jdump(header, fd, option=orjson.OPT_APPEND_NEWLINE)
        torch.save(sae.state_dict(), fd)


@beartype.beartype
def load(fpath: pathlib.Path | str, *, device="cpu") -> SparseAutoencoder:
    """
    Loads a sparse autoencoder from disk.
    """
    with open(fpath, "rb") as fd:
        header = json.loads(fd.readline())
        buffer = io.BytesIO(fd.read())

    if "schema" not in header:
        # Original, pre-schema format: just raw config parameters
        # Remove old parameters that no longer exist
        for keyword in (
            "sparsity_coeff",
            "ghost_grads",
            "l1_coeff",
            "use_ghost_grads",
            "seed",
        ):
            header.pop(keyword, None)
        # Legacy format - create SparseAutoencoderConfig with Relu activation
        header["d_model"] = header.pop("d_vit")
        cfg_kwargs = _normalize_cfg_kwargs(header)
        cfg = SparseAutoencoderConfig(**cfg_kwargs, activation=Relu())
    elif header["schema"] == 1:
        # Schema version 1: A cautionary tale of poor version management
        #
        # This schema version unfortunately has TWO incompatible formats because we made breaking changes without incrementing the schema version. This is exactly what schema versioning is supposed to prevent!
        #
        # Format 1A (original): cls field contains activation type ("Relu", "TopK", etc.)
        # Format 1B (later): cls field is "SparseAutoencoderConfig" and activation is a dict
        #
        # The complex logic below exists to handle both formats. This should have been avoided by incrementing to schema version 2 when we changed the format.
        #
        # Apologies from Sam for this mess - proper schema versioning discipline would have prevented this confusing situation. Every breaking change should increment the version number!

        cls_name = header.get("cls", "SparseAutoencoderConfig")
        cfg_dict = dict(header["cfg"])

        if cls_name in ["Relu", "TopK", "BatchTopK"]:
            # Format 1A: Old format where cls indicates the activation type
            activation_cls = globals()[cls_name]
            if cls_name in ["TopK", "BatchTopK"]:
                activation = activation_cls(top_k=cfg_dict.get("top_k", 32))
            else:
                activation = activation_cls()
            cfg_kwargs = _normalize_cfg_kwargs(cfg_dict)
            cfg = SparseAutoencoderConfig(**cfg_kwargs, activation=activation)
        else:
            # Format 1B: Newer format with activation as dict
            if "activation" in cfg_dict:
                activation_info = cfg_dict["activation"]
                activation_cls = globals()[activation_info["cls"]]
                activation = activation_cls(**activation_info["params"])
                cfg_dict["activation"] = activation
            cfg_kwargs = _normalize_cfg_kwargs(cfg_dict)
            cfg = SparseAutoencoderConfig(**cfg_kwargs)
    elif header["schema"] in (2, 3, 4):
        # Schema version 2: cleaner format with activation serialization
        cfg_dict = dict(header["cfg"])
        activation_info = cfg_dict["activation"]
        activation = _deserialize_dataclass_payload(
            activation_info, allow_legacy_nested=True
        )
        cfg_dict["activation"] = activation
        cfg_kwargs = _normalize_cfg_kwargs(cfg_dict)
        cfg = SparseAutoencoderConfig(**cfg_kwargs)
    elif header["schema"] == 5:
        cfg_dict = dict(header["cfg"])
        activation = _deserialize_dataclass_payload(
            cfg_dict["activation"], allow_legacy_nested=False
        )
        cfg_dict["activation"] = activation
        cfg_kwargs = _normalize_cfg_kwargs(cfg_dict)
        cfg = SparseAutoencoderConfig(**cfg_kwargs)
    else:
        raise ValueError(f"Unknown schema version: {header['schema']}")

    model = SparseAutoencoder(cfg)
    model.load_state_dict(torch.load(buffer, weights_only=True, map_location=device))
    return model
