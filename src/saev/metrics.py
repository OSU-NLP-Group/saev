import dataclasses
import math
from collections import abc

import beartype


@beartype.beartype
def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Metrics:
    """Validated reconstruction metrics aggregated over one evaluation corpus.

    The primary totals are `sse_recon` (SAE reconstruction SSE) and `sse_baseline` (mean-baseline SSE). Derived terms are:
    - `normalized_mse = sse_recon / sse_baseline`
    - `mse_per_dim = sse_recon / n_elements`
    - `mse_per_token = sse_recon / n_tokens`
    - `baseline_mse_per_dim = sse_baseline / n_elements`
    - `baseline_mse_per_token = sse_baseline / n_tokens`

    Size terms are:
    - `n_tokens`: number of tokens included in aggregation
    - `d_model`: embedding width per token
    - `n_elements = n_tokens * d_model`
    """

    mse_per_dim: float
    mse_per_token: float
    normalized_mse: float
    baseline_mse_per_dim: float
    baseline_mse_per_token: float
    sse_recon: float
    sse_baseline: float
    n_tokens: int
    d_model: int
    n_elements: int

    def __post_init__(self):
        msg = f"n_tokens must be an int, got {type(self.n_tokens)}."
        assert type(self.n_tokens) is int, msg
        msg = f"d_model must be an int, got {type(self.d_model)}."
        assert type(self.d_model) is int, msg
        msg = f"n_elements must be an int, got {type(self.n_elements)}."
        assert type(self.n_elements) is int, msg

        msg = f"n_tokens must be positive, got {self.n_tokens}."
        assert self.n_tokens > 0, msg
        msg = f"d_model must be positive, got {self.d_model}."
        assert self.d_model > 0, msg
        expected_n_elements = self.n_tokens * self.d_model
        msg = f"n_elements={self.n_elements} != n_tokens*d_model={expected_n_elements}."
        assert self.n_elements == expected_n_elements, msg

        msg = f"sse_recon must be >= 0, got {self.sse_recon}."
        assert self.sse_recon >= 0.0, msg
        msg = f"sse_baseline must be > 0, got {self.sse_baseline}."
        assert self.sse_baseline > 0.0, msg

        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, int | float):
                continue
            msg = f"{field.name} must be finite, got {value}."
            assert math.isfinite(value), msg

        msg = f"mse_per_dim={self.mse_per_dim} is inconsistent with sse_recon/n_elements={self.sse_recon / self.n_elements}."
        assert close(self.mse_per_dim, self.sse_recon / self.n_elements), msg
        msg = f"mse_per_token={self.mse_per_token} is inconsistent with sse_recon/n_tokens={self.sse_recon / self.n_tokens}."
        assert close(self.mse_per_token, self.sse_recon / self.n_tokens), msg
        msg = f"baseline_mse_per_dim={self.baseline_mse_per_dim} is inconsistent with sse_baseline/n_elements={self.sse_baseline / self.n_elements}."
        assert close(self.baseline_mse_per_dim, self.sse_baseline / self.n_elements), (
            msg
        )
        msg = f"baseline_mse_per_token={self.baseline_mse_per_token} is inconsistent with sse_baseline/n_tokens={self.sse_baseline / self.n_tokens}."
        assert close(self.baseline_mse_per_token, self.sse_baseline / self.n_tokens), (
            msg
        )
        msg = f"normalized_mse={self.normalized_mse} is inconsistent with sse_recon/sse_baseline={self.sse_recon / self.sse_baseline}."
        assert close(self.normalized_mse, self.sse_recon / self.sse_baseline), msg

    @classmethod
    def from_accumulators(
        cls, *, sse_recon: float, sse_baseline: float, n_tokens: int, d_model: int
    ) -> "Metrics":
        """Construct metrics from aggregate sums and shape information.

        Args:
            sse_recon: Sum of squared reconstruction errors over all selected tokens and dimensions.
            sse_baseline: Sum of squared mean-baseline errors over the same tokens and dimensions.
            n_tokens: Number of selected tokens in the aggregation set.
            d_model: Activation dimension per token.

        Returns:
            A validated `Metrics` object with all derived fields populated.
        """

        msg = f"n_tokens must be positive, got {n_tokens}."
        assert n_tokens > 0, msg
        msg = f"d_model must be positive, got {d_model}."
        assert d_model > 0, msg
        msg = f"sse_recon must be >= 0, got {sse_recon}."
        assert sse_recon >= 0.0, msg
        msg = f"sse_baseline must be > 0, got {sse_baseline}."
        assert sse_baseline > 0.0, msg

        n_elements = n_tokens * d_model
        return cls(
            mse_per_dim=sse_recon / n_elements,
            mse_per_token=sse_recon / n_tokens,
            normalized_mse=sse_recon / sse_baseline,
            baseline_mse_per_dim=sse_baseline / n_elements,
            baseline_mse_per_token=sse_baseline / n_tokens,
            sse_recon=sse_recon,
            sse_baseline=sse_baseline,
            n_tokens=n_tokens,
            d_model=d_model,
            n_elements=n_elements,
        )

    @classmethod
    def from_dict(cls, dct: abc.Mapping[str, object]) -> "Metrics":
        values: dict[str, int | float] = {}
        for field in dataclasses.fields(cls):
            key = field.name
            if field.type is int:
                values[key] = cls._get_int(dct, key)
                continue
            msg = f"{key} has unsupported type {field.type}; expected int or float."
            assert field.type is float, msg
            values[key] = cls._get_float(dct, key)

        return cls(**values)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, float | int]:
        return dataclasses.asdict(self)

    @staticmethod
    def _get_float(dct: abc.Mapping[str, object], key: str) -> float:
        msg = f"Missing metric key: {key}."
        assert key in dct, msg
        value = dct[key]
        msg = f"{key} must be int/float, got {type(value)}."
        assert not isinstance(value, bool), msg
        assert isinstance(value, int | float), msg
        return float(value)

    @staticmethod
    def _get_int(dct: abc.Mapping[str, object], key: str) -> int:
        msg = f"Missing metric key: {key}."
        assert key in dct, msg
        value = dct[key]
        msg = f"{key} must be int, got {type(value)}."
        assert not isinstance(value, bool), msg
        assert isinstance(value, int), msg
        return value
