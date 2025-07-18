import beartype
import torch
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor


@jaxtyped(typechecker=beartype.beartype)
def gather_batched(
    value: Float[Tensor, "batch n dim"], i: Int[Tensor, "batch k"]
) -> Float[Tensor, "batch k dim"]:
    batch_size, n, dim = value.shape
    _, k = i.shape

    batch_i = torch.arange(batch_size, device=value.device)[:, None].expand(-1, k)
    return value[batch_i, i]
