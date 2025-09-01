import beartype
import torch
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

import saev.helpers


@jaxtyped(typechecker=beartype.beartype)
def batched_upsample_and_pred(
    tensor: Float[Tensor, "batch  width height"],
    *,
    size: tuple[int, int],
    mode: str,
    batch_size: int = 128,
) -> Int[Tensor, "n {size[0]} {size[1]}"]:
    preds = []

    for start, end in saev.helpers.batched_idx(len(tensor), batch_size):
        upsampled_bcwh = torch.nn.functional.interpolate(
            tensor[start:end].contiguous(), size=size, mode=mode
        )
        pred_bwh = upsampled_bcwh.argmax(axis=1).cpu()
        del upsampled_bcwh
        preds.append(pred_bwh)

    return torch.cat(preds)
