import beartype
import torch
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor


# If cloudpickle complains about a weakref, then switch this to just beartype. https://github.com/patrick-kidger/jaxtyping/issues/332
@jaxtyped(typechecker=beartype.beartype)
def calc_avg_prec(
    scores_NC: Float[Tensor, "N C"], y_true_NT: Bool[Tensor, "N T"]
) -> Float[Tensor, "C T"]:
    """
    Vectorized implementation of average precision (AP).

    Step-by-step:
    * sort images by score  (per prototype)
    * walk down the list, accumulate TP and precision = TP / rank
    * AP_t = mean of precision at the ranks where y == 1

    Args:
        scores_NC: Scores for n images and c prototypes (where c << k to batch this calculation).
    """
    n, c = scores_NC.shape
    _, t = y_true_NT.shape

    # total positives per trait   P_t  (shape  (T,))
    pos_T = y_true_NT.sum(dim=0).to(torch.float32)
    pos_mask_T = pos_T > 0  # traits that exist in this split
    pos_T[pos_T == 0] = 1.0  # avoid divide‑by‑zero later

    # 1. Order indices for each prototype  — shape (N, C)
    idx_NC = torch.argsort(scores_NC, dim=0, descending=True)

    # 2. Gather labels in that order
    # y_sorted_NCT[n, c, t] == y_true_NT[idx_NC[n, c], t]
    # Add trait dimension to indices: (N,C) -> (N,C,T) via view expansion
    # einops.repeat(idx_NC, 'n c -> n c t', t=t)
    idx_NCT = idx_NC.unsqueeze(-1).expand(-1, -1, t)

    # Add prototype dimension to labels: (N,T) -> (N,C,T) via view expansion
    # einops.repeat(y_true_NT, 'n t -> n c t', c=c)
    y_NCT = y_true_NT.unsqueeze(1).expand(-1, c, -1)

    y_sorted_NCT = torch.gather(y_NCT, dim=0, index=idx_NCT)

    cum_tp_NCT = torch.cumsum(y_sorted_NCT.float(), dim=0)

    ranks = torch.arange(1, n + 1, device=scores_NC.device, dtype=torch.float32)
    ranks = ranks.view(-1, 1, 1)
    precision_NCT = cum_tp_NCT / ranks

    masked_precision = precision_NCT * y_sorted_NCT.float()
    sum_precision_CT = masked_precision.sum(dim=0)

    avg_precision_CT = sum_precision_CT / pos_T.unsqueeze(0)
    avg_precision_CT = avg_precision_CT * pos_mask_T.unsqueeze(0)

    return avg_precision_CT
