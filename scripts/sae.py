import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


def get_wsd_scheduler(
    optimizer, n_steps, end_lr_factor=0.1, n_warmup_steps=None, percent_cooldown=0.1
):
    """
    See https://www.lighton.ai/lighton-blogs/passing-the-torch-training-a-mamba-model-for-smooth-handover
    """
    if n_warmup_steps is None:
        n_warmup_steps = 0.05 * n_steps

    def lr_lambda(step):
        if step < n_warmup_steps:
            return step / n_warmup_steps
        elif step < (1 - percent_cooldown) * n_steps:
            return 1
        else:
            return 1 - (1 - end_lr_factor) * min(
                (step - (1 - percent_cooldown) * n_steps),
                (1 - percent_cooldown) * n_steps,
            ) / (percent_cooldown * n_steps + 1e-2)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def sample_prefixes(
    n_latents: int,
    n_prefixes: int,
    min_prefix_length: int = 1,
    pareto_power: float = 0.5,
) -> torch.Tensor:
    """
    Samples prefix lengths using a Pareto distribution to favor shorter prefixes.

    Args:
        n_latents: Total number of latent dimensions
        n_prefixes: Number of prefixes to sample
        min_prefix_length: Minimum length of any prefix
        pareto_power: Power parameter for Pareto distribution (lower = more uniform)

    Returns:
        torch.Tensor: Sorted prefix lengths
        Example: with n_latents=10 might return [3, 7, 10]
    """
    if n_prefixes == 1:
        return torch.tensor([n_latents])

    # Calculate probability distribution favoring shorter prefixes
    possible_lengths = torch.arange(n_latents - min_prefix_length + 1)
    pareto_cdf = (
        1 - (possible_lengths / (n_latents - min_prefix_length + 1)) ** pareto_power
    )

    # Convert CDF to PDF
    pareto_pdf = np.concatenate(
        [np.zeros(min_prefix_length), pareto_cdf[1:] - pareto_cdf[:-1]]
    )
    probability_dist = pareto_pdf / pareto_pdf.sum()

    # Sample and sort prefix lengths
    prefixes = np.random.choice(
        n_latents, size=n_prefixes - 1, replace=False, p=probability_dist
    )

    # Add n_latents as the final prefix
    prefixes = np.append(prefixes, n_latents)

    prefixes.sort()

    return torch.tensor(prefixes)


class RunningAvgNormalizer(nn.Module):
    def __init__(self, alpha=0.99):
        super().__init__()
        self.running_avg = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.alpha = alpha

    @torch.no_grad()
    def normalize(self, x, update=False):
        if update is True:
            with torch.no_grad():
                if self.running_avg is None:
                    self.running_avg.data = x.norm(dim=-1).mean()
                else:
                    self.running_avg.data = (
                        self.alpha * self.running_avg
                        + (1 - self.alpha) * x.norm(dim=-1).mean()
                    )
        return x * (np.sqrt(x.shape[-1]) / self.running_avg.detach())

    @torch.no_grad()
    def unnormalize(self, x):
        return x * (self.running_avg.detach() / np.sqrt(x.shape[-1]))


class AdaptiveSparsityController(nn.Module):
    """
    Learns the appropriate sparsity regularization weight to hit a target l0.
    This idea was shared with me by Glen Taggart.
    """

    def __init__(
        self, target_l0, starting_sparsity_loss_scale=1.2, warmup_steps=400, eps=0.0003
    ):
        super().__init__()
        self.sparsity_loss_scale = starting_sparsity_loss_scale
        self.eps = eps
        self.target_l0 = target_l0
        self.step = 0
        self.warmup_steps = warmup_steps

    @torch.no_grad()
    def forward(self, avg_l0=None):
        """
        Given the avg_l0 on a batch, updates self.sparsity_loss_scale and returns the result.
        If avg_l0 is None, returns the current sparsity loss scale without updating it.
        """
        if avg_l0 is None:
            # Return current sparsity loss scale without updating it
            return self.sparsity_loss_scale * min(self.step / self.warmup_steps, 1)

        if self.step > self.warmup_steps:
            # Update sparsity loss scale
            if avg_l0 < self.target_l0:
                self.sparsity_loss_scale *= 1 - self.eps
            elif avg_l0 >= self.target_l0:
                self.sparsity_loss_scale *= 1 + self.eps

        self.step += 1

        return self.sparsity_loss_scale * min(self.step / self.warmup_steps, 1)


class MatryoshkaSAE(nn.Module):
    def __init__(
        self,
        d_model,
        n_latents,
        n_prefixes,
        target_l0,
        n_steps,
        lr=3e-2,
        permute_latents=True,
        min_prefix_length=1,
        sparsity_type="l1",
        starting_sparsity_loss_scale=1.2,
    ):
        super().__init__()

        self.W_enc = nn.Parameter(torch.randn(d_model, n_latents) / (np.sqrt(d_model)))
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.W_dec = nn.Parameter(
            (0.1 * self.W_enc.data / self.W_enc.data.norm(dim=0, keepdim=True)).T
        )
        self.b_dec = nn.Parameter(
            torch.zeros(
                d_model,
            )
        )

        self.n_latents = n_latents
        self.d_model = d_model
        self.n_prefixes = n_prefixes
        self.min_prefix_length = min_prefix_length
        self.n_steps = n_steps

        self.normalizer = RunningAvgNormalizer()

        self.sparsity_controller = AdaptiveSparsityController(
            target_l0=target_l0,
            warmup_steps=int(self.n_steps * 0.2),
            starting_sparsity_loss_scale=starting_sparsity_loss_scale,
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.5, 0.9375))
        self.scaler = torch.amp.GradScaler("cuda")
        self.scheduler = get_wsd_scheduler(
            self.optimizer,
            n_steps=n_steps,
            n_warmup_steps=100,
            percent_cooldown=0.2,
            end_lr_factor=0.1,
        )

        self.permute_latents = permute_latents
        if self.permute_latents:
            self.sq_act_running_avg = nn.Parameter(
                torch.zeros(self.n_latents), requires_grad=False
            )

        assert sparsity_type in {"l1", "log"}
        self.sparsity_type = sparsity_type

    def forward(self, x):
        """
        Forward pass through the SAE.
        Args:
            x: Input tensor of shape (batch_size, d_model)
        Returns:
            Output tensor of shape (batch_size, d_model)
        """

        preacts = x @ self.W_enc + self.b_enc
        acts = F.relu(preacts)

        return acts, acts @ self.W_dec + self.b_dec

    @property
    def dtype(self):
        return self.W_enc.dtype

    @property
    def device(self):
        return self.W_enc.device

    @torch.no_grad()
    def get_acts(self, x, indices=None, normalize=True):
        if normalize:
            x = self.normalizer.normalize(x, update=False)
        if isinstance(indices, int):
            indices = [indices]
        if indices is None:
            preacts = x @ self.W_enc + self.b_enc
            acts = torch.einsum("...d,d->...d", F.relu(preacts), self.W_dec.norm(dim=1))
        else:
            preacts = x @ self.W_enc[:, indices] + self.b_enc[indices]
            acts = torch.einsum(
                "...d,d->...d", F.relu(preacts), self.W_dec[indices].norm(dim=1)
            )
        return self.normalizer.unnormalize(acts)

    def step(self, x, return_metrics=False):
        x = self.normalizer.normalize(x, update=True)

        prefixes = sample_prefixes(
            self.n_latents, self.n_prefixes, self.min_prefix_length
        ).to(self.device)
        block_bounds = torch.cat([torch.tensor([0]).to(self.device), prefixes])

        acts = [
            F.relu(
                x @ self.W_enc[:, block_start:block_end]
                + self.b_enc[block_start:block_end]
            )
            for block_start, block_end in zip(block_bounds[:-1], block_bounds[1:])
        ]

        # Get the norms of W_dec
        W_dec_norms = self.W_dec.norm(dim=1)

        block_sparsity_losses, mse_contributions = [], []
        for block_acts, block_start, block_end in zip(
            acts, block_bounds[:-1], block_bounds[1:]
        ):
            normed_block_acts = block_acts * W_dec_norms[block_start:block_end][None]
            if self.sparsity_type == "log":
                sparsity_loss = (
                    (torch.log(normed_block_acts + 0.1) - np.log(0.1)).mean(dim=0)
                ).sum(dim=-1)
            elif self.sparsity_type == "l1":
                sparsity_loss = normed_block_acts.mean(dim=0).sum(dim=-1)
            else:
                raise ValueError(
                    f"Unknown sparsity_type '{self.sparsity_type}'. Expected one of: 'l1', 'log'"
                )

            block_sparsity_losses.append(sparsity_loss)

            if self.permute_latents:
                with torch.no_grad():
                    mse_contributions.append((normed_block_acts**2).mean(dim=0))

        if self.permute_latents:
            with torch.no_grad():
                mse_contributions = torch.cat(mse_contributions, dim=0)

        block_outputs = [
            block_acts @ self.W_dec[block_start:block_end]
            for block_acts, block_start, block_end in zip(
                acts, block_bounds[:-1], block_bounds[1:]
            )
        ]

        with torch.no_grad():
            avg_l0 = (
                np.sum([(block_acts > 0).float().sum().cpu() for block_acts in acts])
                / acts[0].shape[0]
            )
            sparsity_scale = self.sparsity_controller(avg_l0)

            if self.permute_latents:
                self.sq_act_running_avg.data = (
                    0.95 * self.sq_act_running_avg.data + 0.05 * mse_contributions
                )
                latent_perm = torch.argsort(
                    self.sq_act_running_avg.data, descending=True
                )

        sparsity_loss = torch.cumsum(torch.stack(block_sparsity_losses), dim=0).mean()

        block_outputs[0] = block_outputs[0] + self.b_dec
        prefix_preds = torch.cumsum(torch.stack(block_outputs), dim=0)

        prefix_errs = ((prefix_preds - x[None]) ** 2).sum(dim=-1).mean(dim=-1)

        recon_loss = prefix_errs.mean()

        loss = recon_loss + sparsity_scale * sparsity_loss

        result = {"loss": loss, "avg_l0": avg_l0, "sparsity_scale": sparsity_scale}

        if return_metrics:
            with torch.no_grad():
                tot_var = ((x - x.mean(dim=0, keepdim=True)) ** 2).sum(dim=-1).mean()

                # block preds: (block, batch, d_model)
                block_errs = ((block_outputs - x[None]) ** 2).sum(dim=-1).mean(dim=-1)

                for i in range(block_outputs.shape[0]):
                    result[f"block_{i}_fvu"] = block_errs[i] / tot_var
                result["last_block_fvu"] = block_errs[-1] / tot_var
                block_sparsity_losses = [float(b) for b in block_sparsity_losses]
                for i, b in enumerate(block_sparsity_losses):
                    result[f"block_{i}_l1_loss"] = b

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.optimizer)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        if self.permute_latents:
            # Permute W_dec, W_enc, and b_enc using the previously defined permutation
            with torch.no_grad():
                self.W_dec.data = self.W_dec.data[latent_perm]
                self.W_enc.data = self.W_enc.data[:, latent_perm]
                self.b_enc.data = self.b_enc.data[latent_perm]

            # Permute optimizer state for W_dec, W_enc, and b_enc
            for param_group in self.optimizer.param_groups:
                for param in param_group["params"]:
                    if param is self.W_dec and param in param_group.get("state", []):
                        state = param_group.get("state", [])[param]
                        if "exp_avg" in state:
                            state["exp_avg"] = state["exp_avg"][latent_perm]
                        if "exp_avg_sq" in state:
                            state["exp_avg_sq"] = state["exp_avg_sq"][latent_perm]
                    elif param is self.W_enc and param in param_group.get("state", []):
                        state = param_group.get("state", [])[param]
                        if "exp_avg" in state:
                            state["exp_avg"] = state["exp_avg"][:, latent_perm]
                        if "exp_avg_sq" in state:
                            state["exp_avg_sq"] = state["exp_avg_sq"][:, latent_perm]
                    elif param is self.b_enc and param in param_group.get("state", []):
                        state = param_group.get("state", [])[param]
                        if "exp_avg" in state:
                            state["exp_avg"] = state["exp_avg"][latent_perm]
                        if "exp_avg_sq" in state:
                            state["exp_avg_sq"] = state["exp_avg_sq"][latent_perm]

        return result