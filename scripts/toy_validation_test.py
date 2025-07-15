import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import torch
    from torch.utils.data import Dataset
    import jsonschema
    import json
    import numpy as np

    def validate_tree(instance):
        with open("tree.schema.json", "r") as f:
            schema = json.load(f)

        jsonschema.validate(instance, schema)


    class Tree:
        def __init__(self, tree_dict, start_idx=0):
            assert "active_prob" in tree_dict
            self.active_prob = tree_dict["active_prob"]
            self.is_read_out = tree_dict.get("is_read_out", True)
            self.mutually_exclusive_children = tree_dict.get(
                "mutually_exclusive_children", False
            )
            self.id = tree_dict.get("id", None)

            self.is_binary = tree_dict.get("is_binary", True)
            if self.is_read_out:
                self.index = start_idx
                start_idx += 1
            else:
                self.index = False

            self.children = []
            for child_dict in tree_dict.get("children", []):
                child = Tree(child_dict, start_idx)
                start_idx = child.next_index
                self.children.append(child)

            self.next_index = start_idx

            if self.mutually_exclusive_children:
                assert len(self.children) >= 2

        def __repr__(self, indent=0):
            s = " " * (indent * 2)
            s += (
                str(self.index) + " "
                if self.index is not False
                else " " * len(str(self.next_index)) + " "
            )
            s += "B" if self.is_binary else " "
            s += "x" if self.mutually_exclusive_children else " "
            s += f" {self.active_prob}"

            for child in self.children:
                s += "\n" + child.__repr__(indent + 2)
            return s

        @property
        def n_features(self):
            return len(self.sample())

        @property
        def child_probs(self):
            return torch.tensor([child.active_prob for child in self.children])

        def sample(self, shape=None, force_inactive=False, force_active=False):
            assert not (force_inactive and force_active)

            # special sampling for shape argument
            if shape is not None:
                if isinstance(shape, int):
                    shape = (shape,)
                n_samples = np.prod(shape)
                samples = [self.sample() for _ in range(n_samples)]
                return torch.tensor(samples).view(*shape, -1).float()

            sample = []

            # is this feature active?
            is_active = (
                (torch.rand(1) <= self.active_prob).item() * (1 - (force_inactive))
                if not force_active
                else 1
            )

            # append something if this is a readout
            if self.is_read_out:
                if self.is_binary:
                    sample.append(is_active)
                else:
                    sample.append((is_active * torch.rand(1)))

            if self.mutually_exclusive_children:
                active_child = (
                    np.random.choice(self.children, p=self.child_probs)
                    if is_active
                    else None
                )

            for child in self.children:
                child_force_inactive = not bool(is_active) or (
                    self.mutually_exclusive_children and child != active_child
                )

                child_force_active = (
                    self.mutually_exclusive_children and child == active_child
                )

                sample += child.sample(
                    force_inactive=child_force_inactive, force_active=child_force_active
                )

            return sample

    return Dataset, Tree, json, np, torch


@app.cell
def _(Tree, json, torch):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    tree_dict = json.load(open('/Users/Beattie.74/Documents/Git Repos/saev/scripts/tree.json', 'r'))
    tree = Tree(tree_dict=tree_dict)

    D_MODEL = tree.n_features

    matryoshka_config = {
        'n_latents': tree.n_features,
        'target_l0': 1.2338, # L0 of the true features
        'n_prefixes': 10,
        'd_model': D_MODEL,
        'n_steps': 40_000, # You could try fewer steps if this runs too slow for your taste; 15K e.g.
        'lr': 3e-2,
        'permute_latents': True,
        'sparsity_type': 'l1',
        'starting_sparsity_loss_scale': 0.2
    }


    vanilla_config = matryoshka_config | {'n_prefixes': 1, 'permute_latents': False}
    return DEVICE, D_MODEL, matryoshka_config, tree, vanilla_config


@app.cell
def _(DEVICE, D_MODEL, Dataset, np, torch, tree, vanilla_config):
    from torch.utils.data import DataLoader

    # Generate dataset
    class TreeDataset(Dataset):
        def __init__(self, tree, true_feats, batch_size, num_batches):
            self.tree = tree
            self.true_feats = true_feats.cpu()
            self.batch_size = batch_size
            self.num_batches = num_batches

        def __len__(self):
            return self.num_batches

        def __getitem__(self, idx):
            true_acts = self.tree.sample(self.batch_size)
            # random_scale = 1+torch.randn_like(true_acts, device=self.true_feats.device) * 0.05
            # true_acts = true_acts * random_scale
            x = true_acts @ self.true_feats
            return x

    # Get true_feats
    true_feats = (torch.randn(tree.n_features, D_MODEL)/np.sqrt(D_MODEL)).to(DEVICE)

    # Generate random orthogonal directions by taking QR decomposition of random matrix
    Q, R = torch.linalg.qr(torch.randn(D_MODEL, D_MODEL))
    true_feats = Q[:tree.n_features].to(DEVICE)


    # randomly scale the norms of features
    random_scaling = 1+torch.randn(tree.n_features, device=DEVICE)* 0.05
    true_feats *= random_scaling[:, None]

    # Setup dataloader
    dataset = TreeDataset(tree, true_feats.cpu(), batch_size=200, num_batches=vanilla_config['n_steps'])
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=True)

    # Create a reference batch to visualize vanilla and matryoshka with over the course of training
    ref_acts = dataset.tree.sample(100).to(DEVICE)
    ref_x = ref_acts @ dataset.true_feats.to(DEVICE)
    ref_acts = ref_acts*random_scaling[None]

    return TreeDataset, dataset, ref_acts, true_feats


@app.cell
def _(ref_acts):
    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.imshow(ref_acts)
    plt.gca().set_aspect('auto')
    plt.show()
    return (plt,)


@app.cell
def _(DEVICE, np, torch):
    from scipy.optimize import linear_sum_assignment

    @torch.no_grad()
    def get_latent_perm(sae, tree_ds, include_all_latents=False):
        '''
        Get permutation of sae latents that tries to assign each latent to its closest-matching ground-truth feature
        H.T. Julian D'Costa for the linear_sum_assignment idea here.
        '''
        global DEVICE
        sample = tree_ds.tree.sample(1000).to(DEVICE)
        x = sample @ tree_ds.true_feats.to(DEVICE)

        feat_norms = tree_ds.true_feats.norm(dim=-1).to(DEVICE)
        scaled_sample = sample * feat_norms[None, :]

        true_acts = scaled_sample
        true_acts = true_acts/true_acts.max(dim=0, keepdim=True).values.clamp(min=1e-10)

        with torch.no_grad():
            sae_acts, x_hat = sae(x)
            sae_acts = sae_acts/sae_acts.max(dim=0, keepdim=True).values.clamp(min=1e-10)
            sims = (sae_acts.T @ true_acts).cpu()

        max_value = sims.max()
        cost_matrix = max_value - sims

        row_ind, col_ind = row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy().T)
        leftover_features = np.array(list({i for i in range(sims.shape[0]) if sae_acts.max(dim=0).values[i] > 0} - set(col_ind))).astype(int)

        if include_all_latents:
            return torch.tensor(np.concatenate([col_ind, leftover_features]))
        else:
            return torch.tensor(col_ind)
    return (get_latent_perm,)


@app.cell
def _(
    Array,
    DEVICE,
    Float,
    TreeDataset,
    matryoshka_config,
    torch,
    tree,
    true_feats,
    vanilla_config,
):
    from saev import config
    from saev.training import make_saes
    from saev import helpers
    from saev.training import Warmup, BatchLimiter
    from saev import nn

    def train() -> tuple[torch.nn.ModuleList, torch.nn.ModuleList]:
        """
        Toy version of the train() function from saev.training, used to test Matryoshka SAEs on toy tree data.
        Note that the toy script given by https://doi.org/10.48550/arXiv.2503.17547 uses an adaptive sparsity scaling,
        in order to specifically target the true sparsity of the features. We (currently) do not do this, as in practical
        use cases we will not know the true sparsity of the features.

        Uses the following parameters for training:
            input/output dim: 20
            latent (sparse) dim: 20
            n_prefixes: 10
            n_steps: 40,000
            lr: 3e-2
            sparsity coef: 4e-4 (default for the objective in saev package)
        """

        sparsity_coeff = 4e-3
        lr = 3e-2
        batch_size = 200
        total_steps = 40000

        if torch.cuda.is_available():
            # This enables tf32 on Ampere GPUs which is only 8% slower than
            # float16 and almost as accurate as float32
            # This was a default in pytorch until 1.12
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        # First, load in the data

        dataset = TreeDataset(tree, true_feats.cpu(), batch_size=200, num_batches=vanilla_config['n_steps'])
        saes, objectives, param_groups = make_saes([
            (config.Relu(d_vit=20, exp_factor=1, n_reinit_samples=0), 
            config.Matryoshka(sparsity_coeff=sparsity_coeff, n_prefixes=10)),
            (config.Relu(d_vit=20, exp_factor=1, n_reinit_samples=0),
            config.Vanilla(sparsity_coeff=sparsity_coeff))
        ])

        param_groups[0]['betas'] = (0.5, 0.9375)
        param_groups[1]['betas'] = (0.5, 0.9375)

        optimizer = torch.optim.Adam(param_groups, fused=True)
        lr_schedulers = [Warmup(0.0, lr, 100), Warmup(0.0, lr, 100)]
        sparsity_schedulers = [
            Warmup(0.0, sparsity_coeff, 0),
            Warmup(0.0, sparsity_coeff, 0)
        ]

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=0, shuffle=False
        )

        dataloader = BatchLimiter(dataloader, total_steps)

        saes.train()
        saes = saes.to(DEVICE)
        objectives.train()
        objectives = objectives.to(DEVICE)

        global_step, n_patches_seen = 0, 0

        batch_dim_set: Float[Array, "batch"] = torch.zeros((200))

        for batch in helpers.progress(dataloader, every=20):
            acts_BD = batch.to(DEVICE, non_blocking=True)[0]
            for sae in saes:
                sae.normalize_w_dec()
            # Forward passes and loss calculations.
            losses = []
            for sae, objective in zip(saes, objectives):
                if isinstance(objective, nn.objectives.MatryoshkaObjective):
                    # Specific case has to be given for Matryoshka SAEs since we need to decode several times
                    # with varying prefix lengths
                    prefix_preds, f_x = sae.matryoshka_forward(acts_BD, matryoshka_config['n_prefixes'])
                    losses.append(objective(acts_BD, f_x, prefix_preds))
                else:
                    x_hat, f_x = sae(acts_BD)
                    losses.append(objective(acts_BD, f_x, x_hat))

            n_patches_seen += len(acts_BD)

            for loss in losses:
                loss.loss.backward()

            for sae in saes:
                sae.remove_parallel_grads()

            optimizer.step()

            # Update LR and sparsity coefficients.
            for param_group, scheduler in zip(optimizer.param_groups, lr_schedulers):
                param_group["lr"] = scheduler.step()

            for objective, scheduler in zip(objectives, sparsity_schedulers):
                objective.sparsity_coeff = scheduler.step()

            # Don't need these anymore.
            optimizer.zero_grad()

            global_step += 1

        return saes, objectives, global_step

    return (train,)


@app.cell
def _(train):
    import saev.training

    saes, objectivces, global_step = train()
    return (saes,)


@app.cell
def _():
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = "browser" 

    def heatmap(data, title=None, dim_names=(None, None)):
        """
        Creates a simple heatmap visualization from a 2D array.

        Args:
            data: 2D numpy array or list of lists containing the values to plot
            title: Optional string for the plot title
            x_axis: Optional string for the x-axis label
            y_axis: Optional string for the y-axis label

        Returns:
            plotly.graph_objects.Figure object
        """
        fig = go.Figure(
            data=go.Heatmap(z=data, colorscale="Viridis"),
            layout=go.Layout(yaxis=dict(autorange="reversed")),
        )

        if title:
            fig.update_layout(title=title)

        row_label = (
            f"{dim_names[0]} ({data.shape[0]})"
            if dim_names[0] is not None
            else f"row ({data.shape[0]})"
        )
        col_label = (
            f"{dim_names[1]} ({data.shape[1]})"
            if dim_names[1] is not None
            else f"col ({data.shape[1]})"
        )

        fig.update_layout(
            xaxis_title=col_label,
            yaxis_title=row_label,
            height=600,
            width=1400,
        )

        fig.update_yaxes(showticklabels=False)
        fig.update_xaxes(showticklabels=False)

        return fig
    return (heatmap,)


@app.cell
def _(dataset, get_latent_perm, saes):
    matryoshka_sae = saes[0]
    vanilla_sae = saes[1]

    matryoshka_perm = get_latent_perm(matryoshka_sae, dataset)
    vanilla_perm = get_latent_perm(vanilla_sae, dataset)

    #matryoshka_perm = get_latent_perm(matryoshka_sae, dataset)
    #vanilla_perm = get_latent_perm(vanilla_sae, dataset)
    return matryoshka_perm, matryoshka_sae, vanilla_perm, vanilla_sae


@app.cell(hide_code=True)
def _(
    heatmap,
    matryoshka_perm,
    matryoshka_sae,
    true_feats,
    vanilla_perm,
    vanilla_sae,
):
    import torch.nn.functional as F

    # Compute cosine similarity matrix of ground truth features
    gt_cosine = F.normalize(true_feats, dim=1) @ F.normalize(true_feats, dim=1).T
    gt_sims = heatmap(gt_cosine.cpu(), title='Ground Truth Feature Cosine Similarity')

    # Compute cosine similarity between learned and ground truth features
    matryoshka_feats = matryoshka_sae.W_dec.data
    vanilla_feats = vanilla_sae.W_dec.data

    matryoshka_cosine = F.normalize(matryoshka_feats, dim=1)[matryoshka_perm] @ F.normalize(true_feats, dim=1).T
    vanilla_cosine = F.normalize(vanilla_feats, dim=1)[vanilla_perm] @ F.normalize(true_feats, dim=1).T

    m_dec_sims = heatmap(matryoshka_cosine.cpu(), title='Matryoshka Decoder, Ground-Truth Cosine Similarity', dim_names=('Matryoshka', 'True Feature'))

    v_dec_sims = heatmap(vanilla_cosine.cpu(), title='Vanilla Decoder, Ground-Truth Cosine Similarity', dim_names=('Vanilla', 'Ground-Truth'))
    # Compare encoder and decoder weights
    matryoshka_enc = matryoshka_sae.W_enc.data.T
    vanilla_enc = vanilla_sae.W_enc.data.T

    matryoshka_enc_true = F.normalize(matryoshka_enc, dim=1)[matryoshka_perm] @ F.normalize(true_feats, dim=1).T
    vanilla_enc_true = F.normalize(vanilla_enc, dim=1)[vanilla_perm] @ F.normalize(true_feats, dim=1).T


    m_enc_sims = heatmap(matryoshka_enc_true.cpu(), title='Matryoshka Encoder, Ground-Truth Cosine Similarity', dim_names=('Latent', 'Feature'))
    v_enc_sims = heatmap(vanilla_enc_true.cpu(), title='Vanilla Encoder, Ground-Truth Cosine Similarity', dim_names=('Latent', 'Feature'))
    return gt_cosine, matryoshka_cosine, vanilla_cosine


@app.cell
def _(gt_cosine, matryoshka_cosine, np, plt, vanilla_cosine):
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))

    ax[0].set_title('Vanilla SAE Cosine Similarity')
    sorted_v_cosine = sorted(vanilla_cosine, key=lambda x: np.argmax(x))
    ax[0].imshow(sorted_v_cosine)

    ax[1].set_title('Matryoshka SAE Cosine Similarity')
    sorted_m_cosine = sorted(matryoshka_cosine, key=lambda x: np.argmax(x))
    ax[1].imshow(sorted_m_cosine)

    ax[2].set_title('GT Cosine Similarity')
    ax[2].imshow(gt_cosine)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The above results are **actually** identical to the results using the provided toy example script from the original Matryoshka SAE paper when the adaptive sparsity loss is omitted. We do not implement the adaptive sparsity loss because we do not want to have to assume the true feature sparsity of our activations. Instead, we have changed the sparsity coefficient from 4e-4 to 4e-3. Finally, note that the ordering in the vanilla cosine similarity is just a bit off, the similarities themselves still line up how they should""")
    return


if __name__ == "__main__":
    app.run()
