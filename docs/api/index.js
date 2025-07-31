URLS=[
"saev/index.html",
"saev/helpers.html",
"saev/colors.html",
"saev/utils/index.html",
"saev/utils/scheduling.html",
"saev/utils/wandb.html",
"saev/ops.html",
"saev/scripts/index.html",
"saev/scripts/visuals.html",
"saev/imaging.html",
"saev/interactive/index.html",
"saev/interactive/features.html",
"saev/interactive/metrics.html",
"saev/nn/index.html",
"saev/nn/modeling.html",
"saev/nn/objectives.html",
"saev/data/index.html",
"saev/data/config.html",
"saev/data/writers.html",
"saev/data/models.html",
"saev/data/images.html",
"saev/data/buffers.html",
"saev/data/indexed.html",
"saev/data/ordered.html",
"saev/data/shuffled.html"
];
INDEX=[
{
"ref":"saev",
"url":0,
"doc":"saev is a Python package for training sparse autoencoders (SAEs) on vision transformers (ViTs) in PyTorch. The main entrypoint to the package is in  __main__ ; use  python -m saev  help to see the options and documentation for the script.  Guide to Training SAEs on Vision Models 1. Record ViT activations and save them to disk. 2. Train SAEs on the activations. 3. Visualize the learned features from the trained SAEs. 4. (your job) Propose trends and patterns in the visualized features. 5. (your job, supported by code) Construct datasets to test your hypothesized trends. 6. Confirm/reject hypotheses using  probing package.  saev helps with steps 1, 2 and 3.  note  saev assumes you are running on NVIDIA GPUs. On a multi-GPU system, prefix your commands with  CUDA_VISIBLE_DEVICES=X to run on GPU X.  Record ViT Activations to Disk To save activations to disk, we need to specify: 1. Which model we would like to use 2. Which layers we would like to save. 3. Where on disk and how we would like to save activations. 4. Which images we want to save activations for. The  saev.activations module does all of this for us. Run  uv run python -m saev activations  help to see all the configuration. In practice, you might run:   uv run python -m saev.data \\  vit-family siglip \\  vit-ckpt hf-hub:timm/ViT-L-16-SigLIP2-256 \\  d-vit 1024 \\  n-patches-per-img 256 \\  no-cls-token \\  vit-layers 13 15 17 19 21 23 \\  dump-to /fs/scratch/PAS2136/samuelstevens/cache/saev/ \\  max-patches-per-shard 500_000 \\  slurm-acct PAS2136 \\  n-hours 48 \\  slurm-partition nextgen \\ data:image-folder \\  data.root /fs/ess/PAS2136/foundation_model/inat21/raw/train_mini/   Let's break down these arguments. This will save activations for the CLIP-pretrained model ViT-B/32, which has a residual stream dimension of 768, and has 49 patches per image (224 / 32 = 7; 7 x 7 = 49). It will save the second-to-last layer (  layer -2 ). It will write 2.4M patches per shard, and save shards to a new directory  /local/scratch/$USER/cache/saev .  note A note on storage space: A ViT-B/16 will save 1.2M images x 197 patches/layer/image x 1 layer = ~240M activations, each of which take up 768 floats x 4 bytes/float = 3072 bytes, for a  total of 723GB for the entire dataset. As you scale to larger models (ViT-L has 1024 dimensions, 14x14 patches are 224 patches/layer/image), recorded activations will grow even larger. This script will also save a  metadata.json file that will record the relevant metadata for these activations, which will be read by future steps. The activations will be in  .bin files, numbered starting from 000000. To add your own models, see the guide to extending in  saev.activations .  Train SAEs on Activations To train an SAE, we need to specify: 1. Which activations to use as input. 2. SAE architectural stuff. 3. Optimization-related stuff. The  saev.training module handles this. Run  uv run python -m saev train  help to see all the configuration. Continuing on from our example before, you might want to run something like:   uv run python -m saev train \\  data.shard-root /local/scratch/$USER/cache/saev/ac89246f1934b45e2f0487298aebe36ad998b6bd252d880c0c9ec5de78d793c8 \\  data.layer -2 \\  data.patches patches \\  data.no-scale-mean \\  data.no-scale-norm \\  sae.d-vit 768 \\  lr 5e-4     uv run train.py  sweep configs/preprint/baseline.toml  data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/f9deaa8a07786087e8071f39a695200ff6713ee02b25e7a7b4a6d5ac1ad968db  data.patches image  data.layer 23  data.no-scale-mean  data.no-scale-norm sae:relu  sae.d-vit 1024     data. flags describe which activations to use.   data.shard-root should point to a directory with  .bin files and the  metadata.json file.   data.layer specifies the layer, and   data.patches says that want to train on individual patch activations, rather than the [CLS] token activation.   data.no-scale-mean and   data.no-scale-norm mean not to scale the activation mean or L2 norm. Anthropic's and OpenAI's papers suggest normalizing these factors, but  saev still has a bug with this, so I suggest not scaling these factors.   sae. flags are about the SAE itself.   sae.d-vit is the only one you need to change; the dimension of our ViT was 768 for a ViT-B, rather than the default of 1024 for a ViT-L. Finally, choose a slightly larger learning rate than the default with   lr 5e-4 . This will train one (1) sparse autoencoder on the data. See the section on sweeps to learn how to train multiple SAEs in parallel using only a single GPU.  Visualize the Learned Features Now that you've trained an SAE, you probably want to look at its learned features. One way to visualize an individual learned feature \\(f\\) is by picking out images that maximize the activation of feature \\(f\\). Since we train SAEs on patch-level activations, we try to find the top  patches for each feature \\(f\\). Then, we pick out the images those patches correspond to and create a heatmap based on SAE activation values.  note More advanced forms of visualization are possible (and valuable!), but should not be included in  saev unless they can be applied to every SAE/dataset combination. If you have specific visualizations, please add them to  contrib/ or another location.  saev.visuals records these maximally activating images for us. You can see all the options with  uv run python -m saev visuals  help . The most important configuration options: 1. The SAE checkpoint that you want to use (  ckpt ). 2. The ViT activations that you want to use (  data. options, should be roughly the same as the options you used to train your SAE, like the same layer, same   data.patches ). 3. The images that produced the ViT activations that you want to use ( images and   images. options, should be the same as what you used to generate your ViT activtions). 4. Some filtering options on which SAE latents to include (  log-freq-range ,   log-value-range ,   include-latents ,   n-latents ). Then, the script runs SAE inference on all of the ViT activations, calculates the images with maximal activation for each SAE feature, then retrieves the images from the original image dataset and highlights them for browsing later on.  note Because of limitations in the SAE training process, not all SAE latents (dimensions of \\(f\\ are equally interesting. Some latents are dead, some are  dense , some only fire on two images, etc. Typically, you want neurons that fire very strongly (high value) and fairly infrequently (low frequency). You might be interested in particular, fixed latents (  include-latents ).  I recommend using  saev.interactive.metrics to figure out good thresholds. So you might run:   uv run python -m saev visuals \\  ckpt checkpoints/abcdefg/sae.pt \\  dump-to /nfs/$USER/saev/webapp/abcdefg \\  data.shard-root /local/scratch/$USER/cache/saev/ac89246f1934b45e2f0487298aebe36ad998b6bd252d880c0c9ec5de78d793c8 \\  data.layer -2 \\  data.patches patches \\ images:imagenet-dataset   This will record the top 128 patches, and then save the unique images among those top 128 patches for each feature in the trained SAE. It will cache these best activations to disk, then start saving images to visualize later on.  saev.interactive.features is a small web application based on [marimo](https: marimo.io/) to interactively look at these images. You can run it with  uv run marimo edit saev/interactive/features.py .  Sweeps > tl;dr: basically the slow part of training SAEs is loading vit activations from disk, and since SAEs are pretty small compared to other models, you can train a bunch of different SAEs in parallel on the same data using a big GPU. That way you can sweep learning rate, lambda, etc. all on one GPU.  Why Parallel Sweeps SAE training optimizes for a unique bottleneck compared to typical ML workflows: disk I/O rather than GPU computation. When training on vision transformer activations, loading the pre-computed activation data from disk is often the slowest part of the process, not the SAE training itself. A single set of ImageNet activations for a vision transformer can require terabytes of storage. Reading this data repeatedly for each hyperparameter configuration would be extremely inefficient.  Parallelized Training Architecture To address this bottleneck, we implement parallel training that allows multiple SAE configurations to train simultaneously on the same data batch:  flowchart TD A[Pre-computed ViT Activations]  >|Slow I/O| B[Memory Buffer] B  >|Shared Batch| C[SAE Model 1] B  >|Shared Batch| D[SAE Model 2] B  >|Shared Batch| E[SAE Model 3] B  >|Shared Batch| F[ .]   import mermaid from 'https: cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';  This approach: - Loads each batch of activations  once from disk - Uses that same batch for multiple SAE models with different hyperparameters - Amortizes the slow I/O cost across all models in the sweep  Running a Sweep The  train command accepts a   sweep parameter that points to a TOML file defining the hyperparameter grid:   uv run python -m saev train  sweep configs/my_sweep.toml   Here's an example sweep configuration file:   [sae] sparsity_coeff = [1e-4, 2e-4, 3e-4] d_vit = 768 exp_factor = [8, 16] [data] scale_mean = true   This would train 6 models (3 sparsity coefficients \u00d7 2 expansion factors), each sharing the same data loading operation.  Limitations Not all parameters can be swept in parallel. Parameters that affect data loading (like  batch_size or dataset configuration) will cause the sweep to split into separate parallel groups. The system automatically handles this division to maximize efficiency.  Training Metrics and Visualizations When you train a sweep of SAEs, you probably want to understand which checkpoint is best.  saev provides some tools to help with that. First, we offer a tool to look at some basic summary statistics of all your trained checkpoints.  saev.interactive.metrics is a [marimo](https: marimo.io/) notebook (similar to Jupyter, but more interactive) for making L0 vs MSE plots by reading runs off of WandB. However, there are some pieces of code that need to be changed for you to use it.  todo Explain how to use the  saev.interactive.metrics notebook.  Need to change your wandb username from samuelstevens to USERNAME from wandb  Tag filter  Need to run the notebook on the same machine as the original ViT shards and the shards need to be there.  Think of better ways to do model and data keys  Look at examples  run visuals before features How to run visuals faster? explain how these features are visualized  Inference Instructions Briefly, you need to: 1. Download a checkpoint. 2. Get the code. 3. Load the checkpoint. 4. Get activations. Details are below.  Download a Checkpoint First, download an SAE checkpoint from the [Huggingface collection](https: huggingface.co/collections/osunlp/sae-v-67ab8c4fdf179d117db28195). For instance, you can choose the SAE trained on OpenAI's CLIP ViT-B/16 with ImageNet-1K activations [here](https: huggingface.co/osunlp/SAE_CLIP_24K_ViT-B-16_IN1K). You can use  wget if you want:   wget https: huggingface.co/osunlp/SAE_CLIP_24K_ViT-B-16_IN1K/resolve/main/sae.pt    Get the Code The easiest way to do this is to clone the code:   git clone https: github.com/OSU-NLP-Group/saev   You can also install the package from git if you use uv (not sure about pip or cuda):   uv add git+https: github.com/OSU-NLP-Group/saev   Or clone it and install it as an editable with pip, lik  pip install -e . in your virtual environment. Then you can do things like  from saev import  . .  note If you struggle to get  saev installed, open an issue on [GitHub](https: github.com/OSU-NLP-Group/saev) and I will figure out how to make it easier.  Load the Checkpoint   import saev.nn sae = saev.nn.load(\"PATH_TO_YOUR_SAE_CKPT.pt\")   Now you have a pretrained SAE.  Get Activations This is the hardest part. We need to: 1. Pass an image into a ViT 2. Record the dense ViT activations at the same layer that the SAE was trained on. 3. Pass the activations into the SAE to get sparse activations. 4. Do something interesting with the sparse SAE activations. There are examples of this in the demo code: for [classification](https: huggingface.co/spaces/samuelstevens/saev-image-classification/blob/main/app.py L318) and [semantic segmentation](https: huggingface.co/spaces/samuelstevens/saev-semantic-segmentation/blob/main/app.py L222). If the permalinks change, you are looking for the  get_sae_latents() functions in both files. Below is example code to do it using the  saev package.   import saev.nn import saev.activations img_transform = saev.activations.make_img_transform(\"clip\", \"ViT-B-16/openai\") vit = saev.activations.make_vit(\"clip\", \"ViT-B-16/openai\") recorded_vit = saev.activations.RecordedVisionTransformer(vit, 196, True, [10]) img = Image.open(\"example.jpg\") x = img_transform(img)  Add a batch dimension x = x[None,  .] _, vit_acts = recorded_vit(x)  Select the only layer in the batch and ignore the CLS token. vit_acts = vit_acts[:, 0, 1:, :] x_hat, f_x, loss = sae(vit_acts)   Now you have the reconstructed x ( x_hat ) and the sparse representation of all patches in the image ( f_x ). You might select the dimensions with maximal values for each patch and see what other images are maximimally activating.  todo Provide documentation for how get maximally activating images."
},
{
"ref":"saev.helpers",
"url":1,
"doc":""
},
{
"ref":"saev.helpers.RemovedFeatureError",
"url":1,
"doc":"Feature existed before but is no longer supported."
},
{
"ref":"saev.helpers.get_cache_dir",
"url":1,
"doc":"Get cache directory from environment variables, defaulting to the current working directory (.) Returns: A path to a cache directory (might not exist yet).",
"func":1
},
{
"ref":"saev.helpers.progress",
"url":1,
"doc":"Wraps an iterable with a logger like tqdm but doesn't use any control codes to manipulate a progress bar, which doesn't work well when your output is redirected to a file. Instead, simple logging statements are used, but it includes quality-of-life features like iteration speed and predicted time to finish. Args: it: Iterable to wrap. every: How many iterations between logging progress. desc: What to name the logger. total: If non-zero, how long the iterable is."
},
{
"ref":"saev.helpers.flattened",
"url":1,
"doc":"Flatten a potentially nested dict to a single-level dict with  . -separated keys.",
"func":1
},
{
"ref":"saev.helpers.get",
"url":1,
"doc":"",
"func":1
},
{
"ref":"saev.helpers.batched_idx",
"url":1,
"doc":"Iterate over (start, end) indices for total_size examples, where end - start is at most batch_size. Args: total_size: total number of examples batch_size: maximum distance between the generated indices. Returns: A generator of (int, int) tuples that can slice up a list or a tensor. Args: total_size: total number of examples batch_size: maximum distance between the generated indices"
},
{
"ref":"saev.helpers.expand",
"url":1,
"doc":"Expand a nested dict that may contain lists into many dicts.",
"func":1
},
{
"ref":"saev.helpers.grid",
"url":1,
"doc":"Generate configs from  cfg according to  sweep_dct .",
"func":1
},
{
"ref":"saev.helpers.current_git_commit",
"url":1,
"doc":"Best-effort short SHA of the repo containing  this file. Returns  None when   git executable is missing,  we\u2019re not inside a git repo (e.g. installed wheel),  or any git call errors out.",
"func":1
},
{
"ref":"saev.colors",
"url":2,
"doc":"Utility color palettes used across saev visualizations."
},
{
"ref":"saev.utils",
"url":3,
"doc":""
},
{
"ref":"saev.utils.scheduling",
"url":4,
"doc":""
},
{
"ref":"saev.utils.scheduling.Scheduler",
"url":4,
"doc":""
},
{
"ref":"saev.utils.scheduling.Scheduler.step",
"url":4,
"doc":"",
"func":1
},
{
"ref":"saev.utils.scheduling.Warmup",
"url":4,
"doc":"Linearly increases from  init to  final over  n_warmup_steps steps."
},
{
"ref":"saev.utils.scheduling.Warmup.step",
"url":4,
"doc":"",
"func":1
},
{
"ref":"saev.utils.scheduling.WarmupCosine",
"url":4,
"doc":"Linearly increases from  init to  peak over  n_warmup steps, then decrease down to final using cosine decay over n_steps - n_warmup."
},
{
"ref":"saev.utils.scheduling.WarmupCosine.step",
"url":4,
"doc":"",
"func":1
},
{
"ref":"saev.utils.scheduling.DataLoaderLike",
"url":4,
"doc":"Base class for protocol classes. Protocol classes are defined as class Proto(Protocol): def meth(self) -> int:  . Such classes are primarily used with static type checkers that recognize structural subtyping (static duck-typing). For example class C: def meth(self) -> int: return 0 def func(x: Proto) -> int: return x.meth() func(C(  Passes static type check See PEP 544 for details. Protocol classes decorated with @typing.runtime_checkable act as simple-minded runtime protocols that check only the presence of given attributes, ignoring their type signatures. Protocol classes can be generic, they are defined as class GenProto[T](Protocol): def meth(self) -> T:  ."
},
{
"ref":"saev.utils.scheduling.DataLoaderLike.drop_last",
"url":4,
"doc":""
},
{
"ref":"saev.utils.scheduling.DataLoaderLike.batch_size",
"url":4,
"doc":""
},
{
"ref":"saev.utils.scheduling.BatchLimiter",
"url":4,
"doc":"Limits the number of batches to only return  n_samples total samples."
},
{
"ref":"saev.utils.wandb",
"url":5,
"doc":""
},
{
"ref":"saev.utils.wandb.ParallelWandbRun",
"url":5,
"doc":"Inspired by https: community.wandb.ai/t/is-it-possible-to-log-to-multiple-runs-simultaneously/4387"
},
{
"ref":"saev.utils.wandb.ParallelWandbRun.log",
"url":5,
"doc":"",
"func":1
},
{
"ref":"saev.utils.wandb.ParallelWandbRun.finish",
"url":5,
"doc":"",
"func":1
},
{
"ref":"saev.ops",
"url":6,
"doc":""
},
{
"ref":"saev.ops.gather_batched",
"url":6,
"doc":"",
"func":1
},
{
"ref":"saev.scripts",
"url":7,
"doc":""
},
{
"ref":"saev.scripts.visuals",
"url":8,
"doc":"There is some important notation used only in this file to dramatically shorten variable names. Variables suffixed with  _im refer to entire images, and variables suffixed with  _p refer to patches."
},
{
"ref":"saev.scripts.visuals.Config",
"url":8,
"doc":"Configuration for generating visuals from trained SAEs."
},
{
"ref":"saev.scripts.visuals.Config.root",
"url":8,
"doc":""
},
{
"ref":"saev.scripts.visuals.Config.top_values_fpath",
"url":8,
"doc":""
},
{
"ref":"saev.scripts.visuals.Config.top_img_i_fpath",
"url":8,
"doc":""
},
{
"ref":"saev.scripts.visuals.Config.top_patch_i_fpath",
"url":8,
"doc":""
},
{
"ref":"saev.scripts.visuals.Config.mean_values_fpath",
"url":8,
"doc":""
},
{
"ref":"saev.scripts.visuals.Config.sparsity_fpath",
"url":8,
"doc":""
},
{
"ref":"saev.scripts.visuals.Config.distributions_fpath",
"url":8,
"doc":""
},
{
"ref":"saev.scripts.visuals.Config.percentiles_fpath",
"url":8,
"doc":""
},
{
"ref":"saev.scripts.visuals.Config.ckpt",
"url":8,
"doc":"Path to the sae.pt file."
},
{
"ref":"saev.scripts.visuals.Config.data",
"url":8,
"doc":"Data configuration"
},
{
"ref":"saev.scripts.visuals.Config.device",
"url":8,
"doc":"Which accelerator to use."
},
{
"ref":"saev.scripts.visuals.Config.dump_to",
"url":8,
"doc":"Where to save data."
},
{
"ref":"saev.scripts.visuals.Config.epsilon",
"url":8,
"doc":"Value to add to avoid log(0)."
},
{
"ref":"saev.scripts.visuals.Config.images",
"url":8,
"doc":"Which images to use."
},
{
"ref":"saev.scripts.visuals.Config.include_latents",
"url":8,
"doc":"Latents to always include, no matter what."
},
{
"ref":"saev.scripts.visuals.Config.log_freq_range",
"url":8,
"doc":"Log10 frequency range for which to save images."
},
{
"ref":"saev.scripts.visuals.Config.log_to",
"url":8,
"doc":"Where to log Slurm job stdout/stderr."
},
{
"ref":"saev.scripts.visuals.Config.log_value_range",
"url":8,
"doc":"Log10 frequency range for which to save images."
},
{
"ref":"saev.scripts.visuals.Config.n_distributions",
"url":8,
"doc":"Number of features to save distributions for."
},
{
"ref":"saev.scripts.visuals.Config.n_hours",
"url":8,
"doc":"Slurm job length in hours."
},
{
"ref":"saev.scripts.visuals.Config.n_latents",
"url":8,
"doc":"Maximum number of latents to save images for."
},
{
"ref":"saev.scripts.visuals.Config.percentile",
"url":8,
"doc":"Percentile to estimate for outlier detection."
},
{
"ref":"saev.scripts.visuals.Config.sae_batch_size",
"url":8,
"doc":"Batch size for SAE inference."
},
{
"ref":"saev.scripts.visuals.Config.seed",
"url":8,
"doc":"Random seed."
},
{
"ref":"saev.scripts.visuals.Config.slurm_acct",
"url":8,
"doc":"Slurm account string. Empty means to not use Slurm."
},
{
"ref":"saev.scripts.visuals.Config.slurm_partition",
"url":8,
"doc":"Slurm partition."
},
{
"ref":"saev.scripts.visuals.Config.sort_by",
"url":8,
"doc":"How to find the top k images. 'cls' picks images where the SAE latents of the ViT's [CLS] token are maximized without any patch highligting. 'img' picks images that maximize the sum of an SAE latent over all patches in the image, highlighting the patches. 'patch' pickes images that maximize an SAE latent over all patches (not summed), highlighting the patches and only showing unique images."
},
{
"ref":"saev.scripts.visuals.Config.top_k",
"url":8,
"doc":"How many images per SAE feature to store."
},
{
"ref":"saev.scripts.visuals.Config.topk_batch_size",
"url":8,
"doc":""
},
{
"ref":"saev.scripts.visuals.safe_load",
"url":8,
"doc":"",
"func":1
},
{
"ref":"saev.scripts.visuals.gather_batched",
"url":8,
"doc":"",
"func":1
},
{
"ref":"saev.scripts.visuals.GridElement",
"url":8,
"doc":"GridElement(img: PIL.Image.Image, label: str, patches: jaxtyping.Float[Tensor, 'n_patches'])"
},
{
"ref":"saev.scripts.visuals.GridElement.img",
"url":8,
"doc":""
},
{
"ref":"saev.scripts.visuals.GridElement.label",
"url":8,
"doc":""
},
{
"ref":"saev.scripts.visuals.GridElement.patches",
"url":8,
"doc":""
},
{
"ref":"saev.scripts.visuals.make_img",
"url":8,
"doc":"",
"func":1
},
{
"ref":"saev.scripts.visuals.get_new_topk",
"url":8,
"doc":"Picks out the new top k values among val1 and val2. Also keeps track of i1 and i2, then indices of the values in the original dataset. Args: val1: top k original SAE values. i1: the patch indices of those original top k values. val2: top k incoming SAE values. i2: the patch indices of those incoming top k values. k: k. Returns: The new top k values and their patch indices.",
"func":1
},
{
"ref":"saev.scripts.visuals.batched_idx",
"url":8,
"doc":"Iterate over (start, end) indices for total_size examples, where end - start is at most batch_size. Args: total_size: total number of examples batch_size: maximum distance between the generated indices. Returns: A generator of (int, int) tuples that can slice up a list or a tensor.",
"func":1
},
{
"ref":"saev.scripts.visuals.get_sae_acts",
"url":8,
"doc":"Get SAE hidden layer activations for a batch of ViT activations. Args: vit_acts: Batch of ViT activations sae: Sparse autoencder. cfg: Experimental config.",
"func":1
},
{
"ref":"saev.scripts.visuals.TopKPatch",
"url":8,
"doc":" todo Document this class."
},
{
"ref":"saev.scripts.visuals.TopKPatch.top_values",
"url":8,
"doc":""
},
{
"ref":"saev.scripts.visuals.TopKPatch.top_i",
"url":8,
"doc":""
},
{
"ref":"saev.scripts.visuals.TopKPatch.mean_values",
"url":8,
"doc":""
},
{
"ref":"saev.scripts.visuals.TopKPatch.sparsity",
"url":8,
"doc":""
},
{
"ref":"saev.scripts.visuals.TopKPatch.distributions",
"url":8,
"doc":""
},
{
"ref":"saev.scripts.visuals.TopKPatch.percentiles",
"url":8,
"doc":""
},
{
"ref":"saev.scripts.visuals.get_topk_patch",
"url":8,
"doc":"Gets the top k images for each latent in the SAE. The top k images are for latent i are sorted by max over all patches: f_x(patch)[i] Thus, we could end up with duplicate images in the top k, if an image has more than one patch that maximally activates an SAE latent. Args: cfg: Config. Returns: A tuple of TopKPatch and m randomly sampled activation distributions.",
"func":1
},
{
"ref":"saev.scripts.visuals.dump_activations",
"url":8,
"doc":"Dump ViT activation statistics for later use. The dataset described by  cfg is processed to find the images or patches that maximally activate each SAE latent. Various tensors summarising these activations are then written to  cfg.root so they can be loaded by other tools. Args: cfg: options controlling which activations are processed and where the resulting files are saved. Returns: None. All data is saved to disk.",
"func":1
},
{
"ref":"saev.scripts.visuals.plot_activation_distributions",
"url":8,
"doc":"",
"func":1
},
{
"ref":"saev.scripts.visuals.dump_imgs",
"url":8,
"doc":" todo document this function. Dump top-k images to a directory. Args: cfg: Configuration object.",
"func":1
},
{
"ref":"saev.scripts.visuals.PercentileEstimator",
"url":8,
"doc":""
},
{
"ref":"saev.scripts.visuals.PercentileEstimator.update",
"url":8,
"doc":"Update the estimator with a new value. This method maintains the marker positions using the P2 algorithm rules. When a new value arrives, it's placed in the appropriate position relative to existing markers, and marker positions are adjusted to maintain their desired percentile positions. Arguments: x: The new value to incorporate into the estimation",
"func":1
},
{
"ref":"saev.scripts.visuals.PercentileEstimator.estimate",
"url":8,
"doc":""
},
{
"ref":"saev.imaging",
"url":9,
"doc":""
},
{
"ref":"saev.imaging.add_highlights",
"url":9,
"doc":"",
"func":1
},
{
"ref":"saev.interactive",
"url":10,
"doc":""
},
{
"ref":"saev.interactive.features",
"url":11,
"doc":""
},
{
"ref":"saev.interactive.metrics",
"url":12,
"doc":""
},
{
"ref":"saev.nn",
"url":13,
"doc":""
},
{
"ref":"saev.nn.SparseAutoencoder",
"url":13,
"doc":"Sparse auto-encoder (SAE) using L1 sparsity penalty. Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.nn.SparseAutoencoder.forward",
"url":13,
"doc":"Given x, calculates the reconstructed x_hat and the intermediate activations f_x. Arguments: x: a batch of ViT activations.",
"func":1
},
{
"ref":"saev.nn.SparseAutoencoder.decode",
"url":13,
"doc":"",
"func":1
},
{
"ref":"saev.nn.SparseAutoencoder.normalize_w_dec",
"url":13,
"doc":"Set W_dec to unit-norm columns.",
"func":1
},
{
"ref":"saev.nn.SparseAutoencoder.remove_parallel_grads",
"url":13,
"doc":"Update grads so that they remove the parallel component (d_sae, d_vit) shape",
"func":1
},
{
"ref":"saev.nn.dump",
"url":13,
"doc":"Save an SAE checkpoint to disk along with configuration, using the [trick from equinox](https: docs.kidger.site/equinox/examples/serialisation). Arguments: fpath: filepath to save checkpoint to. sae: sparse autoencoder checkpoint to save.",
"func":1
},
{
"ref":"saev.nn.load",
"url":13,
"doc":"Loads a sparse autoencoder from disk.",
"func":1
},
{
"ref":"saev.nn.get_objective",
"url":13,
"doc":"",
"func":1
},
{
"ref":"saev.nn.modeling",
"url":14,
"doc":"Neural network architectures for sparse autoencoders."
},
{
"ref":"saev.nn.modeling.Relu",
"url":14,
"doc":"Relu(d_vit: int = 1024, exp_factor: int = 16, n_reinit_samples: int = 524288, remove_parallel_grads: bool = True, normalize_w_dec: bool = True, seed: int = 0)"
},
{
"ref":"saev.nn.modeling.Relu.d_sae",
"url":14,
"doc":""
},
{
"ref":"saev.nn.modeling.Relu.d_vit",
"url":14,
"doc":""
},
{
"ref":"saev.nn.modeling.Relu.exp_factor",
"url":14,
"doc":"Expansion factor for SAE."
},
{
"ref":"saev.nn.modeling.Relu.n_reinit_samples",
"url":14,
"doc":"Number of samples to use for SAE re-init. Anthropic proposes initializing b_dec to the geometric median of the dataset here: https: transformer-circuits.pub/2023/monosemantic-features/index.html appendix-autoencoder-bias. We use the regular mean."
},
{
"ref":"saev.nn.modeling.Relu.normalize_w_dec",
"url":14,
"doc":"Whether to make sure W_dec has unit norm columns. See https: transformer-circuits.pub/2023/monosemantic-features/index.html appendix-autoencoder for original citation."
},
{
"ref":"saev.nn.modeling.Relu.remove_parallel_grads",
"url":14,
"doc":"Whether to remove gradients parallel to W_dec columns (which will be ignored because we force the columns to have unit norm). See https: transformer-circuits.pub/2023/monosemantic-features/index.html appendix-autoencoder-optimization for the original discussion from Anthropic."
},
{
"ref":"saev.nn.modeling.Relu.seed",
"url":14,
"doc":"Random seed."
},
{
"ref":"saev.nn.modeling.JumpRelu",
"url":14,
"doc":"Implementation of the JumpReLU activation function for SAEs. Not implemented."
},
{
"ref":"saev.nn.modeling.TopK",
"url":14,
"doc":"TopK(d_vit: int = 1024, exp_factor: int = 16, n_reinit_samples: int = 524288, remove_parallel_grads: bool = True, normalize_w_dec: bool = True, seed: int = 0, top_k: int = 32)"
},
{
"ref":"saev.nn.modeling.TopK.d_sae",
"url":14,
"doc":""
},
{
"ref":"saev.nn.modeling.TopK.d_vit",
"url":14,
"doc":""
},
{
"ref":"saev.nn.modeling.TopK.exp_factor",
"url":14,
"doc":"Expansion factor for SAE."
},
{
"ref":"saev.nn.modeling.TopK.n_reinit_samples",
"url":14,
"doc":"Number of samples to use for SAE re-init. Anthropic proposes initializing b_dec to the geometric median of the dataset here: https: transformer-circuits.pub/2023/monosemantic-features/index.html appendix-autoencoder-bias. We use the regular mean."
},
{
"ref":"saev.nn.modeling.TopK.normalize_w_dec",
"url":14,
"doc":"Whether to make sure W_dec has unit norm columns. See https: transformer-circuits.pub/2023/monosemantic-features/index.html appendix-autoencoder for original citation."
},
{
"ref":"saev.nn.modeling.TopK.remove_parallel_grads",
"url":14,
"doc":"Whether to remove gradients parallel to W_dec columns (which will be ignored because we force the columns to have unit norm). See https: transformer-circuits.pub/2023/monosemantic-features/index.html appendix-autoencoder-optimization for the original discussion from Anthropic."
},
{
"ref":"saev.nn.modeling.TopK.seed",
"url":14,
"doc":"Random seed."
},
{
"ref":"saev.nn.modeling.TopK.top_k",
"url":14,
"doc":""
},
{
"ref":"saev.nn.modeling.BatchTopK",
"url":14,
"doc":"BatchTopK(d_vit: int = 1024, exp_factor: int = 16, n_reinit_samples: int = 524288, remove_parallel_grads: bool = True, normalize_w_dec: bool = True, seed: int = 0, top_k: int = 32)"
},
{
"ref":"saev.nn.modeling.BatchTopK.d_sae",
"url":14,
"doc":""
},
{
"ref":"saev.nn.modeling.BatchTopK.d_vit",
"url":14,
"doc":""
},
{
"ref":"saev.nn.modeling.BatchTopK.exp_factor",
"url":14,
"doc":"Expansion factor for SAE."
},
{
"ref":"saev.nn.modeling.BatchTopK.n_reinit_samples",
"url":14,
"doc":"Number of samples to use for SAE re-init. Anthropic proposes initializing b_dec to the geometric median of the dataset here: https: transformer-circuits.pub/2023/monosemantic-features/index.html appendix-autoencoder-bias. We use the regular mean."
},
{
"ref":"saev.nn.modeling.BatchTopK.normalize_w_dec",
"url":14,
"doc":"Whether to make sure W_dec has unit norm columns. See https: transformer-circuits.pub/2023/monosemantic-features/index.html appendix-autoencoder for original citation."
},
{
"ref":"saev.nn.modeling.BatchTopK.remove_parallel_grads",
"url":14,
"doc":"Whether to remove gradients parallel to W_dec columns (which will be ignored because we force the columns to have unit norm). See https: transformer-circuits.pub/2023/monosemantic-features/index.html appendix-autoencoder-optimization for the original discussion from Anthropic."
},
{
"ref":"saev.nn.modeling.BatchTopK.seed",
"url":14,
"doc":"Random seed."
},
{
"ref":"saev.nn.modeling.BatchTopK.top_k",
"url":14,
"doc":""
},
{
"ref":"saev.nn.modeling.SparseAutoencoder",
"url":14,
"doc":"Sparse auto-encoder (SAE) using L1 sparsity penalty. Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.nn.modeling.SparseAutoencoder.forward",
"url":14,
"doc":"Given x, calculates the reconstructed x_hat and the intermediate activations f_x. Arguments: x: a batch of ViT activations.",
"func":1
},
{
"ref":"saev.nn.modeling.SparseAutoencoder.decode",
"url":14,
"doc":"",
"func":1
},
{
"ref":"saev.nn.modeling.SparseAutoencoder.normalize_w_dec",
"url":14,
"doc":"Set W_dec to unit-norm columns.",
"func":1
},
{
"ref":"saev.nn.modeling.SparseAutoencoder.remove_parallel_grads",
"url":14,
"doc":"Update grads so that they remove the parallel component (d_sae, d_vit) shape",
"func":1
},
{
"ref":"saev.nn.modeling.MatryoshkaSparseAutoencoder",
"url":14,
"doc":"Subclass of SparseAutoencoder for use with the Matryoshka objective function. Needed since the matryoshka objective requires access to the weights of the decoder in order to calculate the reconstructions from prefixes of the sparse encoding. Still uses L1 for sparsity penalty, though when using BatchTopK as activation (recommended), this is not relevant. Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.nn.modeling.MatryoshkaSparseAutoencoder.matryoshka_forward",
"url":14,
"doc":"Given x, calculates the reconstructed x_hat from the prefixes of encoded intermediate activations f_x. Arguments: x: a batch of ViT activations.",
"func":1
},
{
"ref":"saev.nn.modeling.MatryoshkaSparseAutoencoder.block_decode",
"url":14,
"doc":"Decodes sparse encoding using only the given interval of indices. Arguments: f_x: Sparse encoding",
"func":1
},
{
"ref":"saev.nn.modeling.MatryoshkaSparseAutoencoder.sample_prefixes",
"url":14,
"doc":"Samples prefix lengths using a Pareto distribution. Derived from \"Learning Multi-Level Features with Matryoshka Sparse Autoencoders\" (https: doi.org/10.48550/arXiv.2503.17547) Args: sae_dim: Total number of latent dimensions n_prefixes: Number of prefixes to sample min_prefix_length: Minimum length of any prefix pareto_power: Power parameter for Pareto distribution (lower = more uniform) Returns: torch.Tensor: Sorted prefix lengths",
"func":1
},
{
"ref":"saev.nn.modeling.MatryoshkaSparseAutoencoder.forward",
"url":14,
"doc":"Given x, calculates the reconstructed x_hat and the intermediate activations f_x. Arguments: x: a batch of ViT activations.",
"func":1
},
{
"ref":"saev.nn.modeling.MatryoshkaSparseAutoencoder.normalize_w_dec",
"url":14,
"doc":"Set W_dec to unit-norm columns.",
"func":1
},
{
"ref":"saev.nn.modeling.MatryoshkaSparseAutoencoder.remove_parallel_grads",
"url":14,
"doc":"Update grads so that they remove the parallel component (d_sae, d_vit) shape",
"func":1
},
{
"ref":"saev.nn.modeling.TopKActivation",
"url":14,
"doc":"Top-K activation function. For use as activation function of sparse encoder. Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.nn.modeling.TopKActivation.forward",
"url":14,
"doc":"Apply top-k activation to the input tensor.",
"func":1
},
{
"ref":"saev.nn.modeling.BatchTopKActivation",
"url":14,
"doc":"Batch Top-K activation function. For use as activation function of sparse encoder. Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.nn.modeling.BatchTopKActivation.forward",
"url":14,
"doc":"Apply top-k activation to the input tensor.",
"func":1
},
{
"ref":"saev.nn.modeling.get_activation",
"url":14,
"doc":"",
"func":1
},
{
"ref":"saev.nn.modeling.dump",
"url":14,
"doc":"Save an SAE checkpoint to disk along with configuration, using the [trick from equinox](https: docs.kidger.site/equinox/examples/serialisation). Arguments: fpath: filepath to save checkpoint to. sae: sparse autoencoder checkpoint to save.",
"func":1
},
{
"ref":"saev.nn.modeling.load",
"url":14,
"doc":"Loads a sparse autoencoder from disk.",
"func":1
},
{
"ref":"saev.nn.objectives",
"url":15,
"doc":""
},
{
"ref":"saev.nn.objectives.Vanilla",
"url":15,
"doc":"Vanilla(sparsity_coeff: float = 0.0004)"
},
{
"ref":"saev.nn.objectives.Vanilla.sparsity_coeff",
"url":15,
"doc":"How much to weight sparsity loss term."
},
{
"ref":"saev.nn.objectives.Matryoshka",
"url":15,
"doc":"Config for the Matryoshka loss for another arbitrary SAE class. Reference code is here: https: github.com/noanabeshima/matryoshka-saes and the original reading is https: sparselatents.com/matryoshka.html and https: arxiv.org/pdf/2503.17547"
},
{
"ref":"saev.nn.objectives.Matryoshka.n_prefixes",
"url":15,
"doc":"Number of random length prefixes to use for loss calculation."
},
{
"ref":"saev.nn.objectives.Matryoshka.sparsity_coeff",
"url":15,
"doc":"How much to weight sparsity loss term (if not using TopK/BatchTopK)."
},
{
"ref":"saev.nn.objectives.Loss",
"url":15,
"doc":"The loss term for an autoencoder training batch."
},
{
"ref":"saev.nn.objectives.Loss.loss",
"url":15,
"doc":"Total loss."
},
{
"ref":"saev.nn.objectives.Loss.metrics",
"url":15,
"doc":"",
"func":1
},
{
"ref":"saev.nn.objectives.Objective",
"url":15,
"doc":"Base class for all neural network modules. Your models should also subclass this class. Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes import torch.nn as nn import torch.nn.functional as F class Model(nn.Module): def __init__(self) -> None: super().__init__() self.conv1 = nn.Conv2d(1, 20, 5) self.conv2 = nn.Conv2d(20, 20, 5) def forward(self, x): x = F.relu(self.conv1(x return F.relu(self.conv2(x Submodules assigned in this way will be registered, and will have their parameters converted too when you call :meth: to , etc.  note As per the example above, an  __init__() call to the parent class must be made before assignment on the child. :ivar training: Boolean represents whether this module is in training or evaluation mode. :vartype training: bool Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.nn.objectives.Objective.forward",
"url":15,
"doc":"Define the computation performed at every call. Should be overridden by all subclasses.  note Although the recipe for forward pass needs to be defined within this function, one should call the :class: Module instance afterwards instead of this since the former takes care of running the registered hooks while the latter silently ignores them.",
"func":1
},
{
"ref":"saev.nn.objectives.VanillaLoss",
"url":15,
"doc":"The vanilla loss terms for an training batch."
},
{
"ref":"saev.nn.objectives.VanillaLoss.loss",
"url":15,
"doc":"Total loss."
},
{
"ref":"saev.nn.objectives.VanillaLoss.metrics",
"url":15,
"doc":"",
"func":1
},
{
"ref":"saev.nn.objectives.VanillaLoss.l0",
"url":15,
"doc":"L0 magnitude of hidden activations."
},
{
"ref":"saev.nn.objectives.VanillaLoss.l1",
"url":15,
"doc":"L1 magnitude of hidden activations."
},
{
"ref":"saev.nn.objectives.VanillaLoss.mse",
"url":15,
"doc":"Reconstruction loss (mean squared error)."
},
{
"ref":"saev.nn.objectives.VanillaLoss.sparsity",
"url":15,
"doc":"Sparsity loss, typically lambda  L1."
},
{
"ref":"saev.nn.objectives.VanillaObjective",
"url":15,
"doc":"Base class for all neural network modules. Your models should also subclass this class. Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes import torch.nn as nn import torch.nn.functional as F class Model(nn.Module): def __init__(self) -> None: super().__init__() self.conv1 = nn.Conv2d(1, 20, 5) self.conv2 = nn.Conv2d(20, 20, 5) def forward(self, x): x = F.relu(self.conv1(x return F.relu(self.conv2(x Submodules assigned in this way will be registered, and will have their parameters converted too when you call :meth: to , etc.  note As per the example above, an  __init__() call to the parent class must be made before assignment on the child. :ivar training: Boolean represents whether this module is in training or evaluation mode. :vartype training: bool Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.nn.objectives.VanillaObjective.forward",
"url":15,
"doc":"Define the computation performed at every call. Should be overridden by all subclasses.  note Although the recipe for forward pass needs to be defined within this function, one should call the :class: Module instance afterwards instead of this since the former takes care of running the registered hooks while the latter silently ignores them.",
"func":1
},
{
"ref":"saev.nn.objectives.MatryoshkaLoss",
"url":15,
"doc":"The composite loss terms for an training batch."
},
{
"ref":"saev.nn.objectives.MatryoshkaLoss.loss",
"url":15,
"doc":"Total loss."
},
{
"ref":"saev.nn.objectives.MatryoshkaLoss.metrics",
"url":15,
"doc":"",
"func":1
},
{
"ref":"saev.nn.objectives.MatryoshkaLoss.l0",
"url":15,
"doc":"Sum of L0 magnitudes of hidden activations for all prefix lengths."
},
{
"ref":"saev.nn.objectives.MatryoshkaLoss.l1",
"url":15,
"doc":"Sum of L1 magnitudes of hidden activations for all prefix lengths."
},
{
"ref":"saev.nn.objectives.MatryoshkaLoss.mse",
"url":15,
"doc":"Average of reconstruction loss (mean squared error) for all prefix lengths."
},
{
"ref":"saev.nn.objectives.MatryoshkaLoss.sparsity",
"url":15,
"doc":"Sparsity loss, typically lambda  L1."
},
{
"ref":"saev.nn.objectives.MatryoshkaObjective",
"url":15,
"doc":"Torch module for calculating the matryoshka loss for an SAE. Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.nn.objectives.MatryoshkaObjective.forward",
"url":15,
"doc":"Define the computation performed at every call. Should be overridden by all subclasses.  note Although the recipe for forward pass needs to be defined within this function, one should call the :class: Module instance afterwards instead of this since the former takes care of running the registered hooks while the latter silently ignores them.",
"func":1
},
{
"ref":"saev.nn.objectives.get_objective",
"url":15,
"doc":"",
"func":1
},
{
"ref":"saev.nn.objectives.ref_mean_squared_err",
"url":15,
"doc":"",
"func":1
},
{
"ref":"saev.nn.objectives.mean_squared_err",
"url":15,
"doc":"",
"func":1
},
{
"ref":"saev.data",
"url":16,
"doc":" SAEV Sharded-Activation File Protocol v1 (2025-06-17) saev caches activations to disk rather than run ViT or LLM inference when training SAEs. Gemma Scope makes this decision as well (see Section 3.3.2 of https: arxiv.org/pdf/2408.05147).  saev.data has a specific protocol to support this in on [OSC](https: www.osc.edu), a super computer center, and take advantage of OSC's specific disk performance. Goal: loss-lessly persist very large Transformer (ViT or LLM) activations in a form that is:  mem-mappable  Parameterized solely by the  experiment configuration ( writers.Config )  Referenced by a content-hash, so identical configs collide, divergent ones never do  Can be read quickly in a random order for training, and can be read (slowly) with random-access for visuals. This document is the single normative source. Any divergence in code is a  bug .  -  1. Directory layout    / / metadata.json  UTF-8 JSON, human-readable, describes data-generating config shards.json  UTF-8 JSON, human-readable, describes shards. acts000000.bin  shard 0 acts000001.bin  shard 1  . actsNNNNNN.bin  shard NNNNNN (zero-padded width=6)    HASH =  sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':' .encode('utf-8'  Guards against silent config drift.  -  2. JSON file schemas  2.1.  metadata.json | field | type | semantic | |            - |    |                      | |  vit_family | string |  \"clip\" \\| \"siglip\" \\| \"dinov2\" | |  vit_ckpt | string | model identifier (OpenCLIP, HF, etc.) | |  layers | int[] | ViT residual\u2010block indices recorded | |  n_patches_per_img | int |  image patches only (excludes CLS) | |  cls_token | bool |  true -> patch 0 is CLS, else no CLS | |  d_vit | int | activation dimensionality | |  n_imgs | int | total images in dataset | |  max_patches_per_shard | int |  logical activations per shard (see  3) | |  data | object | opaque dataset description | |  dtype | string | numpy dtype. Fixed  \"float32\" for now. | |  protocol | string |  \"1.0.0\" for now. | The  data object is  dataclasses.asdict(cfg.data) , with an additional  __class__ field with  cfg.data.__class__.__name__ as the value.  2.2.  shards.json A single array of  shard objects, each of which has the following fields: | field | type | semantic | |    |    |                  | | name | string | shard filename ( acts000000.bin ). | | n_imgs | int | the number of images in the shard. |  -  3 Shard sizing maths   n_tokens_per_img = n_patches_per_img + (1 if cls_token else 0) n_imgs_per_shard = floor(max_patches_per_shard / (n_tokens_per_img  len(layers ) shape_per_shard = ( n_imgs_per_shard, len(layers), n_tokens_per_img, d_vit, )    max_patches_per_shard is a  budget (default ~2.4 M) chosen so a shard is approximately 10 GiB for Float32 @  d_vit = 1024 .  The last shard will have a smaller value for  n_imgs_per_shard ; this value is documented in  n_imgs in  shards.json  -  4. Data Layout and Global Indexing The entire dataset of activations is treated as a single logical 4D tensor with the shape  (n_imgs, len(layers), n_tokens_per_img, d_vit) . This logical tensor is C-contiguous with axes ordered  [Image, Layer, Token, Dimension] . Physically, this tensor is split along the first axis ( Image ) into multiple shards, where each shard is a single binary file. The number of images in each shard is constant, except for the final shard, which may be smaller. To locate an arbitrary activation vector, a reader must convert a logical coordinate ( global_img_idx ,  layer_value ,  token_idx ) into a file path and an offset within that file.  4.1 Definitions Let the parameters from  metadata.json be:  L =  len(layers)  P =  n_patches_per_img  T =  P + (1 if cls_token else 0) (Total tokens per image)  D =  d_vit  S =  n_imgs from  shards.json or  n_imgs_per_shard from Section 3 (shard sizing).  4.2 Coordinate Transformations Given a logical coordinate:   global_img_idx : integer, with  0   Not sure if this is the correct way to think about it, but: 100 / 14.6 = 6.8, close to 7.9 hours.  Ordered Dataloader Design The  saev/data/ordered.py module implements a high-throughput ordered dataloader that guarantees sequential data delivery. This is useful for iterating through all patches in an image at once.  Key Design Decisions 1. Single-threaded I/O in Manager Process Initially, the dataloader used multiple worker threads for parallel I/O, similar to PyTorch's DataLoader. However, this created a fundamental ordering problem: when multiple workers read batches in parallel, they complete at different times and deliver batches out of order. We switched to single-threaded I/O because: - Sequential reads from memory-mapped files are already highly optimized by the OS - The OS page cache provides excellent performance for sequential access patterns - Eliminating multi-threading removes all batch reordering complexity - The simpler design is more maintainable and debuggable 2. Process Separation with Ring Buffer The dataloader still uses a separate manager process connected via a multiprocessing Queue (acting as a ring buffer). This provides: - Overlap between I/O and computation - Configurable read-ahead via  buffer_size parameter - Natural backpressure when computation is slower than I/O - Process isolation for better resource management 3. Shard-Aware Sequential Reading The dataloader correctly handles the actual distribution of data across shards by: - Reading  shards.json to get the exact number of images per shard - Maintaining cumulative offsets for efficient index-to-shard mapping - Handling batches that span multiple shards without gaps or duplicates  Performance Considerations - Memory-mapped files: Using  np.memmap allows efficient access to large files without loading them entirely into memory - Sequential access pattern: The dataloader reads data in the exact order it's stored on disk, maximizing OS cache effectiveness - Minimal data copying: Activations are copied only once from the memory-mapped file to PyTorch tensors - Read-ahead buffering: The configurable buffer size allows tuning the trade-off between memory usage and I/O overlap  Trade-offs The single-threaded design trades potential parallel I/O throughput for: - Guaranteed ordering - Simplicity and maintainability - Elimination of synchronization overhead - Predictable performance characteristics In practice, the sequential read performance is sufficient for most use cases, especially when the computation (e.g., SAE forward pass) is the bottleneck rather than I/O."
},
{
"ref":"saev.data.Config",
"url":16,
"doc":"Configuration for loading indexed activation data from disk."
},
{
"ref":"saev.data.Config.shard_root",
"url":16,
"doc":"Directory with .bin shards and a metadata.json file."
},
{
"ref":"saev.data.Config.patches",
"url":16,
"doc":"Which kinds of patches to use. 'cls' indicates just the [CLS] token (if any). 'image' indicates it will return image patches. 'all' returns all patches."
},
{
"ref":"saev.data.Config.layer",
"url":16,
"doc":"Which ViT layer(s) to read from disk.  -2 selects the second-to-last layer.  \"all\" enumerates every recorded layer."
},
{
"ref":"saev.data.Config.seed",
"url":16,
"doc":"Random seed."
},
{
"ref":"saev.data.Config.debug",
"url":16,
"doc":"Whether the dataloader process should log debug messages."
},
{
"ref":"saev.data.Dataset",
"url":16,
"doc":"Dataset of activations from disk."
},
{
"ref":"saev.data.Dataset.cfg",
"url":16,
"doc":"Configuration; set via CLI args."
},
{
"ref":"saev.data.Dataset.metadata",
"url":16,
"doc":"Activations metadata; automatically loaded from disk."
},
{
"ref":"saev.data.Dataset.layer_index",
"url":16,
"doc":"Layer index into the shards if we are choosing a specific layer."
},
{
"ref":"saev.data.Dataset.Example",
"url":16,
"doc":"Individual example."
},
{
"ref":"saev.data.Dataset.transform",
"url":16,
"doc":"",
"func":1
},
{
"ref":"saev.data.Dataset.d_vit",
"url":16,
"doc":"Dimension of the underlying vision transformer's embedding space."
},
{
"ref":"saev.data.Dataset.get_img_patches",
"url":16,
"doc":"",
"func":1
},
{
"ref":"saev.data.DataLoader",
"url":16,
"doc":"High-throughput streaming loader that reads data from disk shards in order (no shuffling)."
},
{
"ref":"saev.data.DataLoader.ExampleBatch",
"url":16,
"doc":"Individual example."
},
{
"ref":"saev.data.DataLoader.n_batches",
"url":16,
"doc":""
},
{
"ref":"saev.data.DataLoader.n_samples",
"url":16,
"doc":""
},
{
"ref":"saev.data.DataLoader.batch_size",
"url":16,
"doc":""
},
{
"ref":"saev.data.DataLoader.drop_last",
"url":16,
"doc":""
},
{
"ref":"saev.data.DataLoader.shutdown",
"url":16,
"doc":"",
"func":1
},
{
"ref":"saev.data.Config",
"url":16,
"doc":"Configuration for loading ordered (non-shuffled) activation data from disk."
},
{
"ref":"saev.data.Config.shard_root",
"url":16,
"doc":"Directory with .bin shards and a metadata.json file."
},
{
"ref":"saev.data.Config.patches",
"url":16,
"doc":"Which kinds of patches to use. 'cls' indicates just the [CLS] token (if any). 'image' indicates it will return image patches. 'all' returns all patches."
},
{
"ref":"saev.data.Config.layer",
"url":16,
"doc":"Which ViT layer(s) to read from disk.  -2 selects the second-to-last layer.  \"all\" enumerates every recorded layer."
},
{
"ref":"saev.data.Config.batch_size",
"url":16,
"doc":"Batch size."
},
{
"ref":"saev.data.Config.batch_timeout_s",
"url":16,
"doc":"How long to wait for at least one batch."
},
{
"ref":"saev.data.Config.drop_last",
"url":16,
"doc":"Whether to drop the last batch if it's smaller than the others."
},
{
"ref":"saev.data.Config.buffer_size",
"url":16,
"doc":"Number of batches to queue in the shared-memory ring buffer. Higher values add latency but improve resilience to brief stalls."
},
{
"ref":"saev.data.Config.debug",
"url":16,
"doc":"Whether the dataloader process should log debug messages."
},
{
"ref":"saev.data.DataLoader",
"url":16,
"doc":"High-throughput streaming loader that deterministically shuffles data from disk shards."
},
{
"ref":"saev.data.DataLoader.ExampleBatch",
"url":16,
"doc":"Individual example."
},
{
"ref":"saev.data.DataLoader.n_batches",
"url":16,
"doc":""
},
{
"ref":"saev.data.DataLoader.n_samples",
"url":16,
"doc":""
},
{
"ref":"saev.data.DataLoader.batch_size",
"url":16,
"doc":""
},
{
"ref":"saev.data.DataLoader.drop_last",
"url":16,
"doc":""
},
{
"ref":"saev.data.DataLoader.manager_pid",
"url":16,
"doc":""
},
{
"ref":"saev.data.DataLoader.shutdown",
"url":16,
"doc":"",
"func":1
},
{
"ref":"saev.data.Config",
"url":16,
"doc":"Configuration for loading shuffled activation data from disk."
},
{
"ref":"saev.data.Config.shard_root",
"url":16,
"doc":"Directory with .bin shards and a metadata.json file."
},
{
"ref":"saev.data.Config.patches",
"url":16,
"doc":"Which kinds of patches to use. 'cls' indicates just the [CLS] token (if any). 'image' indicates it will return image patches. 'all' returns all patches."
},
{
"ref":"saev.data.Config.layer",
"url":16,
"doc":"Which ViT layer(s) to read from disk.  -2 selects the second-to-last layer.  \"all\" enumerates every recorded layer."
},
{
"ref":"saev.data.Config.batch_size",
"url":16,
"doc":"Batch size."
},
{
"ref":"saev.data.Config.drop_last",
"url":16,
"doc":"Whether to drop the last batch if it's smaller than the others."
},
{
"ref":"saev.data.Config.scale_norm",
"url":16,
"doc":"Whether to scale norms to sqrt(D)."
},
{
"ref":"saev.data.Config.n_threads",
"url":16,
"doc":"Number of dataloading threads."
},
{
"ref":"saev.data.Config.buffer_size",
"url":16,
"doc":"Number of batches to queue in the shared-memory ring buffer. Higher values add latency but improve resilience to brief stalls."
},
{
"ref":"saev.data.Config.batch_timeout_s",
"url":16,
"doc":"How long to wait for at least one batch."
},
{
"ref":"saev.data.Config.seed",
"url":16,
"doc":"Random seed."
},
{
"ref":"saev.data.Config.debug",
"url":16,
"doc":"Whether the dataloader process should log debug messages."
},
{
"ref":"saev.data.Config.log_every_s",
"url":16,
"doc":"How frequently the dataloader process should log (debug) performance messages."
},
{
"ref":"saev.data.Metadata",
"url":16,
"doc":"Metadata(vit_family: Literal['clip', 'siglip', 'dinov2'], vit_ckpt: str, layers: tuple[int,  .], n_patches_per_img: int, cls_token: bool, d_vit: int, n_imgs: int, max_patches_per_shard: int, data: dict[str, object], dtype: Literal['float32'] = 'float32', protocol: Literal['1.0.0'] = '1.0.0')"
},
{
"ref":"saev.data.Metadata.vit_family",
"url":16,
"doc":""
},
{
"ref":"saev.data.Metadata.vit_ckpt",
"url":16,
"doc":""
},
{
"ref":"saev.data.Metadata.layers",
"url":16,
"doc":""
},
{
"ref":"saev.data.Metadata.n_patches_per_img",
"url":16,
"doc":""
},
{
"ref":"saev.data.Metadata.cls_token",
"url":16,
"doc":""
},
{
"ref":"saev.data.Metadata.d_vit",
"url":16,
"doc":""
},
{
"ref":"saev.data.Metadata.n_imgs",
"url":16,
"doc":""
},
{
"ref":"saev.data.Metadata.max_patches_per_shard",
"url":16,
"doc":""
},
{
"ref":"saev.data.Metadata.data",
"url":16,
"doc":""
},
{
"ref":"saev.data.Metadata.dtype",
"url":16,
"doc":""
},
{
"ref":"saev.data.Metadata.protocol",
"url":16,
"doc":""
},
{
"ref":"saev.data.Metadata.from_cfg",
"url":16,
"doc":"",
"func":1
},
{
"ref":"saev.data.Metadata.load",
"url":16,
"doc":"",
"func":1
},
{
"ref":"saev.data.Metadata.dump",
"url":16,
"doc":"",
"func":1
},
{
"ref":"saev.data.Metadata.hash",
"url":16,
"doc":""
},
{
"ref":"saev.data.Metadata.n_tokens_per_img",
"url":16,
"doc":""
},
{
"ref":"saev.data.Metadata.n_shards",
"url":16,
"doc":""
},
{
"ref":"saev.data.Metadata.n_imgs_per_shard",
"url":16,
"doc":"Calculate the number of images per shard based on the protocol. Returns: Number of images that fit in a shard."
},
{
"ref":"saev.data.Metadata.shard_shape",
"url":16,
"doc":""
},
{
"ref":"saev.data.config",
"url":17,
"doc":""
},
{
"ref":"saev.data.config.Activations",
"url":17,
"doc":"Configuration for loading activation data from disk."
},
{
"ref":"saev.data.config.Activations.shard_root",
"url":17,
"doc":"Directory with .bin shards and a metadata.json file."
},
{
"ref":"saev.data.config.Activations.patches",
"url":17,
"doc":"Which kinds of patches to use."
},
{
"ref":"saev.data.config.Activations.layer",
"url":17,
"doc":"Which ViT layer(s) to read from disk.  -2 selects the second-to-last layer.  \"all\" enumerates every recorded layer, and  \"meanpool\" averages activations across layers."
},
{
"ref":"saev.data.config.Activations.clamp",
"url":17,
"doc":"Maximum value for activations; activations will be clamped to within [-clamp, clamp] ."
},
{
"ref":"saev.data.config.Activations.n_random_samples",
"url":17,
"doc":"Number of random samples used to calculate approximate dataset means at startup."
},
{
"ref":"saev.data.config.Activations.scale_mean",
"url":17,
"doc":"Whether to subtract approximate dataset means from examples. If a string, manually load from the filepath."
},
{
"ref":"saev.data.config.Activations.scale_norm",
"url":17,
"doc":"Whether to scale average dataset norm to sqrt(d_vit). If a string, manually load from the filepath."
},
{
"ref":"saev.data.writers",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.Config",
"url":18,
"doc":"Configuration for calculating and saving ViT activations."
},
{
"ref":"saev.data.writers.Config.data",
"url":18,
"doc":"Which dataset to use."
},
{
"ref":"saev.data.writers.Config.vit_layers",
"url":18,
"doc":"Which layers to save. By default, the second-to-last layer."
},
{
"ref":"saev.data.writers.Config.dump_to",
"url":18,
"doc":"Where to write shards."
},
{
"ref":"saev.data.writers.Config.vit_family",
"url":18,
"doc":"Which model family."
},
{
"ref":"saev.data.writers.Config.vit_ckpt",
"url":18,
"doc":"Specific model checkpoint."
},
{
"ref":"saev.data.writers.Config.vit_batch_size",
"url":18,
"doc":"Batch size for ViT inference."
},
{
"ref":"saev.data.writers.Config.n_workers",
"url":18,
"doc":"Number of dataloader workers."
},
{
"ref":"saev.data.writers.Config.d_vit",
"url":18,
"doc":"Dimension of the ViT activations (depends on model)."
},
{
"ref":"saev.data.writers.Config.n_patches_per_img",
"url":18,
"doc":"Number of ViT patches per image (depends on model)."
},
{
"ref":"saev.data.writers.Config.cls_token",
"url":18,
"doc":"Whether the model has a [CLS] token."
},
{
"ref":"saev.data.writers.Config.max_patches_per_shard",
"url":18,
"doc":"Maximum number of activations per shard; 2.4M is approximately 10GB for 1024-dimensional 4-byte activations."
},
{
"ref":"saev.data.writers.Config.ssl",
"url":18,
"doc":"Whether to use SSL."
},
{
"ref":"saev.data.writers.Config.device",
"url":18,
"doc":"Which device to use."
},
{
"ref":"saev.data.writers.Config.n_hours",
"url":18,
"doc":"Slurm job length."
},
{
"ref":"saev.data.writers.Config.slurm_acct",
"url":18,
"doc":"Slurm account string."
},
{
"ref":"saev.data.writers.Config.slurm_partition",
"url":18,
"doc":"Slurm partition."
},
{
"ref":"saev.data.writers.Config.log_to",
"url":18,
"doc":"Where to log Slurm job stdout/stderr."
},
{
"ref":"saev.data.writers.RecordedVisionTransformer",
"url":18,
"doc":"Base class for all neural network modules. Your models should also subclass this class. Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes import torch.nn as nn import torch.nn.functional as F class Model(nn.Module): def __init__(self) -> None: super().__init__() self.conv1 = nn.Conv2d(1, 20, 5) self.conv2 = nn.Conv2d(20, 20, 5) def forward(self, x): x = F.relu(self.conv1(x return F.relu(self.conv2(x Submodules assigned in this way will be registered, and will have their parameters converted too when you call :meth: to , etc.  note As per the example above, an  __init__() call to the parent class must be made before assignment on the child. :ivar training: Boolean represents whether this module is in training or evaluation mode. :vartype training: bool Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.data.writers.RecordedVisionTransformer.hook",
"url":18,
"doc":"",
"func":1
},
{
"ref":"saev.data.writers.RecordedVisionTransformer.reset",
"url":18,
"doc":"",
"func":1
},
{
"ref":"saev.data.writers.RecordedVisionTransformer.activations",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.RecordedVisionTransformer.forward",
"url":18,
"doc":"Define the computation performed at every call. Should be overridden by all subclasses.  note Although the recipe for forward pass needs to be defined within this function, one should call the :class: Module instance afterwards instead of this since the former takes care of running the registered hooks while the latter silently ignores them.",
"func":1
},
{
"ref":"saev.data.writers.ShardWriter",
"url":18,
"doc":"ShardWriter is a stateful object that handles sharded activation writing to disk."
},
{
"ref":"saev.data.writers.ShardWriter.root",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.ShardWriter.shape",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.ShardWriter.shard",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.ShardWriter.acts_path",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.ShardWriter.acts",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.ShardWriter.filled",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.ShardWriter.flush",
"url":18,
"doc":"",
"func":1
},
{
"ref":"saev.data.writers.ShardWriter.next_shard",
"url":18,
"doc":"",
"func":1
},
{
"ref":"saev.data.writers.get_acts_dir",
"url":18,
"doc":"Return the activations directory based on the relevant values of a config. Also saves a metadata.json file to that directory for human reference. Args: cfg: Config for experiment. Returns: Directory to where activations should be dumped/loaded from.",
"func":1
},
{
"ref":"saev.data.writers.Metadata",
"url":18,
"doc":"Metadata(vit_family: Literal['clip', 'siglip', 'dinov2'], vit_ckpt: str, layers: tuple[int,  .], n_patches_per_img: int, cls_token: bool, d_vit: int, n_imgs: int, max_patches_per_shard: int, data: dict[str, object], dtype: Literal['float32'] = 'float32', protocol: Literal['1.0.0'] = '1.0.0')"
},
{
"ref":"saev.data.writers.Metadata.vit_family",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.Metadata.vit_ckpt",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.Metadata.layers",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.Metadata.n_patches_per_img",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.Metadata.cls_token",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.Metadata.d_vit",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.Metadata.n_imgs",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.Metadata.max_patches_per_shard",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.Metadata.data",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.Metadata.dtype",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.Metadata.protocol",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.Metadata.from_cfg",
"url":18,
"doc":"",
"func":1
},
{
"ref":"saev.data.writers.Metadata.load",
"url":18,
"doc":"",
"func":1
},
{
"ref":"saev.data.writers.Metadata.dump",
"url":18,
"doc":"",
"func":1
},
{
"ref":"saev.data.writers.Metadata.hash",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.Metadata.n_tokens_per_img",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.Metadata.n_shards",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.Metadata.n_imgs_per_shard",
"url":18,
"doc":"Calculate the number of images per shard based on the protocol. Returns: Number of images that fit in a shard."
},
{
"ref":"saev.data.writers.Metadata.shard_shape",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.Shard",
"url":18,
"doc":"A single shard entry in shards.json, recording the filename and number of images."
},
{
"ref":"saev.data.writers.Shard.name",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.Shard.n_imgs",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.ShardInfo",
"url":18,
"doc":"A read-only container for shard metadata as recorded in shards.json."
},
{
"ref":"saev.data.writers.ShardInfo.shards",
"url":18,
"doc":""
},
{
"ref":"saev.data.writers.ShardInfo.load",
"url":18,
"doc":"",
"func":1
},
{
"ref":"saev.data.writers.ShardInfo.dump",
"url":18,
"doc":"",
"func":1
},
{
"ref":"saev.data.writers.ShardInfo.append",
"url":18,
"doc":"",
"func":1
},
{
"ref":"saev.data.writers.get_dataloader",
"url":18,
"doc":"Get a dataloader for a default map-style dataset. Args: cfg: Config. img_transform: Image transform to be applied to each image. Returns: A PyTorch Dataloader that yields dictionaries with  'image' keys containing image batches,  'index' keys containing original dataset indices and  'label' keys containing label batches.",
"func":1
},
{
"ref":"saev.data.writers.worker_fn",
"url":18,
"doc":"Args: cfg: Config for activations.",
"func":1
},
{
"ref":"saev.data.writers.IndexLookup",
"url":18,
"doc":"Index  shard helper.  map() \u2013 turn a global dataset index into precise physical offsets.  length() \u2013 dataset size for a particular (patches, layer) view. Parameters      metadata : Metadata Pre-computed dataset statistics (images, patches, layers, shard size). patches: 'cls' | 'image' | 'all' layer: int | 'all'"
},
{
"ref":"saev.data.writers.IndexLookup.map_global",
"url":18,
"doc":"Return    - ( shard_i, index in shard (img_i_in_shard, layer_i, token_i) )",
"func":1
},
{
"ref":"saev.data.writers.IndexLookup.map_img",
"url":18,
"doc":"Return    - (shard_i, img_i_in_shard)",
"func":1
},
{
"ref":"saev.data.writers.IndexLookup.length",
"url":18,
"doc":"",
"func":1
},
{
"ref":"saev.data.models",
"url":19,
"doc":""
},
{
"ref":"saev.data.models.DinoV2",
"url":19,
"doc":"Base class for all neural network modules. Your models should also subclass this class. Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes import torch.nn as nn import torch.nn.functional as F class Model(nn.Module): def __init__(self) -> None: super().__init__() self.conv1 = nn.Conv2d(1, 20, 5) self.conv2 = nn.Conv2d(20, 20, 5) def forward(self, x): x = F.relu(self.conv1(x return F.relu(self.conv2(x Submodules assigned in this way will be registered, and will have their parameters converted too when you call :meth: to , etc.  note As per the example above, an  __init__() call to the parent class must be made before assignment on the child. :ivar training: Boolean represents whether this module is in training or evaluation mode. :vartype training: bool Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.data.models.DinoV2.get_residuals",
"url":19,
"doc":"",
"func":1
},
{
"ref":"saev.data.models.DinoV2.get_patches",
"url":19,
"doc":"",
"func":1
},
{
"ref":"saev.data.models.DinoV2.forward",
"url":19,
"doc":"Define the computation performed at every call. Should be overridden by all subclasses.  note Although the recipe for forward pass needs to be defined within this function, one should call the :class: Module instance afterwards instead of this since the former takes care of running the registered hooks while the latter silently ignores them.",
"func":1
},
{
"ref":"saev.data.models.Clip",
"url":19,
"doc":"Base class for all neural network modules. Your models should also subclass this class. Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes import torch.nn as nn import torch.nn.functional as F class Model(nn.Module): def __init__(self) -> None: super().__init__() self.conv1 = nn.Conv2d(1, 20, 5) self.conv2 = nn.Conv2d(20, 20, 5) def forward(self, x): x = F.relu(self.conv1(x return F.relu(self.conv2(x Submodules assigned in this way will be registered, and will have their parameters converted too when you call :meth: to , etc.  note As per the example above, an  __init__() call to the parent class must be made before assignment on the child. :ivar training: Boolean represents whether this module is in training or evaluation mode. :vartype training: bool Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.data.models.Clip.get_residuals",
"url":19,
"doc":"",
"func":1
},
{
"ref":"saev.data.models.Clip.get_patches",
"url":19,
"doc":"",
"func":1
},
{
"ref":"saev.data.models.Clip.forward",
"url":19,
"doc":"Define the computation performed at every call. Should be overridden by all subclasses.  note Although the recipe for forward pass needs to be defined within this function, one should call the :class: Module instance afterwards instead of this since the former takes care of running the registered hooks while the latter silently ignores them.",
"func":1
},
{
"ref":"saev.data.models.Siglip",
"url":19,
"doc":"Base class for all neural network modules. Your models should also subclass this class. Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes import torch.nn as nn import torch.nn.functional as F class Model(nn.Module): def __init__(self) -> None: super().__init__() self.conv1 = nn.Conv2d(1, 20, 5) self.conv2 = nn.Conv2d(20, 20, 5) def forward(self, x): x = F.relu(self.conv1(x return F.relu(self.conv2(x Submodules assigned in this way will be registered, and will have their parameters converted too when you call :meth: to , etc.  note As per the example above, an  __init__() call to the parent class must be made before assignment on the child. :ivar training: Boolean represents whether this module is in training or evaluation mode. :vartype training: bool Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.data.models.Siglip.get_residuals",
"url":19,
"doc":"",
"func":1
},
{
"ref":"saev.data.models.Siglip.get_patches",
"url":19,
"doc":"",
"func":1
},
{
"ref":"saev.data.models.Siglip.forward",
"url":19,
"doc":"Define the computation performed at every call. Should be overridden by all subclasses.  note Although the recipe for forward pass needs to be defined within this function, one should call the :class: Module instance afterwards instead of this since the former takes care of running the registered hooks while the latter silently ignores them.",
"func":1
},
{
"ref":"saev.data.models.make_vit",
"url":19,
"doc":"",
"func":1
},
{
"ref":"saev.data.models.make_img_transform",
"url":19,
"doc":"",
"func":1
},
{
"ref":"saev.data.images",
"url":20,
"doc":""
},
{
"ref":"saev.data.images.Imagenet",
"url":20,
"doc":"Configuration for HuggingFace Imagenet."
},
{
"ref":"saev.data.images.Imagenet.name",
"url":20,
"doc":"Dataset name on HuggingFace. Don't need to change this "
},
{
"ref":"saev.data.images.Imagenet.split",
"url":20,
"doc":"Dataset split. For the default ImageNet-1K dataset, can either be 'train', 'validation' or 'test'."
},
{
"ref":"saev.data.images.Imagenet.n_imgs",
"url":20,
"doc":"Number of images in the dataset. Calculated on the fly, but is non-trivial to calculate because it requires loading the dataset. If you need to reference this number very often, cache it in a local variable."
},
{
"ref":"saev.data.images.ImageFolder",
"url":20,
"doc":"Configuration for a generic image folder dataset."
},
{
"ref":"saev.data.images.ImageFolder.root",
"url":20,
"doc":"Where the class folders with images are stored."
},
{
"ref":"saev.data.images.ImageFolder.n_imgs",
"url":20,
"doc":"Number of images in the dataset. Calculated on the fly, but is non-trivial to calculate because it requires walking the directory structure. If you need to reference this number very often, cache it in a local variable."
},
{
"ref":"saev.data.images.Ade20k",
"url":20,
"doc":""
},
{
"ref":"saev.data.images.Ade20k.root",
"url":20,
"doc":"Where the class folders with images are stored."
},
{
"ref":"saev.data.images.Ade20k.split",
"url":20,
"doc":"Data split."
},
{
"ref":"saev.data.images.Ade20k.n_imgs",
"url":20,
"doc":""
},
{
"ref":"saev.data.images.Fake",
"url":20,
"doc":"Fake(n_imgs: int = 10)"
},
{
"ref":"saev.data.images.Fake.n_imgs",
"url":20,
"doc":""
},
{
"ref":"saev.data.images.setup",
"url":20,
"doc":"Run dataset-specific setup. These setup functions can assume they are the only job running, but they should be idempotent; they should be safe (and ideally cheap) to run multiple times in a row.",
"func":1
},
{
"ref":"saev.data.images.setup_imagenet",
"url":20,
"doc":"",
"func":1
},
{
"ref":"saev.data.images.setup_imagefolder",
"url":20,
"doc":"",
"func":1
},
{
"ref":"saev.data.images.setup_ade20k",
"url":20,
"doc":"",
"func":1
},
{
"ref":"saev.data.images.get_dataset",
"url":20,
"doc":"Gets the dataset for the current experiment; delegates construction to dataset-specific functions. Args: cfg: Experiment config. img_transform: Image transform to be applied to each image. Returns: A dataset that has dictionaries with  'image' ,  'index' ,  'target' , and  'label' keys containing examples.",
"func":1
},
{
"ref":"saev.data.images.ImagenetDataset",
"url":20,
"doc":"An abstract class representing a :class: Dataset . All datasets that represent a map from keys to data samples should subclass it. All subclasses should overwrite :meth: __getitem__ , supporting fetching a data sample for a given key. Subclasses could also optionally overwrite :meth: __len__ , which is expected to return the size of the dataset by many :class: ~torch.utils.data.Sampler implementations and the default options of :class: ~torch.utils.data.DataLoader . Subclasses could also optionally implement :meth: __getitems__ , for speedup batched samples loading. This method accepts list of indices of samples of batch and returns list of samples.  note :class: ~torch.utils.data.DataLoader by default constructs an index sampler that yields integral indices. To make it work with a map-style dataset with non-integral indices/keys, a custom sampler must be provided."
},
{
"ref":"saev.data.images.ImageFolderDataset",
"url":20,
"doc":"A generic data loader where the images are arranged in this way by default:  root/dog/xxx.png root/dog/xxy.png root/dog/[ .]/xxz.png root/cat/123.png root/cat/nsdf3.png root/cat/[ .]/asd932_.png This class inherits from :class: ~torchvision.datasets.DatasetFolder so the same methods can be overridden to customize the dataset. Args: root (str or  pathlib.Path ): Root directory path. transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g,  transforms.RandomCrop target_transform (callable, optional): A function/transform that takes in the target and transforms it. loader (callable, optional): A function to load an image given its path. is_valid_file (callable, optional): A function that takes path of an Image file and check if the file is a valid file (used to check of corrupt files) allow_empty(bool, optional): If True, empty folders are considered to be valid classes. An error is raised on empty folders if False (default). Attributes: classes (list): List of the class names sorted alphabetically. class_to_idx (dict): Dict with items (class_name, class_index). imgs (list): List of (image path, class_index) tuples"
},
{
"ref":"saev.data.images.Ade20kDataset",
"url":20,
"doc":"An abstract class representing a :class: Dataset . All datasets that represent a map from keys to data samples should subclass it. All subclasses should overwrite :meth: __getitem__ , supporting fetching a data sample for a given key. Subclasses could also optionally overwrite :meth: __len__ , which is expected to return the size of the dataset by many :class: ~torch.utils.data.Sampler implementations and the default options of :class: ~torch.utils.data.DataLoader . Subclasses could also optionally implement :meth: __getitems__ , for speedup batched samples loading. This method accepts list of indices of samples of batch and returns list of samples.  note :class: ~torch.utils.data.DataLoader by default constructs an index sampler that yields integral indices. To make it work with a map-style dataset with non-integral indices/keys, a custom sampler must be provided."
},
{
"ref":"saev.data.images.Ade20kDataset.samples",
"url":20,
"doc":""
},
{
"ref":"saev.data.images.Ade20kDataset.Sample",
"url":20,
"doc":""
},
{
"ref":"saev.data.images.FakeDataset",
"url":20,
"doc":"An abstract class representing a :class: Dataset . All datasets that represent a map from keys to data samples should subclass it. All subclasses should overwrite :meth: __getitem__ , supporting fetching a data sample for a given key. Subclasses could also optionally overwrite :meth: __len__ , which is expected to return the size of the dataset by many :class: ~torch.utils.data.Sampler implementations and the default options of :class: ~torch.utils.data.DataLoader . Subclasses could also optionally implement :meth: __getitems__ , for speedup batched samples loading. This method accepts list of indices of samples of batch and returns list of samples.  note :class: ~torch.utils.data.DataLoader by default constructs an index sampler that yields integral indices. To make it work with a map-style dataset with non-integral indices/keys, a custom sampler must be provided."
},
{
"ref":"saev.data.buffers",
"url":21,
"doc":""
},
{
"ref":"saev.data.buffers.RingBuffer",
"url":21,
"doc":"Fixed-capacity, multiple-producer / multiple-consumer queue backed by a shared-memory tensor. Parameters      slots : int capacity in number of items (tensor rows) shape : tuple[int] shape of one item, e.g. (batch, dim) dtype : torch.dtype tensor dtype put(tensor) : blocks if full get() -> tensor : blocks if empty qsize() -> int advisory size (approximate) close() frees shared storage (call in the main process)"
},
{
"ref":"saev.data.buffers.RingBuffer.put",
"url":21,
"doc":"Copy  tensor into the next free slot; blocks if the queue is full.",
"func":1
},
{
"ref":"saev.data.buffers.RingBuffer.get",
"url":21,
"doc":"Return a view of the next item; blocks if the queue is empty.",
"func":1
},
{
"ref":"saev.data.buffers.RingBuffer.qsize",
"url":21,
"doc":"Approximate number of filled slots (race-safe enough for tests).",
"func":1
},
{
"ref":"saev.data.buffers.RingBuffer.fill",
"url":21,
"doc":"Approximate proportion of filled slots (race-safe enough for tests).",
"func":1
},
{
"ref":"saev.data.buffers.RingBuffer.close",
"url":21,
"doc":"Release the shared-memory backing store (call once in the parent).",
"func":1
},
{
"ref":"saev.data.buffers.ReservoirBuffer",
"url":21,
"doc":"Pool of (tensor, meta) pairs. Multiple producers call put(batch_x, batch_meta). Multiple consumers call get(batch_size) -> (x, meta). Random order, each sample delivered once, blocking semantics."
},
{
"ref":"saev.data.buffers.ReservoirBuffer.put",
"url":21,
"doc":"",
"func":1
},
{
"ref":"saev.data.buffers.ReservoirBuffer.get",
"url":21,
"doc":"",
"func":1
},
{
"ref":"saev.data.buffers.ReservoirBuffer.close",
"url":21,
"doc":"Release the shared-memory backing store (call once in the parent).",
"func":1
},
{
"ref":"saev.data.buffers.ReservoirBuffer.qsize",
"url":21,
"doc":"Approximate number of filled slots (race-safe enough for tests).",
"func":1
},
{
"ref":"saev.data.buffers.ReservoirBuffer.fill",
"url":21,
"doc":"Approximate proportion of filled slots (race-safe enough for tests).",
"func":1
},
{
"ref":"saev.data.indexed",
"url":22,
"doc":""
},
{
"ref":"saev.data.indexed.Config",
"url":22,
"doc":"Configuration for loading indexed activation data from disk."
},
{
"ref":"saev.data.indexed.Config.shard_root",
"url":22,
"doc":"Directory with .bin shards and a metadata.json file."
},
{
"ref":"saev.data.indexed.Config.patches",
"url":22,
"doc":"Which kinds of patches to use. 'cls' indicates just the [CLS] token (if any). 'image' indicates it will return image patches. 'all' returns all patches."
},
{
"ref":"saev.data.indexed.Config.layer",
"url":22,
"doc":"Which ViT layer(s) to read from disk.  -2 selects the second-to-last layer.  \"all\" enumerates every recorded layer."
},
{
"ref":"saev.data.indexed.Config.seed",
"url":22,
"doc":"Random seed."
},
{
"ref":"saev.data.indexed.Config.debug",
"url":22,
"doc":"Whether the dataloader process should log debug messages."
},
{
"ref":"saev.data.indexed.Dataset",
"url":22,
"doc":"Dataset of activations from disk."
},
{
"ref":"saev.data.indexed.Dataset.cfg",
"url":22,
"doc":"Configuration; set via CLI args."
},
{
"ref":"saev.data.indexed.Dataset.metadata",
"url":22,
"doc":"Activations metadata; automatically loaded from disk."
},
{
"ref":"saev.data.indexed.Dataset.layer_index",
"url":22,
"doc":"Layer index into the shards if we are choosing a specific layer."
},
{
"ref":"saev.data.indexed.Dataset.Example",
"url":22,
"doc":"Individual example."
},
{
"ref":"saev.data.indexed.Dataset.transform",
"url":22,
"doc":"",
"func":1
},
{
"ref":"saev.data.indexed.Dataset.d_vit",
"url":22,
"doc":"Dimension of the underlying vision transformer's embedding space."
},
{
"ref":"saev.data.indexed.Dataset.get_img_patches",
"url":22,
"doc":"",
"func":1
},
{
"ref":"saev.data.ordered",
"url":23,
"doc":"Ordered (sequential) dataloader for activation data. This module provides a high-throughput dataloader that reads activation data from disk shards in sequential order, without shuffling. The implementation uses a single-threaded manager process to ensure data is delivered in the exact order it appears on disk. See the design decisions in src/saev/data/performance.md. Usage: >>> cfg = Config(shard_root=\"./shards\", layer=13, batch_size=4096) >>> dataloader = DataLoader(cfg) >>> for batch in dataloader:  . activations = batch[\"act\"]  [batch_size, d_vit]  . image_indices = batch[\"image_i\"]  [batch_size]  . patch_indices = batch[\"patch_i\"]  [batch_size]"
},
{
"ref":"saev.data.ordered.Config",
"url":23,
"doc":"Configuration for loading ordered (non-shuffled) activation data from disk."
},
{
"ref":"saev.data.ordered.Config.shard_root",
"url":23,
"doc":"Directory with .bin shards and a metadata.json file."
},
{
"ref":"saev.data.ordered.Config.patches",
"url":23,
"doc":"Which kinds of patches to use. 'cls' indicates just the [CLS] token (if any). 'image' indicates it will return image patches. 'all' returns all patches."
},
{
"ref":"saev.data.ordered.Config.layer",
"url":23,
"doc":"Which ViT layer(s) to read from disk.  -2 selects the second-to-last layer.  \"all\" enumerates every recorded layer."
},
{
"ref":"saev.data.ordered.Config.batch_size",
"url":23,
"doc":"Batch size."
},
{
"ref":"saev.data.ordered.Config.batch_timeout_s",
"url":23,
"doc":"How long to wait for at least one batch."
},
{
"ref":"saev.data.ordered.Config.drop_last",
"url":23,
"doc":"Whether to drop the last batch if it's smaller than the others."
},
{
"ref":"saev.data.ordered.Config.buffer_size",
"url":23,
"doc":"Number of batches to queue in the shared-memory ring buffer. Higher values add latency but improve resilience to brief stalls."
},
{
"ref":"saev.data.ordered.Config.debug",
"url":23,
"doc":"Whether the dataloader process should log debug messages."
},
{
"ref":"saev.data.ordered.ImageOutOfBoundsError",
"url":23,
"doc":"Common base class for all non-exit exceptions."
},
{
"ref":"saev.data.ordered.ImageOutOfBoundsError.message",
"url":23,
"doc":""
},
{
"ref":"saev.data.ordered.DataLoader",
"url":23,
"doc":"High-throughput streaming loader that reads data from disk shards in order (no shuffling)."
},
{
"ref":"saev.data.ordered.DataLoader.ExampleBatch",
"url":23,
"doc":"Individual example."
},
{
"ref":"saev.data.ordered.DataLoader.n_batches",
"url":23,
"doc":""
},
{
"ref":"saev.data.ordered.DataLoader.n_samples",
"url":23,
"doc":""
},
{
"ref":"saev.data.ordered.DataLoader.batch_size",
"url":23,
"doc":""
},
{
"ref":"saev.data.ordered.DataLoader.drop_last",
"url":23,
"doc":""
},
{
"ref":"saev.data.ordered.DataLoader.shutdown",
"url":23,
"doc":"",
"func":1
},
{
"ref":"saev.data.shuffled",
"url":24,
"doc":""
},
{
"ref":"saev.data.shuffled.Config",
"url":24,
"doc":"Configuration for loading shuffled activation data from disk."
},
{
"ref":"saev.data.shuffled.Config.shard_root",
"url":24,
"doc":"Directory with .bin shards and a metadata.json file."
},
{
"ref":"saev.data.shuffled.Config.patches",
"url":24,
"doc":"Which kinds of patches to use. 'cls' indicates just the [CLS] token (if any). 'image' indicates it will return image patches. 'all' returns all patches."
},
{
"ref":"saev.data.shuffled.Config.layer",
"url":24,
"doc":"Which ViT layer(s) to read from disk.  -2 selects the second-to-last layer.  \"all\" enumerates every recorded layer."
},
{
"ref":"saev.data.shuffled.Config.batch_size",
"url":24,
"doc":"Batch size."
},
{
"ref":"saev.data.shuffled.Config.drop_last",
"url":24,
"doc":"Whether to drop the last batch if it's smaller than the others."
},
{
"ref":"saev.data.shuffled.Config.scale_norm",
"url":24,
"doc":"Whether to scale norms to sqrt(D)."
},
{
"ref":"saev.data.shuffled.Config.n_threads",
"url":24,
"doc":"Number of dataloading threads."
},
{
"ref":"saev.data.shuffled.Config.buffer_size",
"url":24,
"doc":"Number of batches to queue in the shared-memory ring buffer. Higher values add latency but improve resilience to brief stalls."
},
{
"ref":"saev.data.shuffled.Config.batch_timeout_s",
"url":24,
"doc":"How long to wait for at least one batch."
},
{
"ref":"saev.data.shuffled.Config.seed",
"url":24,
"doc":"Random seed."
},
{
"ref":"saev.data.shuffled.Config.debug",
"url":24,
"doc":"Whether the dataloader process should log debug messages."
},
{
"ref":"saev.data.shuffled.Config.log_every_s",
"url":24,
"doc":"How frequently the dataloader process should log (debug) performance messages."
},
{
"ref":"saev.data.shuffled.ImageOutOfBoundsError",
"url":24,
"doc":"Common base class for all non-exit exceptions."
},
{
"ref":"saev.data.shuffled.ImageOutOfBoundsError.message",
"url":24,
"doc":""
},
{
"ref":"saev.data.shuffled.DataLoader",
"url":24,
"doc":"High-throughput streaming loader that deterministically shuffles data from disk shards."
},
{
"ref":"saev.data.shuffled.DataLoader.ExampleBatch",
"url":24,
"doc":"Individual example."
},
{
"ref":"saev.data.shuffled.DataLoader.n_batches",
"url":24,
"doc":""
},
{
"ref":"saev.data.shuffled.DataLoader.n_samples",
"url":24,
"doc":""
},
{
"ref":"saev.data.shuffled.DataLoader.batch_size",
"url":24,
"doc":""
},
{
"ref":"saev.data.shuffled.DataLoader.drop_last",
"url":24,
"doc":""
},
{
"ref":"saev.data.shuffled.DataLoader.manager_pid",
"url":24,
"doc":""
},
{
"ref":"saev.data.shuffled.DataLoader.shutdown",
"url":24,
"doc":"",
"func":1
}
]