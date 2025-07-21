# Logbook

This provides a set of notes, step-by-step, of my process developing this library.
In combination with code comments and git blame, it's probably the best way to understand *why* a decision was made, rather than *what* decision was made.

*Open science babbyyy!*

# 10/11/2024

Something in the post-training is not working.
I should run the post-training steps (analysis.py and generate_app_data.py) once for a known good checkpoint and once for the new checkpoint that used the dataloader and see if it's an issue with the post-training code or the checkpoint.
Honestly, I'm not sure which one I want to be broken.
Probbaly the post-training code?

# 10/17/2024

I trained a couple SAEs using both my updated codebase and the original codebase.
Like, the original codebase, without even ruff/isort added.
I did this because I figured there were four possible problems:

1. The updated training code is wrong.
2. The updated analysis.py file is wrong.
3. The updated get_app_data.py file is wrong.
4. The random seed was causing problems.

The key is that I now have several checkpoints, trained with the original codebase or with the udpated codebase.
I also have updated and original versions of the other scripts.
So I can take each piece through step by step and see where the mistake is.

| Checkpoint | Wandb | Training Code |
|---|---|---|
| 2dlebd60 | [samuelstevens/saev/wawwh1rj](https://wandb.ai/samuelstevens/saev/runs/wawwh1rj) | original |
| dd991rt3 | [samuelstevens/saev/g0sqbjux](https://wandb.ai/samuelstevens/saev/runs/g0sqbjux) | updated |


| Checkpoint | Updated Training? | Updated Analysis? | Updated App Data? | Worked? |
|---|---|---|--|---|
| 2dlebd60 | No | No | No | Yes |
| 2dlebd60 | No | No | Yes | No[^dataset-shuffle] |

[^dataset-shuffle]: This worked after I added the `.shuffle(seed=1)` line (commit [6319148](https://github.com/samuelstevens/saev/commit/6319148f269013721baf9da15a08e5b1ca0b6e32)) to the updated `generate_app_data.py` script. I think this is because in the original `vit_sae_analysis/dashboard_fns.py` there is a `shuffle(seed=seed)` and the default seed is 1.

So per this footnote[^dataset-shuffle] I think the `dataset.shuffle()` call in both `analysis.py` and `generate_app_data.py` needs to be the same.
I am going to update both the original code and the updated code to use `cfg.seed`.

# 10/19/2024

Same trial as before, but everywhere that we used `seed=1` is now `seed=cfg.seed`:

| Checkpoint | Updated Training? | Updated Analysis? | Updated App Data? | Worked? |
|---|---|---|--|---|
| 2dlebd60 | No | No | No | Yes |
| 2dlebd60 | No | No | Yes | Yes |

**Good!**
Now we can use the original analysis with the updated `generate_app_data.py` and it will work just fine.
So we can at least assume that `generate_app_data.py` is correct.
Now, let's figure out if there's something wrong with the `analysis.py` file.

| Checkpoint | Updated Training? | Updated Analysis? | Updated App Data? | Worked? |
|---|---|---|--|---|
| 2dlebd60 | No | Yes | No | No[^generate-bug] |
| 2dlebd60 | No | Yes | Yes | No |

[^generate-bug]: It threw an exception about some data not being available.

So here's the thing.
It's clearly my analysis that is wrong.
Even after removing the `dataset.shuffle` call in the updated `generate_app_data.py` file, the grids have some weird images.
Like it will be a mix of dogs and spiders in a single grid.
Why does that happen?
So I am going to do my best to eliminate as many differences as possible between the two scripts.

# 10/21/2024

I changed the activation store to behave *exactly* as the original code (shuffle before converting to an iterable dataset, commit [51f7947](https://github.com/samuelstevens/saev/commit/51f7947edcb19a7ca59dd3bd45d9869d090a91bf)).
Then the updated analysis and app data scripts work!
Now that I trust the updated analysis code, I can evaluate whether *my* pre-trained models are as good as the others.

| Checkpoint | Updated Training? | Updated Analysis? | Updated App Data? | Worked? |
|---|---|---|--|---|
| 2dlebd60 | No | Yes | Yes | **Yes** |
| dd991rt3 | Yes | Yes | Yes | **Yes** |

...and now we see that my updated training code works as well!
What a relief!

With this in mind, there are several minor changes I want to make before I do some BioCLIP and TreeOfLife runs:

1. Removing `transformer-lens` [done, commit [18612b7](https://github.com/samuelstevens/saev/commit/18612b75988c32ae8ab3db6656b44a442f3f7641)]
2. Removing HookedVisionTransformer [done, commit [c7ba7c7](https://github.com/samuelstevens/saev/commit/c7ba7c72c76472fd8cf2e7b2dc668d03a15b803d)]
3. OpenCLIP instead of huggingface `transformers` [done, commit [d362f64](https://github.com/samuelstevens/saev/commit/d362f64437b3599f56bb698136712d7590ee897b)]
4. Pre-computing ViT activations [done, commit [ee79f5b](https://github.com/samuelstevens/saev/commit/ee79f5b84186e655b2e5d485e972fe69bb73dd65)]

I'm going to do each of these independently using a set of runs as references.

# 10/22/2024

Removed HookedVisionTransformer (see above)
Checkpoint [v6jto37s](https://wandb.ai/samuelstevens/saev/runs/wwb20pa0) worked for training, analysis, and app data.

Testing an implementation using OpenCLIP instead of `transformers`.
Assuming it works (which seems likely given that the loss curve is identical), then I will pre-compute the activations, save them as a numpy array to disk, and memmap them during training rather than computing them.
I expect this to take a little bit because I had issues with shuffling and such in the analysis step earlier.
I think the best strategy is to work backwards.
The `generate_app_data.py` script doesn't need an activation store at all.
So I will start with the `analysis.py` script and add a new activations store class that meets the same interface as the original (maybe not for the constructor).
Then I will verify that the analysis script works correctly.

Only after that will I use the new class in training.
Working with the analysis script is a shorter feedback loop.

# 10/23/2024

OpenCLIP instead of transformers works (training, analysis, generate).
So now I am pre-computing activations.
I'm waiting on the activations to be saved (~3 hours).

CachedActivationsStore produced some duplicates in the analysis step.
Why is that?

For example, neuron 78 has the same image for image 6 and 7 (1-indexed, images 5 and 6 if zero-indexed).

Fixed it.
We no longer randomly sample batches; instead, we use a dataloader and `__getitem__`.

With training, however, the metrics no longer match the reference metrics.
Why is that?
We can find out by comparing to the original activations store.
Likely, we will need to build a custom data order using `np.random.default_rng(seed=cfg.seed)`.


# 10/24/2024

My strategy for calculating the mean activations only used 15 examples instead of 15 x 1024.
With 15 x 1024 examples, the b_dec is better initialized and it works exactly like before.

Now I have a complete training strategy.
The goal now is to train a SAE on BioCLIP's patch-level activations from TreeOfLife-10M.
How do we do this?

1. Train an SAE on ViT-L-14/openai image-level activations on *TreeOfLife-10M*.
2. Train an SAE on *BioCLIP* image-level activations on TreeOfLife-10M.
3. Train an SAE on BioCLIP *patch*-level activations on TreeOfLife-10M.

# 10/29/2024

I want to write a paper that does a bunch of introductory experiments for SAEs applied to vision.
Such a paper would be a good reference for new researchers hoping to train SAEs on vision models.
So I can make a set of experiments and slowly knock them off.
But I should think clearly about what specific graphs/figures I would want to make.
Then I can rank them in terms of difficulty, compute and reward (novelty, bits of information gained, etc) and work on the most important experiments first.

How do we know if an SAE is good?
This will enable us to say whether a recipe is good or bad.
So what does prior work do on automatically evaluating SAE quality?

# 10/30/2024

Here's a rough set of contributions + experiments:

We make the first comprehensive effort to train SAEs on vision models.

Contributions:

* Experimental findings.
* Code package to reproduce and extend.
* Trained SAE checkpoints + interactive explorer (web tool).

Experiments:

1. Activations: Compare [CLS] token activations vs patch-level activations.
2. Pretraining Task: Compare effects of supervised pre-training (classification), vision-only self-supervised (DINOv2, MAE) and vision-language (CLIP, SigLIP).
3. Pretraining Corpus: Compare various CLIP models (LAION, BioCLIP, PubmedCLIP, etc.)
4. Vision Architecture: Compare vision model architectures (ResNet, ViTs, ConvNeXt, etc.)
5. SAE Architecture: Compare SAE architectures and objectives (L1, TopK, Gated, JumpReLU, etc.)


The OpenAI paper (Scaling and evaluating sparse autoencoders) proposes four evaluation metrics:

1. Downstream loss: replace the original model representations with SAE reconstructions and calculate loss.
2. Probe loss: They train a 1D binary probe for specific, curated tasks. They pick features from SAEs that minimize loss on the task, and a low loss indicates that the SAE has learned the true feature used for the binary classification task.
3. Explanability:
4. Ablation sparsity:

Overall, I felt that these evaluation metrics are still not very good.
I think that we can use delta imagenet-1k linear probing accuracy as a measure of an SAE's performance, along with another dataset (iNat21? ToL-1M?) to verify that this approach is generalizable.
We also should include some qualitative study of the features to ensure that they are meaningful.
What will that qualitative study be?
Maybe a decision tree trained on SAE features to predict imagenet-1k classes?
We could use heuristics to pick out non-trivial SAE features; that is, we don't want features that are perfectly correlated with a class label.
Or we could use LAION as an example dataset.
ToL-10M was 37GB for one token.
LAION-400M would be 40x larger, so 1.4TB.
If we also stored patch-level activations, that would be 257x larger, or 360TB.
This is unfeasible.
But if we train on 1B tokens per autoencoder, that's 1B x 1024 floats x 4 bytes/float = ~4.1TB.
Which I think is not unreasonable.

In fact, it is very reasonable given our limits (100TB/user) on OSC.
Recall, however, that it's 4.1TB per vision backbone that I want to analyze.
So I can analyze about 23 backbones (realistically 20, given that we don't want to bump up against limits).

So what would I do?

1. Record 1B activations for DINOv2, OpenCLIP, and a pretrained ViT-L/14 that's hopefully trained on classification from reLAION datasets. This will let us first compare the effect of pretraining objective on SAE reconstruction, which I think is the most interesting.
2. Train a bunch of SAEs on these models. Explore different activations, L1 coefficients, etc.
3. Measure ImageNet performance using linear classification on reconstructed ImageNet activations.

What graphs and figures would I put in this paper?

* I want qualitative comparisons of features discovered across different objectives. One specific question is: do vision-only pretrained model learn abstract concepts like "love" or "taking care of their young"? I can also try to answer this question with specific, hand-crafted experiments.
* I want to see if particular vision architectures have significantly different MSE-L0 tradeoffs.
* Do different pre-training corpuses lead to meaningfully different features? Probably?

In some regard, I like the idea of a hand-crafted task with a linear probe to measure whether an SAE is learning a particular feature that can accurately predict a task.
Is there a neuron that reliably distinguishes between pictures of love and other pictures?
Such tasks would be naturally biased and arbitrary, but I think we can try to construct evals that lead to meaningful differences between models and their SAEs.
What are some examples of such tasks?

1. Love (abstract concept in ImageNet)
2. Groups of animals (abstract concept in ToL-10M)
3. Geographic areas/cultures - We know CLIP can do this already, but can DINOv2?
4. Taxonomic trees: is this a mammal, is this a bird, etc. Probably to a finer degree (is this a feline, is this a hawk, etc).

But again, if you can reliably make linear ImageNet performance go down using the SAE reconstructions, *probably* the SAE is of higher quality.
I would be very surprised if this is not the case.

This opens up questions like:

* At what layer do you get best ImageNet performance?
* Is this the same layer when using the model activations directly?
* Do you do better when training an SAE on all patches and all layers, or better to train patch-specific, layer-specific SAEs (similar to GemmaScope).

I forgot that I probably want to record activations for every layer and every patch.
For ViT-L-14 that's 24 layers, which means one ViT will fill up my storage completely.

# 10/31/2024 (Happy Halloween)

For ImageNet-1K, with 1.2M images, how big is a set of ViT-L/14 activations?
1.2M images x 24 layers x 257 patches x 1024 floats x 4 bytes/float = 30TB.
Still too big to put on my lab servers.
But with only 4 layers?
Then it's 5TB.
Still too big.
But I can debug these processes on the lab servers.

With a ViT-B/32, saving the last 3 layers, ImagetNet-1K is

1.2M x 3 x 50 x 768 x 4 bytes/float = 553GB

It seems that training is working well.
I can train on 100M patches in about 40m on an A6000, which is good because it's 10x more tokens than 10M and about 10x slower (40m vs 4m).

Now I need to make sure training still works when only using the [CLS] token.
Then I need to work on evaluation to get the entire pipeline works.

How do we evaluate?

1. Log sparsity histograms
2. Delta ImageNet-1K linear classification accuracy
3. Large-scale linear probing tasks using existing datasets
4. Hand-crafted linear probing tasks

# 11/01/2024

Notes on evaluation

[Open Source Sparse Autoencoders for all Residual Stream Layers of GPT2-Small](https://www.lesswrong.com/posts/f9EgfLSurAiqRJySD/open-source-sparse-autoencoders-for-all-residual-stream#General_Advice_for_Training_SAEs)

> Since we don’t have good metrics for interpretability / reconstruction quality, it’s hard to know when we are actually optimizing what we care about. 

> On top of this, we’re trying to pick a good point on the pareto frontier between interpretability and reconstruction quality which is a hard thing to assess well.

> The main objective is to have your Sparse Autoencoder learn a population of sparse features (which are likely to be interpretable) without having some **dense features** (features which activate all the time and are likely uninterpretable) or too many **dead features** (features which never fire).

> Too Dense:  dense features will occur at a frequency > 1 / 100. Some dense-ish features are likely fine (such as a feature representing that a token begins with a space) but too many is likely an issue.

> Too Sparse: Dead features won’t be sampled so will turn up at log10(epsilon), for epsilon added to avoid logging 0 numbers. Too many of these mean you’re over penalizing with L1. 

This is only in the validation set. 
It's unlikely to have dead features in your entire training set (I think).

> Just-Right: Without too many dead or dense features, we see a distribution that has most mass between -5 or -4 and -3 log10 feature sparsity. The exact range can vary depending on the model / SAE size but the dense or dead features tend to stick out. 

This post suggests that the historgrams are one of the best ways to quickly determine whether an SAE is good or not.
We can automate this to some degree by counting the number of dead features and the number of dense features. 
A better SAE will have fewer dead features and fewer dense features than a worse SAE.
So that's a very easy and quick way to evaluate models.

# 11/02/2024

How exactly do we evaluate models?
Let's qualitatively compare several single-number metrics and see how well they correlate with the more in-depth analyses.

Single-number metrics:

* MSE + lambda * L1 (original training loss, but fix lambda).
* Number of dead fautres + lambda * number of dense features
* MSE + lambda L0

Complex metrics:

* Normalized ImageNet-1K linear classification accuracy
* Large-scale probe tasks

Qualititative metrics:

* Looking at the top-k image grids
* Hand-crafted probe tasks

How well do the probe tasks work?

I also think I don't have to re-train the linear classifiers.
I can just apply them directly to the SAE.

# 11/04/2024

With more than one worker, my iterable HF dataset is being shuffled.
This is very bad for me.
I need to make sure this doesn't happen, then (sadly) recalculate the image activations.

# 11/05/2024

My "clean" dataset using a ViT-B/32 has empty values at 62721167 and onwards.
The length of the dataset is 62777183.

The config is `DataLoad(shard_root='/local/scratch/stevens.994/cache/saev/4dc22752a94c350ea6045599290cfbc31e3ee96b213d485318e434362b3bbdda/', patches='patches', layer=-2)`
`dataset[62721166]` and `dataset[62721165]` are non-empty, but `dataset[62721167]` onwards is empty, which causes issues.

Time to figure out why.
It was an indexing issue.
It's fixed now.
I fixed this bug.

What's the priority?

1. Qualitatively an SAE trained on the CLS token using DinoV2 and a CLIP model.
2. Apply some specific hand-crafted evaluations to measure "abstract" features in DINOv2 space.

I want to uncover interesting patterns in DINOv2 features.
Does it also have superposition?
Or is it only due to language?

Also, how do we evaluate patch-level features qualitatively?
Should we look at images that have maximal patches?
Or is a mean a better use?
Can I highlight individual patches?


Now I want to pick out images that have patches that maximize individual SAE features.
So I need to look at every patch, then refer back to the original image index and patch index, then save these values.

Tomorrow:

Patch-level activations don't seem to be working. 
But this could be an outcome of using CLIP rather than DINO. 
So I should verify that CLS token visualizations still work with CLIP and pre-trained SAEs.
Then I should train a patch-level SAE with DINO activations on OSC.
Finally, I can try getting patch-level activations with DINO.

Why do the same images show up? 
Are the img indices actually changing from the first batch, or is it always 0-333?


# 11/06/2024

1. Why do the same images show up in different features? Are indices changing from loop to loop, or always [0, 334)?
2. Set the script up so it works with CLS tokens.
3. Verify that it works with older checkpoints.
4. Train both patch-level and CLS-level DINOv2 SAEs on OSC.
5. Change webapp.py script to support:
  1. Max activation over entire image (current)
  2. Max CLS activation (old)
  3. Max patch, then only show unique images out of top k (new)


# 11/07/2024

Potential outline

1. Qualitatively compare DINOv2 and CLIP using top-k grids. Demonstrate the different concepts that are learned using cherry picked examples.
2. Discuss methods to evaluate SAEs, cheap proxies for these evaluations, and which proxies are good approximations of more expensive evaluation methods.
3. Discuss training strategies that reliably improve these proxy metrics (hyperparams, different datasets, etc).
4. Use this analysis to compare DINOv2 vs CLIP, CLIP vs BioCLIP, training SAEs on ImageNet-1K, iNat21, ToL-10M and LAION-2B multilingual.

Time to train on the [CLS] token and the patch tokens for DIONv2.

# 11/08/2024

OSC is busy right now.
What experiments can I do on the lab servers?

1. Work on quantitative evaluations.
2. Write.
3. Work on hand-crafted tasks
4. Explore large-patch models like ViT-S/32

# 11/10/2024

Thinking about a preprint.

**What is the goal and why do people care?**

The goal is to produce a reliable training recipe for SAEs for ViTs.
People care because SAEs are a powerful tool for interpretability that are underused in vision.
Training SAEs is challenging, but there exists a simple recipe so that everyone can get started in this area.

**Who is your audience? Who will use this or build on this?**

Computer vision interpretability researchers.

**If you want it yourself to solve some problem, then that is a good sign. Explain why you want it.**

I want to find traits in images of living organisms.
I want to build a search engine using these patch-level traits.

**What is your hypothesis? How will you know if it's true?**

My hypothesis is that simple, cheap tricks like normalizing input vectors, using L1 magnitude in the sparsity term of the loss, etc will remove the need for more complex stuff like forcing W_dec to have unit norms, removing parallel gradients, ghost grads, etc.
I'll know if it's true if adding ghost grads to my recipe R is not meaningfully better than just my recipe R.
Same for removing parallel gradients, etc.

A second hypothesis is that DINOv2 learns fine-grained visual patterns, while CLIP learns larger, semantic concepts.
I hypothesize that DINOv2 will be better for recovering morphological traits within species, and CLIP will be better across more diverse clades.
I will know if this is true by comparing scores on Broden.

**What is the problem/impediment that means it has not been done yet?**

No one is interested.
Existing interpretability methods like GradCAM and such seem to be "good enough".
There's not really an impediment to applying SAEs to ViTs, there just isn't as drastic a need.

**Nugget — what is your key insight that makes it doable? This is probably the single most important thing in a good paper.**

The nugget is that sparse autoencoders uncover features that are not easily recoverable by gradient based interpretability or clustering methods.

> This isn't a good nugget. Keep thinking about it.

**What is your elevator pitch? That is, if you meet a senior scientist in an elevator and she asks you what your paper is about, how do you describe it in 3 sentences or fewer.**

We train sparse autoencoders on vision transformers and find new patterns in the features learned through different pre-training objectivs and data distributions.

**Teaser — what would a teaser image show to explain this core idea?**



**What are the key previous works and what do they get wrong? All papers have limitations. Find these and they point the way forward.**

**How will you evaluate your method quantitatively?**

**What is your “demo”? That is, how do you show that the idea works.**

**What are the key risks?**

**Do you have all the data you need?**


---

The answers to some of these questions imply that I need to compare to GradCAM and standard interpretability methods.
First, I need to convince computer vision interpretability folks that our method has merit compared to other methods.
Second, I need to show that GradCAM specifically isn't good enough for the types of things I want to do.
On the other hand, that sounds boring and slow.
Why do I care about SAEs over GradCAM?
Because I want to find other images that have similar features at a patch level.
Can I do that with the raw activations?
Like just a clustering method?


I should expand on this:

> The goal is to produce a reliable training recipe for SAEs for ViTs.
> People care because SAEs are a powerful tool for interpretability that are underused in vision.
> Training SAEs is challenging, but there exists a simple recipe so that everyone can get started in this area.

How do you know they're a powerful tool?
What evidence do you have?

I can use SAEs to produce interpretable classification trees.

Why is training SAEs challenging?
What's wrong with the simplest possible recipe?

I know that training SAEs for vision is challenging because the default recipe doesn't work for CLIP for ImageNet-1K.
But my recipe does.

How do you know the default recipe "doesn't work"?

* No interpretable features -> many features are dense/dead, not many features corespond with Broden concepts
* 


**What do I want?**

I want a simple, reliable recipe for training sparse autoencoders on vision transformer patch activations for any pre-trained vision transformer and for any dataset of activations.

**Why do I want that?**

With such a recipe, I can train a sparse autoencoder for any ViT on any dataset.
With sparse autoencoders, I can

1. Train interpretable classifiers (decision trees, concept bottleneck models).
2. Discover differences in the learned features of pre-trained vision models.
3. Find morphological traits in living organism datasets that are consistent across species.

**How do I know I have a reliable recipe?**

If I have a recipe that works on DINOv2, CLIP, and BioCLIP across ImageNet-1K and iNat21 (mini), where

* Larger SAEs outperform smaller SAEs on the reconstruction-sparsity tradeoff
* larger SAEs outperform smaller SAEs on our downstream measures

Then I am confident that I have a reliable recipe.

**Why doesn't it exist?**

I don't know.
It doesn't seem that hard.

What about from a perspective farther out?


# 11/11/2024

Wider models aren't doing better on the MSE-LO tradeoff.

From the GemmaScope paper:

> Holding all else equal, wide SAEs learn more latent directions and provide better reconstruction fidelity at a given level of sparsity than narrow SAEs.

OpenAI's scaling paper also shows this.

So there's probably something wrong with my setup.
Let's start with the dumb baseline again and verify that each additional step of complexity leads to an improved model.

Performance:

I can take the idea from [this post](https://community.wandb.ai/t/is-it-possible-to-log-to-multiple-runs-simultaneously/4387/2) and train many models in parallel so that I can amortize the cost of loading a batch over many training runs.
I did this.

## Checkpoint Notes

| WandB ID | Model | Dataset | Remove Parallel Grads | Normalize W_dec | Re-Init b_dec | Shard Root |
|---|---|---|---|---|---|---|
| h52suuax | CLIP | iNat21 | True | True | True | 029ca197e |
| 9d8r1qhs | CLIP | iNat21 | False | False | False | 029ca197e |
| cesfj6kj | DINOv2 | iNat21 | False | False | False | b8e0fc701 |
| c57ddw4o | CLIP | ImageNet-1K | False | False | False | f46d3e7a3 |
| 1lm0m9d2 | CLIP | ImageNet-1K | True | True | True | f46d3e7a3 |
| lfesqa63 | DINOv2 | ImageNet-1K | False | False | False | 3c824621e |

For iNat21
```sh
CKPT=cesfj6kj CKPT_DATA=b8e0fc701b95ffe84a99d1beeb57a16c2d8aa462c888e114f8e22658bf601fb4 CUDA_VISIBLE_DEVICES=1 \
  uv run main.py webapp \
  --ckpt checkpoints/$CKPT/sae.pt \
  --dump-to /research/nfs_su_809/workspace/stevens.994/saev/webapp/$CKPT \
  --data.shard-root /local/scratch/stevens.994/cache/saev/$CKPT_DATA \
  --data.patches patches --data.layer -2 --top-k 64 --n-workers 32 --sort-by patch images:inat21-dataset --images.root /research/nfs_su_809/workspace/stevens.994/datasets/inat21/ --images.split train_mini
```

For ImageNet-1K
```sh
CKPT=1lm0m9d2 CKPT_DATA=f46d3e7a3223a50f1423ad03305cda971fac4512f03264392dd719c8f2381cab CUDA_VISIBLE_DEVICES=3 \
  uv run main.py webapp \
  --ckpt checkpoints/$CKPT/sae.pt \
  --dump-to /research/nfs_su_809/workspace/stevens.994/saev/webapp/$CKPT \
  --data.shard-root /local/scratch/stevens.994/cache/saev/$CKPT_DATA \
  --data.patches patches --data.layer -2 --top-k 64 --n-workers 32 --sort-by patch images:imagenet-dataset
```

## Notes on Checkpoint `cesfj6kj`

DINOv2, iNat21 train_mini split.

* 235 is bovine horns. 3300 is also bovine horns but a different kind; larger animals like bison and cows.
* 402 is bird heads
* 561 is bird wings
* 103, 575 712 are flower anthers
* 741 is ungulate noses
* 850 is bird wing *tips*
* 1196 has porcupine spines and antelope horns, but 1203 is sea urchin spines.
* 1638, 2709 have timestamps (artifacts of data)
* 1665 is reptile and amphibian noses
* 1823 is fish caudal (tail) fin
* 1900 is flower petals
* 2010, 2635 are bird tail
* 2094 is owl heads
* 2193 is insect legs
* 2247 is spiral shells
* 2358 is bird feeders
* 2537, 3568 are split images
* 2606 is protective shells, like porcupines, echidnas and armadillos.
* 2635 is bird tails
* 2746 is mushrool stalks
* 2770 is legs but only for Aphonopelma (tarantula genus)
* 2948 is callouts in the images
* 3007, 3691 are phoenix (palm tree genus)
* 3009 is Trochilidae (hummingbird family)
* 3093 is birds with black markings on the eyes <- TRAIT?
* 3218 is bird wings with white stripes on wing
* 3226 is rings on a finger (human jewelery)
* 3332 is cacti spines
* 3410 is Echinacea purpurea (but both images are labeled as animals because they contain insects).
* 3476 is human hair
* 3652 is coins (used for scale)
* 3664 is a bird wing pattern (not sure what it's called, but very distinctive).
* 3738 is a long bird beak. All sandpipers and pelicans.
* 3786 seems to be a thin white stripe at the front edge of bird wings.
* 3897 is a moth body
* 3988 is Ramphastos (toucan genus) bills (there are definitely other neurons for this feature)
* 4748 might be legs of ungulates.
* 6995 looks like dorsal fins of fish.
* 10863 seems to activate on fur.
* 10992 might be fish pectoral fins



## Notes on Checkpoint `h52suuax`

CLIP, iNat21 train_mini split, *with CLIP tricks

* 647 is mammal feet; covers rodents, carnivores, marsupials, *and* hyraxes
* 674 is stripes across zebras, fish, *and* butterflies.
* 1128 is camoflage across spiders, insects and frogs.
* 1246 is only on flowering plants --> is this reliable??
* 1282 is spines across sea urchins and porcupines

* 6826 is the spines of Erinaceus roumanicus (specific hedgehog species)
* 6819 is the spikes of Asphodelaceae family, which has the following description on Wikipedia: "The flowers (the inflorescence) are typically borne on a leafless stalk (scape) which arises from a basal rosette of leaves."
* 6371 is both dorsal and anal fins
* 6293 is birds with white markings on their head

## Notes on Checkpoint `lfesqa63`

DINO, ImageNet-1K train without tricks

* 128 is round things (donuts, tubas, trumpets)
* 218 and 1931 are fluffly pink things
* 435 is the round trumpet bell, but from the side.
* 461 is the crossed legs of folding tables, folding chairs, folding ironing boards, etc.
* 626 is window screens, which have a very distinct texture.
* 687 is flower bouqets, typically held, but I expect that to be a result of ImageNet, not the SAE.
* 819 is tank treads from a 45 degree angle, but 83 is tank treads straight from the side. **I think DINOv2 doesn't learn object rotation invariances because of a lack of labels.** 3600 is also tank treads, from a different angle again.
* 882 is animal hind (all from the side view).
* 932 is a flower pattern
* 951 is open animal mouths
* 1075 is flags (all american--ImageNet or SAE?)
* 1172 is metal wings (often from "pickelhaube")
* 1235 is the specific "knee" joint of birds with long legs
* 1328 is the Coca-Cola logo
* 1368 is ram/bighorn sheep curved horns.
* 1572 is "cursive"
* 1888 is kingsnakes; 2417 is kingsnake and other snakes
* 10143 is human ears
* 10312 is human shoulders (hoodie, lab coat, black and white)
* 10778 is masks (ski mask, knight helmet, burka)
* 10858 is dog litters. But is this all young, or specifically dog littles because they look the same?
* 2264 is animal ears (cats, fox, lynx)
* 2389 is the dung ball
* 2504 is echidna spines
* 2791 is the spiky bit below the petals in a cardoon flower.
* 2940 is the hood of a cobra (3892)
* 3090 is the legs of a tarantula
* 3428 is antenna of leaf beetles
* 3604 is generic text on shiny packaging
* 3901 is specifically the *top* of the treads
* 3919 is generic characters -> no OCR!
* 3956 is whisker (cat otter, lync, etc)
* 10966 is sneakers
* 11170 is bird feet
* 11183 is gauges
* 11248 is cat whiskers (tigers, cats, leopards)
* 11264 is onions on burgers.
* 11492 is rice (specifically with other food? Visual or semantic?)
* 11565 is just generic "black text on white background", but not reading the symbols
* 11650 is jack o lanterns. Compare to CLIP's "october" feature
* 11744 is striped rodent ears (possom, badger, lemur)
* 11851 is wing tips in birds of prey (again)

## Notes on Checkpoint `1lm0m9d2`

CLIP, ImageNet-1K train, *with* SAE tricks

* 113 is "cloud" (text and symbol)
* 188 is god and bible
* 299 is dog tongues
* 315 is ungulate legs/hooves
* 350 is bald heads
* 376 is "October" with just text. Does it work for Halloween stuff too?
* 407 is "Support" text.
* 425 might an example of needing a register.
* 537 is grated cheese on top of food
* 594 is photography (OCR and concepts)
* 599 is teeth (drawing and photographs)
* 613 is snow dog tails
* 768 is images that are split (multiple shots in one image)
* 889 is linoleum tile (flooring)
* 927 is Russia (OCR, symbols, concepts like U-boat, locomotive)
* 949 is "royal"

What patterns am I seeing now?

* CLIP groups text, symbols, drawings and images together in the semantic space. This is really useful for things like LMMs.
* DINOv2 identifies traits more often than CLIP when looking at iNat21. Like, 5x as often



# 11/14/2024

Anthropic discusses ways to search for particular features [here](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#searching).
In general, it seems using combinations of positive/negative examples and filtering by "fire/no fire" is good, rather than the LDA-based classifier I have now.

# 11/18/2024

### Influential Topics and Papers in Model Debugging  

Here’s a breakdown of key topics and landmark papers to guide your search:  

---

### 1. **Interpretability and Conceptual Debugging**  
   - **Papers**:  
     - *“Distill and Detect: Mapping Knowledge Representations for Neural Networks”* by Hinton et al. Introduced distillation, with implications for debugging via knowledge transfer.  
     - *“Network Dissection: Quantifying Interpretability of Deep Visual Representations”* by Bau et al. (2017). Introduced a method for attributing neurons to human-understandable concepts.  
     - *“Testing with Concept Activation Vectors (TCAV)”* by Kim et al. (2018). Uses concept vectors for debugging biases and assessing representation relevance.  

   - **Search Terms**:  
     - "Concept-based interpretability"  
     - "Neuron-level interpretability"  
     - "Network dissection techniques"  

---

### 2. **Feature Attribution**  
   - **Papers**:  
     - *“Integrated Gradients”* by Sundararajan et al. (2017). A widely-used method for explaining model predictions via input gradients.  
     - *“SHAP: A Unified Approach to Interpreting Model Predictions”* by Lundberg and Lee (2017). Provides local explanations for individual predictions.  
     - *“Grad-CAM”* by Selvaraju et al. (2017). Explains model decisions by visualizing class-specific activation maps.  

   - **Search Terms**:  
     - "Feature attribution methods"  
     - "Saliency maps"  
     - "Explainability in deep learning"  

---

### 3. **Debugging via Counterfactuals**  
   - **Papers**:  
     - *“Counterfactual Explanations without Opening the Black Box”* by Wachter et al. (2017). Explores generating counterfactual examples for model debugging.  
     - *“Robustness Disparities by Design?”* by D’Amour et al. (2020). Discusses biases exposed via adversarial counterfactuals.  

   - **Search Terms**:  
     - "Counterfactual model debugging"  
     - "Robustness and adversarial testing"  

---

### 4. **Bias and Fairness Debugging**  
   - **Papers**:  
     - *“A Framework for Understanding Unintended Consequences of Machine Learning”* by Suresh and Guttag (2019). Highlights sources of bias and debugging strategies.  
     - *“The Mythos of Model Interpretability”* by Lipton (2016). Discusses the trade-offs and pitfalls in debugging interpretability.  

   - **Search Terms**:  
     - "Bias debugging in ML models"  
     - "Fairness-aware machine learning"  

---

### 5. **Debugging Neural Representations**  
   - **Papers**:  
     - *“Probing Neural Representations”* by Conneau et al. (2018). Uses diagnostic classifiers to analyze internal representations.  
     - *“Representation Erasure”* by Elazar and Goldberg (2018). Tests the influence of specific features by removing them and observing performance.  

   - **Search Terms**:  
     - "Probing neural representations"  
     - "Feature erasure methods"  

---

### 6. **Model Behavior and Failure Analysis**  
   - **Papers**:  
     - *“A Taxonomy of Machine Learning Failures”* by Barredo Arrieta et al. (2020). Categorizes failure modes and discusses debugging strategies.  
     - *“Debugging Deep Models with Logical Constraints”* by Narayanan et al. (2018). Formal methods for pinpointing and fixing logic violations in neural network outputs.  

   - **Search Terms**:  
     - "Failure modes in ML"  
     - "Behavioral analysis of neural models"  

---

### 7. **Transfer Learning and Representation Debugging**  
   - **Papers**:  
     - *“Do Better ImageNet Models Transfer Better?”* by Kornblith et al. (2019). Studies how representation quality affects transfer learning.  
     - *“Representation Learning with Contrastive Predictive Coding”* by Oord et al. (2018). Debugging representation learning via contrastive methods.  

   - **Search Terms**:  
     - "Transfer learning debugging"  
     - "Contrastive representation learning"  

---

### Practical Suggestions  
Use combinations of terms like *“debugging deep models,”* *“interpretability techniques,”* and *“bias evaluation in neural networks”* to drill down further. Sites like [Papers with Code](https://paperswithcode.com/) and ArXiv categories (e.g., cs.LG, cs.CV) can quickly lead you to relevant papers.  

Would you like to focus on any particular subdomain or technique?


https://huggingface.co/docs/transformers/main/en/tasks/semantic_segmentation
https://github.com/facebookresearch/dinov2/issues/25


# 11/19/2024

For some reason when I trained on ViT-L/14 activations nothing worked.
So we need to debug that.
There are several changes

* ViT-B/16 -> ViT-L/14
* Removed b_dec re-init
* Scaled mean activation norm to approximately sqrt(d_vit)
* Subtracted approximate mean activation to center activations
* Various code changes to the way the activations are recorded :(.

So I will re-record ViT-B/16 activations, then re-add the b_dec re-init, ignore the scale_norm and scale_mean and *pray* that it works again.
Then I will re-add the original changes, debugging what went wrong at each step.
It might just be a learning rate thing.

# 11/20/2024

What exactly are my next steps?

1. Reproduce original workflow with ViT-B/16. There's some bug introduced and I need to fix it. Add tests for it.
2. Propose a trend in sparse autoencoders between CLIP and DINOv2 with at least 3 examples.
3. Ensure that your probing methodology works.


# 11/21/2024

Scaling Monosemanticity suggests that features should be both specific (high precision) and they influence behavior (causally faithful).

They suggest that measuring sensitivity (how reliably a feature activates for text that matches our proposed concept; recall) is hard.

# 12/02/2024

I want to see which hparam works the best.
I need to see how the predictions are to see if the linear model is any good.

# 12/04/2024

I can demonstrate a lot of manipulation in various ways:

* Linear probe + semantic segmentation
* Masked autoencoder pre-trained ViT with pixel reconstruction
* VLM with language generations
* BioCLIP with probabilities

I want to build Gradio demos for all of these.
I also want to cherry pick qualitative examples.
And finally, I want to present a set of hparams for training these things at the scale I'm training.

> We believe the only way to really understand the precise details of a technology is to use it: to see its strengths and weaknesses for yourself, and envision where it could go in the future. 

From https://goodfire.ai/blog/research-preview/

Framework for intervention:

1. Train an SAE on a particular set of vision transformer activations.
2. Train a (or use an existing pre-trained) task-specific head: make predictions based on [CLS] activations, use an LLM to generate captions based on an image and a prompt, etc.
3. Manipulate the vision transformer activations using the SAE-proposed features.
4. Compare task-specific outputs before and after the intervention.


## Notes from Meeting

Talking about faithfulness, causal intervention, etc.
But this method is not supposed to compete with INTR.
It should be a good visual trait extractor.
It should lead to highly factored visual traits.

We want to show the quality of the visual traits.
How can we demonstrate that?
By manipulating traits for particular classification traits.
This is just a means to an end.

If two species are inseperable by human eye, does the model find traits that consistently fire/don't fire between the two species to answer that question?

Can we find a difference between reticulated giraffe species?
What about a lack of differences between two species of red wolves that were recently merged into a single species?

Experiments

two "understanding"

dino vs clip (pre-training modality)
bioclip vs clip (domain/finetuning effects)

four "control"

image classification -> CLIP
sem seg -> DINO
mae (image gen) -> MAE
image captioning (vqa? moondream) -> Moondream

don't compare to protopnet or stuff in intro, it can come up in related work

So what's stopping us from releasing a preprint?

1. Experimental results
2. Dashboards
3. Writing

So I will:

1. Kick off jobs training on normalized activations for DINOv2 (patch) and CLIP (CLS)
2. Train an SAE on unnormalized activations for CLIP CLS to get something ready.
3. Build gradio demo for SAE inference on CLIP activations.

# 12/05/2024

Instead of gradio or streamlit, I'll just use gradio as an inference endpoint and then Elm -> static html + js as the polished web demos. 
I probably should write one pure inference demo in gradio to demonstrate how simple it can be, but for polished, interactive experiences, I want to write the frontends myself.
But this path has many pitfalls---do not get caught up in frontends in favor of writing a paper.

# 12/07/2024

Silly interface isn't working.
Part of the problem is that features for specific flower classes aren't actually flowers.
So it looks really bad.
Or if a feature is a flower, then it fires on all the flowers, so it's not specific enough.

Ok, I fixed it by using Caltech-101 instead.
I think part of the problem is that it's image-level, so there's not as much room to subtly manipulate the predictions.
I think a SigLIP model with image-text matching would be absolutely dope because you could write super specific captions, then manipulate at a patch level, then compute the last layer of the SigLIP + do the softmax to compute probabilities.

It would be especailly cool to do this on a patch level.

What else is important right now?

I need a complete to-do list of every single item that needs to be done from an experimental perspective.
Because there are a lot of different things to do.

Use [facebook/vit-mae-base](https://huggingface.co/facebook/vit-mae-base) as an MAE.

# 12/09/2024

Studying models again!

| WandB ID | Model | Dataset | Shard Root |
|---|---|---|---|
| a1a0vucf | CLIP | iNat21/train-mini | 50149a5a12c7 (strawberry1) |
| gpnn7x3p | BioCLIP | iNat21/train-mini | 07aed612e3f7 (strawberry1) |

## a1a0vucf - CLIP + iNat21

* `24K/2`: Generic moth wings
* `24K/3`: Laptop
* `24K/6`: Mottled bird wings (TRAIT)
* `24K/24`: Moth thorax (top view only)
* `24K/32`: Top view of mushroom heads
* `24K/40`: Protective equipment (life vest, rubber gloves, etc)
* `24K/61`: Open animal mouths (very general)
* `24K/1848`: moth heads

* 4661: wings from white-spotted butterflies (Nymphalidae)
* 565: very similar to the feature above---not sure what the difference is
* 3339: yellow flower bunches
* 10860: birds perched on branch
* 5317: moths from the back (overwing). Lots of different families.
* 9411: birds with spread feathers in flight
* 12722: bird heads, specifically behind the eye/neck area
* 11239: purple flower bunches
* 17051: white/purple flowers with big petals
* 14910: birds
* 23048: insect legs (stink bugs, beetles, longhorns)
* 8540: tusk looking things (elephant tusks, mushroom stalks)
* 21860: Lions (great localization)
* 15918: insects, both moths and beetles
* 4004: power lines and windmills
* 4331: wing tips of moths
* 4594: bird reflection in water
* 14525: lizard tails
* 9405: human hands, not subject matter
* 2142: camoflage patterning, on snakes, owls, hyenas and frogs
* 18268: ridge on a white surface, like sea shells, mushrooms
* 24103: mountains in background
* 909: mammal ears

## rscsjxgd - CLIP + iNat21


## gpnn7x3p - BioCLIP + iNat21

* 7: All birds of order "Caprimulgiformes".
* 29: all sort of "long" things, like spider legs, grasses, long leaves, etc.
* 417: all spiny thigs, like furry caterpillars, or the underside of mushrooms.
* 449: Moths from "Lepidoptera Geometridae".
* 451: Moths from "Lepidoptera Pieridae"
* 459: Legs of insects AND a fish?
* 468: Bowls that humming birds are sitting on.
* 473: Spottled neck of birds (TRAIT)
* 480: Mammal ears (TRAIT)
* 504: Squirrel tails (but also a bird head).
* 508: Moth wings/bird feathers
* 518: Moth wings with white stripe (TRAIT)
* 3205: Moth antennae
* 6448: Moth antennae again
* 13270: Noctuidae, owlet moths, with wings closed
* 9587: Purple flowers on top of grassy plants.
* 13156: Bird chests
* 23K/24104: Rounded to kidney-shaped leaves
* 7672: gall wasp "galls"
* 18380: moth wings with "eye"s
* 13917: purple flowers
* 873: belly-up rodents and shrews
* 21542: wings of birds with spots on their bellies
* 18444: hanging, bell-like flower (purple)
* 17435: owl heads
* 1677: striped insect thorax (TRAIT?)
* 16938: hanging bell-like flower (yellow)
* 10185: butterfly spotted wings, white with speckled dark spots
* 20085: butterflies with orange wings with black rims
* 10412: moth wings again
* 17920: dragonfly bodies
* 4465: Plant stems (TRAIT)
* 4467: Leaves of plants from family Asteraceae (+ mislabeld arachnid)
* 4468: Purple (petals, insect body)
* 4509: Leaf midribs (part of leaf)
* 18076: Bird crests (across different families)
* 18494: Bird tails (across different families)
* 5388: Beetle legs
* 10598: Beetles
* 19870: Bird legs
* 5435: moth wing edges
* 16081: more moth wings
* 5804: squirrels (great localization)
* 23215: Longhorn beetles (great localization)
* 18591: Canis (great localization)
* 24239: Cervidae (deer, great localization)
* 15393: Cheetahs, (great localization)
* 19155: Accipitridae (hawks, great localization)
* 16357: Starfish (great localization)
* 19505: Crabs!
* 12067: flowers, but mostly flowers with butterflies

In general, for both of these checkpoints, the features do not tend to look very monosemantic.
I would expect checkpoints with a higher sparsity contraint, and thus lower eval/L0, to be more monosemantic.
Claude agrees.
So we will try that.


# 12/17/2024

As an example image, I want to manipulate sand in a beach.

I have the image and patch details below:

```json
{
  "image": 3122,
  "patches": [202,203,204,205,206,207,217,218,219,220,221,222,223,231,232,233,234,235,236,237,238,239,245,246,247,248,249,250,251,252,253,254,255]
}
```

Now I need to grab the top features from a particular checkpoint.
Then I need to manipulate them and re-run the last layer of the neural network + the linear probe head.

And I need to make this possible in a Gradio app.


# 12/19/2024

## Raw Transcription of Tanya's Meeting Notes

1 version of SAE - understanding model : input -> R^d

1 version is just doing input -> semantic R^d
Model composition is one example of this mapping

Not just control, but more about what a model can do over a distribution of unknown tasks
-> SAEs help understand learned representations and enables "true sensitivity analysis" in data-agnostic way

Understanding model limits and potential by connecting representation space to interpretable features
-> Explore application by enumerating dimensions of interpretable space -> downstream tasks mostly care about interpretable features <- what can we do in principle?

Only care about interpretable control

Human-AI partnership

Control without interpretability is meaningless -> no idea about output
If possible, we have to retrain (significant retraining)

Pitch:

Apply to new tasks, interpretable and trustworth all without re-training
                                      ^ existing work is not trustworth because not controllable

Use these three to prevent "where are the tables" from reviewers:

* Rolnik, Beery -> Accuracy driven is not the right way to eval
* Fei-Fei's interpretable AI - call to action
* If humans make decisions, we eval on unknown task and compare to other metrics

-> Compare pixel masking (phylo-nn)
-> Cite with retraining tabular models from scratch
-> OOD papers - motivate to new tasks

Redwinged blackbird
-> push features -> see class modifications (patch-level classification)
no red wing -> then blackbird

find qualitative traits that manipulate classifications

connect interpretable features -> representation space

counting + remove features for object
         + colors?

# 12/20/2024

Update for Yu

Progress:

* Further refined story based on feedback from others: SAEs enable the scientific method in interpretability.
* Got a preliminary version of the semantic segmentation figure. Focusing on classification figure now.

To Dos:

* Finish Figure 1 and 2 in "understanding"
* Make figure demonstrating trait-based image classification
* Train SAE on VLM vision encoder
* Make figure for language generation
* Improve introduction so no reader asks for quantitative results

Questions:

* Which VLM should I use? Moondream2, Phi 3.5 vision, Llama 3.2 11B? -> Go with Qwen 2.5 VL 7B

Unknowns (scary):

* How to write the intro effectively

How Yu can help:

* Add Harry and Tanya's overleaf accounts to the preprint
* Think about intro; share any weaknesses (as concrete as possible) via Teams.
* [done] Provide concrete feedback on semantic segmentation figure

# 01/08/2025

Comparing BioCLIP and CLIP: I think BioCLIP probably learns very highly specific features compared to CLIP.
I think we can show this with moth and butterfly wings.
If we can demonstrate BioCLIP has different features for different traits of butterfly wings, and that CLIP doesn't, then we can show that BioCLIP can classify those species and CLIP cannot.

* 449: Moths from "Lepidoptera Geometridae" (.
* 508: Moth wings/bird feathers
* 518: butterfly wings with white stripe (TRAIT)
* 10185: butterfly spotted wings, white with speckled dark spots
* 10412: moth wings again (Lepidoptera Hesperiidae Erynnis)
* 18380: moth wings with "eye"s (all but one are Lepidoptera Nymphalidae, all are underwings)
* 5435: moth wing edges
* 16081: more moth wings

* 451: Moths from "Lepidoptera Pieridae"
* 20085: butterflies with orange wings with black rims (all but one are Lepidoptera Nymphalidae Danaus, all are "overwing")


# 01/13/2025

usvhngx4: CLIP + ImageNet

* 4988: smoking/cigarettes, text/image alignment
* 21009: low-res images
* 7368: sorry, apologize text AND lighhouses/seawalls
* 20604: pawprint graphics
* 15731: cancel symbol
* 5573: black and white stripes
* 20652: damaged machinery (crack, accident, dent)
* 7622: america/united states
* 21366: bison horns
* 12182: BMW logo
* 11106: plant stems with animal
* 7840: gloved hands
* 22136: tongues, both human and animal
* 17910: chopsticks
* 6923: sea animal head?
* 24456: sunglasses and "cool", both visual and text and photographic styles
* 6909: brazil
* 11579: Maple leaves
* 11965: mustaches
* 17966: swastikas
* 15901: knives or blades
* 6252: gas, smoke, vapors
* 6114: baseballs (great localization)
* 18670: rifles
* 218: adidas
* 611: pencils
* 20735: lightsabers
* 9664: sheep
* 18085: aprons, vests, body armor
* 11560: red setter heads
* 5725: pillows

# 01/15/2025

Getting DINOv2 visuals did not work very well (vqn8tscm-high-freq).
I need to double check that I picked the correct checkpiont for ImageNet, etc.
I will double check and then run it again.

The DINOv2 checkpoint is actually oebd6e6i.
This one has:

* 4002: photographs of human teeth
* 21856, 20471, 22680: dog ears
* 18174, 2190: text (generic, no OCR)
* 13263: human lips
* 11025: clasps
* 14605: olympic rings -> can CLIP detect this across visual styles?
* 24034: statues of virgin mary?
* 15371: rolling chair legs

I think I need to find some test sets.
Otherwise I won't really have good comparisons.


# 01/17/2025

Construct  train/test split by automatically finding images that maximally activate "Brazil".
Then make the split by putting similar images from DINOv2 in the train set and different images in test set.
This hopefully leads to a train/test split where semantic concepts are in different splits: brazil flags in the test set, the rio jesus statue is in the test set.
This hopefully makes it hard for models that cannot reliably identify semantic concepts (DINOv2).

# 01/19/2025

I now know that I need to collect additional probing datasets.
Specifically, I want to find probing datasets that find interesting things in the CLIP checkpoints that I have.
So that means things like:

* smoking/cigarettes
* low-res images
* brazil (and other countries)
* sunglasses/cool

The brazil feature reliably triggers!
Very reliably, on many different things.
It's now important to check this for dinov2---this is the first reliable feature we've found.

So I want to improve this interface in a couple ways:

1. I want to be able to (un)select particular latents. Since for each model/dataset pair there are 24K choices, I want a dropdown with fuzzy search.

# 01/21/2025

Now I need to find the test sets.
We have Brazil and sunglasses across visual styles.
Let's do it and make some test sets.

# 01/22/2025

Debugging the script:

image 31 is the fish. It is cool-negative. the `pred_labels_SN[24456, 31]` is 1. That's weird, because on the browser, it's not.

The image after vit -> sae has different patch values for 24456 than after saved vit -> sae.

probbaly I am not applying dataset-level normalization to the SAE activations.
ACTUALLY probably I am not doing that for TONS of examples

I probably need to get it by loading the dataset for the imagenet shards, then recording the scalar/mean vector and using it.

# 02/03/2025

I think my semantic segmentation head isn't working well. It should make predictions based on the last layer of the transformer, but the SAE is trained on the second-to-last layer.
I need to

1. Train semseg head on the last layer
2. Confirm that all my layers are working correctly.

This is also a good chance to make sure that I can reliably reproduce prior results.


# 02/05/2025

[https://cs.uwaterloo.ca/~shai/clustering.html](https://cs.uwaterloo.ca/~shai/clustering.html) is interesting work on clustering.
If you have good clustering, then any algorithm works.
If not, no algorithm works.

# 02/10/2025

Comments:

* [x] Figure 2. I think "Container ships" should be "Containers on ships." A container ship may not have containers on top, right?
* [ ] Figure 2. Should we replace the "blurry photos" example with something else? Blurry to me is a "global" concept and should not be localized, unless there is motion. This example to me is less convincing than others.
  -> I actually think that this is a really cool aspect of SAEs, that some global concepts are present in the patch tokens, rather than the [CLS] token. It's also cool to me that it's not just straightforward concepts like "objects"---it can cover a broad range of semantic concepts, even those that are not visually similar.
* [ ] Figure 1. I still have problems understanding the SAEs row.
  * [x] I think the first column is observation, so no hypothesis should be made. If so, why "We manually choose to inspect the bird’s blue feathers." fit there? Or, should we say "We manually choose to inspect the bird's wing" instead of saying "blue feathers?" to keep it still an observation?
  * [x] For the second column, how about this?
    * Our SAE finds similar patches from images not necessarily of blue jays; all correspond to “blue feathers.” We hypothesize this “blue feathers” feature is integral to the ViT’s prediction.
* [x] Figure 3. I'm not sure what sum across vectors means. Across human-selected patches? If so, after "get user-specific patches", only red patches should remain, not the white ones, right?
* [x] Section 3.3. Change n to something else. You already use n for the dimensionality in SAEs.
* [ ] Figure 7: The top right, lower right, and lower middle read weirdly to me. They seem to not follow the current figure layout.
* [ ] Figure 7 caption doesn't make sense (Lower right, lower middle?)

# 02/11/2025

For this project, I need to do a couple things.

1. Polish the web demos.
2. Tweet/PR about the release.
3. Plan a submission to ICCV.

I want to think deeply about the ICCV submission, and think about what flaws must be addressed, nice-to-haves, etc.
Here's a brain dump of everything:

* Can we suppress GradCAM activations? Is that enough to simulate feature suppression? How can we differentiate with such an approach?
* Can we get quantitative results? How does INTR do it?
* Can we expand to more vision tasks? Object detection, LLava-style VLMs, something else?
* How can we semantically probe for the presence/absence of concepts?
* Can we form observation -> hypothesis -> validation for Section 4
* Need a random baseline manipulation for feature manipulation. Then it will establish that the given feature is truly driving decisions.
* Additional examples (cultural, semantic concepts, etc)
* Needs to be down to 8 pages (https://iccv.thecvf.com/Conferences/2025/AuthorGuidelines)
* Apply to other architectures (CNNs? Mamba?)
* Use something besides ReLU SAEs?
* Writing quality :)

What is likely highest-priority?

1. Quantitative results - a comparison to a random feature baseline for manipulation quality could be incorporated here.
2. Suppressing GradCAM activations - we need to see if this enables the same kinds of manipulations. If so, I think there are other kinds of manipulation for which GradCAM is not the best (semantic segmentation). We will need to highlight those particular tasks.
3. Getting it down to 8 pages

Things I'm aware of and that I don't think are high priority:

* Adding another control task, like manipulating a VLM
* Other SAE architectures beyond ReLU, other vision architectures like CNNs or SSMs.
* Rigorous methodology around showing the absence of a concept (semantic probing)

@Harry do you have suggestions for ICCV submissions vs CVPR or NeurIPS? Are there particular things that we should watch out for?

Submission Checklists

* https://www.ianhuston.net/2011/03/checklist-for-arxiv-submission/
* https://trevorcampbell.me/html/arxiv.html


# 02/12/2025

What experiments do I want to run for ICCV?
I know I need some qualitative results.
I also need to compare against random feature perturbations.
I also want to compare to suppressing GradCAM activations.
Then, I need to think about what parts are still qualitative, even after this, and try to figure out how to write in a way that admits this failure.

INTR doesn't have any quantitative results.
Finer-CAM does have some quantiative results, but the metrics are so unfamiliar to me that I don't really have any intuition around the numbers.
Great that there is an improvement, but what kind of improvement should I expect?
I think random feature perturbation is a much better baseline.
We could also think about which traits reliably distinguish two species, and then deliberately mask them.
I should email Justin Kitzes about that again.

What about semantic segmentation?
If I can reliably identify a feature for a given class, can I suppress it without changing other features?

That would be a good comparison.

Three methods:

1. Random SAE feature
2. Automatically identified feature
3. Gold feature (human identified via semantic probing)

If you set the feature to -2x the observed value, what happens to class predictions?
I would expect:

1. Random SAE feature: no meaningful change in any predictions
2. Auto identified feature: that feature disappears in original output patches (predicts something else), other patches stay the same.
3. Gold feature: same thing.

How would I measure it?
How would I build human intuition around these values?

% of original patches suppressed
% of "other" patches changed.

Many examples.

Let's get a chatbot to write the experimental design precisely.
Well, they both didn't write an experimental plan that I liked enough.
So I will write one in my docs.


# 02/13/2025

Updates on quantitative experiments!

1. Random vectors do not manipulate the predictions very much. I suspect that we should pick random activation vectors as our features, so that we're sampling vectors that all reliably point in the same direction (many models produce anisotropic features; [1, 2, 3])
2. Random SAE feature manipulation also does not manipulate the predictions very much. This is great!
3. Picking out automatic features is really slow. Like, 3 hours slow. Futhermore, it does not find features that I want.

[1]: Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere
[2]: How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings
[3]: On the Sentence Embeddings from Pre-trained Language Models


These are the logs from contrib.semseg.quantitative:

[2025-02-13 20:53:34,898] [INFO] [contrib.semseg.quantitative] Made 512000 predictions.
[2025-02-13 20:53:43,252] [INFO] [contrib.semseg.quantitative] Got masks for 'wall' (1).
Top 3 features for wall:
 14721 >0.0: 0.270
  6008 >0.0: 0.269
  5693 >0.0: 0.267
[2025-02-13 20:55:41,681] [INFO] [contrib.semseg.quantitative] Got masks for 'building, edifice' (2).
Top 3 features for building, edifice:
     0 >0.0: 0.000
     1 >0.0: 0.000
     2 >0.0: 0.000
[2025-02-13 20:57:45,082] [INFO] [contrib.semseg.quantitative] Got masks for 'sky' (3).
Top 3 features for sky:
     0 >0.0: 0.000
     1 >0.0: 0.000
     2 >0.0: 0.000
[2025-02-13 20:59:43,573] [INFO] [contrib.semseg.quantitative] Got masks for 'floor, flooring' (4).
Top 3 features for floor, flooring:
     0 >0.0: 0.000
     1 >0.0: 0.000
     2 >0.0: 0.000
[2025-02-13 21:01:43,486] [INFO] [contrib.semseg.quantitative] Got masks for 'tree' (5).
Top 3 features for tree:
     0 >0.0: 0.000
     1 >0.0: 0.000
     2 >0.0: 0.000
[2025-02-13 21:03:44,888] [INFO] [contrib.semseg.quantitative] Got masks for 'ceiling' (6).
Top 3 features for ceiling:
     0 >0.0: 0.000
     1 >0.0: 0.000
     2 >0.0: 0.000

When I cancel these runs, it's always dying on `x.sum(...)`.

So a couple things:

1. It's really slow. How can I speed it up? Maybe batching, moving to GPU, and speeding it up?
2. It's not finding any features. This is probably a bug, given that it finds "good" SAE for the first class, but not any of the next ones.

Either it's a bug and I need to fix it, or I need to only pick patches with

One such bug is activation normalizations.
You need to normalize the activations.
Just like in the comparison app in saev, you need to load the mean/scalar from disk for DINOv2 activations on IN1K.

TODO TODO TODO

# 02/14/2025

Happy Valentine's Day!

There are several different screen sizes to try.

1. Mobile (real phone)
2. 1/3 macbook
3. 1/2 macbook
4. 2/3 macbook
5. Full macbook

I think this covers the important stuff.

# 02/16/2025

Classification app to do:

* [done] add a "toggle highlights" button
* make it so the highlights turn on/off when you hover w a mouse over the images
* [done] change the patches to be all red (no legend needed)
* [done] Add classnames to the images in similar patches
* [done] Add examples of covering a trait.

# 02/19/2025

I chated with Dan Rubenstein about distinguishing birds, dichtomous keys, and other topics.

Some notes of interest:

Dan really wanted to upload various images of zebras, horses and other equids and see which sparse features BioCLIP used during its classification.
This kind of interface is quite general and could be done with BioCLIP + SAE + SVM (I think).
But this would be a pretty tough web demo and I don't know if I can prioritize it.

As examples of genera that are similar save for a few traits:

* MacArthur's warblers are all the exact same body shape but have distinct coloring.
* European tits are also the same body shape but have different coloring.

I would love to compare INTR, Finer-CAM, Prompt-CAM, SAE-based classifiers, and Tory Petersen's published field guides.

Some other stuff:

Heterogenous summation vs gestalt classification: SAEs+SVMs are heterogenous summation models, while a dense ViT + linear layer is a gestalt classifier.

If we could reliably extract traits, then we could write a decision tree (by hand) for predicting age and sex, which would be awesome for conservation.

# 02/28/2025

There are tons of features for "trees" present in the SAEs.
For example, look at example 553 in ADE20K validation and selec the bottom right trees.
There are four different "tree" features (6097, 4299, 7110, 7648) that all look like great semantic features to me.
Of course we don't get good recall.

Same thing for toilets.
Look at example 1099 and select the toilet.
5876, 10875, and 15002 are all porcelain or tile.
I wonder what their precision is like.

But I am surprised that we don't have good precision.

# 07/18/2025

I am updating the visuals.py script to work with the new dataloader.
The new dataloader loads shards in random order for training.
For visualization, we need them in order so that all the activations for a given image's patches are seen together.
However, if we assume that every latent actually needs k different images and we save latents for 500 latents (default is 400), then we need:

128 x 500 = 64K images

For 256 patches/img, that's:

256 x 64K = 16,384,000 (16.3M) patches

How fast is that with random access on a slurm node?
I benchmarked  this with 4 workers on an Ascend worker node and got 192 samples/sec over 100 batches of 256.
That's:

16,384,000 / 192 = 23.7 hours.

With 16 workers and a batch size of 128, I got 1,409 samples/sec.

16,384,000 / 1,409  = 3.2 hours

With 16 workers and a batch size of 256, I got 2,396 samples/sec.

16,384,000 / 2,396 = 1.8 hours

With 16 workers and a batch size of 512, I got 4,084 samples/sec.

16,384,000 / 4,084 = 1.1 hours

With 16 workers and a batch size of 1024, I got 5,838 samples/sec.

16,384,000 / 5,838 = 46 minutes

In a table:

| Workers | Batch Size | Samples/sec | Time for 16.3M patches |
|---------|------------|-------------|------------------------|
| 4       | 256        | 192         | 23.7 hours             |
| 16      | 128        | 1,409       | 3.2 hours              |
| 16      | 256        | 2,396       | 1.8 hours              |
| 16      | 512        | 4,084       | 1.1 hours              |
| 16      | 1024       | 5,838       | 46 minutes             |

If I change the random seed, then I get more like 800 samples/sec using the same settings as above. I wonder if the OSC disk system has some smart caching or whatnot that makes repeat runs faster.
Claude code ran a bunch of commands and concluded:

> Yes, OSC's disk system definitely has caching mechanisms that can explain the performance difference. Based on my investigation:
> 
> 1. GPFS Filesystem: Your data is on /fs/scratch which uses GPFS (General Parallel File System), a high-performance clustered filesystem designed for supercomputing environments.
> 2. System Cache: The free -h output shows 181GB of buffer/cache memory being used. When you access files, they get cached in system memory.
> 3. GPFS Page Pool: GPFS has its own internal caching mechanism called the "page pool" that caches frequently accessed data blocks.
> 
> The ~7x speedup (from 800 to 5,838 samples/sec) between different random seeds makes sense because:
> - With the same seed, you're accessing the same files in the same order
> - These files get cached in memory after the first access
> - Subsequent runs with the same seed hit the cache instead of disk
> - Different seeds access different files/orders, causing cache misses
> 
> This is why benchmarking I/O on HPC systems should always include:
> - Cold cache runs (first run after clearing caches)
> - Warm cache runs (subsequent runs)
> - Multiple random seeds to get realistic performance estimates
> 
> For your use case, the 800 samples/sec with a new seed is probably more representative of real-world performance when processing new data.

So this probably means these numbers are bogus.
Thus, I probably need an in-order efficient dataloader.
However, this should be pretty easy to write because I don't actually need to use multiple threads. I can simply read the entire shard in order. In fact, I might be able to simply use the main process.
