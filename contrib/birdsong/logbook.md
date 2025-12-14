# 12/11/2025

Is there a norm layer after the final layer in Bird MAE?
Maybe my activations are total dogshit.
Most models (DINOv3, Perception Encoder) include a LayerNorm after the final output on the patch embeddings.
I bet that's it.

I saved the activations with `final_norm='patch-norm'` (self.norm) and the activations are still pretty similar.
I don't know why.
I need to diagnose this.

There's also the whole thing with `loss/n_dead` spiking straight to 12K.
That implies that all 12K are dead from the very start.
That's really weird.

# 12/12/2025

I think I need to look at activations, their scales, and how they compare to activations from other models.
Luckily this is all shard-based, so we don't need to load up ViTs or datasets.
But if we want to compare with/without norms for BirdMAE, we will need to get the model itself.


1. Check all norm layers in BirdMAE if there's a weird bias/weight term
2. Check why fishbase didn't log auxk before 10M tokens
3. Normalize dim 295?

# 12/13/2025

Figured it out. Transformers have "emergent outlier features" where a high-magnitude feature travels "through" the entire transformer's residual stream. These show up after a lot of training. I don't think this is actually an "feature"; I think it's a poor optimization decision from the BirdMAE guys but they never have to deal with it because the pre-attention and pre-mlp LayerNorms fixes this stuff by learning to set d=296 to effectively -1 (the learned multiplicative weight is very very small and the bias is ~1). 

So either 

- We do per-dimension scaling, where we normalize each dimension to mean=0 and std=0 
- We record the activations after layernorm1/2 instead of the raw residual stream.

Some arguments for each choice:

1. [Gemini](https://gemini.google.com/app/fd8adf2843a1e37f) says to do per-dim scaling (raw residual is better for steering?)
2. [Claude](https://claude.ai/chat/9eda992d-a30f-4b6b-8cf8-6ba1e32f77df) says use the layernorm (the birdmae transformer has learned how to handle these weird dimensions, so we should use its "solution")
3. [GPT](https://chatgpt.com/c/693d98c4-a938-832d-a8b9-a5f69a00cd30) says that Gao et al used layernorms. 

All three models recommend this paper from ~1 month ago (https://arxiv.org/pdf/2511.13981) that does PCA whitening, which is apparently this per-dimension scaling + some additional covariance matrix inverse thing to "de-correlate" features. I'll probably just say that BirdMAE should use layernorm2 and call it a day, but testing the per-dim scaling approach might be a good task for a master's/undergrad student.

