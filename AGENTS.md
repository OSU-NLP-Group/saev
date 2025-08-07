# Conventions

This document outlines some programming conventions that are not caught by automated tools.

* File descriptors from `open()` are called `fd`.
* Use types where possible, including `jaxtyping` hints.
* Decorate functions with `beartype.beartype` unless they use a `jaxtyping` hint, in which case use `jaxtyped(typechecker=beartype.beartype)`.
* Variables referring to a filepath should be suffixed with `_fpath`. Directories are `_dpath`.
* Prefer `make` over `build` when naming functions that construct objects, and use `get` when constructing primitives (like string paths or config values).
* Only use `setup` for naming functions that don't return anything.

Throughout the code, variables are annotated with shape suffixes, as [recommended by Noam Shazeer](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd).

The key for these suffixes:

* B: batch size
* W: width in patches (typically 14 or 16)
* H: height in patches (typically 14 or 16)
* D: ViT activation dimension (typically 768 or 1024)
* S: SAE latent dimension (768 x 16, etc)
* L: Number of latents being manipulated at once (typically 1-5 at a time)
* C: Number of classes in ADE20K (151)

For example, an activation tensor with shape (batch, width, height d_vit) is `acts_BWHD`.

# Running Code

Run code with `uv run python <args>` or `uv run <scriptname.py> <args>` to run Python code.
