- Use `uv run SCRIPT.py` or `uv run python ARGS` to run python instead of Just plain `python`.
- After making edits, run `uvx ruff format --preview .` to format the file, then run `uvx ruff check --fix .` to lint, then run `uvx ty check FILEPATH` to type check (`ty` is prerelease software, and typechecking often will have false positives). Only do this if you think you're finished, or if you can't figure out a bug. Maybe linting will make it obvious. Don't fix linting or typing errors in files you haven't modified.

# Gather Context

- Public docs for developers and users are in markdown in docs/src. Internal, messier design and implementation docs are in markdown in docs/research/issues. Both are valuable sources of context when getting started.
- You can use `gh` to access issues and PRs on GitHub to gather more context. We use GitHub issues a lot to share ideas and communicate about problems, so you should almost always check to see if there's a relevant GitHub issue for whatever you're working on.

# Code Style

- Don't hard-wrap comments. Only use linebreaks for new paragraphs. Let the editor soft wrap content.
- Don't hard-wrap string literals. Keep each log or user-facing message in a single source line and rely on soft wrapping when reading it.
- Prefer negative if statements in combination with early returns/continues. Rather than nesting multiple positive if statements, just check if a condition is False, then return/continue in a loop. This reduces indentation.
- This project uses Python 3.12. You can use `dict`, `list`, `tuple` instead of the imports from `typing`. You can use `| None` instead of `Optional`.
- File descriptors from `open()` are called `fd`.
- Use types where possible, including `jaxtyping` hints.
- Decorate functions with `beartype.beartype` unless they use a `jaxtyping` hint, in which case use `jaxtyped(typechecker=beartype.beartype)`.
- Variables referring to a absolute filepath should be suffixed with `_fpath`. Filenames are `_fname`. Directories are `_dpath`.
- Prefer `make` over `build` when naming functions that construct objects, and use `get` when constructing primitives (like string paths or config values).
- Only use `setup` for naming functions that don't return anything.
- submitit and jaxtyping don't work in the same file. See [this issue]. To solve this, all jaxtyped functions/classes need to be in a different file to the submitit launcher script.
- Never create a simple script to demonstrate functionality unless explicitly asked..
- Write single-line commit messages; never say you co-authored a commit.
- Before committing, run `git status` to check for already-staged files. If asked to commit only specific files, unstage everything first, then stage only the requested files, then after the commit, restage the already-staged files.
- Only use ascii characters. If you would use unicode to represent math, use pseudo-LaTeX instead in comments: 10⁶ should be 10^6, 3×10⁷ should be 3x10^7.
- Prefix variables with `n_` for totals and cardinalities, but ignore it for dimensions `..._per_...` and dimensions. Examples: `n_examples`, `n_models`, but `tokens_per_example`, `examples_per_shard`
- Try to keep code short. Shorter code is in principle easier to read. If variable names are really long, shorten based on conventions in this codebase (..._indices -> ..._i). Since you use `uvx ruff format --preview`, if you can make a small variable name change to fit everything on one line, that's a good idea. When variables are used once, simply inline it.
- If you make edits to a file and notice that I made edits to your edits, note the changes I make compared to your initial version and explicitly describe the style of changes. Keep these preferences in mind as you write the rest of the code.

# Defensive Programming

- Consider the [style guidelines for TigerBeetle](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md) and adapt it to Python.
- Use asserts to validate assumptions frequently. For example, I didn't have an assert here at first because I assumed the shape couldn't change. It turns out it can! So now we have an assert to make it clear that we expect the input and output shapes are identical.
```py
def sp_csr_to_pt(csr: scipy.sparse.csr_matrix, *, device: str) -> Tensor:
    shape_sp = csr.shape
    pt = torch.sparse_csr_tensor(
        csr.indptr,
        csr.indices,
        csr.data,
        size=shape_sp,
        device=device,
    )
    # MISSING
    assert pt.shape == shape_sp, f"{tuple(pt.shape)} != {tuple(shape_sp)}"
    return pt
```
- Use asserts rather than if statements + errors:
```py
# Bad.
train_token_acts_fpath = train_inference_dpath / "token_acts.npz"
if not train_token_acts_fpath.exists():
    msg = f"Train SAE activations missing: '{train_token_acts_fpath}'. Run inference.py."
    logger.error(msg)
    raise FileNotFoundError(msg)

# Good.
train_token_acts_fpath = train_inference_dpath / "token_acts.npz"
msg = f"Train SAE acts missing: '{train_token_acts_fpath}'. Run inference.py."
assert train_token_acts_fpath.exists(), msg
```

# No hacks: ask for help instead

Due to the difficulty of implementing this codebase, we must strive to keep the code high quality, clean, modular, simple and functional; more like an Agda codebase, less like a C codebase.
Hacks and duct tape must be COMPLETELY AVOIDED, in favor of robust, simple and general solutions.
In some cases, you will be asked to perform a seemingly impossible task, either because it is (and the developer is unaware), or because you don't grasp how to do it properly.
In these cases, do not attempt to implement a half-baked solution just to satisfy the developer's request.
If the task seems too hard, be honest that you couldn't solve it in the proper way, leave the code unchanged, explain the situation to the developer and ask for further feedback and clarifications.
The developer is a domain expert that will be able to assist you in these cases.

# Tensor Variables

Throughout the code, variables are annotated with shape suffixes, as [recommended by Noam Shazeer](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd).

The key for these suffixes:

- b: batch size
- w: width in patches (typically 14 or 16)
- h: height in patches (typically 14 or 16)
- d: Transformer activation dimension (typically 768 or 1024)
- s: SAE latent dimension (1024 x 16, etc)
- l: Number of latents being manipulated at once (typically 1-5 at a time)
- c: Number of classes

For example, an activation tensor with shape (batch, width, height d_vit) is `acts_bwhd`.
