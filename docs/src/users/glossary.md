# Glossary

Definitions for words used in the code and documentation.

- **example**: one dataset item (image, sentence, audio clip, point cloud, graph instance).
- **token**: one model position in the encoder’s residual stream (the thing with hidden size `d_model`). Always "token" inside the model.
- **content token**: tokens derived from the raw input (image patches, wordpieces, audio windows, nodes, etc.).
- **special token**: tokens not directly derived from the raw input (class/summary token, [SEP], [MASK], [PAD], register tokens, etc.).
- **sequence length L**: total tokens per example (content + special). If variable, call it “ragged”.
- **layer**: an integer index into the encoder’s stack.
- **activation kind (optional but useful)**: which stream you saved (e.g., resid_pre, resid_post, mlp_out, attn_out, qkv, head_out).

Modality-specific vocab:

- **patch** (vision): a 2D content token. Often laid out on a grid with shape (H_patches, W_patches).
- **frame/token or tube** (video): content token in time × space; often (T, H, W).
- **wordpiece / subword** (text): content token from a tokenizer.
- **window / frame** (audio): time–frequency window.
- **node** (graph), point (point cloud).

