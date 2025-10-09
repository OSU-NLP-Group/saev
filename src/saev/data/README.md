# Data Loading Architecture

This directory contains the data loading infrastructure for SAE training and evaluation. The system is designed to efficiently handle large-scale activation datasets stored as binary shards.

## Overview

The data pipeline consists of three main stages:

1. **Activation Generation** (`writers.py`): Extracts and saves ViT activations to disk
2. **Data Storage**: Binary shards with metadata for efficient access
3. **Data Loading**: Multiple dataloader implementations for different use cases

## File Structure

```
data/
├── __init__.py          # Re-exports main classes
├── writers.py           # Activation extraction and shard writing
├── indexed.py           # Random-access dataset for training
├── ordered.py           # Sequential dataloader for evaluation  
├── iterable.py          # Legacy shuffled dataloader
├── images.py            # Image dataset configurations
├── performance.md       # Performance analysis and design decisions
└── README.md           # This file
```

## Key Concepts

### Shape Suffix Convention

Following [Noam Shazeer's recommendation](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd), tensors are annotated with shape suffixes:

- `B`: Batch size
- `W`: Width in patches (typically 14 or 16)
- `H`: Height in patches (typically 14 or 16)
- `D`: ViT activation dimension (typically 768 or 1024)
- `S`: SAE latent dimension (768 × 16, etc)
- `L`: Number of latents being manipulated
- `C`: Number of classes (e.g., 151 for ADE20K)

Example: `acts_BWHD` has shape `(batch, width, height, d_vit)`

### Shards

Activations are stored in binary files called "shards":
- Each shard contains activations for multiple images
- Format: `acts{NNNNNN}.bin` (e.g., `acts000000.bin`)
- Accompanied by `metadata.json` and `shards.json`
- Uses numpy's binary format for efficient memory-mapped access

### Metadata Files

1. **metadata.json**: Global dataset information
   - Model architecture details (family, checkpoint, layers)
   - Dataset dimensions (n_imgs, n_patches_per_img, d_vit)
   - Storage parameters (max_tokens_per_shard)

2. **shards.json**: Per-shard information
   - List of shards with actual image counts
   - Critical for handling non-uniform shard sizes

### Coordinate System

Each activation is identified by:
- `image_i`: Global image index (0 to n_imgs-1)
- `patch_i`: Patch index within the image (0 to n_patches_per_img-1)
- `layer`: ViT layer index

## Dataloader Implementations

### 1. Indexed Dataset (`indexed.py`)

**Use case**: Training with random sampling
- Provides random access to individual activations
- Supports shuffling via PyTorch's DataLoader
- Memory-efficient through memory-mapped files
- Best for: SAE training where random sampling is desired

```python
from saev.data.indexed import Config, Dataset

cfg = Config(shard_root="./shards", layer=13)
dataset = Dataset(cfg)
# Use with torch.utils.data.DataLoader for batching
```

### 2. Ordered DataLoader (`ordered.py`)

**Use case**: Sequential evaluation and debugging
- Guarantees strict sequential order
- Single-threaded design for simplicity
- Process-based with ring buffer for I/O overlap
- Best for: Reproducible evaluation, debugging, visualization

```python
from saev.data.ordered import Config, DataLoader

cfg = Config(shard_root="./shards", layer=13, batch_size=4096)
dataloader = DataLoader(cfg)
for batch in dataloader:
    # Batches arrive in exact sequential order
    pass
```

### 3. Iterable DataLoader (`iterable.py`)

**Use case**: High-throughput training (legacy)
- Multi-process shuffled loading
- Complex buffer management for performance
- Being phased out in favor of indexed dataset
- Best for: Backwards compatibility

## Common Pitfalls and Solutions

### 1. Shard Distribution

**Problem**: Assuming uniform distribution of images across shards
**Solution**: Always use `shards.json` to get actual counts

```python
# Wrong: Assumes uniform distribution
imgs_per_shard = metadata.n_imgs // n_shards

# Right: Use actual distribution
shard_info = ShardInfo.load(shard_root)
imgs_in_shard = shard_info[shard_idx].n_imgs
```

### 2. Index Calculations

**Problem**: Incorrect mapping from global to local indices
**Solution**: Use cumulative offsets

```python
# Calculate cumulative image offsets
cumulative_imgs = [0]
for shard in shard_info:
    cumulative_imgs.append(cumulative_imgs[-1] + shard.n_imgs)

# Find shard for global image index
for i in range(len(cumulative_imgs) - 1):
    if cumulative_imgs[i] <= global_img_i < cumulative_imgs[i + 1]:
        shard_i = i
        local_img_i = global_img_i - cumulative_imgs[i]
        break
```

### 3. CLS Token Handling

**Problem**: Forgetting to account for CLS token in patch indices
**Solution**: Add offset when CLS token is present

```python
# When accessing patches in storage
patch_idx_with_cls = patch_i + int(metadata.cls_token)
activation = mmap[img_i, layer_i, patch_idx_with_cls]
```

## Performance Tips

1. **Batch Size**: Larger batches improve throughput but use more memory
   - Typical sizes: 1024-16384 for training, 4096 for evaluation

2. **Buffer Size**: Controls read-ahead in ordered dataloader
   - Default: 64 batches
   - Increase for more aggressive prefetching

3. **Memory Mapping**: Files are memory-mapped, not loaded
   - OS handles caching automatically
   - Sequential access is highly optimized

4. **Process Spawning**: Use `spawn` start method for multiprocessing
   ```python
   import torch.multiprocessing as mp
   mp.set_start_method("spawn", force=True)
   ```

## Testing

Run tests with activation data:
```bash
pytest tests/test_*.py --shards /path/to/shards/
```

Key test files:
- `test_ordered_dataloader.py`: Ordered dataloader tests
- `test_indexed.py`: Indexed dataset tests
- `test_iterable_dataloader.py`: Legacy dataloader tests

## Future Improvements

1. **Async I/O**: Investigate io_uring or similar for better performance
2. **Compression**: Add optional compression for disk space savings
3. **Distributed Loading**: Support for multi-node training
4. **Streaming**: Direct loading from cloud storage

## Debugging

Enable debug logging:
```python
cfg = Config(..., debug=True)
```

Check shard contents:
```python
import numpy as np
from saev.data.writers import Metadata

metadata = Metadata.load(shard_root)
shape = metadata.shard_shape  # (n_imgs, n_layers, n_patches, d_vit)
mmap = np.memmap("acts000000.bin", mode="r", dtype=np.float32, shape=shape)
print(f"Shard shape: {shape}")
print(f"First activation: {mmap[0, 0, 0, :5]}")  # First 5 dims
```
