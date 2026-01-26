# Shard Validation Specification

## Problem

The filesystem (e.g., scratch storage on HPC clusters) may silently delete files that haven't been accessed in 30-90 days. When this happens, shard files are completely removed from disk. Training jobs that start without these files will fail with confusing errors deep in the data loading pipeline.

## Solution

Add eager validation at data loader initialization that checks all binary shard files:
1. Exist on disk
2. Are regular files (not directories)
3. Have non-zero size
4. Are readable

## Requirements

### Functional

1. **Validation timing**: Eager validation during `__init__` of all three loaders (ShuffledDataLoader, IndexedDataset, OrderedDataLoader)

2. **Files to validate**: Only binary shard files (`acts*.bin`) listed in `shards.json`. Metadata files (`metadata.json`, `shards.json`) are already validated by the JSON parsing step.

3. **Validation checks** (using `os.stat()`):
   - File exists
   - Path is a regular file (not a directory or other special file)
   - File has non-zero size (`st_size > 0`)
   - File is readable (catch `PermissionError` / `OSError`)

4. **Error reporting**:
   - Collect ALL problematic files before raising an error
   - Raise `FileNotFoundError` with a message listing all problematic files
   - Group by problem type: missing, empty, unreadable, not-a-file
   - Use **absolute paths** in error messages to avoid ambiguity

5. **No skip option**: Always validate. Safety is more important than startup time.

### Non-functional

1. **Performance**: Sequential `stat()` calls are acceptable for 100-1000 shards (~1-2 seconds)

2. **Code organization**: Add a `validate()` method on the `ShardInfo` class in `shards.py`. All three loaders call this shared method.

## API Design

```python
# In shards.py

@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ShardInfo:
    shards: list[Shard] = dataclasses.field(default_factory=list)

    # ... existing methods ...

    def validate(self, shards_dir: pathlib.Path | str) -> None:
        """
        Validate that all shard files exist, are regular files, and have non-zero size.

        Args:
            shards_dir: Path to the shards directory containing the shard files.
                Accepts str or Path; normalized to Path internally.

        Raises:
            FileNotFoundError: If any shard files are missing, empty, unreadable,
                or not regular files. The error message lists all problematic files
                grouped by problem type, using absolute paths.
        """
```

### Error Message Format

```
Shard validation failed in '/scratch/user/saev/shards/abc12345':

Missing files (2):
  - /scratch/user/saev/shards/abc12345/acts000003.bin
  - /scratch/user/saev/shards/abc12345/acts000007.bin

Empty files (1):
  - /scratch/user/saev/shards/abc12345/acts000012.bin

Unreadable files (1):
  - /scratch/user/saev/shards/abc12345/acts000015.bin

Not regular files (1):
  - /scratch/user/saev/shards/abc12345/acts000020.bin
```

## Implementation Changes

### shards.py

Add `validate()` method to `ShardInfo` class:

```python
def validate(self, shards_dir: pathlib.Path | str) -> None:
    shards_dir = pathlib.Path(shards_dir)

    missing: list[str] = []
    empty: list[str] = []
    unreadable: list[str] = []
    not_file: list[str] = []

    for shard in self.shards:
        fpath = shards_dir / shard.name
        abs_path = str(fpath.resolve())

        try:
            stat = fpath.stat()
        except FileNotFoundError:
            missing.append(abs_path)
            continue
        except (PermissionError, OSError):
            unreadable.append(abs_path)
            continue

        if not stat.st_mode & 0o100000:  # S_IFREG - regular file check
            not_file.append(abs_path)
        elif stat.st_size == 0:
            empty.append(abs_path)

    # Build error message if any problems found
    if missing or empty or unreadable or not_file:
        lines = [f"Shard validation failed in '{shards_dir}':", ""]

        if missing:
            lines.append(f"Missing files ({len(missing)}):")
            lines.extend(f"  - {p}" for p in missing)
            lines.append("")

        if empty:
            lines.append(f"Empty files ({len(empty)}):")
            lines.extend(f"  - {p}" for p in empty)
            lines.append("")

        if unreadable:
            lines.append(f"Unreadable files ({len(unreadable)}):")
            lines.extend(f"  - {p}" for p in unreadable)
            lines.append("")

        if not_file:
            lines.append(f"Not regular files ({len(not_file)}):")
            lines.extend(f"  - {p}" for p in not_file)

        raise FileNotFoundError("\n".join(lines))
```

Note: Use `import stat as stat_module` and `stat_module.S_ISREG(st.st_mode)` for cleaner regular file check.

### shuffled.py, indexed.py, ordered.py

Replace the existing inline validation loop with a call to `shard_info.validate(shards_dir)`.

Current pattern (to be replaced):
```python
shard_info = shards.ShardInfo.load(self.cfg.shards)
for shard in shard_info:
    shard_path = os.path.join(self.cfg.shards, shard.name)
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Shard file not found: {shard_path}")
```

New pattern:
```python
shard_info = shards.ShardInfo.load(self.cfg.shards)
shard_info.validate(self.cfg.shards)
```

## Test Plan

Create `tests/test_shard_validation.py` with real temporary files using pytest's `tmp_path` fixture.

### Test Cases

1. **test_validate_all_shards_present_and_nonempty**: Create valid shards directory with metadata.json, shards.json, and non-empty acts*.bin files. Validation should pass silently.

2. **test_validate_missing_shard_file**: Create shards directory but delete one of the shard files listed in shards.json. Should raise FileNotFoundError with absolute path of missing file.

3. **test_validate_empty_shard_file**: Create shards directory with one shard file truncated to 0 bytes. Should raise FileNotFoundError mentioning the empty file.

4. **test_validate_multiple_problems**: Create shards directory with one missing file and one empty file. Error message should list both problems with absolute paths.

5. **test_validate_no_shards**: Edge case where shards.json lists zero shards. Should pass (nothing to validate).

6. **test_validate_unreadable_file**: Create shard file then `chmod 0` to make it unreadable. Should raise FileNotFoundError listing it under "Unreadable files". (Skip on Windows or if running as root.)

7. **test_validate_directory_instead_of_file**: Create a directory named `acts000000.bin` instead of a file. Should raise FileNotFoundError listing it under "Not regular files".

8. **test_validate_accepts_str_path**: Pass `shards_dir` as a `str` instead of `Path`. Should work identically to `Path` input.

### Test Fixtures

Create helper to build minimal valid shards directory:
- `metadata.json` with minimal valid Metadata
- `shards.json` listing shard entries
- `acts*.bin` files with some non-zero content

```python
@pytest.fixture
def make_shards_dir(tmp_path):
    """Factory fixture that creates a shards directory structure."""
    def _make(n_shards: int = 3, *, empty_shards: list[int] = None, missing_shards: list[int] = None):
        # Create directory structure matching disk.is_shards_dir() expectations
        shards_dir = tmp_path / "saev" / "shards" / "abc12345"
        shards_dir.mkdir(parents=True)

        # Write minimal metadata.json
        # Write shards.json with n_shards entries
        # Create acts*.bin files (skip missing_shards, truncate empty_shards)

        return shards_dir
    return _make
```

## Out of Scope

- Validating exact file sizes based on expected shard dimensions
- Detecting partially-written/corrupted shards
- Parallel validation for large datasets
- Skip/disable option for validation
- Validating labels.bin (optional file, already handled separately)
- Retry logic for transient filesystem errors (ESTALE, EIO)
- Touch-read mode to force data fetch on lazy NFS
