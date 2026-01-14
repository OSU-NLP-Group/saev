# Freshwater Fish from TreeOfLife-200M

Extract images from TreeOfLife-200M into ImageFolder format for use with saev.

## Usage

### All fish by order (~2M images from GBIF)

```bash
uv run python contrib/freshwater_fish/scripts/extract_tol.py \
    --order-filter Cypriniformes Perciformes Siluriformes Salmoniformes \
    --output-dpath data/fish \
    --slurm-acct PAS2136 \
    --n-hours 12
```

### Specific families (from taxa file)

```bash
uv run python contrib/freshwater_fish/scripts/extract_tol.py \
    --taxa-file path/to/freshwater_taxa.csv \
    --output-dpath data/freshwater-fish \
    --slurm-acct PAS2136
```

### All images (no filtering)

```bash
uv run python contrib/freshwater_fish/scripts/extract_tol.py \
    --output-dpath data/all-images \
    --slurm-acct PAS2136 \
    --n-hours 24
```

## Filtering Options

- **`--order-filter`**: Filter by taxonomic orders (recommended for GBIF fish)
- **`--class-filter`**: Filter by taxonomic class (works for EOL, not GBIF)
- **`--taxa-file`**: Filter by specific families/genera/species from CSV/parquet
- **No filters**: Extract everything
- **`--sources`**: Which TreeOfLife sources to include (default: all)

**Note**: GBIF has ~2M fish images but the `class` column is null. Use `--order-filter` for fish.

Common fish orders: `Cypriniformes`, `Perciformes`, `Siluriformes`, `Characiformes`, `Cichliformes`, `Salmoniformes`, `Tetraodontiformes`, `Pleuronectiformes`, `Anguilliformes`, `Clupeiformes`

## Taxa File Format

CSV or parquet with any combination of columns: `family`, `genus`, `species`.

```csv
family
Cichlidae
Characidae
Cyprinidae
```

## Output

ImageFolder structure: `output_dpath/{label}/{uuid}.jpg`

Use with saev:
```python
from saev.data import ImgFolder
cfg = ImgFolder(root=pathlib.Path("data/freshwater-fish"))
```
