# FoldExplorer Residue Embedding

We retrained a residue-level representation model following the original
FoldExplorer training strategy and datasets.

Given a folder of protein structure files, the inference pipeline extracts
residue-level embeddings by integrating both structural and sequence information.

For each structure file, the pipeline performs:

1. ESM-IF residue embedding extraction from the 3D structure;
2. ESM2 residue embedding extraction from the corresponding amino acid sequence;
3. FoldExplorer-style feature fusion to generate the final residue-level representation.

## Installation

Python 3.9 or newer is recommended.

```bash
pip install -r requirements.txt
```

The first run may download the ESM2 and ESM-IF model weights through
`fair-esm`. A CUDA GPU is recommended.

## Model Weights

The FoldExplorer residue embedding model weights are not included in this
repository.

Please download the pretrained checkpoint from: [Download checkpoint.pt](http://www.csbio.sjtu.edu.cn/bioinf/FoldExplorer/ResidueBasedEmbedding/checkpoint.pt). After downloading, place the checkpoint at:
```text
weights/checkpoint.pt
```

## Usage

Run commands from this folder:

```bash
python infer.py \
  --structure-dir path/to/structures \
  --output-dir outputs/example \
  --device cuda
```

Supported structure extensions are `.pdb`, `.ent`, `.cif`, `.mmcif`, and their
`.gz` variants.

Useful options:

- `--recursive`: scan subfolders too.
- `--chain A`: extract one chain from every structure.
- `--shard-size 500`: number of structures to process per output file.
- `--batch-size 8`: FoldExplorer batch size.
- `--toks-per-batch 4096`: ESM2 token budget; lower it if GPU memory is tight.

## Outputs

The output directory contains compressed numpy shards:

```text
foldexplorer_residue_embeddings_000001.npz
foldexplorer_residue_embeddings_000002.npz
manifest.json
failed_structures.tsv  # only when some structures fail
```

Each `.npz` file contains:

- `ids`: structure IDs in this shard.
- `paths`: original structure file paths.
- `lengths`: residue counts.
- `embedding_000000`, `embedding_000001`, ...: numpy arrays with shape
  `[L, 256]`, one array per structure.

Embeddings are saved as `float16` by default to reduce file size. The full
pipeline always computes ESM2, ESM-IF, and FoldExplorer embeddings before
saving the final numpy arrays.

Load one shard:

```python
import numpy as np

data = np.load("outputs/example/foldexplorer_residue_embeddings_000001.npz")
ids = data["ids"]
first_embedding = data["embedding_000000"]  # shape: [L, 256]
```
