from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from esm_embeddings import ESM2Extractor
from esm_embeddings import ESMIFExtractor
from esm_embeddings import StructureFailure
from esm_embeddings import collect_structure_files
from esm_embeddings import strip_structure_suffix
from model import ResidueBasedEmbedModel
from model import encode_batch
from model import load_encoder_checkpoint


DEFAULT_CONFIG = Path(__file__).with_name("checkpoint_config.json")


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def write_failures(path: Path, failures: list[StructureFailure]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("structure_id\tpath\terror\n")
        for failure in failures:
            error = str(failure.error).replace("\t", " ").replace("\r", " ").replace("\n", " ")
            handle.write(f"{failure.structure_id}\t{failure.path}\t{error}\n")


def make_structure_ids(structure_paths: list[Path], structure_dir: Path) -> dict[Path, str]:
    ids: dict[Path, str] = {}
    counts: dict[str, int] = {}
    for path in structure_paths:
        try:
            rel_name = path.relative_to(structure_dir).as_posix()
            base_id = strip_structure_suffix(rel_name).replace("/", "__")
        except ValueError:
            base_id = strip_structure_suffix(path.name)

        count = counts.get(base_id, 0)
        counts[base_id] = count + 1
        ids[path] = base_id if count == 0 else f"{base_id}__{count + 1}"
    return ids


def batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def collate_foldexplorer_batch(
    items: list[tuple[str, torch.Tensor, torch.Tensor]],
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    ids = [item_id for item_id, _, _ in items]
    seq_list = [seq_emb for _, seq_emb, _ in items]
    struc_list = [struc_emb for _, _, struc_emb in items]
    lengths = [seq_emb.shape[0] for seq_emb in seq_list]

    seq_pad = pad_sequence(seq_list, batch_first=True)
    struc_pad = pad_sequence(struc_list, batch_first=True)
    padding_mask = torch.arange(seq_pad.size(1))[None, :] >= torch.tensor(lengths)[:, None]
    return ids, seq_pad, struc_pad, padding_mask, lengths


@torch.no_grad()
def run_foldexplorer_residue(
    model: ResidueBasedEmbedModel,
    seq_embeddings: dict[str, torch.Tensor],
    struc_embeddings: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    common_ids = sorted(set(seq_embeddings).intersection(struc_embeddings))
    if not common_ids:
        return {}

    items: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for structure_id in common_ids:
        seq_emb = seq_embeddings[structure_id]
        struc_emb = struc_embeddings[structure_id]
        if seq_emb.shape[0] != struc_emb.shape[0]:
            raise ValueError(
                f"{structure_id} has mismatched lengths: "
                f"ESM2={seq_emb.shape[0]}, ESM-IF={struc_emb.shape[0]}"
            )
        items.append((structure_id, seq_emb.float(), struc_emb.float()))

    outputs: dict[str, np.ndarray] = {}
    num_batches = (len(items) + batch_size - 1) // batch_size
    for batch in tqdm(batched(items, batch_size), total=num_batches, desc="Running FoldExplorer", leave=False):
        ids, seq_pad, struc_pad, padding_mask, lengths = collate_foldexplorer_batch(batch)
        residue_output = encode_batch(
            model,
            seq_pad.to(device),
            struc_pad.to(device),
            padding_mask.to(device),
        ).cpu()

        for index, structure_id in enumerate(ids):
            outputs[structure_id] = residue_output[index, : lengths[index]].contiguous().numpy()

    return outputs


def output_dtype(name: str) -> np.dtype:
    if name == "float16":
        return np.dtype(np.float16)
    if name == "float32":
        return np.dtype(np.float32)
    raise ValueError("--output-dtype must be float16 or float32.")


def save_npz_shard(
    path: Path,
    embeddings: dict[str, np.ndarray],
    source_paths: dict[str, str],
    dtype: np.dtype,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = sorted(embeddings)
    payload: dict[str, np.ndarray] = {
        "ids": np.asarray(ids, dtype=str),
        "paths": np.asarray([source_paths[item_id] for item_id in ids], dtype=str),
        "lengths": np.asarray([embeddings[item_id].shape[0] for item_id in ids], dtype=np.int32),
    }
    for index, item_id in enumerate(ids):
        payload[f"embedding_{index:06d}"] = embeddings[item_id].astype(dtype, copy=False)

    np.savez_compressed(path, **payload)
    return {
        "file": str(path),
        "count": len(ids),
        "ids": ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract residue-level FoldExplorer embeddings from a folder of protein structures.",
    )
    parser.add_argument("--structure-dir", required=True, type=Path, help="Folder containing structure files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder for .npz embedding shards.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--recursive", action="store_true", help="Scan structure-dir recursively.")
    parser.add_argument("--chain", default=None, help="Optional chain id to extract from every structure.")
    parser.add_argument("--shard-size", type=int, default=500, help="Number of structures per output .npz file.")
    parser.add_argument("--batch-size", type=int, default=8, help="FoldExplorer batch size.")
    parser.add_argument("--toks-per-batch", type=int, default=4096, help="ESM2 token budget per batch.")
    parser.add_argument("--checkpoint", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=argparse.SUPPRESS)
    parser.add_argument("--min-residues", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--output-dtype", choices=["float16", "float32"], default="float16", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args.structure_dir.is_dir():
        raise NotADirectoryError(f"{args.structure_dir} is not a directory.")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be greater than 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    if args.toks_per_batch <= 0:
        raise ValueError("--toks-per-batch must be greater than 0.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = output_dtype(args.output_dtype)

    config = read_config(args.config)
    checkpoint = args.checkpoint or (args.config.parent / config["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FoldExplorer checkpoint not found: {checkpoint}")

    structure_paths = collect_structure_files(args.structure_dir, recursive=args.recursive)
    if not structure_paths:
        raise FileNotFoundError(f"No supported structure files found in {args.structure_dir}.")
    structure_ids = make_structure_ids(structure_paths, args.structure_dir)

    foldexplorer = load_encoder_checkpoint(
        str(checkpoint),
        device=device,
        config=config.get("model", {}),
        state_key=config.get("state_key", "encoder_q"),
    )
    esm_if = ESMIFExtractor(device=device, chain=args.chain, min_residues=args.min_residues)
    esm2 = ESM2Extractor(device=device, toks_per_batch=args.toks_per_batch)

    all_failures: list[StructureFailure] = []
    shard_records: list[dict[str, Any]] = []
    embedded_count = 0

    total_shards = (len(structure_paths) + args.shard_size - 1) // args.shard_size
    for shard_index, shard_paths in enumerate(
        tqdm(
            batched(structure_paths, args.shard_size),
            total=total_shards,
            desc="Processing shards",
        ),
        start=1,
    ):
        esm_if_records, failures = esm_if.extract(shard_paths, structure_ids=structure_ids)
        all_failures.extend(failures)
        if not esm_if_records:
            continue

        sequences = {structure_id: record.sequence for structure_id, record in esm_if_records.items()}
        struc_embeddings = {structure_id: record.embedding for structure_id, record in esm_if_records.items()}
        source_paths = {structure_id: record.path for structure_id, record in esm_if_records.items()}
        seq_embeddings = esm2.extract(sequences)
        fold_embeddings = run_foldexplorer_residue(
            foldexplorer,
            seq_embeddings,
            struc_embeddings,
            device=device,
            batch_size=args.batch_size,
        )
        if not fold_embeddings:
            continue

        shard_path = output_dir / f"foldexplorer_residue_embeddings_{len(shard_records) + 1:06d}.npz"
        shard_info = save_npz_shard(shard_path, fold_embeddings, source_paths, dtype=dtype)
        shard_records.append(shard_info)
        embedded_count += shard_info["count"]

        del esm_if_records, seq_embeddings, struc_embeddings, fold_embeddings
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if all_failures:
        write_failures(output_dir / "failed_structures.tsv", all_failures)

    manifest = {
        "structure_dir": str(args.structure_dir),
        "num_structure_files_found": len(structure_paths),
        "num_structures_embedded": embedded_count,
        "num_failed_structures": len(all_failures),
        "embedding_level": "residue",
        "embedding_dim": config.get("checkpoint_info", {}).get("embedding_dim", 256),
        "output_dtype": str(dtype),
        "shard_size": args.shard_size,
        "shards": shard_records,
    }
    save_json(output_dir / "manifest.json", manifest)

    if embedded_count == 0:
        raise RuntimeError("No FoldExplorer embeddings were generated. See failed_structures.tsv if it exists.")
    print(f"Saved {embedded_count} residue-level FoldExplorer embeddings to {output_dir}")


if __name__ == "__main__":
    main()
