from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm


SUPPORTED_STRUCTURE_EXTENSIONS = {
    ".pdb",
    ".ent",
    ".cif",
    ".mmcif",
    ".pdb.gz",
    ".ent.gz",
    ".cif.gz",
    ".mmcif.gz",
}


NON_CANONICAL_RESIDUES = {
    "CME": "C",
    "CSO": "C",
    "CSD": "C",
    "OCS": "C",
    "CSS": "C",
    "CSX": "C",
    "CAS": "C",
    "YCM": "C",
    "CRQ": "C",
    "CSB": "C",
    "CS4": "C",
    "CSW": "C",
    "CYG": "C",
    "CZZ": "C",
    "CSR": "C",
    "CSP": "C",
    "CMH": "C",
    "OCY": "C",
    "SMC": "C",
    "MSE": "M",
    "MLY": "M",
    "MLZ": "M",
    "FME": "M",
    "M3L": "M",
    "MSO": "M",
    "MHO": "M",
    "MEN": "M",
    "MME": "M",
    "MHS": "M",
    "MGN": "M",
    "MDO": "M",
    "KOR": "K",
    "KCX": "K",
    "LLP": "K",
    "LYR": "K",
    "KPI": "K",
    "PYL": "K",
    "TRN": "R",
    "TRQ": "R",
    "AGM": "R",
    "SEP": "S",
    "TOX": "S",
    "SNC": "C",
    "TPO": "T",
    "PTR": "Y",
    "IYR": "Y",
    "PHD": "D",
    "DSN": "D",
    "DDE": "D",
    "DYA": "D",
    "PCA": "E",
    "NEP": "E",
    "HYP": "P",
    "PSW": "P",
    "HIQ": "H",
    "HIC": "H",
    "HSK": "H",
    "ALS": "A",
    "AYA": "A",
    "DAL": "A",
    "GL3": "G",
    "GYS": "G",
    "LCK": "L",
    "LED": "L",
    "NIY": "I",
    "CGV": "V",
    "FGL": "F",
    "BYR": "Y",
    "SCH": "S",
    "SEB": "S",
    "SEE": "S",
}


@dataclass(frozen=True)
class StructureFailure:
    structure_id: str
    path: str
    error: str


@dataclass
class ESMIFRecord:
    structure_id: str
    path: str
    sequence: str
    embedding: torch.Tensor


def strip_structure_suffix(name: str) -> str:
    lower = name.lower()
    for suffix in sorted(SUPPORTED_STRUCTURE_EXTENSIONS, key=len, reverse=True):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def structure_id_from_path(path: Path) -> str:
    return strip_structure_suffix(path.name)


def is_structure_file(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in SUPPORTED_STRUCTURE_EXTENSIONS)


def collect_structure_files(structure_dir: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in structure_dir.glob(pattern) if path.is_file() and is_structure_file(path))


def register_non_canonical_residues() -> None:
    from biotite.sequence import ProteinSequence

    ProteinSequence._dict_3to1.update(NON_CANONICAL_RESIDUES)


def residue_keys(structure) -> Iterable[tuple[Optional[str], int, Optional[str]]]:
    chain_ids = getattr(structure, "chain_id", [None] * len(structure))
    res_ids = getattr(structure, "res_id")
    ins_codes = getattr(structure, "ins_code", [None] * len(structure))

    for chain_id, res_id, ins_code in zip(chain_ids, res_ids, ins_codes):
        yield chain_id, int(res_id), ins_code


def filter_structure_to_supported_residues(structure, min_residues: int):
    from biotite.sequence import ProteinSequence
    from biotite.structure import get_residues

    _, residue_names = get_residues(structure)
    keep_by_residue = [resname in ProteinSequence._dict_3to1 for resname in residue_names]
    if sum(keep_by_residue) < min_residues:
        raise ValueError(f"fewer than {min_residues} supported residues")

    atom_mask: list[bool] = []
    residue_index = -1
    previous_key = None
    for key in residue_keys(structure):
        if key != previous_key:
            residue_index += 1
            previous_key = key
        if residue_index >= len(keep_by_residue):
            raise ValueError("residue parsing failed while filtering structure")
        atom_mask.append(keep_by_residue[residue_index])

    return structure[atom_mask]


class ESMIFExtractor:
    def __init__(
        self,
        device: torch.device,
        chain: Optional[str] = None,
        min_residues: int = 10,
    ) -> None:
        try:
            import esm
            from esm.inverse_folding.util import CoordBatchConverter
            from esm.inverse_folding.util import extract_coords_from_structure
            from esm.inverse_folding.util import load_structure
        except ImportError as exc:
            raise ImportError("Install fair-esm and biotite to extract ESM-IF embeddings.") from exc

        register_non_canonical_residues()
        self.device = device
        self.chain = chain
        self.min_residues = min_residues
        self.extract_coords_from_structure = extract_coords_from_structure
        self.load_structure = load_structure
        self.model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.model = self.model.eval().to(device)
        self.batch_converter = CoordBatchConverter(alphabet)

    @torch.no_grad()
    def extract(
        self,
        structure_paths: list[Path],
        structure_ids: Optional[dict[Path, str]] = None,
    ) -> tuple[dict[str, ESMIFRecord], list[StructureFailure]]:
        records: dict[str, ESMIFRecord] = {}
        failures: list[StructureFailure] = []

        for path in tqdm(structure_paths, desc="Extracting ESM-IF", leave=False):
            structure_id = structure_ids.get(path, structure_id_from_path(path)) if structure_ids else structure_id_from_path(path)
            try:
                if structure_id in records:
                    raise ValueError(f"duplicate structure id '{structure_id}'")
                structure = self.load_structure(str(path), chain=self.chain)
                structure = filter_structure_to_supported_residues(
                    structure,
                    min_residues=self.min_residues,
                )
                coords, sequence = self.extract_coords_from_structure(structure)
                batch = [(coords, None, None)]
                coords, confidence, _, _, padding_mask = self.batch_converter(batch, device=self.device)

                encoder_out = self.model.encoder.forward(
                    coords,
                    padding_mask,
                    confidence,
                    return_all_hiddens=False,
                )
                embedding = encoder_out["encoder_out"][0][1:-1, 0].detach().cpu().float()
                if embedding.shape[0] != len(sequence):
                    raise ValueError(
                        f"ESM-IF length mismatch: sequence={len(sequence)}, embedding={embedding.shape[0]}"
                    )
                records[structure_id] = ESMIFRecord(
                    structure_id=structure_id,
                    path=str(path),
                    sequence=sequence,
                    embedding=embedding,
                )
            except Exception as exc:
                failures.append(StructureFailure(structure_id, str(path), str(exc)))

        return records, failures


def token_batches(
    records: list[tuple[str, str]],
    toks_per_batch: int,
) -> Iterable[list[tuple[str, str]]]:
    batch: list[tuple[str, str]] = []
    max_len = 0

    for item in sorted(records, key=lambda pair: len(pair[1])):
        item_len = len(item[1])
        proposed_max_len = max(max_len, item_len)
        proposed_tokens = (len(batch) + 1) * (proposed_max_len + 2)
        if batch and proposed_tokens > toks_per_batch:
            yield batch
            batch = []
            max_len = 0

        batch.append(item)
        max_len = max(max_len, item_len)

    if batch:
        yield batch


class ESM2Extractor:
    def __init__(
        self,
        device: torch.device,
        model_name: str = "esm2_t33_650M_UR50D",
        repr_layer: int = 33,
        toks_per_batch: int = 4096,
    ) -> None:
        try:
            from esm import pretrained
        except ImportError as exc:
            raise ImportError("Install fair-esm to extract ESM2 embeddings.") from exc

        self.device = device
        self.repr_layer = repr_layer
        self.toks_per_batch = toks_per_batch
        self.model, alphabet = pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.eval().to(device)
        self.batch_converter = alphabet.get_batch_converter()

    @torch.no_grad()
    def extract(self, sequences: dict[str, str]) -> dict[str, torch.Tensor]:
        embeddings: dict[str, torch.Tensor] = {}
        batches = list(token_batches(list(sequences.items()), toks_per_batch=self.toks_per_batch))

        for batch in tqdm(batches, desc="Extracting ESM2", leave=False):
            labels, seqs, toks = self.batch_converter(batch)
            toks = toks.to(self.device)
            out = self.model(toks, repr_layers=[self.repr_layer], return_contacts=False)
            reps = out["representations"][self.repr_layer]

            for index, label in enumerate(labels):
                length = len(seqs[index])
                embeddings[label] = reps[index, 1 : length + 1].detach().cpu().float()

        return embeddings
