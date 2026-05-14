from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from typing import Optional
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidueBasedEmbedModel(nn.Module):
    """Fuse residue-level ESM2 and ESM-IF embeddings."""

    def __init__(
        self,
        seq_dim: int = 1280,
        struc_dim: int = 512,
        hidden_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_proj = nn.Sequential(
            nn.LayerNorm(seq_dim),
            nn.Linear(seq_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.struc_proj = nn.Sequential(
            nn.LayerNorm(struc_dim),
            nn.Linear(struc_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.fuse_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        seq_emb: torch.Tensor,
        struc_emb: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_h = self.seq_proj(seq_emb)
        struc_h = self.struc_proj(struc_emb)
        h = torch.cat([seq_h, struc_h], dim=-1)
        h = self.fuse_proj(h)
        h = self.transformer(h, src_key_padding_mask=padding_mask)
        return self.layer_norm(h)


def build_model(config: Optional[Mapping[str, Any]] = None) -> ResidueBasedEmbedModel:
    cfg = dict(config or {})
    return ResidueBasedEmbedModel(
        seq_dim=int(cfg.get("seq_dim", 1280)),
        struc_dim=int(cfg.get("struc_dim", 512)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        n_layers=int(cfg.get("n_layers", 2)),
        n_heads=int(cfg.get("n_heads", 8)),
        dropout=float(cfg.get("dropout", 0.1)),
    )


def load_encoder_checkpoint(
    checkpoint_path: str,
    device: Union[torch.device, str] = "cpu",
    config: Optional[Mapping[str, Any]] = None,
    state_key: str = "encoder_q",
) -> ResidueBasedEmbedModel:
    model = build_model(config).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, Mapping) and state_key in checkpoint:
        state_dict = checkpoint[state_key]
    elif isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def encode_batch(
    model: ResidueBasedEmbedModel,
    seq_emb: torch.Tensor,
    struc_emb: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    residue_emb = model(seq_emb, struc_emb, padding_mask)
    return F.normalize(residue_emb, dim=-1)
