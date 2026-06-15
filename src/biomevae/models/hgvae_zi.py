"""DEPRECATED: HGVAE_ZI graph autoencoder.

The HGVAE_ZI model is no longer recommended; use :class:`TreeDTMVAE`
instead.

The shared taxonomy-graph utilities (:class:`TaxonomyGraph`,
:func:`build_taxonomy_graph_from_phyla_tsv`,
:func:`load_sgb_table_as_samples_by_leaf`,
:func:`build_internal_sums_vector`, :func:`parse_phylarow_to_lineage`,
:data:`RANK_PREFIX_RE`) live in :mod:`biomevae.models.taxonomy_tree` and are
re-exported here for backwards compatibility with old configs/checkpoints.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # optional dependency
    from torch_geometric.data import Data, Dataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "HGVAE_ZI requires torch_geometric. Install with `pip install torch_geometric`."
    ) from exc

from biomevae.models.taxonomy_tree import (
    RANK_PREFIX_RE,
    TaxonomyGraph,
    _strip_prefix,
    build_internal_sums_vector,
    build_taxonomy_graph_from_phyla_tsv,
    load_sgb_table_as_samples_by_leaf,
    parse_phylarow_to_lineage,
)


class TaxonomyGraphDataset(Dataset):
    def __init__(self, Xdf_samples_by_sgb: pd.DataFrame, taxg: TaxonomyGraph, *, eps: float = 1e-6):
        super().__init__()
        self.X = Xdf_samples_by_sgb
        self.taxg = taxg
        self.sample_ids = self.X.index.tolist()
        self.sgb_ids = self.X.columns.tolist()
        self.eps = float(eps)

        missing = [sgb for sgb in self.sgb_ids if sgb not in self.taxg.leaf_name_to_id]
        if missing:
            raise ValueError(f"{len(missing)} SGBs in table missing in phyla.tsv. Example: {missing[:10]}")

        self.leaf_ids = list(self.taxg.leaf_ids)
        col_index = {sgb: j for j, sgb in enumerate(self.sgb_ids)}
        self.leaf_col_index = [col_index.get(self.taxg.node_names[nid], None) for nid in self.leaf_ids]

        self.edge_index = self.taxg.edge_index
        self.node_type = self.taxg.node_type
        self.node_rank = self.taxg.node_rank

    def len(self) -> int:
        return len(self.sample_ids)

    def get(self, idx: int) -> Data:
        sid = self.sample_ids[idx]
        row = self.X.loc[sid].to_numpy(dtype=np.float32)

        leaf_abund = np.zeros(len(self.leaf_ids), dtype=np.float32)
        for i, j in enumerate(self.leaf_col_index):
            if j is not None:
                leaf_abund[i] = row[j]

        x_all = build_internal_sums_vector(self.taxg, leaf_abund, self.leaf_ids)
        x_in = np.log(x_all + self.eps).astype(np.float32)
        return Data(
            x=torch.from_numpy(x_in).view(-1, 1),
            y=torch.from_numpy(x_all).view(-1, 1),
            edge_index=self.edge_index,
            node_type=self.node_type,
            node_rank=self.node_rank,
            sample_idx=torch.tensor([idx], dtype=torch.long),
        )


class GraphEncoder(nn.Module):
    def __init__(self, hidden: int, latent_dim: int, rank_vocab: int, rank_emb_dim: int = 16):
        super().__init__()
        self.rank_emb = nn.Embedding(rank_vocab, rank_emb_dim)
        self.type_emb = nn.Embedding(2, 8)
        self.conv1 = SAGEConv(1 + rank_emb_dim + 8, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.mu = nn.Linear(hidden * 2, latent_dim)
        self.logvar = nn.Linear(hidden * 2, latent_dim)

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([data.x, self.rank_emb(data.node_rank), self.type_emb(data.node_type)], dim=-1)
        h = F.relu(self.conv1(h, data.edge_index))
        h = F.relu(self.conv2(h, data.edge_index))
        hg = torch.cat([global_mean_pool(h, data.batch), global_max_pool(h, data.batch)], dim=-1)
        return self.mu(hg), self.logvar(hg)


class ZILogNormalGraphDecoder(nn.Module):
    def __init__(self, hidden: int, latent_dim: int, rank_vocab: int, rank_emb_dim: int = 16):
        super().__init__()
        self.rank_emb = nn.Embedding(rank_vocab, rank_emb_dim)
        self.type_emb = nn.Embedding(2, 8)
        self.conv1 = SAGEConv(latent_dim + rank_emb_dim + 8, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.out_mu = nn.Linear(hidden, 1)
        self.out_logsig = nn.Linear(hidden, 1)
        self.out_logit_pi = nn.Linear(hidden, 1)

    def forward(self, data: Data, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_nodes = z[data.batch]
        h = torch.cat([z_nodes, self.rank_emb(data.node_rank), self.type_emb(data.node_type)], dim=-1)
        h = F.relu(self.conv1(h, data.edge_index))
        h = F.relu(self.conv2(h, data.edge_index))
        return self.out_mu(h), self.out_logsig(h).clamp(-6.0, 3.0), self.out_logit_pi(h).clamp(-12.0, 12.0)


class HGVAE_ZI(nn.Module):
    def __init__(self, hidden: int, latent_dim: int, rank_vocab: int):
        super().__init__()
        self.enc = GraphEncoder(hidden, latent_dim, rank_vocab)
        self.dec = ZILogNormalGraphDecoder(hidden, latent_dim, rank_vocab)

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    @staticmethod
    def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=-1).mean()

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        mu, logvar = self.enc(data)
        z = self.reparam(mu, logvar)
        mu_log, log_sig_log, logit_pi = self.dec(data, z)
        return {"mu": mu, "logvar": logvar, "z": z, "mu_log": mu_log, "log_sig_log": log_sig_log, "logit_pi": logit_pi}

    def encode(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.enc(data)

    def expected_abundance(self, data: Data, mu: torch.Tensor) -> torch.Tensor:
        mu_log, log_sig_log, logit_pi = self.dec(data, mu)
        sig = torch.exp(log_sig_log)
        mean_pos = torch.exp(mu_log + 0.5 * sig * sig).clamp_min(0.0)
        pi = torch.sigmoid(logit_pi)
        return (1.0 - pi) * mean_pos


def zi_lognormal_nll(x_true: torch.Tensor, mu_log: torch.Tensor, log_sig_log: torch.Tensor, logit_pi: torch.Tensor, *, eps: float = 1e-6, zero_tol: float = 0.0) -> torch.Tensor:
    x = x_true.view(-1)
    mu = mu_log.view(-1)
    log_sig = log_sig_log.view(-1)
    sig = torch.exp(log_sig)
    pi = torch.sigmoid(logit_pi.view(-1))
    is_zero = x <= zero_tol
    y = torch.log(x + eps)
    normal_nll = 0.5 * ((y - mu) / sig) ** 2 + log_sig + 0.5 * math.log(2.0 * math.pi)
    jac = torch.log(x + eps)
    nll_pos = -torch.log(1.0 - pi + 1e-12) + normal_nll + jac
    nll_zero = -torch.log(pi + 1e-12)
    return torch.where(is_zero, nll_zero, nll_pos).mean()


def hierarchical_consistency_loss(x_pred: torch.Tensor, batch: torch.Tensor, children_of: Dict[int, List[int]], internal_ids: List[int], *, eps: float = 1e-6) -> torch.Tensor:
    bsz = int(batch.max().item()) + 1
    total_nodes = x_pred.size(0)
    N = total_nodes // bsz
    if N * bsz != total_nodes:
        raise ValueError("Graphs in batch appear to have different node counts; this implementation assumes fixed N.")

    x_pred = x_pred.view(bsz, N, 1)
    loss = 0.0
    count = 0
    for v in internal_ids:
        ch = children_of.get(int(v), [])
        if not ch:
            continue
        xv = x_pred[:, v, 0]
        xsum = x_pred[:, ch, 0].sum(dim=1)
        loss = loss + torch.mean(torch.abs(torch.log(xv + eps) - torch.log(xsum + eps)))
        count += 1
    if count == 0:
        return torch.tensor(0.0, device=x_pred.device)
    return loss / count


def latent_smoothness_loss_from_affinity(mu: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    diff = mu.unsqueeze(1) - mu.unsqueeze(0)
    d2 = (diff * diff).sum(dim=-1)
    return (A * d2).mean()


def load_sample_affinity_npy(path: Path) -> np.ndarray:
    A = np.load(path)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("sample affinity must be a square matrix.")
    A = np.asarray(A, dtype=np.float32)
    A[np.isnan(A)] = 0.0
    A[A < 0] = 0.0
    return A


def build_hgvae_zi_dataset(sgb_table_tsv: Path, phyla_tsv: Path, *, eps: float = 1e-6, keep_prefixes: bool = False) -> tuple[TaxonomyGraph, TaxonomyGraphDataset, list[str]]:
    taxg = build_taxonomy_graph_from_phyla_tsv(phyla_tsv, keep_prefixes=keep_prefixes)
    Xdf, sample_ids, _ = load_sgb_table_as_samples_by_leaf(sgb_table_tsv)
    dataset = TaxonomyGraphDataset(Xdf, taxg, eps=eps)
    return taxg, dataset, sample_ids


def build_hgvae_zi_loader(dataset: TaxonomyGraphDataset, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
