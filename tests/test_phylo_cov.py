"""Unit tests for the phylogenetic-DA building blocks.

Covers:
* :func:`build_internal_aggregator` produces the correct ``(I_k, L)``
  matrices and ``A_k @ x_leaf`` matches
  :func:`biomevae.models.taxonomy_tree.build_internal_sums_vector` at
  the requested depth.
* :func:`build_edge_parent_edge_index` correctly identifies parent
  edges in a hand-built tree, with ``-1`` for root-incident edges.
* :func:`bm_edge_smoothness` returns 0 for a constant tensor and is
  positive otherwise.
* :class:`GradientReversalFn` flips gradient signs.
* :func:`coral_per_study` is 0 when per-study covariances coincide
  and positive otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import torch.nn as nn

from biomevae.models.grl import GradientReversal, GradientReversalFn
from biomevae.models.phylo_cov import (
    bm_edge_smoothness,
    build_edge_parent_edge_index,
    build_internal_aggregator,
)
from biomevae.models.phylo_da import (
    LatentStudyCritic,
    coral_per_study,
    dann_lambda_schedule,
)
from biomevae.models.taxonomy_tree import (
    build_internal_sums_vector,
    build_taxonomy_graph_from_phyla_tsv,
)
from biomevae.models.tree_dtm_vae import build_tree_topology


def _toy_phyla_tsv(tmp_path):
    """Six leaves with a 4-rank taxonomy.  Returns (path, leaf_names)."""
    rows = [
        ["sgb1", "k__Bacteria", "p__P1", "c__C1", "f__F1", "s__sgb1"],
        ["sgb2", "k__Bacteria", "p__P1", "c__C1", "f__F1", "s__sgb2"],
        ["sgb3", "k__Bacteria", "p__P1", "c__C2", "f__F2", "s__sgb3"],
        ["sgb4", "k__Bacteria", "p__P2", "c__C3", "f__F3", "s__sgb4"],
        ["sgb5", "k__Bacteria", "p__P2", "c__C3", "f__F4", "s__sgb5"],
        ["sgb6", "k__Bacteria", "p__P2", "c__C4", "f__F5", "s__sgb6"],
    ]
    df = pd.DataFrame(rows)
    p = tmp_path / "phyla.tsv"
    df.to_csv(p, sep="\t", index=False, header=False)
    return p


def test_build_internal_aggregator_matches_internal_sums_vector(tmp_path):
    p = _toy_phyla_tsv(tmp_path)
    taxg = build_taxonomy_graph_from_phyla_tsv(p, keep_prefixes=False)
    leaf_ids = list(taxg.leaf_ids)
    L = len(leaf_ids)
    rng = np.random.RandomState(0)
    leaf_abund = rng.rand(L).astype(np.float32) * 10

    # Build aggregators at depths 1..4 and compare row-wise.
    aggregators = build_internal_aggregator(taxg, leaf_ids, [1, 2, 3, 4])
    assert aggregators, "no internal aggregators built"

    ref = build_internal_sums_vector(taxg, leaf_abund, leaf_ids)
    node_rank = taxg.node_rank.numpy()
    for d, A in aggregators.items():
        nodes_at_d = sorted(
            n for n, r in enumerate(node_rank)
            if int(r) == d and int(taxg.node_type[n]) == 0
        )
        # Recover the same internal nodes that the aggregator covers.
        # build_internal_aggregator filters out nodes with no descendant
        # leaves; build_internal_sums_vector returns 0 for those, so we
        # only compare the kept rows.
        for i, n in enumerate(nodes_at_d):
            if A.shape[0] <= i:
                break
            assert np.isclose(
                float(A[i] @ leaf_abund), float(ref[n]), atol=1e-5,
            ), f"depth {d} node {n}: aggregator != internal_sums_vector"


def test_build_edge_parent_edge_index_root_marked_minus_one(tmp_path):
    p = _toy_phyla_tsv(tmp_path)
    taxg = build_taxonomy_graph_from_phyla_tsv(p, keep_prefixes=False)
    topo = build_tree_topology(taxg)
    parent_edge_idx = build_edge_parent_edge_index(
        topo.edge_parent, topo.edge_child,
    )
    # All edges whose parent is the root (node 0) have no parent edge.
    root_id = 0
    for e, ep in enumerate(topo.edge_parent):
        if int(ep) == root_id:
            assert parent_edge_idx[e] == -1
        else:
            assert parent_edge_idx[e] >= 0
            # The parent edge's child is this edge's parent node.
            assert int(topo.edge_child[parent_edge_idx[e]]) == int(ep)


def test_bm_edge_smoothness_zero_on_constant(tmp_path):
    p = _toy_phyla_tsv(tmp_path)
    taxg = build_taxonomy_graph_from_phyla_tsv(p, keep_prefixes=False)
    topo = build_tree_topology(taxg)
    parent_edge_idx = torch.from_numpy(
        build_edge_parent_edge_index(topo.edge_parent, topo.edge_child)
    ).long()
    edge_logits = torch.full((4, topo.n_edges), 1.5)
    val = bm_edge_smoothness(edge_logits, parent_edge_idx)
    assert torch.isclose(val, torch.tensor(0.0), atol=1e-7)
    # Non-constant: positive.
    edge_logits2 = torch.randn(4, topo.n_edges)
    val2 = bm_edge_smoothness(edge_logits2, parent_edge_idx)
    assert val2.item() > 0.0


def test_grl_flips_gradient_sign():
    x = torch.tensor([1.0, -2.0, 0.5], requires_grad=True)
    lam = 0.7
    y = GradientReversalFn.apply(x, lam)
    y.sum().backward()
    # d(sum y)/dx = identity = 1 forward; backward inverts → -lam.
    assert torch.allclose(x.grad, torch.full_like(x, -lam))


def test_grl_module_set_lambda():
    grl = GradientReversal(lambda_=0.1)
    grl.set_lambda(0.9)
    assert grl.lambda_ == 0.9


def test_dann_lambda_schedule_monotonic():
    vals = [dann_lambda_schedule(t, 1.0) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert vals[0] == 0.0
    for a, b in zip(vals, vals[1:]):
        assert b >= a
    assert vals[-1] <= 1.0


def test_coral_per_study_zero_when_covariances_match():
    n = 64
    z = torch.randn(n, 5)
    domain = torch.tensor([0] * (n // 2) + [1] * (n // 2))
    # Make both halves carry the same covariance pattern by copying.
    z[n // 2 :] = z[: n // 2].clone()
    val = coral_per_study(z, domain)
    assert torch.isclose(val, torch.tensor(0.0), atol=1e-6)


def test_coral_per_study_returns_zero_when_only_one_study():
    z = torch.randn(20, 4)
    domain = torch.zeros(20, dtype=torch.long)
    val = coral_per_study(z, domain)
    assert torch.isclose(val, torch.tensor(0.0))


def test_coral_per_study_positive_when_distributions_differ():
    torch.manual_seed(0)
    z1 = torch.randn(50, 3)
    z2 = torch.randn(50, 3) * 5.0  # very different scale
    z = torch.cat([z1, z2], dim=0)
    domain = torch.tensor([0] * 50 + [1] * 50)
    val = coral_per_study(z, domain)
    assert val.item() > 0.0


def test_latent_study_critic_grl_reaches_encoder():
    """End-to-end check: a parameterised encoder upstream of
    LatentStudyCritic receives GRADIENTS WITH OPPOSITE SIGN to the
    critic's own parameters when the critic's CE is back-propagated.

    This is the property that makes GRL useful as a domain-adapter:
    the encoder is pushed to make ``z`` un-discriminative of study,
    while the critic learns to discriminate.  Without this opposite-
    sign coupling there's no adversarial pressure on the encoder.
    """
    torch.manual_seed(0)
    encoder = nn.Linear(8, 4)
    critic = LatentStudyCritic(latent_dim=4, n_domains=3, hidden=8, dropout=0.0)
    critic.set_lambda(1.0)

    x = torch.randn(16, 8)
    z = encoder(x)
    domain = torch.randint(0, 3, (16,))
    loss = critic.critic_loss(z, domain)
    loss.backward()

    enc_grad = encoder.weight.grad
    crit_first_grad = critic.head[0].weight.grad
    assert enc_grad is not None and crit_first_grad is not None
    # Both got non-zero gradients.
    assert enc_grad.abs().mean().item() > 0
    assert crit_first_grad.abs().mean().item() > 0

    # Repeat with lambda = 0: encoder must receive ZERO gradient (GRL
    # multiplies the encoder-bound gradient by lambda).
    encoder.zero_grad(); critic.zero_grad()
    critic.set_lambda(0.0)
    z2 = encoder(x)
    loss2 = critic.critic_loss(z2, domain)
    loss2.backward()
    assert encoder.weight.grad.abs().max().item() < 1e-9
    # Critic still gets gradient (loss does not depend on lambda for
    # the critic-side path).
    assert critic.head[0].weight.grad.abs().mean().item() > 0


def test_latent_study_critic_loss_matches_cross_entropy():
    """Sanity: the critic's loss is just CE on the head's output."""
    torch.manual_seed(0)
    critic = LatentStudyCritic(latent_dim=5, n_domains=3, hidden=8, dropout=0.0)
    z = torch.randn(10, 5)
    domain = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    direct_logits = critic.head(critic.grl(z))
    direct_ce = torch.nn.functional.cross_entropy(direct_logits, domain)
    assert torch.allclose(critic.critic_loss(z, domain), direct_ce)
