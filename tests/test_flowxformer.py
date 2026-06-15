import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from biomevae.cli import vae_embed  # noqa: E402
from biomevae.data import train_val_split_groups  # noqa: E402
from biomevae.models.flowxformer import FlowFeaturizer, build_tree_spec  # noqa: E402


class FlowFeaturizerTests(unittest.TestCase):
    def _write_taxonomy(self, path: Path) -> None:
        rows = [
            ["t__A", "k1", "p1", "c1", "o1", "f1", "g1", "s1"],
            ["t__B", "k1", "p1", "c1", "o1", "f1", "g1", "s2"],
        ]
        header = ["clade", "k", "p", "c", "o", "f", "g", "s"]
        with path.open("w", encoding="utf-8") as fh:
            fh.write("\t".join(header) + "\n")
            for row in rows:
                fh.write("\t".join(row) + "\n")

    def test_flow_featurizer_balanced_tree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tax_path = Path(tmpdir) / "phyla.tsv"
            self._write_taxonomy(tax_path)
            feature_clades = ["t__A", "t__B"]
            tree_spec = build_tree_spec(feature_clades, str(tax_path), branchlen_mode="unit")
            reference = np.array([0.5, 0.5], dtype=np.float32)
            featurizer = FlowFeaturizer(tree_spec, reference, uot_mode="off")

            x = torch.tensor([[3.0, 1.0]])
            _, flows, root_mismatch = featurizer(x)

            leaf_labels = tree_spec.node_labels
            leaf_edges = {
                leaf_labels[child]: idx
                for idx, child in enumerate(tree_spec.edge_child)
                if leaf_labels[child].startswith("feature::")
            }
            self.assertIn("feature::t__A", leaf_edges)
            self.assertIn("feature::t__B", leaf_edges)
            flow_a = flows[0, leaf_edges["feature::t__A"]].item()
            flow_b = flows[0, leaf_edges["feature::t__B"]].item()
            self.assertAlmostEqual(flow_a, 0.25, places=5)
            self.assertAlmostEqual(flow_b, -0.25, places=5)
            self.assertAlmostEqual(root_mismatch.abs().item(), 0.0, places=5)


class EmbedFlowxformerTests(unittest.TestCase):
    def test_embed_requires_taxonomy(self):
        cfg = {
            "model_type": "flowxformer",
            "model_kwargs": {"uot_mode": "root_l1"},
            "hidden": [64],
            "latent_dim": 8,
            "dropout": 0.1,
            "activation": "relu",
            "layer_norm": False,
        }
        with self.assertRaises(SystemExit):
            vae_embed._build_model(cfg, input_dim=2, taxonomy_path=None, feature_clades=None)


class EmbedHGVAEZITests(unittest.TestCase):
    def test_embed_hgvae_zi_requires_taxonomy(self):
        cfg = {
            "model_type": "hgvae_zi",
            "model_kwargs": {"rank_vocab": 4},
            "hidden": [64],
            "latent_dim": 3,
            "dropout": 0.0,
            "activation": "relu",
            "layer_norm": False,
        }
        with self.assertRaises(SystemExit):
            vae_embed._build_model(cfg, input_dim=2, taxonomy_path=None, feature_clades=None)


class GroupSplitTests(unittest.TestCase):
    def test_group_split_no_leakage(self):
        groups = np.array(["A", "A", "B", "B", "C", "C"], dtype=object)
        train_idx, val_idx = train_val_split_groups(len(groups), 0.33, 7, groups)
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        self.assertTrue(train_groups.isdisjoint(val_groups))


if __name__ == "__main__":
    unittest.main()
