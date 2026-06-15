"""Taxonomy / phylogeny tree utilities for tree-structured microbiome VAEs.

The module deliberately separates three concerns:

1. Build a rooted taxonomy tree from a taxonomy table.
2. Load and strictly align an abundance/count table to the tree leaves.
3. Aggregate leaf observations to all internal nodes for tree likelihoods.

Only numpy, pandas and torch are required.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch


RANK_PREFIX_RE = re.compile(r"^([a-zA-Z])__(.*)$")
MISSING_TOKENS = {"", "na", "nan", "none", "null", "unassigned", "unclassified", "unknown"}
# Header labels that may appear as the first row of a taxonomy table when the
# file was exported with a header but is read positionally (``has_header`` is
# False). Such a row is never a real taxon, so it is dropped before building the
# tree to avoid injecting a spurious leaf (e.g. ``clade_name``) that would break
# leaf/table alignment against the abundance table.
HEADER_LEAF_SENTINELS = {
    "clade_name",
    "clade",
    "taxonomy",
    "taxon",
    "lineage",
    "feature",
    "feature_id",
    "featureid",
    "features",
    "otu",
    "otu_id",
    "otuid",
    "sgb",
    "sgb_id",
    "id",
    "name",
    "index",
}
PREFIX_TO_RANK = {
    "k": "kingdom",
    "d": "domain",
    "p": "phylum",
    "c": "class",
    "o": "order",
    "f": "family",
    "g": "genus",
    "s": "species",
    "t": "strain",
}
DEFAULT_RANKS = (
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
    "strain",
)


def _clean_str(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s


def is_missing_taxon_label(x: object) -> bool:
    return _clean_str(x).lower() in MISSING_TOKENS


def strip_rank_prefix(x: object) -> str:
    """Strip prefixes such as ``k__`` or ``s__`` from a taxon label."""
    s = _clean_str(x)
    m = RANK_PREFIX_RE.match(s)
    return m.group(2).strip() if m else s


# Backward-compatible private name used in the original code.
def _strip_prefix(x: str) -> str:
    return strip_rank_prefix(x)


@dataclass(frozen=True)
class TaxonomyToken:
    rank_name: str
    rank_index: int
    label: str
    clean_label: str
    missing_placeholder: bool = False


def _rank_name_for_field(raw: str, idx: int) -> str:
    m = RANK_PREFIX_RE.match(raw)
    if m:
        return PREFIX_TO_RANK.get(m.group(1).lower(), f"rank_{idx + 1}")
    if idx < len(DEFAULT_RANKS):
        return DEFAULT_RANKS[idx]
    return f"rank_{idx + 1}"


def parse_taxonomy_fields(
    fields: Sequence[object],
    *,
    keep_prefixes: bool = False,
    fill_missing_intermediate: bool = True,
) -> List[TaxonomyToken]:
    """Parse one taxonomic lineage.

    ``fields`` may be multiple rank columns or a single pipe-delimited lineage.
    Missing terminal ranks are omitted. Missing internal ranks are filled with
    rank-aware placeholders by default so that genus-level information is not
    incorrectly moved up to family/order depth.
    """
    raw_fields: List[str]
    if len(fields) == 1 and "|" in _clean_str(fields[0]):
        raw_fields = [_clean_str(x) for x in _clean_str(fields[0]).split("|")]
    else:
        raw_fields = [_clean_str(x) for x in fields]

    nonmissing = [i for i, x in enumerate(raw_fields) if not is_missing_taxon_label(x)]
    if not nonmissing:
        return []
    last_nonmissing = max(nonmissing)

    out: List[TaxonomyToken] = []
    for i, raw in enumerate(raw_fields):
        rank_name = _rank_name_for_field(raw, i)
        rank_index = i + 1
        if is_missing_taxon_label(raw):
            if fill_missing_intermediate and i < last_nonmissing:
                clean = f"unclassified_{rank_name}"
                prefix = next((k for k, v in PREFIX_TO_RANK.items() if v == rank_name), "x")
                label = f"{prefix}__{clean}" if keep_prefixes else clean
                out.append(
                    TaxonomyToken(
                        rank_name=rank_name,
                        rank_index=rank_index,
                        label=label,
                        clean_label=clean,
                        missing_placeholder=True,
                    )
                )
            continue
        clean = strip_rank_prefix(raw)
        label = raw if keep_prefixes else clean
        out.append(
            TaxonomyToken(
                rank_name=rank_name,
                rank_index=rank_index,
                label=label,
                clean_label=clean,
                missing_placeholder=False,
            )
        )
    return out


# Backward-compatible helper. It returns only labels, as in the original code.
def parse_phylarow_to_lineage(ranks: List[str], keep_prefixes: bool = False) -> List[str]:
    return [t.label for t in parse_taxonomy_fields(ranks, keep_prefixes=keep_prefixes)]


@dataclass
class TaxonomyGraph:
    node_names: List[str]
    node_type: torch.Tensor                 # 0 internal/root, 1 leaf
    node_rank: torch.Tensor                 # taxonomic rank index; root is 0
    edge_index: torch.Tensor                # undirected, for optional graph tooling
    parent_of: Dict[int, int]               # child -> parent
    children_of: Dict[int, List[int]]       # parent -> children
    leaf_ids: List[int]
    leaf_name_to_id: Dict[str, int]
    internal_ids: List[int]
    node_depth: torch.Tensor                # topological depth; root is 0
    node_rank_name: List[str]
    node_paths: List[Tuple[Tuple[str, str], ...]]


@dataclass
class AlignmentReport:
    n_samples: int
    n_tree_leaves: int
    n_table_features: int
    n_matched_leaves: int
    missing_tree_leaves: List[str]
    extra_table_features: List[str]
    duplicate_table_matches: Dict[str, List[str]]
    alias_matches: Dict[str, str]

    @property
    def matched_fraction(self) -> float:
        return self.n_matched_leaves / max(1, self.n_tree_leaves)

    def summary(self, max_items: int = 8) -> str:
        def head(xs: Sequence[str]) -> str:
            suffix = "" if len(xs) <= max_items else f" ... (+{len(xs) - max_items})"
            return ", ".join(xs[:max_items]) + suffix

        return (
            f"samples={self.n_samples}, tree_leaves={self.n_tree_leaves}, "
            f"table_features={self.n_table_features}, matched_leaves={self.n_matched_leaves} "
            f"({self.matched_fraction:.1%}); "
            f"missing=[{head(self.missing_tree_leaves)}]; "
            f"extra=[{head(self.extra_table_features)}]"
        )


def _column_index(columns: Sequence[object], col: Union[int, str]) -> int:
    if isinstance(col, int):
        return col
    try:
        return list(columns).index(col)
    except ValueError as exc:
        raise ValueError(f"Column {col!r} not found. Available columns: {list(columns)!r}") from exc


def build_taxonomy_graph_from_phyla_tsv(
    phyla_tsv: Union[Path, str],
    *,
    keep_prefixes: bool = False,
    internal_prefix: str = "tax__",
    sep: str = "\t",
    has_header: bool = False,
    leaf_col: Union[int, str] = 0,
    tax_cols: Optional[Sequence[Union[int, str]]] = None,
    fill_missing_intermediate: bool = True,
    on_duplicate_leaf: str = "error",  # error | ignore_same | first
) -> TaxonomyGraph:
    """Build a rooted taxonomy tree from a table.

    The first column is normally the leaf ID, for example an SGB ID, followed by
    rank columns. Duplicate leaf IDs with conflicting lineages are errors by
    default because silently keeping the first occurrence corrupts alignment.
    """
    header = 0 if has_header else None
    tax = pd.read_csv(phyla_tsv, sep=sep, header=header, dtype=str).fillna("")
    if tax.shape[1] < 2:
        raise ValueError("Taxonomy table must have a leaf column and at least one taxonomy column.")

    leaf_idx = _column_index(tax.columns, leaf_col)

    # Guard against a stray header line being parsed as a taxon row. When the
    # source file actually carries a header (e.g. a MetaPhlAn ``clade_name``
    # column) but is read positionally (``has_header=False``), the first row is
    # otherwise turned into a spurious leaf such as ``clade_name`` that has no
    # matching abundance-table feature and breaks strict alignment. Drop it so
    # the tree leaves line up with the table features.
    if not has_header and tax.shape[0] > 0:
        first_leaf = _clean_str(tax.iloc[0, leaf_idx]).lstrip("#").strip().lower()
        if first_leaf in HEADER_LEAF_SENTINELS:
            tax = tax.iloc[1:].reset_index(drop=True)
            if tax.shape[0] == 0:
                raise ValueError(
                    "Taxonomy table contained only a header row after dropping "
                    f"the leading header line {first_leaf!r}."
                )

    if tax_cols is None:
        tax_indices = [i for i in range(tax.shape[1]) if i != leaf_idx]
    else:
        tax_indices = [_column_index(tax.columns, c) for c in tax_cols]

    path_to_node_id: Dict[Tuple[Tuple[str, str], ...], int] = {}
    node_names: List[str] = []
    node_type: List[int] = []
    node_rank: List[int] = []
    node_depth: List[int] = []
    node_rank_name: List[str] = []
    node_paths: List[Tuple[Tuple[str, str], ...]] = []
    parent_of: Dict[int, int] = {}
    children_of: Dict[int, List[int]] = {}

    def add_node(
        name: str,
        *,
        ntype: int,
        rank_index: int,
        depth: int,
        rank_name: str,
        path: Tuple[Tuple[str, str], ...],
    ) -> int:
        nid = len(node_names)
        node_names.append(name)
        node_type.append(ntype)
        node_rank.append(rank_index)
        node_depth.append(depth)
        node_rank_name.append(rank_name)
        node_paths.append(path)
        children_of[nid] = []
        return nid

    root_id = add_node(
        "root",
        ntype=0,
        rank_index=0,
        depth=0,
        rank_name="root",
        path=(),
    )
    path_to_node_id[()] = root_id

    leaf_ids: List[int] = []
    leaf_name_to_id: Dict[str, int] = {}
    leaf_path_by_name: Dict[str, Tuple[Tuple[str, str], ...]] = {}

    for row_idx, row in tax.iterrows():
        leaf_name = _clean_str(row.iloc[leaf_idx])
        if not leaf_name:
            raise ValueError(f"Empty leaf name at taxonomy row {row_idx}.")

        fields = [row.iloc[i] for i in tax_indices]
        tokens = parse_taxonomy_fields(
            fields,
            keep_prefixes=keep_prefixes,
            fill_missing_intermediate=fill_missing_intermediate,
        )

        cur_path: Tuple[Tuple[str, str], ...] = ()
        for token in tokens:
            token_key = (token.rank_name, token.clean_label)
            nxt = cur_path + (token_key,)
            if nxt not in path_to_node_id:
                parent_id = path_to_node_id[cur_path]
                display = token.label
                name = f"{internal_prefix}{token.rank_index:02d}__{token.rank_name}__{display}"
                nid = add_node(
                    name,
                    ntype=0,
                    rank_index=token.rank_index,
                    depth=len(nxt),
                    rank_name=token.rank_name,
                    path=nxt,
                )
                path_to_node_id[nxt] = nid
                parent_of[nid] = parent_id
                children_of[parent_id].append(nid)
            cur_path = nxt

        if leaf_name in leaf_name_to_id:
            old_path = leaf_path_by_name[leaf_name]
            if old_path != cur_path and on_duplicate_leaf != "first":
                raise ValueError(
                    f"Duplicate leaf {leaf_name!r} has conflicting lineages: "
                    f"{old_path!r} vs {cur_path!r}."
                )
            if on_duplicate_leaf in {"ignore_same", "first"}:
                continue
            raise ValueError(f"Duplicate leaf {leaf_name!r} at taxonomy row {row_idx}.")

        parent_id = path_to_node_id[cur_path]
        leaf_path = cur_path + (("leaf", leaf_name),)
        leaf_id = add_node(
            leaf_name,
            ntype=1,
            rank_index=(tokens[-1].rank_index + 1 if tokens else 1),
            depth=len(cur_path) + 1,
            rank_name="leaf",
            path=leaf_path,
        )
        leaf_name_to_id[leaf_name] = leaf_id
        leaf_path_by_name[leaf_name] = cur_path
        leaf_ids.append(leaf_id)
        parent_of[leaf_id] = parent_id
        children_of[parent_id].append(leaf_id)

    src: List[int] = []
    dst: List[int] = []
    for child, parent in parent_of.items():
        src.extend([parent, child])
        dst.extend([child, parent])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    internal_ids = [i for i, t in enumerate(node_type) if t == 0]

    return TaxonomyGraph(
        node_names=node_names,
        node_type=torch.tensor(node_type, dtype=torch.long),
        node_rank=torch.tensor(node_rank, dtype=torch.long),
        edge_index=edge_index,
        parent_of=parent_of,
        children_of=children_of,
        leaf_ids=leaf_ids,
        leaf_name_to_id=leaf_name_to_id,
        internal_ids=internal_ids,
        node_depth=torch.tensor(node_depth, dtype=torch.long),
        node_rank_name=node_rank_name,
        node_paths=node_paths,
    )


def load_feature_table_as_samples_by_feature(
    table_tsv: Union[Path, str],
    *,
    sep: str = "\t",
    feature_col: Optional[str] = None,
    metadata_cols: Iterable[str] = ("NCBI_tax_id", "tax_id", "additional_species"),
    sample_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Load a feature x sample table and return samples x features.

    MetaPhlAn-style ``#clade_name`` is normalized to ``clade_name``. If duplicate
    feature rows are present, they are summed after numeric conversion.
    """
    df = pd.read_csv(table_tsv, sep=sep, dtype=str).fillna("0")
    df.columns = [str(c).lstrip("#").strip() for c in df.columns]

    if feature_col is None:
        if "clade_name" in df.columns:
            feature_col = "clade_name"
        else:
            feature_col = df.columns[0]
    if feature_col not in df.columns:
        raise ValueError(f"Feature column {feature_col!r} not found in {table_tsv}.")

    meta = {str(c).lstrip("#").strip() for c in metadata_cols}
    if sample_cols is None:
        sample_cols = [c for c in df.columns if c != feature_col and c not in meta]
    if not sample_cols:
        raise ValueError("No sample columns found after excluding feature and metadata columns.")

    features = df[feature_col].astype(str).map(str.strip)
    values = df.loc[:, list(sample_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    values.index = features
    values = values.groupby(level=0, sort=False).sum()
    Xdf = values.T
    Xdf.index = Xdf.index.astype(str)
    Xdf.columns = Xdf.columns.astype(str)
    return Xdf, Xdf.index.tolist(), Xdf.columns.tolist()


# Backward-compatible loader name from the original implementation.
def load_sgb_table_as_samples_by_leaf(sgb_table_tsv: Union[Path, str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    return load_feature_table_as_samples_by_feature(sgb_table_tsv)


def _candidate_feature_aliases(name: str) -> List[str]:
    s = _clean_str(name)
    aliases = [s]
    if "|" in s:
        aliases.append(s.split("|")[-1])
    aliases.append(strip_rank_prefix(s))
    if "|" in s:
        aliases.append(strip_rank_prefix(s.split("|")[-1]))
    out: List[str] = []
    seen = set()
    for a in aliases:
        if a and a not in seen:
            seen.add(a)
            out.append(a)
    return out


def build_unique_leaf_alias_map(taxg: TaxonomyGraph) -> Dict[str, str]:
    """Return alias -> canonical leaf name for aliases that are unambiguous."""
    tmp: Dict[str, set[str]] = defaultdict(set)
    for leaf_id in taxg.leaf_ids:
        leaf_name = taxg.node_names[leaf_id]
        for alias in _candidate_feature_aliases(leaf_name):
            tmp[alias].add(leaf_name)
    return {alias: next(iter(names)) for alias, names in tmp.items() if len(names) == 1}


def align_table_to_tree_leaves(
    Xdf: pd.DataFrame,
    taxg: TaxonomyGraph,
    *,
    strict: bool = True,
    allow_missing_leaves: bool = False,
    allow_extra_features: bool = True,
    use_unique_aliases: bool = True,
    min_matched_fraction: float = 0.95,
) -> Tuple[np.ndarray, List[str], AlignmentReport]:
    """Align samples x features to the tree leaf order.

    Missing leaves are filled with zero only when ``allow_missing_leaves`` is true.
    Extra table features are ignored only when ``allow_extra_features`` is true.
    """
    Xdf = Xdf.copy()
    Xdf.columns = Xdf.columns.astype(str)
    leaf_names = [taxg.node_names[nid] for nid in taxg.leaf_ids]
    leaf_name_set = set(leaf_names)
    alias_to_leaf = build_unique_leaf_alias_map(taxg) if use_unique_aliases else {}

    matched_columns: Dict[str, List[str]] = defaultdict(list)
    alias_matches: Dict[str, str] = {}
    extra_features: List[str] = []

    for feature in Xdf.columns:
        mapped: Optional[str] = None
        if feature in leaf_name_set:
            mapped = feature
        elif use_unique_aliases:
            for alias in _candidate_feature_aliases(feature):
                if alias in alias_to_leaf:
                    mapped = alias_to_leaf[alias]
                    alias_matches[feature] = mapped
                    break
        if mapped is None:
            extra_features.append(feature)
        else:
            matched_columns[mapped].append(feature)

    missing = [name for name in leaf_names if name not in matched_columns]
    duplicate_matches = {k: v for k, v in matched_columns.items() if len(v) > 1}

    report = AlignmentReport(
        n_samples=Xdf.shape[0],
        n_tree_leaves=len(leaf_names),
        n_table_features=Xdf.shape[1],
        n_matched_leaves=len(leaf_names) - len(missing),
        missing_tree_leaves=missing,
        extra_table_features=extra_features,
        duplicate_table_matches=duplicate_matches,
        alias_matches=alias_matches,
    )

    problems: List[str] = []
    if missing and (strict and not allow_missing_leaves):
        problems.append(f"missing {len(missing)} tree leaves")
    if extra_features and (strict and not allow_extra_features):
        problems.append(f"found {len(extra_features)} table features not present as tree leaves")
    if report.matched_fraction < min_matched_fraction:
        problems.append(
            f"matched fraction {report.matched_fraction:.1%} below minimum {min_matched_fraction:.1%}"
        )
    if problems:
        raise ValueError("Leaf/table alignment failed: " + "; ".join(problems) + ". " + report.summary())

    X = np.zeros((Xdf.shape[0], len(leaf_names)), dtype=np.float32)
    for j, leaf_name in enumerate(leaf_names):
        cols = matched_columns.get(leaf_name, [])
        if cols:
            X[:, j] = Xdf.loc[:, cols].sum(axis=1).to_numpy(dtype=np.float32)
    return X, leaf_names, report


def close_composition(X: np.ndarray, *, min_total: float = 0.0) -> np.ndarray:
    """Normalize each sample to sum to one, leaving all-zero rows as zero."""
    X = np.asarray(X, dtype=np.float32)
    X = np.clip(X, 0.0, None)
    totals = X.sum(axis=1, keepdims=True)
    out = np.zeros_like(X, dtype=np.float32)
    ok = totals[:, 0] > min_total
    out[ok] = X[ok] / totals[ok]
    return out


def validate_nonnegative_integer_counts(X: np.ndarray, *, atol: float = 1e-4) -> np.ndarray:
    """Validate and return a rounded float32 count matrix."""
    X = np.asarray(X)
    if np.any(~np.isfinite(X)) or np.any(X < -atol):
        raise ValueError("Count matrix contains negative or non-finite values.")
    rounded = np.rint(np.clip(X, 0, None))
    if not np.allclose(X, rounded, atol=atol, rtol=0.0):
        raise ValueError("Count likelihood requested, but table contains non-integer values.")
    return rounded.astype(np.float32)


def aggregate_leaf_matrix_to_nodes(
    taxg: TaxonomyGraph,
    leaf_matrix: np.ndarray,
    leaf_ids: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Aggregate samples x leaves to samples x all nodes by summing descendants."""
    leaf_ids = list(taxg.leaf_ids if leaf_ids is None else leaf_ids)
    X_leaf = np.asarray(leaf_matrix, dtype=np.float32)
    if X_leaf.ndim != 2 or X_leaf.shape[1] != len(leaf_ids):
        raise ValueError(
            f"leaf_matrix must have shape (n_samples, {len(leaf_ids)}); got {X_leaf.shape}."
        )

    n_samples = X_leaf.shape[0]
    n_nodes = len(taxg.node_names)
    X_nodes = np.zeros((n_samples, n_nodes), dtype=np.float32)
    X_nodes[:, leaf_ids] = X_leaf

    order = np.argsort(taxg.node_depth.numpy())[::-1]
    for nid in order:
        if int(taxg.node_type[nid]) == 0:
            children = taxg.children_of.get(int(nid), [])
            if children:
                X_nodes[:, int(nid)] = X_nodes[:, children].sum(axis=1)
    return X_nodes


# Backward-compatible single-sample helper from the original code.
def build_internal_sums_vector(taxg: TaxonomyGraph, leaf_abund: np.ndarray, leaf_ids: List[int]) -> np.ndarray:
    return aggregate_leaf_matrix_to_nodes(taxg, np.asarray(leaf_abund, dtype=np.float32)[None, :], leaf_ids)[0]
