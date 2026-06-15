from typing import Dict, List, Any
import numpy as np
import pandas as pd

__all__ = [
    "TAX_LEVELS_ALL",
    "load_taxonomy_table",
    "build_taxonomy_structures",
    "load_feature_clades",
    "build_taxonomy_graph_from_taxonomy",
    "build_phylo_embeddings",
]

TAX_LEVELS_ALL = ["k","p","c","o","f","g","s"]

def _infer_sep(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        head = f.readline()
    if "\t" in head and "," in head:
        return "," if head.count(",") >= head.count("\t") else "\t"
    if "\t" in head:
        return "\t"
    return ","

def load_taxonomy_table(path: str) -> pd.DataFrame:
    sep = _infer_sep(path)
    df = pd.read_csv(path, sep=sep, dtype=str)

    # Many public Phyla files ship without a header row. In that case Pandas
    # interprets the first lineage entry as the header, which leads to empty
    # data and missing taxonomy levels. When the detected "header" looks like a
    # lineage identifier (e.g. "t__SGB123"), reload the file without a header so
    # that column indices are numeric and the downstream logic can rename them.
    first_col = str(df.columns[0]).strip().lower()
    if first_col.startswith("t__"):
        df = pd.read_csv(path, sep=sep, dtype=str, header=None)

    cols_norm = {}
    for c in df.columns:
        if isinstance(c, str):
            cols_norm[c] = c.strip().lower()
        else:
            cols_norm[c] = c
    df.rename(columns=cols_norm, inplace=True)

    clade_col = next((c for c in ["clade","clade_name","taxid","tax_id","t","feature","id"] if c in df.columns), None)
    if clade_col is None:
        clade_col = df.columns[0]

    if df.shape[1] == 8 and set(df.columns) == set(range(8)):
        df.columns = ["clade"] + TAX_LEVELS_ALL
        clade_col = "clade"

    out = pd.DataFrame()
    out["clade"] = df[clade_col].astype(str)

    alias = {
        "k": ["k","kingdom"], "p": ["p","phylum"], "c": ["c","class"],
        "o": ["o","order"],   "f": ["f","family"],
        "g": ["g","genus"],   "s": ["s","species","specie"]
    }
    for lvl in TAX_LEVELS_ALL:
        col = next((c for c in alias[lvl] if c in df.columns), None)
        out[lvl] = df[col].astype(str) if col else f"NA_{lvl}"

    return out.set_index("clade")


def load_feature_clades(input_path: str) -> List[str]:
    """Return the ordered list of feature (clade) identifiers from the input TSV."""

    raw = pd.read_csv(input_path, sep="\t", dtype=str)
    if raw.shape[1] < 1:
        raise SystemExit("Input file must contain at least one column with clade identifiers.")
    return raw.iloc[:, 0].astype(str).tolist()

def build_taxonomy_structures(
    input_path: str,
    taxonomy_path: str,
    levels: List[str],
    lap_w: List[float],
    verbose: bool = True
) -> Dict[str, Any]:
    raw = pd.read_csv(input_path, sep="\t", dtype=str)
    feature_clades = raw.iloc[:,0].astype(str).tolist()

    tax = load_taxonomy_table(taxonomy_path)
    tax_aligned = tax.reindex(feature_clades)
    if tax_aligned.isna().any().any():
        if verbose:
            print(f"[taxonomy] {sum(tax_aligned.isna().any(axis=1))} clades missing; filling NA_*.")
        tax_aligned = tax_aligned.fillna({lvl: f"NA_{lvl}" for lvl in TAX_LEVELS_ALL})

    F = len(feature_clades)
    A_mats: Dict[str, np.ndarray] = {}
    for lvl in levels:
        if lvl not in TAX_LEVELS_ALL:
            raise SystemExit(f"Unknown taxonomy level '{lvl}'. Choose among {TAX_LEVELS_ALL}.")
        labs = tax_aligned[lvl].astype(str).values
        cats, inv = np.unique(labs, return_inverse=True)
        G = len(cats)
        A = np.zeros((G, F), dtype=np.float32)
        for j, g_idx in enumerate(inv):
            A[g_idx, j] = 1.0
        A_mats[lvl] = A
        if verbose:
            print(f"[taxonomy] level={lvl}: {G} groups, A={A.shape}")

    ws = lap_w[0] if len(lap_w)>0 else 0.0
    wg = lap_w[1] if len(lap_w)>1 else 0.0
    wf = lap_w[2] if len(lap_w)>2 else 0.0

    W = np.zeros((F, F), dtype=np.float32)

    def add_level(level: str, w: float):
        if w <= 0: return
        labs = tax_aligned[level].astype(str).values
        groups: Dict[str, List[int]] = {}
        for i, lab in enumerate(labs):
            groups.setdefault(lab, []).append(i)
        for idxs in groups.values():
            if len(idxs) > 1:
                arr = np.asarray(idxs, dtype=np.int32)
                W[arr[:,None], arr[None,:]] += w
        np.fill_diagonal(W, 0.0)

    add_level("s", ws); add_level("g", wg); add_level("f", wf)
    D = np.diag(W.sum(axis=1))
    L = (D - W).astype(np.float32)
    if verbose:
        print(f"[taxonomy] Laplacian: W nnz={int(np.count_nonzero(W))}, shape={W.shape}")
    return {"A_mats": A_mats, "L": L, "feature_clades": feature_clades}


def _graph_edge_weight(level: str, mode: str) -> float:
    if mode == "unweighted":
        return 1.0
    if mode == "branchlen":
        depth = {
            "s": 0.5,
            "g": 0.75,
            "f": 1.0,
            "o": 1.25,
            "c": 1.5,
            "p": 1.75,
            "k": 2.0,
        }
        return float(depth.get(level, 1.0))
    raise SystemExit(f"Unknown taxonomy graph mode '{mode}'. Choose 'unweighted' or 'branchlen'.")


def build_taxonomy_graph_from_taxonomy(
    feature_clades: List[str],
    taxonomy_path: str,
    mode: str = "unweighted",
) -> Dict[str, Any]:
    """Construct a graph specification describing the taxonomy hierarchy."""

    if not feature_clades:
        raise SystemExit("feature_clades must contain at least one entry.")

    tax = load_taxonomy_table(taxonomy_path)
    tax_aligned = tax.reindex(feature_clades)
    if tax_aligned.isna().any().any():
        tax_aligned = tax_aligned.fillna({lvl: f"NA_{lvl}" for lvl in TAX_LEVELS_ALL})

    nodes: List[str] = []
    node_index: Dict[str, int] = {}

    def get_idx(label: str) -> int:
        if label not in node_index:
            node_index[label] = len(nodes)
            nodes.append(label)
        return node_index[label]

    feature_indices: List[int] = []
    edges: List[List[float]] = []

    for clade in feature_clades:
        feat_label = f"feature::{clade}"
        prev_idx = get_idx(feat_label)
        feature_indices.append(prev_idx)

        lineage = tax_aligned.loc[clade]
        for lvl in reversed(TAX_LEVELS_ALL):
            parent_label = f"{lvl}::{lineage[lvl]}"
            parent_idx = get_idx(parent_label)
            w = _graph_edge_weight(lvl, mode)
            edges.append([prev_idx, parent_idx, w])
            prev_idx = parent_idx

    return {
        "num_nodes": len(nodes),
        "edges": edges,
        "feature_indices": feature_indices,
        "node_labels": nodes,
        "mode": mode,
    }


def build_phylo_embeddings(
    feature_clades: List[str],
    taxonomy_path: str,
    method: str = "pca",
    dim: int = 32,
) -> np.ndarray:
    """Compute phylogeny-aware embeddings for each feature."""

    if method != "pca":
        raise SystemExit("Only 'pca' phylogeny embedding is currently supported.")

    if not feature_clades:
        raise SystemExit("feature_clades must contain at least one entry.")

    tax = load_taxonomy_table(taxonomy_path)
    tax_aligned = tax.reindex(feature_clades)
    if tax_aligned.isna().any().any():
        tax_aligned = tax_aligned.fillna({lvl: f"NA_{lvl}" for lvl in TAX_LEVELS_ALL})

    F = len(feature_clades)
    if F == 1:
        return np.zeros((1, min(dim, 1)), dtype=np.float32)

    W = np.zeros((F, F), dtype=np.float64)

    def add_level(level: str, weight: float) -> None:
        labs = tax_aligned[level].astype(str).values
        groups: Dict[str, List[int]] = {}
        for idx, lab in enumerate(labs):
            groups.setdefault(lab, []).append(idx)
        for members in groups.values():
            if len(members) < 2:
                continue
            arr = np.asarray(members, dtype=np.int32)
            W[arr[:, None], arr[None, :]] += weight
        np.fill_diagonal(W, 0.0)

    add_level("s", 1.0)
    add_level("g", 0.5)
    add_level("f", 0.25)

    D = np.diag(W.sum(axis=1))
    L = D - W

    evals, evecs = np.linalg.eigh(L)
    order = np.argsort(evals)
    max_dim = min(dim, F - 1)
    if max_dim <= 0:
        return np.zeros((F, 1), dtype=np.float32)
    selected = order[1 : max_dim + 1]
    emb = evecs[:, selected]
    return emb.astype(np.float32)
