from __future__ import annotations

from typing import Iterable, Tuple

import networkx as nx
import pandas as pd


def build_interaction_graph(
    edges: pd.DataFrame,
    src_col: str = "src",
    dst_col: str = "dst",
    weight_col: str | None = "weight",
    directed: bool = True,
) -> nx.Graph | nx.DiGraph:
    """
    Build a user interaction graph from an edge list DataFrame.

    Each row represents an interaction (follow, mention, reply, etc.)
    """
    if directed:
        g: nx.Graph | nx.DiGraph = nx.DiGraph()
    else:
        g = nx.Graph()

    for _, row in edges.iterrows():
        src = row[src_col]
        dst = row[dst_col]
        if weight_col and weight_col in row and not pd.isna(row[weight_col]):
            w = row[weight_col]
            if g.has_edge(src, dst):
                g[src][dst]["weight"] += w
            else:
                g.add_edge(src, dst, weight=float(w))
        else:
            g.add_edge(src, dst, weight=1.0)

    return g


def largest_components(g: nx.Graph | nx.DiGraph, k: int = 5) -> Iterable[Tuple[int, list]]:
    """
    Yield up to k connected components (or weakly connected for DiGraph),
    sorted by size descending.
    """
    if isinstance(g, nx.DiGraph):
        comps = nx.weakly_connected_components(g)
    else:
        comps = nx.connected_components(g)
    comps = sorted(comps, key=len, reverse=True)[:k]
    for idx, nodes in enumerate(comps):
        yield idx, list(nodes)

