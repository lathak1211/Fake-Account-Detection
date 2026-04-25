from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

import networkx as nx

try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import from_networkx

    HAS_PYG = True
except Exception:  # pragma: no cover - optional dependency
    HAS_PYG = False
    GCNConv = None
    from_networkx = None


@dataclass
class GraphGNNConfig:
    input_dim: int = 8
    hidden_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.2


class SimpleGCN(nn.Module):
    def __init__(self, config: GraphGNNConfig, num_nodes: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.emb = nn.Embedding(num_nodes, config.input_dim)
        self.convs = nn.ModuleList()
        dims = [config.input_dim] + [config.hidden_dim] * (config.num_layers - 1)
        for in_dim in dims:
            self.convs.append(GCNConv(in_dim, config.hidden_dim))
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(self, data) -> torch.Tensor:
        x = self.emb.weight  # (num_nodes, input_dim)
        edge_index = data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        scores = torch.sigmoid(self.head(x)).squeeze(-1)
        return scores


def run_gnn_on_graph(
    g: nx.Graph, config: Optional[GraphGNNConfig] = None
) -> Tuple[dict, float]:
    """
    Run a simple GCN on the given NetworkX graph to get per-node bot scores.

    Returns:
        node_scores: dict[node_id, score in [0,1]]
        cluster_risk: float in [0,1] summarizing suspiciousness of the graph
    """
    if config is None:
        config = GraphGNNConfig()

    if not HAS_PYG or from_networkx is None:
        # Fallback: simple graph heuristics (degree & clustering)
        degrees = dict(g.degree())
        clustering = nx.clustering(g)
        node_scores = {}
        for n in g.nodes():
            d = degrees.get(n, 0)
            c = clustering.get(n, 0.0)
            # Heuristic: very high degree + low clustering → bot-like hub
            score = 1.0 / (1.0 + torch.exp(torch.tensor(-(d - 5.0))))  # sigmoid(degree - 5)
            score = float(0.7 * score + 0.3 * (1.0 - c))
            node_scores[n] = max(0.0, min(1.0, score))
        cluster_risk = float(sum(node_scores.values()) / max(len(node_scores), 1))
        return node_scores, cluster_risk

    # torch_geometric path
    # Ensure nodes are relabeled to 0..N-1 for PyG
    g_relabeled = nx.convert_node_labels_to_integers(g)
    data = from_networkx(g_relabeled)
    num_nodes = g_relabeled.number_of_nodes()
    model = SimpleGCN(config, num_nodes=num_nodes)

    model.eval()
    with torch.no_grad():
        scores = model(data)  # (num_nodes,)

    node_scores = {int(n): float(scores[i]) for i, n in enumerate(g_relabeled.nodes())}
    cluster_risk = float(scores.mean())
    return node_scores, cluster_risk

