from __future__ import annotations

import torch
from typing import Dict, List, Tuple
import networkx as nx
import pandas as pd
from core.behavior_lstm import BehaviorLSTM, BehaviorLSTMConfig
from core.temporal_transformer import TemporalTransformer, TemporalTransformerConfig
from core.graph_gnn import run_gnn_on_graph, GraphGNNConfig
from features.behavior import build_behavior_sequence
from features.graph_features import build_interaction_graph


def get_lstm_score(seq_tensor: torch.Tensor, model: BehaviorLSTM) -> float:
    """
    Get LSTM score for a single sequence tensor.
    Assumes seq_tensor is (1, seq_len, input_dim)
    """
    model.eval()
    with torch.no_grad():
        score = model(seq_tensor).item()
    return score


def get_transformer_score(seq_tensor: torch.Tensor, model: TemporalTransformer) -> float:
    """
    Get Transformer score for a single sequence tensor.
    Assumes seq_tensor is (1, seq_len, input_dim)
    """
    model.eval()
    with torch.no_grad():
        score = model(seq_tensor).item()
    return score


def get_gnn_score(user_id: str, g: nx.Graph, node_scores: Dict[str, float]) -> float:
    """
    Get GNN score for a single user from precomputed node_scores.
    """
    return node_scores.get(user_id, 0.5)  # default to 0.5 if not found


def compute_deep_scores_for_users(
    events_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    user_ids: List[str],
    models: Dict[str, any],
    batch_size: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    Compute deep learning scores for a list of users.
    Returns dict of {user_id: {'lstm': score, 'transformer': score, 'gnn': score}}
    Processes users in batches to manage memory efficiently.
    """
    # Build graph once (reuse for all batches)
    if not edges_df.empty:
        g = build_interaction_graph(edges_df, directed=True)
        node_scores, _ = run_gnn_on_graph(g)
    else:
        g = None
        node_scores = {}
    
    behavior_model = models["behavior"]
    temporal_model = models["temporal"]
    
    scores = {}
    user_ids_set = set(user_ids)
    
    # Process users in batches to manage memory
    for batch_start in range(0, len(user_ids), batch_size):
        batch_user_ids = user_ids[batch_start:batch_start + batch_size]
        batch_user_set = set(batch_user_ids)
        
        # Filter events to only this batch
        batch_events = events_df[events_df["user_id"].astype(str).isin(batch_user_set)].copy()
        
        if batch_events.empty:
            # No events for this batch, use defaults
            for user_id in batch_user_ids:
                scores[user_id] = {'lstm': 0.5, 'transformer': 0.5, 'gnn': 0.5}
            continue
        
        # Build sequences only for this batch
        seq_tensor, seq_user_ids = build_behavior_sequence(batch_events, max_seq_len=200)
        
        # Extract scores for each user in batch
        for i, user_id in enumerate(seq_user_ids):
            if user_id in batch_user_set:
                seq = seq_tensor[i:i+1]  # (1, seq_len, dim)
                lstm_score = get_lstm_score(seq, behavior_model)
                transformer_score = get_transformer_score(seq, temporal_model)
                gnn_score = get_gnn_score(user_id, g, node_scores) if g else 0.5
                scores[user_id] = {
                    'lstm': lstm_score,
                    'transformer': transformer_score,
                    'gnn': gnn_score
                }
        
        # Clean up batch tensor from memory
        del seq_tensor
        del batch_events
    
    # For users without sequences, use defaults
    for user_id in user_ids:
        if user_id not in scores:
            scores[user_id] = {'lstm': 0.5, 'transformer': 0.5, 'gnn': 0.5}
    
    return scores