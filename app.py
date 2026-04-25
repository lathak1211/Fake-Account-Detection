from __future__ import annotations
import torch
import json
import ijson
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from core.behavior_lstm import BehaviorLSTM, BehaviorLSTMConfig
from core.temporal_transformer import TemporalTransformer, TemporalTransformerConfig
from core.graph_gnn import run_gnn_on_graph
from core.nlp_module import ContentNLPEngine, NLPConfig
from core.fusion import ScoreFusion, FusionConfig
from features.behavior import build_behavior_sequence
from features.graph_features import build_interaction_graph, largest_components
from features.lifecycle import infer_lifecycle_stage
from deep_models import compute_deep_scores_for_users, get_gnn_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier


# TwiBot-22 data loading helpers
@st.cache_resource
def load_user_id_mapping(path: str = "TwiBot-22/user.json") -> Dict[int, str]:
    """Load mapping from node index to user_id for TwiBot-22 using streaming JSON parsing."""
    mapping = {}
    with open(path, "rb") as f:  # ijson requires binary mode
        for i, item in enumerate(ijson.items(f, "item")):
            user_id = str(item.get("id", item.get("user_id", "")))
            if user_id:
                mapping[i] = user_id
    return mapping

@st.cache_resource
def load_edges(path_index: str = "TwiBot-22/edge_index.pt", path_type: str = "TwiBot-22/edge_type.pt") -> pd.DataFrame:
    """Load TwiBot-22 graph data from PyTorch tensors and convert to DataFrame."""
    edge_index = torch.load(path_index)
    edge_type = torch.load(path_type)
    # Convert to DataFrame
    df = pd.DataFrame({
        "src": edge_index[0].numpy(),
        "dst": edge_index[1].numpy(),
        "relation": edge_type.numpy()
    })
    # Filter to user-user relations (assuming 0=following, 1=followers)
    df = df[df["relation"].isin([0, 1])].copy()
    # Map indices to user_ids
    mapping = load_user_id_mapping()
    df["src"] = df["src"].map(mapping).astype(str)
    df["dst"] = df["dst"].map(mapping).astype(str)
    return df[["src", "dst"]]


@st.cache_resource
def load_tweets(path: str = "TwiBot-22/tweet_8.json", sample_rate: float = 0.02, max_rows: int = 200000) -> pd.DataFrame:
    """Stream tweets from JSON array using ijson for memory efficiency."""
    import random
    rows = []
    with open(path, "rb") as f:  # ijson requires binary mode
        for tweet in ijson.items(f, "item"):
            if random.random() > sample_rate:
                continue
            author_id = tweet.get("author_id")
            created_at = tweet.get("created_at")
            if author_id and created_at:
                rows.append({
                    "user_id": str(author_id),
                    "timestamp": created_at,
                    "event_type": "tweet"
                })
            if len(rows) >= max_rows:
                break
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df


@st.cache_resource
def load_labels(path: str = "TwiBot-22/label.csv") -> pd.DataFrame:
    """Load labels (human/bot) and map to binary target."""
    labels = pd.read_csv(path)
    labels = labels.rename(columns={labels.columns[0]: "user_id", labels.columns[1]: "label"})
    labels["label"] = labels["label"].astype(str).str.lower().map({"human": 0, "bot": 1})
    if labels["label"].isna().any():
        raise ValueError("Unknown labels found in label.csv")
    labels["user_id"] = labels["user_id"].astype(str)
    return labels


@st.cache_resource
def load_users(path: str = "TwiBot-22/user.json", labeled_user_ids: set = None) -> pd.DataFrame:
    """Load user profiles with strong bot detection features using streaming JSON parsing.
    
    Args:
        path: Path to user.json file
        labeled_user_ids: Optional set of user IDs to filter. If provided, only loads these users.
    """
    chunk_size = 20000
    rows = []
    chunks = []
    with open(path, "rb") as f:  # ijson requires binary mode
        for user in ijson.items(f, "item"):
            user_id = str(user.get("id", user.get("user_id", "")))
            if user_id and (labeled_user_ids is None or user_id in labeled_user_ids):
                followers_count = int(user.get("followers_count", user.get("followers", 0)) or 0)
                following_count = int(user.get("following_count", user.get("friends_count", 0)) or 0)
                tweet_count = int(user.get("statuses_count", user.get("tweet_count", 0)) or 0)
                listed_count = int(user.get("listed_count", 0) or 0)
                
                # STRONG BOT DETECTION FEATURES
                # 1. Followers/Following ratio (bots often follow many, get few followers)
                ff_ratio = followers_count / (following_count + 1)
                
                # 2. Log transforms (captures exponential bot behavior patterns)
                log_followers = np.log1p(followers_count)
                log_following = np.log1p(following_count)
                log_tweets = np.log1p(tweet_count)
                
                # 3. Engagement score (bots tweet a lot but have few followers)
                engagement = tweet_count / (followers_count + 1)
                
                # 4. Activity score (engagement relative to network size)
                activity_score = tweet_count / (followers_count + following_count + 1)
                
                # 5. Interaction score (killer feature: followers * tweets / following)
                interaction_score = followers_count * tweet_count / (following_count + 1)
                
                rows.append({
                    "user_id": user_id,
                    "followers_count": followers_count,
                    "following_count": following_count,
                    "tweet_count": tweet_count,
                    "listed_count": listed_count,
                    "ff_ratio": ff_ratio,
                    "log_followers": log_followers,
                    "log_following": log_following,
                    "log_tweets": log_tweets,
                    "engagement": engagement,
                    "activity_score": activity_score,
                    "interaction_score": interaction_score,
                })

                if len(rows) >= chunk_size:
                    chunks.append(pd.DataFrame(rows))
                    rows = []

    if rows:
        chunks.append(pd.DataFrame(rows))

    if chunks:
        users = pd.concat(chunks, ignore_index=True)
    else:
        users = pd.DataFrame(columns=[
            "user_id", "followers_count", "following_count", "tweet_count", "listed_count",
            "ff_ratio", "log_followers", "log_following", "log_tweets", "engagement",
            "activity_score", "interaction_score"
        ])

    users = users[users["user_id"] != ""].drop_duplicates(subset=["user_id"]).reset_index(drop=True)
    users["user_id"] = users["user_id"].astype(str)

    # Convert numeric types to reduce memory usage
    for col in users.select_dtypes(include=["int64"]).columns:
        users[col] = pd.to_numeric(users[col], downcast="integer")
    for col in users.select_dtypes(include=["float64"]).columns:
        users[col] = pd.to_numeric(users[col], downcast="float")

    users = users.set_index("user_id")
    return users


@st.cache_resource
def load_split(path: str = "TwiBot-22/split.csv") -> pd.DataFrame:
    """Load pre-defined train/valid/test splits from TwiBot-22."""
    split = pd.read_csv(path)
    split = split.rename(columns={split.columns[0]: "user_id", split.columns[1]: "split"})
    split["user_id"] = split["user_id"].astype(str)
    return split


@st.cache_resource
def load_models() -> Dict[str, Any]:
    behavior_model = BehaviorLSTM(BehaviorLSTMConfig(input_dim=5))
    temporal_model = TemporalTransformer(TemporalTransformerConfig(input_dim=5))
    nlp_engine = ContentNLPEngine(NLPConfig(use_embeddings=False))
    fusion = ScoreFusion(FusionConfig(use_meta_classifier=True))
    return {
        "behavior": behavior_model.eval(),
        "temporal": temporal_model.eval(),
        "nlp": nlp_engine,
        "fusion": fusion,
    }


def set_page_config() -> None:
    st.set_page_config(
        page_title="Fake Social Media Detection Lab",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_style():
    st.markdown("""
    <style>

    /* ===== Base Layout ===== */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #0b1120 100%);
        color: #e2e8f0;
    }

    section.main > div {
        padding-top: 2rem;
        max-width: 1200px;
        margin: auto;
    }

    /* ===== Sidebar ===== */
    section[data-testid="stSidebar"] {
        background-color: #0b1220;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    section[data-testid="stSidebar"] * {
        color: #cbd5e1;
    }

    /* ===== Titles ===== */
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: -0.5px;
    }

    h1 {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }

    h2 {
        font-size: 1.4rem;
        margin-top: 2rem;
    }

    h3{
    margin-top:20px;
    }

    /* ===== Modern Metric Cards ===== */
    .metric-card {
        background: linear-gradient(145deg,#0f172a,#111827);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.4rem;
        text-align: center;
        transition: all 0.25s ease;
        box-shadow: 0 6px 18px rgba(0,0,0,0.35);
        backdrop-filter: blur(6px);
    }

    .metric-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(59,130,246,0.6);
        box-shadow: 0 10px 25px rgba(0,0,0,0.45);
    }

    .section-title {
        font-size: 2rem;
        font-weight: 700;
        color: #3b82f6;
    }

    .subtle {
        font-size: 0.9rem;
        color: #94a3b8;
    }

    .pipeline-container{
    display:flex;
    align-items:center;
    gap:12px;
    flex-wrap:wrap;
    padding:18px;
    margin-top:10px;

    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.05);
    border-radius:14px;
    }

    .pipe{
    width:170px;
    padding:14px;
    border-radius:12px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.08);
}

.pipe:hover{
    border-color:#3b82f6;
    box-shadow:0 0 8px rgba(59,130,246,0.25);
}

.pipe-desc{
    font-size:12px;
    color:#cbd5f5;
}

    .block-container{
    padding-top:2rem;
    padding-bottom:3rem;
    }

    .section-card{
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.06);
    border-radius:14px;
    padding:20px;
    margin-top:15px;
    }

    .summary-card{
    background:rgba(59,130,246,0.08);
    border:1px solid rgba(59,130,246,0.25);
    padding:16px;
    border-radius:12px;
    margin-top:15px;
    }

    .warning-banner{
    background:rgba(234,179,8,0.08);
    border:1px solid rgba(234,179,8,0.3);
    padding:14px;
    border-radius:10px;
    margin-top:10px;
    }

.arrow{
    font-size:20px;
    color:#64748b;
}

    /* ===== Metric Highlight ===== */
    .big-metric {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
    }

    /* ===== Buttons ===== */
    .stButton>button {
        border-radius: 8px;
        background-color: #2563eb;
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-1px);
    }

    /* ===== File Upload Box ===== */
    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 1rem;
        border: 1px dashed rgba(255,255,255,0.15);
    }

    /* ===== Expander ===== */
    .streamlit-expanderHeader {
        font-weight: 500;
    }

    /* ===== Table Styling ===== */
    .stDataFrame {
        background: rgba(255,255,255,0.02);
        border-radius: 10px;
    }

    /* ===== Remove harsh borders ===== */
    hr {
        border: none;
        height: 1px;
        background: rgba(255,255,255,0.08);
        margin: 2rem 0;
    }

    </style>
    """, unsafe_allow_html=True)



def overview_page(models: Dict[str, Any]) -> None:
    st.markdown("### Overview")
    st.markdown(
        "A modular, **research-grade prototype** for fake account and content detection "
        "combining deep learning, graph analysis, and NLP in a unified dashboard."
    )
    st.markdown(
        "<div style='height:1px;background:rgba(255,255,255,0.08);margin:25px 0;'></div>",
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
        <b>Behavioral Model</b><br>
        <span class="subtle">LSTM + Transformer</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
        <b>Graph Module</b><br>
        <span class="subtle">GNN / heuristics on interactions</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
        <b>NLP Engine</b><br>
        <span class="subtle">TF-IDF + Logistic Regression</span>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
        <b>Fusion</b><br>
        <span class="subtle">Meta-classifier / ensemble</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Detection Pipeline")

    pipeline = """
<div class="pipeline-container">

<div class="pipe pipe-blue">
<b>Ingest</b><br>
<span class="pipe-desc">Collect events & posts</span>
</div>

<div class="arrow">→</div>

<div class="pipe pipe-purple">
<b>Behavior</b><br>
<span class="pipe-desc">Analyze activity patterns</span>
</div>

<div class="arrow">→</div>

<div class="pipe pipe-purple">
<b>Graph</b><br>
<span class="pipe-desc">Detect networks</span>
</div>

<div class="arrow">→</div>

<div class="pipe pipe-green">
<b>NLP</b><br>
<span class="pipe-desc">Analyze content</span>
</div>

<div class="arrow">→</div>

<div class="pipe pipe-yellow">
<b>Lifecycle</b><br>
<span class="pipe-desc">Track account stage</span>
</div>

<div class="arrow">→</div>

<div class="pipe pipe-orange">
<b>Fusion</b><br>
<span class="pipe-desc">Combine model scores</span>
</div>

<div class="arrow">→</div>

<div class="pipe pipe-red">
<b>Linking</b><br>

    
<span class="pipe-desc">Cross-platform identity</span>
</div>

</div>
"""

    st.markdown(pipeline, unsafe_allow_html=True)


def single_account_page(models: Dict[str, Any]) -> None:
    st.markdown("### Single Account Analysis")
    st.markdown(
        "Upload per-event logs and content samples for one or more accounts. "
        "This page runs the full pipeline per account."
    )

    with st.expander("Input format", expanded=False):
        st.write(
            "Expected **events CSV/JSON** columns: `user_id`, `timestamp`, `event_type`.\n\n"
            "Optional **content CSV/JSON** columns: `user_id`, `text`."
        )

    with st.container():
        events_file = st.file_uploader(
            "Upload events CSV/JSON",
            type=["csv", "json"],
            key="events_upload"
        )

        content_file = st.file_uploader(
            "Upload content CSV/JSON (optional)",
            type=["csv", "json"],
            key="content_upload"
        )

    if events_file is None:
        st.info("Loading TwiBot-22 tweet events stream.")
        events = load_tweets()
    else:
        if events_file.name.endswith(".csv"):
            events = pd.read_csv(events_file)
        else:
            events = pd.read_json(events_file)

    if content_file is None:
        contents = pd.DataFrame(columns=["user_id", "text"])
    else:
        if content_file.name.endswith(".csv"):
            contents = pd.read_csv(content_file)
        else:
            contents = pd.read_json(content_file)

    users = sorted(events["user_id"].astype(str).unique())
    if not users:
        st.warning("No users found in the events data.")
        return

    selected_user = st.selectbox("Select user", options=users)

    user_events = events[events["user_id"].astype(str) == selected_user]
    user_contents = contents[contents["user_id"].astype(str) == selected_user]

    seq_tensor, user_ids = build_behavior_sequence(user_events, max_seq_len=200)
    seq_tensor = seq_tensor.float()

    if seq_tensor.shape[0] == 0:
        st.warning("Insufficient data to build behavioral sequence for this user.")
        return

    behavior_model: BehaviorLSTM = models["behavior"]
    temporal_model: TemporalTransformer = models["temporal"]
    nlp_engine: ContentNLPEngine = models["nlp"]
    fusion: ScoreFusion = models["fusion"]

    with st.spinner("Running models..."):
        with torch.no_grad():
            lstm_score = float(behavior_model(seq_tensor)[0].item())
            transformer_score = float(temporal_model(seq_tensor)[0].item())
        if not user_contents.empty:
            content_scores = nlp_engine.content_risk_scores(user_contents["text"].astype(str).tolist())
            content_score = float(float(np.mean(content_scores)))
        else:
            content_score = 0.0

        # Compute GNN score
        edges = load_edges()
        g = build_interaction_graph(edges)
        node_scores, _ = run_gnn_on_graph(g)
        gnn_score = get_gnn_score(selected_user, g, node_scores)

        # Placeholder for XGBoost score (since model not trained for single user)
        xgb_score = 0.5

        fused = fusion.fuse_scores(
            lstm=lstm_score,
            transformer=transformer_score,
            gnn=gnn_score,
            xgb=xgb_score,
            content=content_score,
        )

    label = fused["label"]
    risk = float(fused["risk_score"])

    st.info(
        f"""
**Account Summary for {selected_user}**

• Final Label: {label}  
• Risk Score: {risk:.1f}  
• Most Influential Signal: {"LSTM" if fused["lstm"] > fused["transformer"] and fused["lstm"] > fused["gnn"] else "Transformer" if fused["transformer"] > fused["gnn"] else "GNN"}
"""
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Final Label</div>
        <div style="font-size:28px;font-weight:700;color:{'#16a34a' if risk < 40 else '#f59e0b' if risk < 70 else '#dc2626'};">
            {label}
        </div>
    </div>
    """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Risk Score (0–100)</div>
        <div class="section-title">{risk:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

        st.progress(int(max(0, min(100, risk))))
        st.caption("0 = Safe · 100 = Highly Suspicious")

    with col3:
        st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Fake Probability</div>
        <div class="section-title">{fused['prob_fake']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    if risk < 40:
        st.success("This account shows mostly normal behavioral patterns.")
    elif risk < 70:
        st.markdown("""
<div class="warning-banner">
This account shows moderate suspicious signals.
</div>
""", unsafe_allow_html=True)
    else:
        st.error("This account exhibits strong coordinated or bot-like behavior.")

    st.markdown("#### Deep Learning Model Scores")
    comp_cols = st.columns(5)
    scores = {
        "LSTM": fused["lstm"],
        "Transformer": fused["transformer"],
        "GNN": fused["gnn"],
        "XGBoost": fused["xgb"],
        "Content (NLP)": fused["content"],
    }
    for col, (name, value) in zip(comp_cols, scores.items()):
        with col:
            st.metric(name, f"{value:.2f}")
            st.progress(int(max(0, min(100, value * 100))))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Behavior Timeline")
    plot_df = user_events.copy()
    plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"])
    daily_counts = plot_df.groupby(plot_df["timestamp"].dt.date).size().reset_index(name="events")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        daily_counts["timestamp"],
        daily_counts["events"],
        marker="o",
        linewidth=2,
        color="#2563eb",
        label="Number of user actions per day"
    )
    ax.legend()
    ax.set_title(
        "Daily Activity Pattern (User Events Over Time)",
        fontsize=14,
        weight="bold"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of User Events")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        "Each point represents the number of recorded user actions on a given day. "
        "Sudden spikes or highly repetitive patterns may indicate automated or bot-like behavior."
    )

    if not user_contents.empty:
        st.markdown("#### Sample Content")
        for _, row in user_contents.head(5).iterrows():
            st.write(f"- {row['text']}")



def bot_cluster_page(models: Dict[str, Any]) -> None:
    st.markdown("### Bot Cluster Detection")
    st.markdown(
        "Analyze interaction graphs (followers, mentions, replies) to surface **coordinated bot clusters**."
    )

    edges_file = st.file_uploader("Upload interaction edges CSV/JSON", type=["csv", "json"], key="cluster_edges")
    if edges_file is None:
        st.info("Loading TwiBot-22 edge list.")
        edges = load_edges()
    else:
        if edges_file.name.endswith(".csv"):
            edges = pd.read_csv(edges_file)
        else:
            edges = pd.read_json(edges_file)

    if edges.empty:
        st.warning("No edges found.")
        return

    g = build_interaction_graph(edges)
    node_scores, cluster_risk = run_gnn_on_graph(g)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Global Cluster Risk</div>
            <div class="section-title">{cluster_risk:.2f}</div>
            <div class="subtle">Higher values indicate more coordinated behavior</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Number of Nodes / Edges</div>
            <div class="section-title">{g.number_of_nodes()} nodes · {g.number_of_edges()} edges</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Suspicious Nodes")
    top_nodes = sorted(node_scores.items(), key=lambda kv: kv[1], reverse=True)[:20]
    top_df = pd.DataFrame(top_nodes, columns=["node", "bot_score"])
    st.dataframe(top_df, use_container_width=True)

    st.markdown("#### Largest Components (graph view)")
    for comp_idx, nodes in largest_components(g, k=3):
        sub = g.subgraph(nodes)
        st.markdown(f"**Component {comp_idx}** – {len(nodes)} nodes")

        # Skip visualization for very large components to avoid memory errors
        if len(nodes) > 1000:
            st.warning(f"Component too large to visualize ({len(nodes)} nodes). "
                      f"Spring layout requires O(n²) memory and computation.")
            continue

        # Matplotlib-based visualization to avoid requiring pydot/graphviz
        pos = nx.spring_layout(sub, seed=42)
        scores = np.array([node_scores.get(n, 0.0) for n in sub.nodes])
        norm_scores = (scores - scores.min()) / (np.ptp(scores) + 1e-6)
        cmap = plt.cm.viridis
        node_colors = cmap(norm_scores)

        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        nx.draw_networkx_edges(sub, pos, ax=ax, alpha=0.4)
        nx.draw_networkx_nodes(
            sub,
            pos,
            node_color=node_colors,
            ax=ax,
            node_size=300,
            edgecolors="black",
            linewidths=0.5,
        )
        nx.draw_networkx_labels(
            sub,
            pos,
            ax=ax,
            font_size=8,
            font_color="white"
        )
        ax.set_title("Interaction Network", fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(
            "Nodes represent accounts and edges represent interactions such as mentions or replies. "
            "Clusters of tightly connected nodes may indicate coordinated bot activity."
        )


def lifecycle_page() -> None:
    st.markdown("### Lifecycle Analysis")
    st.markdown(
        "Estimate the **lifecycle stage** of accounts (creation, warm-up, attack, dormant) "
        "from their activity timelines."
    )

    events_file = st.file_uploader("Upload events CSV/JSON", type=["csv", "json"], key="lifecycle_events")
    if events_file is None:
        st.info("Loading TwiBot-22 tweet events stream.")
        events = load_tweets()
    else:
        if events_file.name.endswith(".csv"):
            events = pd.read_csv(events_file)
        else:
            events = pd.read_json(events_file)

    if events.empty:
        st.warning("No events found.")
        return

    summary = infer_lifecycle_stage(events)
    st.dataframe(summary, use_container_width=True)

    stage_counts = summary["stage"].value_counts().rename_axis("stage").reset_index(name="count")
    # Use Matplotlib instead of Altair-backed bar_chart to avoid jsonschema/altair issues
    fig, ax = plt.subplots(figsize=(3, 2.2), facecolor="#0b1120")
    ax.set_facecolor("#0b1120")

    # color mapping for lifecycle stages
    stage_colors = {
        "creation": "#22c55e",   # green
        "warm-up": "#eab308",    # yellow
        "attack": "#ef4444",     # red
        "dormant": "#64748b"     # gray
    }
    colors = [stage_colors.get(stage, "#3b82f6") for stage in stage_counts["stage"]]

    bars = ax.bar(
        stage_counts["stage"],
        stage_counts["count"],
        color=colors,
        edgecolor="#94a3b8"
    )
    for bar in bars:
        height = bar.get_height()
        ax.text(
        bar.get_x() + bar.get_width()/2,
        height/2,
        f"{int(height)}",
        ha="center",
        va="center",
        color="white",
        fontsize=9,
        fontweight="bold"
    )

    ax.set_title("Account Lifecycle Distribution", fontsize=10)
    ax.set_xlabel("Lifecycle Stage",fontsize=9)
    ax.set_ylabel("Number of Accounts",fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(colors="#cbd5e1")
    ax.yaxis.label.set_color("#cbd5e1")
    ax.xaxis.label.set_color("#cbd5e1")
    ax.title.set_color("#e2e8f0")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)
    st.caption(
        "Lifecycle stages describe how an account behaves over time. "
        "'Creation' indicates newly created accounts, while 'Warm-up' accounts "
        "are gradually increasing activity to appear legitimate before coordinated actions."
    )
    st.caption(
        "Green = newly created accounts • Yellow = warming up • Red = active attack stage • Gray = dormant accounts"
    )


def cross_platform_page() -> None:
    st.markdown("### Cross-Platform Linking")
    st.markdown(
        "Compare usernames, behaviors, and content style across platforms to estimate "
        "whether two accounts may belong to the same underlying entity."
    )

    with st.expander("Input schema", expanded=False):
        st.write(
            "Provide two JSON blobs (one per platform) with fields:\n"
            "- `username`\n"
            "- `behavior_vector`: list of numeric features\n"
            "- `content_vector`: list of numeric features\n\n"
            "For example:\n"
            "```json\n"
            "{\n"
            '  "username": "bot_army_01",\n'
            '  "behavior_vector": [0.1, 0.9, 0.3],\n'
            '  "content_vector": [0.2, 0.2, 0.8]\n'
            "}\n"
            "```"
        )

    col1, col2 = st.columns(2)
    with col1:
        blob_a = st.text_area("Platform A JSON", height=150)
    with col2:
        blob_b = st.text_area("Platform B JSON", height=150)

    default_demo = st.checkbox("Use demo example", value=True)
    if default_demo:
        blob_a = json.dumps(
            {
                "username": "news_watchdog",
                "behavior_vector": [0.2, 0.7, 0.4],
                "content_vector": [0.3, 0.5, 0.2],
            },
            indent=2,
        )
        blob_b = json.dumps(
            {
                "username": "NewsWatchDog",
                "behavior_vector": [0.21, 0.68, 0.39],
                "content_vector": [0.32, 0.49, 0.18],
            },
            indent=2,
        )

    if st.button("Compute similarity"):
        try:
            a = json.loads(blob_a)
            b = json.loads(blob_b)
        except Exception as e:
            st.error(f"Failed to parse JSON: {e}")
            return

        overall, details = cross_platform_similarity(
            username_a=a["username"],
            username_b=b["username"],
            behavior_vec_a=a["behavior_vector"],
            behavior_vec_b=b["behavior_vector"],
            content_vec_a=a["content_vector"],
            content_vec_b=b["content_vector"],
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Overall Similarity</div>
                <div class="section-title">{overall:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Possible Linked Identity</div>
                <div class="section-title">{'Yes' if overall >= 0.7 else 'Unclear'}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Breakdown")
            st.dataframe(
    pd.DataFrame(
        details.items(),
        columns=["Metric", "Score"]
    ),
    use_container_width=True
)


def cross_platform_similarity(
    username_a: str,
    username_b: str,
    behavior_vec_a: List[float],
    behavior_vec_b: List[float],
    content_vec_a: List[float],
    content_vec_b: List[float],
) -> tuple[float, Dict[str, float]]:
    """Compute similarity between two cross-platform accounts using multiple metrics."""
    from difflib import SequenceMatcher
    
    # Username similarity (Levenshtein-like)
    username_similarity = SequenceMatcher(None, username_a.lower(), username_b.lower()).ratio()
    
    # Behavior vector similarity (cosine)
    behavior_vec_a = np.array(behavior_vec_a, dtype=float)
    behavior_vec_b = np.array(behavior_vec_b, dtype=float)
    behavior_norm_a = np.linalg.norm(behavior_vec_a)
    behavior_norm_b = np.linalg.norm(behavior_vec_b)
    
    if behavior_norm_a > 0 and behavior_norm_b > 0:
        behavior_similarity = float(np.dot(behavior_vec_a, behavior_vec_b) / (behavior_norm_a * behavior_norm_b))
    else:
        behavior_similarity = 0.0
    
    # Content vector similarity (cosine)
    content_vec_a = np.array(content_vec_a, dtype=float)
    content_vec_b = np.array(content_vec_b, dtype=float)
    content_norm_a = np.linalg.norm(content_vec_a)
    content_norm_b = np.linalg.norm(content_vec_b)
    
    if content_norm_a > 0 and content_norm_b > 0:
        content_similarity = float(np.dot(content_vec_a, content_vec_b) / (content_norm_a * content_norm_b))
    else:
        content_similarity = 0.0
    
    # Weighted overall similarity
    overall = 0.2 * username_similarity + 0.4 * behavior_similarity + 0.4 * content_similarity
    
    details = {
        "username_similarity": username_similarity,
        "behavior_similarity": behavior_similarity,
        "content_similarity": content_similarity,
    }
    
    return overall, details


def build_node_feature_matrix(users_df: pd.DataFrame, graph: nx.Graph) -> pd.DataFrame:
    """Combine user metadata with graph structural features for bot detection."""
    base = users_df.copy().reset_index()

    # Extract degree features (CRITICAL for bot detection: bots have unusual degree patterns)
    out_deg = dict(graph.out_degree(base["user_id"], weight="weight")) if isinstance(graph, nx.DiGraph) else dict(graph.degree(base["user_id"], weight="weight"))
    in_deg = dict(graph.in_degree(base["user_id"], weight="weight")) if isinstance(graph, nx.DiGraph) else out_deg

    base["out_degree"] = base["user_id"].map(lambda x: float(out_deg.get(x, 0)))
    base["in_degree"] = base["user_id"].map(lambda x: float(in_deg.get(x, 0)))

    # Compute degree_ratio using vectorized operations (memory efficient)
    base["degree_ratio"] = base["out_degree"] / (base["in_degree"] + 1e-6)

    # Local clustering: bots often have sparse local clusters
    try:
        clustering = nx.clustering(graph)
        base["clustering_coefficient"] = base["user_id"].map(lambda x: float(clustering.get(x, 0)))
    except:
        base["clustering_coefficient"] = 0.0  # Fallback if computation fails

    # Global centrality: bots may have unusual network positions
    # Skip for very large graphs to avoid memory issues
    if len(graph.nodes()) > 50000:
        base["eigenvector_centrality"] = 0.0  # Skip for large graphs
    else:
        try:
            eigenvector = nx.eigenvector_centrality_numpy(graph, max_iter=50)  # Reduced iterations
            base["eigenvector_centrality"] = base["user_id"].map(lambda x: float(eigenvector.get(x, 0)))
        except:
            base["eigenvector_centrality"] = 0.0  # Fallback if computation fails

    return base.set_index("user_id")


def train_evaluate_twi_bot(user_features: pd.DataFrame, labels: pd.DataFrame, split: pd.DataFrame, tweets: pd.DataFrame = None, edges: pd.DataFrame = None, models: Dict = None) -> tuple[pd.DataFrame, dict]:
    """Train XGBoost with threshold tuning on TwiBot-22 splits for optimal F1 score."""
    # FIX: Properly merge all data on user_id (memory safe)
    user_features = user_features.reset_index(names=["user_id"])
    
    data = user_features.merge(labels, on="user_id", how="inner")
    data = data.merge(split, on="user_id", how="inner")
    
    # Reduce memory usage
    for col in data.select_dtypes(include=['float64']).columns:
        data[col] = data[col].astype('float32')
    
    for col in data.select_dtypes(include=['int64']).columns:
        data[col] = data[col].astype('int32')
    
    data = data.dropna().copy()

    if data.empty:
        raise ValueError("No users available after merge; check TwiBot-22 sources and user mapping.")

    # DEBUG: Show actual split values
    st.info(f" Merged {len(data)} users with labels and splits")
    st.info(f"Split distribution: {dict(data['split'].value_counts())}")
    
    feature_cols = [c for c in data.columns if c not in ["user_id", "label", "split"]]
    st.info(f"Using {len(feature_cols)} features: {feature_cols}")
    
    # Load predefined splits from split.csv
    train_data = data[data["split"] == "train"]
    valid_data = data[data["split"] == "val"]
    test_data = data[data["split"] == "test"]
    
    if len(train_data) == 0 or len(valid_data) == 0 or len(test_data) == 0:
        st.warning(f" Split issue: Train={len(train_data)}, Valid={len(valid_data)}, Test={len(test_data)}")
        st.warning(f"Unique splits in data: {data['split'].unique()}")
        from sklearn.model_selection import train_test_split
        # Sample data to avoid memory issues
        if len(data) > 100000:
            data = data.sample(n=100000, random_state=42)
        temp_train, test_data = train_test_split(data, test_size=0.2, stratify=data["label"], random_state=42)
        train_data, valid_data = train_test_split(temp_train, test_size=0.25, stratify=temp_train["label"], random_state=42)

    # Sample training data for memory efficiency (max 50K)
    max_train_samples = 50000
    if len(train_data) > max_train_samples:
        train_data = train_data.sample(n=max_train_samples, random_state=42)

    if len(train_data) == 0 or len(valid_data) == 0 or len(test_data) == 0:
        raise ValueError(f"Insufficient samples: Train={len(train_data)}, Valid={len(valid_data)}, Test={len(test_data)}")

    X_train = train_data[feature_cols].astype(float)
    y_train = train_data["label"].astype(int)
    X_valid = valid_data[feature_cols].astype(float)
    y_valid = valid_data["label"].astype(int)
    X_test = test_data[feature_cols].astype(float)
    y_test = test_data["label"].astype(int)

    # Calculate positive weight for class imbalance (CRITICAL)
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale = neg / pos
    st.info(f"Class balance: {neg} humans, {pos} bots. Scale pos_weight={scale:.2f}")

    # XGBOOST: The better model for imbalanced bot detection
    model = XGBClassifier(
        n_estimators=300,  # Higher for better generalization
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale,  # CRITICAL for class imbalance
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train, verbose=False)

    # THRESHOLD TUNING: Find best threshold for F1 score 
    y_valid_probs = model.predict_proba(X_valid)[:, 1]
    y_test_probs = model.predict_proba(X_test)[:, 1]
    
    best_f1 = 0
    best_threshold = 0.5
    threshold_results = []
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred_valid = (y_valid_probs > threshold).astype(int)
        f1 = f1_score(y_valid, y_pred_valid, zero_division=0)
        threshold_results.append((threshold, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    st.info(f" Best threshold: {best_threshold} (F1={best_f1:.4f})")
    
    # Evaluate with best threshold on all splits
    metrics_list = []
    confusion_matrices = {}
    
    for split_name, X_split, y_split, y_probs in [("Train", X_train, y_train, model.predict_proba(X_train)[:, 1]),
                                                     ("Validation", X_valid, y_valid, y_valid_probs),
                                                     ("Test", X_test, y_test, y_test_probs)]:
        # Use best threshold
        y_pred = (y_probs > best_threshold).astype(int)
        cm = confusion_matrix(y_split, y_pred)

        metrics_list.append({
            "Model": "XGBoost (Tuned)",
            "Split": split_name,
            "Accuracy": accuracy_score(y_split, y_pred),
            "Precision": precision_score(y_split, y_pred, zero_division=0),
            "Recall": recall_score(y_split, y_pred, zero_division=0),
            "F1 Score": f1_score(y_split, y_pred, zero_division=0),
        })
        confusion_matrices[f"XGBoost_{split_name}"] = cm

    # Add Fusion Model evaluation if models provided
    if models is not None and tweets is not None and edges is not None:
        # Compute deep scores for test users
        test_user_ids = test_data["user_id"].tolist()
        # Filter tweets and edges to only test users to save memory
        test_user_set = set(test_user_ids)
        tweets_filtered = tweets[tweets["user_id"].astype(str).isin(test_user_set)].copy()
        edges_filtered = edges[edges["src"].astype(str).isin(test_user_set) | edges["dst"].astype(str).isin(test_user_set)].copy()
        deep_scores = compute_deep_scores_for_users(tweets_filtered, edges_filtered, test_user_ids, models)
        
        # Get XGBoost probs for test
        xgb_probs = model.predict_proba(X_test)[:, 1]
        
        # Fuse scores
        fusion = ScoreFusion(FusionConfig(use_meta_classifier=True))
        fused_labels = []
        for i, user_id in enumerate(test_user_ids):
            ds = deep_scores.get(user_id, {'lstm': 0.5, 'transformer': 0.5, 'gnn': 0.5})
            fused = fusion.fuse_scores(
                lstm=ds['lstm'],
                transformer=ds['transformer'],
                gnn=ds['gnn'],
                xgb=xgb_probs[i],
                content=0.0
            )
            fused_labels.append(1 if fused['prob_fake'] >= 0.5 else 0)
        
        y_pred_fusion = np.array(fused_labels)
        cm_fusion = confusion_matrix(y_test, y_pred_fusion)
        
        metrics_list.append({
            "Model": "Fusion Model",
            "Split": "Test",
            "Accuracy": accuracy_score(y_test, y_pred_fusion),
            "Precision": precision_score(y_test, y_pred_fusion, zero_division=0),
            "Recall": recall_score(y_test, y_pred_fusion, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred_fusion, zero_division=0),
        })
        confusion_matrices["Fusion_Test"] = cm_fusion

    return pd.DataFrame(metrics_list), confusion_matrices


def twibot_pipeline_page(models: Dict[str, Any]) -> None:
    st.markdown("## TwiBot-22 Bot Detection Pipeline")
    st.markdown("Advanced machine learning system for detecting fake accounts using graph and behavioral features.")

    with st.spinner("Loading TwiBot-22 data..."):
        edges = load_edges()
        tweets = load_tweets()
        labels = load_labels()
        split = load_split()

        # Get labeled user IDs for efficient filtering during user loading
        labeled_user_ids = set(labels["user_id"].astype(str))
        users = load_users(labeled_user_ids=labeled_user_ids)

    # Data summary in a clean card
    st.markdown("""
    <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; margin: 10px 0;">
        <h4 style="color: #ffffff; margin-top: 0;">Data Summary</h4>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Edges", f"{len(edges):,}")
    with col2:
        st.metric("Tweets", f"{len(tweets):,}")
    with col3:
        st.metric("Users", f"{len(users):,}")
    with col4:
        st.metric("Labels", f"{len(labels):,}")

    g = build_interaction_graph(edges, src_col="src", dst_col="dst", directed=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Graph Nodes", f"{g.number_of_nodes():,}")
    with col2:
        st.metric("Graph Edges", f"{g.number_of_edges():,}")

    # Users are already filtered to labeled ones during loading
    user_features = users

    # Merge graph features
    graph_features = build_node_feature_matrix(user_features, g)
    merged_features = user_features.join(
        graph_features[["out_degree", "in_degree", "degree_ratio", "clustering_coefficient", "eigenvector_centrality"]],
        how="left"
    ).fillna(0.0)

    st.metric("Features per User", merged_features.shape[1])

    # Train and evaluate
    metrics_df, confusion_matrices = train_evaluate_twi_bot(merged_features, labels, split, tweets, edges, models)

    st.markdown("<br>", unsafe_allow_html=True)

    # Create tabs for clean organization
    tab1, tab2, tab3 = st.tabs(["Results", "Confusion Matrix", "Details"])

    with tab1:
        st.markdown("### Model Performance Summary")

        # Extract test results for highlight
        test_results = metrics_df[metrics_df["Split"] == "Test"]
        if not test_results.empty:
            test_row = test_results.iloc[0]

            # Key metrics in large cards at top
            st.markdown("""
            <div style="background-color: #2e7d32; padding: 20px; border-radius: 10px; margin: 10px 0; text-align: center;">
                <h2 style="color: #ffffff; margin: 0;">Test Set Results</h2>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{test_row['Accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{test_row['Precision']:.3f}")
            with col3:
                st.metric("Recall", f"{test_row['Recall']:.3f}")
            with col4:
                st.metric("F1 Score", f"{test_row['F1 Score']:.3f}")

        st.markdown("#### Complete Results Table")
        st.dataframe(metrics_df, use_container_width=True)

    with tab2:
        st.markdown("### Confusion Matrix Analysis")

        for key, cm in confusion_matrices.items():
            if "Test" in key:  # Show test matrix prominently
                st.markdown(f"#### {key}")
                cm_df = pd.DataFrame(cm, index=["Predicted Human", "Predicted Bot"], columns=["Actual Human", "Actual Bot"])
                st.dataframe(cm_df, use_container_width=True)

        # Other matrices in expander
        with st.expander("View All Confusion Matrices"):
            for key, cm in confusion_matrices.items():
                if "Test" not in key:
                    st.markdown(f"**{key}**")
                    cm_df = pd.DataFrame(cm, index=["Predicted Human", "Predicted Bot"], columns=["Actual Human", "Actual Bot"])
                    st.dataframe(cm_df, use_container_width=True)

    with tab3:
        st.markdown("### Technical Details")

        with st.expander("Split Distribution"):
            st.markdown("#### Sample Distribution Across Splits")
            split_counts = split["split"].value_counts().rename_axis("Split").reset_index(name="Count")
            st.dataframe(split_counts, use_container_width=True)

        with st.expander("Class Balance Information"):
            # Calculate class distribution
            class_dist = labels["label"].value_counts().rename({0: "Human", 1: "Bot"})
            st.markdown("#### Class Distribution in Labels")
            st.dataframe(class_dist.rename_axis("Class").reset_index(name="Count"), use_container_width=True)

        with st.expander("Feature Engineering Details"):
            st.markdown(f"""
            **Total Features:** {merged_features.shape[1]}

            **User Features (12):**
            - followers_count, following_count, tweet_count, listed_count
            - ff_ratio, log_followers, log_following, log_tweets
            - engagement, activity_score, interaction_score

            **Graph Features (5):**
            - out_degree, in_degree, degree_ratio
            - clustering_coefficient, eigenvector_centrality
            """)

        with st.expander("Model Configuration"):
            st.markdown("""
            **XGBoost Parameters:**
            - n_estimators: 300
            - max_depth: 6
            - learning_rate: 0.05
            - scale_pos_weight: dynamic (neg/pos ratio)
            - subsample: 0.8
            - colsample_bytree: 0.8

            **Threshold Tuning:** Grid search 0.1-0.9 for optimal F1
            """)


def main() -> None:
    set_page_config()
    inject_style()

    models = load_models()
    #models = {}

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Pages",
        options=[
            "Overview",
            "Single Account Analysis",
            "Bot Cluster Detection",
            "Lifecycle Analysis",
            "Cross-Platform Linking",
            "TwiBot-22 Pipeline",
        ],
    )

    if page == "Overview":
        overview_page(models)
    elif page == "Single Account Analysis":
        single_account_page(models)
    elif page == "Bot Cluster Detection":
        bot_cluster_page(models)
    elif page == "Lifecycle Analysis":
        lifecycle_page()
    elif page == "Cross-Platform Linking":
        cross_platform_page()
    elif page == "TwiBot-22 Pipeline":
        twibot_pipeline_page(models)


if __name__ == "__main__":
    main()

