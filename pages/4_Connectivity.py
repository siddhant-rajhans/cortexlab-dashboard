"""ROI Connectivity page."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils import (
    make_roi_indices,
    generate_brain_predictions,
    compute_connectivity,
    cluster_rois,
    graph_metrics,
    ROI_GROUPS,
)

st.set_page_config(page_title="ROI Connectivity", page_icon="🔗", layout="wide")
st.title("🔗 ROI Connectivity Analysis")
st.markdown("Functional connectivity between brain regions from predicted responses.")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    n_timepoints = st.slider("Duration (TRs)", 30, 200, 80)
    seed = st.number_input("Random seed", value=42, min_value=0)
    n_clusters = st.slider("Number of clusters", 2, 8, 4)
    threshold = st.slider("Edge threshold", 0.1, 0.8, 0.3, 0.05)

# --- Generate Data ---
roi_indices, n_vertices = make_roi_indices()
predictions = generate_brain_predictions(n_timepoints, n_vertices, seed)

# --- Correlation Matrix ---
corr_matrix, roi_names = compute_connectivity(predictions, roi_indices)

st.subheader("Correlation Matrix")
fig = go.Figure(go.Heatmap(
    z=corr_matrix,
    x=roi_names,
    y=roi_names,
    colorscale="RdBu_r",
    zmid=0,
    zmin=-1,
    zmax=1,
    colorbar=dict(title="Correlation"),
))
fig.update_layout(
    height=600,
    width=700,
    template="plotly_dark",
    xaxis=dict(tickangle=45, tickfont=dict(size=8)),
    yaxis=dict(tickfont=dict(size=8)),
)
st.plotly_chart(fig, use_container_width=True)

# --- Network Clusters ---
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.subheader("Functional Network Clusters")
    clusters, labels = cluster_rois(corr_matrix, roi_names, n_clusters)

    cluster_data = []
    for cid, rois in sorted(clusters.items()):
        for roi in rois:
            cluster_data.append({"Cluster": f"Network {cid}", "ROI": roi})

    cluster_df = pd.DataFrame(cluster_data)
    fig2 = px.bar(
        cluster_df.groupby("Cluster").size().reset_index(name="Count"),
        x="Cluster",
        y="Count",
        color="Cluster",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig2.update_layout(height=350, template="plotly_dark", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    for cid, rois in sorted(clusters.items()):
        st.markdown(f"**Network {cid}:** {', '.join(rois)}")

with col2:
    st.subheader("Degree Centrality")
    degrees = graph_metrics(corr_matrix, roi_names, threshold)

    # Sort by degree
    sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    deg_df = pd.DataFrame(sorted_degrees, columns=["ROI", "Degree Centrality"])

    fig3 = go.Figure(go.Bar(
        x=deg_df["Degree Centrality"],
        y=deg_df["ROI"],
        orientation="h",
        marker_color="#6C5CE7",
    ))
    fig3.update_layout(
        xaxis_title="Degree Centrality",
        xaxis_range=[0, 1],
        height=600,
        template="plotly_dark",
        yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
    )
    st.plotly_chart(fig3, use_container_width=True)

# --- Network Graph ---
st.divider()
st.subheader("Network Graph")

try:
    import networkx as nx

    G = nx.Graph()
    for name in roi_names:
        G.add_node(name)

    for i in range(len(roi_names)):
        for j in range(i + 1, len(roi_names)):
            if abs(corr_matrix[i, j]) > threshold:
                G.add_edge(roi_names[i], roi_names[j], weight=abs(corr_matrix[i, j]))

    pos = nx.spring_layout(G, seed=seed, k=2.0)

    # Cluster colors
    color_map = px.colors.qualitative.Set2
    node_colors = [color_map[(labels[i] - 1) % len(color_map)] for i in range(len(roi_names))]

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [pos[n][0] for n in roi_names]
    node_y = [pos[n][1] for n in roi_names]

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="rgba(150,150,150,0.3)"),
        hoverinfo="none",
    ))
    fig4.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=12, color=node_colors, line=dict(width=1, color="white")),
        text=roi_names,
        textposition="top center",
        textfont=dict(size=8),
        hoverinfo="text",
    ))
    fig4.update_layout(
        height=500,
        template="plotly_dark",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    st.plotly_chart(fig4, use_container_width=True)
except ImportError:
    st.info("Install `networkx` for the network graph visualization: `pip install networkx`")
