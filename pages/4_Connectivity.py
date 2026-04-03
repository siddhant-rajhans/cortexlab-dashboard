"""ROI Connectivity - Research Grade."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from session import init_session, log_analysis, get_carried_rois, download_csv_button, show_analysis_log
from theme import inject_theme, section_header
from utils import (
    make_roi_indices, compute_connectivity, cluster_rois, graph_metrics,
    partial_correlation, betweenness_centrality, modularity_score,
    ROI_GROUPS,
)
from synthetic import generate_realistic_predictions

st.set_page_config(page_title="ROI Connectivity", page_icon="🔗", layout="wide")
init_session()
inject_theme()
show_analysis_log()

st.title("🔗 ROI Connectivity Analysis")
st.markdown("Functional connectivity between brain regions: correlation structure, network organization, and graph topology.")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    stim_type = st.selectbox("Stimulus type", ["visual", "auditory", "language", "multimodal"])
    n_timepoints = st.slider("Duration (TRs)", 30, 200, 80)
    tr_seconds = st.slider("TR (seconds)", 0.5, 2.0, 1.0, 0.1)
    seed = st.number_input("Seed", value=42, min_value=0)

    st.subheader("Analysis Parameters")
    n_clusters = st.slider("Number of clusters", 2, 8, 4)
    threshold = st.slider("Edge threshold", 0.1, 0.8, 0.3, 0.05)
    use_partial = st.checkbox("Use partial correlation", value=False,
                              help="Control for shared mean signal across all ROIs")

    carried = get_carried_rois()
    use_carried = False
    if carried:
        use_carried = st.checkbox(f"Filter to {len(carried)} carried ROIs", value=False)

# --- Generate Data ---
roi_indices, n_vertices = make_roi_indices()
predictions = generate_realistic_predictions(n_timepoints, roi_indices, stim_type, tr_seconds, seed=seed)
log_analysis(f"Connectivity: {stim_type}, partial={use_partial}")

# Filter ROIs if carrying from alignment
active_indices = roi_indices
if use_carried and carried:
    active_indices = {k: v for k, v in roi_indices.items() if k in carried}

# --- Compute Connectivity ---
if use_partial:
    corr_matrix, roi_names = partial_correlation(predictions, active_indices)
    corr_label = "Partial Correlation"
else:
    corr_matrix, roi_names = compute_connectivity(predictions, active_indices)
    corr_label = "Pearson Correlation"

n_rois = len(roi_names)

# --- Correlation Matrix with Cluster Boundaries ---
st.subheader(f"{corr_label} Matrix")

clusters, labels = cluster_rois(corr_matrix, roi_names, n_clusters)

# Sort ROIs by cluster for block-diagonal structure
sorted_idx = np.argsort(labels)
sorted_corr = corr_matrix[np.ix_(sorted_idx, sorted_idx)]
sorted_names = [roi_names[i] for i in sorted_idx]
sorted_labels = labels[sorted_idx]

fig_corr = go.Figure(go.Heatmap(
    z=sorted_corr, x=sorted_names, y=sorted_names,
    colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
    colorbar=dict(title="r"),
))

# Add cluster boundary lines
boundaries = []
for i in range(1, len(sorted_labels)):
    if sorted_labels[i] != sorted_labels[i - 1]:
        boundaries.append(i - 0.5)

for b in boundaries:
    fig_corr.add_shape(type="line", x0=b, x1=b, y0=-0.5, y1=n_rois - 0.5,
                       line=dict(color="white", width=1.5, dash="dot"))
    fig_corr.add_shape(type="line", x0=-0.5, x1=n_rois - 0.5, y0=b, y1=b,
                       line=dict(color="white", width=1.5, dash="dot"))

fig_corr.update_layout(
    height=550, template="plotly_dark",
    xaxis=dict(tickangle=45, tickfont=dict(size=8)),
    yaxis=dict(tickfont=dict(size=8)),
)
st.plotly_chart(fig_corr, use_container_width=True)
st.caption(f"White dotted lines indicate cluster boundaries ({n_clusters} clusters)")

# --- Dendrogram ---
st.divider()
col_dendro, col_clusters = st.columns([1, 1])

with col_dendro:
    st.subheader("Hierarchical Clustering Dendrogram")
    from scipy.cluster.hierarchy import linkage, dendrogram
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    dist = 1.0 - np.abs(corr_matrix)
    np.fill_diagonal(dist, 0.0)
    condensed = [dist[i, j] for i in range(n_rois) for j in range(i + 1, n_rois)]
    Z = linkage(condensed, method="average")

    fig_dendro, ax = plt.subplots(figsize=(8, 4))
    ax.set_facecolor("#0E1117")
    fig_dendro.patch.set_facecolor("#0E1117")
    dendrogram(Z, labels=roi_names, leaf_rotation=90, leaf_font_size=7, ax=ax,
               color_threshold=Z[-n_clusters + 1, 2] if n_clusters < n_rois else 0)
    ax.tick_params(colors="white")
    ax.set_ylabel("Distance (1 - |r|)", color="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    st.pyplot(fig_dendro)
    plt.close()

with col_clusters:
    st.subheader("Network Clusters")
    mod_q = modularity_score(corr_matrix, labels)
    st.metric("Modularity (Q)", f"{mod_q:.3f}",
              help="Newman's modularity. Higher = stronger community structure. Q > 0.3 is typically considered meaningful.")

    for cid in sorted(clusters.keys()):
        rois = clusters[cid]
        # Identify dominant functional group
        group_counts = {}
        for roi in rois:
            for g, g_rois in ROI_GROUPS.items():
                if roi in g_rois:
                    group_counts[g] = group_counts.get(g, 0) + 1
        dominant = max(group_counts, key=group_counts.get) if group_counts else "Mixed"
        st.markdown(f"**Network {cid}** ({dominant}): {', '.join(rois)}")

# --- Centrality Comparison ---
st.divider()
st.subheader("Centrality Analysis")

col_deg, col_btw = st.columns(2)

degrees = graph_metrics(corr_matrix, roi_names, threshold)
btw = betweenness_centrality(corr_matrix, roi_names, threshold)

with col_deg:
    st.markdown("**Degree Centrality** - fraction of ROIs connected to each node")
    deg_df = pd.DataFrame(sorted(degrees.items(), key=lambda x: x[1], reverse=True), columns=["ROI", "Degree"])
    fig_deg = go.Figure(go.Bar(x=deg_df["Degree"], y=deg_df["ROI"], orientation="h", marker_color="#6C5CE7"))
    fig_deg.update_layout(xaxis_range=[0, 1], height=max(300, n_rois * 20), template="plotly_dark",
                          yaxis=dict(autorange="reversed", tickfont=dict(size=9)))
    st.plotly_chart(fig_deg, use_container_width=True)

with col_btw:
    st.markdown("**Betweenness Centrality** - how often a node lies on shortest paths between others")
    btw_df = pd.DataFrame(sorted(btw.items(), key=lambda x: x[1], reverse=True), columns=["ROI", "Betweenness"])
    fig_btw = go.Figure(go.Bar(x=btw_df["Betweenness"], y=btw_df["ROI"], orientation="h", marker_color="#FF6B6B"))
    fig_btw.update_layout(height=max(300, n_rois * 20), template="plotly_dark",
                          yaxis=dict(autorange="reversed", tickfont=dict(size=9)))
    st.plotly_chart(fig_btw, use_container_width=True)

# Combined table
centrality_df = pd.merge(deg_df, btw_df, on="ROI")
download_csv_button(centrality_df, "centrality_metrics.csv")

# --- Edge Weight Distribution ---
st.divider()
col_dist, col_graph = st.columns([1, 2])

with col_dist:
    st.subheader("Edge Weight Distribution")
    upper_tri = corr_matrix[np.triu_indices(n_rois, k=1)]
    fig_hist = go.Figure(go.Histogram(x=upper_tri, nbinsx=40, marker_color="rgba(108,92,231,0.7)"))
    fig_hist.add_vline(x=threshold, line_color="red", line_dash="dash", annotation_text="Threshold")
    fig_hist.add_vline(x=-threshold, line_color="red", line_dash="dash")
    fig_hist.update_layout(
        xaxis_title="Correlation", yaxis_title="Count",
        height=350, template="plotly_dark",
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    n_edges = np.sum(np.abs(upper_tri) > threshold)
    max_edges = n_rois * (n_rois - 1) // 2
    st.caption(f"{n_edges}/{max_edges} edges above threshold ({100 * n_edges / max(max_edges, 1):.1f}% density)")

# --- Network Graph ---
with col_graph:
    st.subheader("Network Graph")
    try:
        import networkx as nx

        G = nx.Graph()
        for name in roi_names:
            G.add_node(name)
        for i in range(n_rois):
            for j in range(i + 1, n_rois):
                w = abs(corr_matrix[i, j])
                if w > threshold:
                    G.add_edge(roi_names[i], roi_names[j], weight=w)

        pos = nx.spring_layout(G, seed=seed, k=2.5)
        color_map = px.colors.qualitative.Set2
        node_colors = [color_map[(labels[i] - 1) % len(color_map)] for i in range(n_rois)]

        # Edges with width proportional to weight
        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            fig_graph = go.Figure() if not hasattr(st, '_graph_fig') else st._graph_fig

        fig_net = go.Figure()
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            w = d.get("weight", 0.3)
            fig_net.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None], mode="lines",
                line=dict(width=w * 3, color=f"rgba(150,150,150,{min(w, 0.8)})"),
                hoverinfo="none", showlegend=False,
            ))

        # Node sizes by degree
        max_deg = max(degrees.values()) if degrees else 1
        node_sizes = [8 + 20 * degrees.get(name, 0) / max(max_deg, 0.01) for name in roi_names]
        node_x = [pos[n][0] for n in roi_names]
        node_y = [pos[n][1] for n in roi_names]

        fig_net.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color="white")),
            text=roi_names, textposition="top center", textfont=dict(size=7, color="white"),
            hovertext=[f"{name}<br>Degree: {degrees.get(name, 0):.2f}<br>Betweenness: {btw.get(name, 0):.3f}" for name in roi_names],
            hoverinfo="text", showlegend=False,
        ))

        fig_net.update_layout(
            height=450, template="plotly_dark",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        st.plotly_chart(fig_net, use_container_width=True)
    except ImportError:
        st.info("Install `networkx` for graph visualization: `pip install networkx`")

# --- Methodology ---
with st.expander("Methodology", expanded=False):
    st.markdown("""
**Functional Connectivity** is computed as pairwise Pearson correlation between ROI
timecourses (mean activation across vertices within each ROI).

**Partial Correlation** controls for the shared mean signal by computing the precision
matrix (inverse covariance) and normalizing. This removes indirect correlations mediated
by a common driver.

**Hierarchical Clustering** uses agglomerative clustering with average linkage on a
distance matrix defined as ``1 - |correlation|``. The dendrogram shows the hierarchical
merging of ROIs into networks.

**Modularity (Q)** quantifies how strongly the network divides into communities compared
to a random network with the same degree distribution. Q > 0.3 typically indicates
meaningful community structure. (Newman, 2006, *PNAS*)

**Degree Centrality** is the fraction of other nodes each node is connected to (above
the correlation threshold). High degree = hub region.

**Betweenness Centrality** counts how often a node lies on the shortest path between
other node pairs. High betweenness = bridge between communities.

**Edge Weight Distribution** shows the histogram of all pairwise correlations. The
threshold (red line) determines which connections are retained for graph analysis.

**References**:
- Rubinov & Sporns, 2010, *NeuroImage* (graph metrics for brain networks)
- Newman, 2006, *PNAS* (modularity in networks)
- Smith et al., 2011, *NeuroImage* (partial correlation for fMRI connectivity)
""")
