"""Cognitive Load Scorer - Research Grade."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from session import init_session, log_analysis, download_csv_button, show_analysis_log
from theme import inject_theme, section_header
from utils import make_roi_indices, score_cognitive_load, COGNITIVE_DIMENSIONS, ROI_GROUPS
from synthetic import generate_realistic_predictions

st.set_page_config(page_title="Cognitive Load", page_icon="📊", layout="wide")
init_session()
inject_theme()
show_analysis_log()

st.title("📊 Cognitive Load Scorer")
st.markdown("Predict cognitive demand from brain activation patterns across four neurocognitive dimensions.")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    n_timepoints = st.slider("Duration (TRs)", 30, 200, 80)
    tr_seconds = st.slider("TR (seconds)", 0.5, 2.0, 1.0, 0.1)
    seed = st.number_input("Seed", value=42, min_value=0)

    st.subheader("Stimulus")
    stim_type = st.selectbox("Primary stimulus", ["visual", "auditory", "language", "multimodal"])

    st.subheader("Comparison Mode")
    compare = st.checkbox("Compare two stimulus types", value=False)
    if compare:
        stim_type_2 = st.selectbox("Second stimulus", [s for s in ["visual", "auditory", "language", "multimodal"] if s != stim_type])

# --- Generate Data ---
roi_indices, n_vertices = make_roi_indices()
predictions = generate_realistic_predictions(n_timepoints, roi_indices, stim_type, tr_seconds, seed=seed)
averages, timeline = score_cognitive_load(predictions, roi_indices, tr_seconds)
log_analysis(f"Cognitive load: {stim_type}, {n_timepoints} TRs")

if compare:
    predictions_2 = generate_realistic_predictions(n_timepoints, roi_indices, stim_type_2, tr_seconds, seed=seed + 100)
    averages_2, timeline_2 = score_cognitive_load(predictions_2, roi_indices, tr_seconds)

# --- Metric Cards ---
dims = ["Overall", "Visual Complexity", "Auditory Demand", "Language Processing", "Executive Load"]
cols = st.columns(5)
for col, dim in zip(cols, dims):
    val = averages.get(dim, 0.0)
    if compare:
        val_2 = averages_2.get(dim, 0.0)
        delta = val - val_2
        col.metric(dim, f"{val:.2f}", delta=f"{delta:+.2f} vs {stim_type_2}", delta_color="normal")
    else:
        col.metric(dim, f"{val:.2f}")

st.divider()

# --- Timeline with Confidence Bands ---
st.subheader("Cognitive Load Timeline")

if compare:
    st.markdown(f"**Solid lines**: {stim_type} | **Dashed lines**: {stim_type_2}")

timeline_df = pd.DataFrame(timeline)
dim_colors = {"Visual Complexity": "#00D2FF", "Auditory Demand": "#FF6B6B", "Language Processing": "#A29BFE", "Executive Load": "#FFEAA7"}

fig = go.Figure()
for dim, color in dim_colors.items():
    y = timeline_df[dim].values

    # Bootstrap confidence band (resample vertices within dimension ROIs)
    rng = np.random.default_rng(seed)
    dim_rois = COGNITIVE_DIMENSIONS.get(dim, [])
    dim_vertices = []
    for roi in dim_rois:
        if roi in roi_indices:
            valid = roi_indices[roi]
            dim_vertices.extend(valid[valid < predictions.shape[1]])

    if dim_vertices:
        boot_scores = []
        for _ in range(50):
            sample_verts = rng.choice(dim_vertices, size=max(1, len(dim_vertices) // 2), replace=True)
            boot_tc = np.abs(predictions[:, sample_verts]).mean(axis=1)
            baseline = max(np.median(np.abs(predictions)), 1e-8)
            boot_scores.append(np.clip(boot_tc / baseline, 0, 1))
        boot_arr = np.array(boot_scores)
        ci_lo = np.percentile(boot_arr, 2.5, axis=0)
        ci_hi = np.percentile(boot_arr, 97.5, axis=0)
        t_axis = timeline_df["time"].values

        fig.add_trace(go.Scatter(x=t_axis, y=ci_hi, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=t_axis, y=ci_lo, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba").replace("#", "rgba(") if color.startswith("#") else color,
                                 showlegend=False))

    fig.add_trace(go.Scatter(x=timeline_df["time"], y=y, name=dim, line=dict(color=color, width=2)))

    if compare:
        timeline_df_2 = pd.DataFrame(timeline_2)
        fig.add_trace(go.Scatter(x=timeline_df_2["time"], y=timeline_df_2[dim].values,
                                 name=f"{dim} ({stim_type_2})", line=dict(color=color, width=1.5, dash="dash")))

fig.update_layout(
    xaxis_title="Time (seconds)", yaxis_title="Cognitive Load (normalized)",
    yaxis_range=[0, 1.05], height=450, template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

# --- Dimension Correlation + Radar ---
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.subheader("Dimension Correlation")
    st.markdown("How do the four cognitive dimensions co-vary over time?")
    dim_timeseries = {}
    for dim in dim_colors:
        dim_timeseries[dim] = timeline_df[dim].values
    dim_arr = np.array(list(dim_timeseries.values()))
    corr = np.corrcoef(dim_arr)
    dim_names = list(dim_colors.keys())

    fig_corr = go.Figure(go.Heatmap(
        z=corr, x=dim_names, y=dim_names,
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="r"),
        text=np.round(corr, 2), texttemplate="%{text}",
    ))
    fig_corr.update_layout(height=350, template="plotly_dark", xaxis_tickangle=30)
    st.plotly_chart(fig_corr, use_container_width=True)

with col2:
    st.subheader("Dimension Profile")
    dim_data = {k: v for k, v in averages.items() if k != "Overall"}
    categories = list(dim_data.keys())
    values = list(dim_data.values()) + [list(dim_data.values())[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values, theta=categories + [categories[0]],
        fill="toself", fillcolor="rgba(108, 92, 231, 0.3)",
        line=dict(color="#6C5CE7"), name=stim_type,
    ))
    if compare:
        dim_data_2 = {k: v for k, v in averages_2.items() if k != "Overall"}
        values_2 = list(dim_data_2.values()) + [list(dim_data_2.values())[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=values_2, theta=categories + [categories[0]],
            fill="toself", fillcolor="rgba(255,107,107,0.2)",
            line=dict(color="#FF6B6B", dash="dash"), name=stim_type_2,
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=350, template="plotly_dark",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# --- Per-ROI Activation Breakdown ---
st.divider()
st.subheader("Per-ROI Activation Within Each Dimension")
selected_dim = st.selectbox("Dimension", list(dim_colors.keys()))
dim_rois = COGNITIVE_DIMENSIONS.get(selected_dim, [])

roi_activations = []
for roi in dim_rois:
    if roi in roi_indices:
        verts = roi_indices[roi]
        valid = verts[verts < predictions.shape[1]]
        if len(valid) > 0:
            act = float(np.abs(predictions[:, valid]).mean())
            roi_activations.append({"ROI": roi, "Mean Activation": act})

if roi_activations:
    roi_act_df = pd.DataFrame(roi_activations).sort_values("Mean Activation", ascending=False)
    fig_roi = go.Figure(go.Bar(
        x=roi_act_df["Mean Activation"], y=roi_act_df["ROI"],
        orientation="h", marker_color=dim_colors[selected_dim],
    ))
    fig_roi.update_layout(height=max(250, len(roi_activations) * 25), template="plotly_dark",
                          yaxis=dict(autorange="reversed"), xaxis_title="Mean |activation|")
    st.plotly_chart(fig_roi, use_container_width=True)
    download_csv_button(roi_act_df, f"cognitive_load_{selected_dim}_rois.csv")

# --- Methodology ---
with st.expander("Methodology", expanded=False):
    st.markdown("""
**Cognitive Load Scoring** maps predicted fMRI activations onto four neurocognitive dimensions
using HCP MMP1.0 ROI groupings:

- **Visual Complexity**: V1-V4, MT, MST, FFC, VVC (ventral & dorsal visual streams)
- **Auditory Demand**: A1, belt areas, STS (auditory cortex + association areas)
- **Language Processing**: Areas 44/45 (Broca's), TPOJ (Wernicke's), STV, PSL (perisylvian language network)
- **Executive Load**: dlPFC (area 46), ACC, FEF (frontoparietal control network)

Each dimension score is the mean absolute activation across its ROIs, normalized by the
median activation across all vertices (baseline). Scores are clipped to [0, 1].

**Confidence bands** on the timeline are computed via bootstrap resampling of vertices
within each dimension's ROI group (50 resamples, 95% CI).

**Limitations**: The ROI-to-dimension mapping is based on established functional neuroanatomy
but is not exhaustive. Cognitive load is a multidimensional construct that cannot be fully
captured by fMRI activation alone. These scores should be interpreted as relative measures,
not absolute cognitive load values.

**References**:
- Glasser et al., 2016, *Nature* (HCP MMP1.0 parcellation)
- Sweller, 1988, *Cognitive Science* (Cognitive Load Theory)
""")
