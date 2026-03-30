"""Temporal Dynamics page."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import (
    make_roi_indices,
    generate_brain_predictions,
    generate_model_features,
    peak_latency,
    temporal_correlation,
    decompose_response,
    ROI_GROUPS,
    ALL_ROIS,
)

st.set_page_config(page_title="Temporal Dynamics", page_icon="⏱️", layout="wide")
st.title("⏱️ Temporal Dynamics")
st.markdown("Analyze how brain responses evolve over time.")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    n_timepoints = st.slider("Duration (TRs)", 30, 200, 80)
    tr_seconds = st.slider("TR duration (seconds)", 0.5, 2.0, 1.0, 0.1)
    seed = st.number_input("Random seed", value=42, min_value=0)

    st.subheader("ROI Selection")
    selected_group = st.selectbox("Region group", list(ROI_GROUPS.keys()))
    available_rois = ROI_GROUPS[selected_group]
    selected_rois = st.multiselect("ROIs to analyze", available_rois, default=available_rois[:3])

    max_lag = st.slider("Max correlation lag (TRs)", 5, 30, 15)
    cutoff = st.slider("Decomposition cutoff (seconds)", 1.0, 10.0, 4.0, 0.5)

if not selected_rois:
    st.warning("Select at least one ROI.")
    st.stop()

# --- Generate Data ---
roi_indices, n_vertices = make_roi_indices()
predictions = generate_brain_predictions(n_timepoints, n_vertices, seed)
features = generate_model_features(n_timepoints, 64, seed + 1)

# --- Peak Latency ---
st.subheader("Peak Response Latency")
latency_data = []
for roi in selected_rois:
    lat = peak_latency(predictions, roi_indices, roi, tr_seconds)
    latency_data.append({"ROI": roi, "Peak Latency (s)": lat})

lat_df = pd.DataFrame(latency_data)
col1, col2 = st.columns([2, 1])

with col1:
    fig = go.Figure(go.Bar(
        x=lat_df["ROI"],
        y=lat_df["Peak Latency (s)"],
        marker_color="#6C5CE7",
    ))
    fig.update_layout(
        yaxis_title="Time to peak (seconds)",
        height=350,
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.dataframe(lat_df, use_container_width=True, hide_index=True)

# --- Lag Correlation ---
st.divider()
st.subheader("Temporal Correlation (Brain vs Model Features)")

lags = np.arange(-max_lag, max_lag + 1) * tr_seconds
fig2 = go.Figure()
colors = ["#00D2FF", "#FF6B6B", "#A29BFE", "#FFEAA7", "#55EFC4", "#FD79A8"]

for i, roi in enumerate(selected_rois):
    corr = temporal_correlation(predictions, features, roi_indices, roi, max_lag)
    fig2.add_trace(go.Scatter(
        x=lags,
        y=corr,
        name=roi,
        line=dict(color=colors[i % len(colors)], width=2),
    ))

fig2.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
fig2.update_layout(
    xaxis_title="Lag (seconds)",
    yaxis_title="Pearson Correlation",
    height=400,
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig2, use_container_width=True)

# --- Sustained vs Transient ---
st.divider()
st.subheader("Sustained vs Transient Decomposition")

roi_for_decomp = st.selectbox("ROI for decomposition", selected_rois)
sustained, transient = decompose_response(predictions, roi_indices, roi_for_decomp, cutoff, tr_seconds)
time_axis = np.arange(len(sustained)) * tr_seconds

fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                     subplot_titles=("Sustained Component", "Transient Component"))

fig3.add_trace(go.Scatter(x=time_axis, y=sustained, name="Sustained",
                          line=dict(color="#6C5CE7", width=2)), row=1, col=1)
fig3.add_trace(go.Scatter(x=time_axis, y=transient, name="Transient",
                          line=dict(color="#FF6B6B", width=1.5)), row=2, col=1)

fig3.update_xaxes(title_text="Time (seconds)", row=2, col=1)
fig3.update_layout(height=500, template="plotly_dark", showlegend=False)
st.plotly_chart(fig3, use_container_width=True)
