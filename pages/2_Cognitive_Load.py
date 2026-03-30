"""Cognitive Load Scorer page."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils import (
    make_roi_indices,
    generate_brain_predictions,
    score_cognitive_load,
    COGNITIVE_DIMENSIONS,
)

st.set_page_config(page_title="Cognitive Load", page_icon="📊", layout="wide")
st.title("📊 Cognitive Load Scorer")
st.markdown("Predict cognitive demand of media content from brain activation patterns.")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    n_timepoints = st.slider("Duration (TRs)", 20, 200, 60)
    tr_seconds = st.slider("TR duration (seconds)", 0.5, 2.0, 1.0, 0.1)
    seed = st.number_input("Random seed", value=42, min_value=0)

    st.subheader("Simulate content type")
    content_type = st.selectbox(
        "Content profile",
        ["Balanced", "Visual-heavy", "Audio-heavy", "Language-heavy", "Low engagement"],
    )

# --- Generate Data ---
roi_indices, n_vertices = make_roi_indices()
predictions = generate_brain_predictions(n_timepoints, n_vertices, seed)

# Apply content profile by amplifying relevant ROIs
multipliers = {"Balanced": {}, "Visual-heavy": {}, "Audio-heavy": {}, "Language-heavy": {}, "Low engagement": {}}
if content_type == "Visual-heavy":
    for roi in COGNITIVE_DIMENSIONS["Visual Complexity"]:
        if roi in roi_indices:
            predictions[:, roi_indices[roi]] *= 3.0
elif content_type == "Audio-heavy":
    for roi in COGNITIVE_DIMENSIONS["Auditory Demand"]:
        if roi in roi_indices:
            predictions[:, roi_indices[roi]] *= 3.0
elif content_type == "Language-heavy":
    for roi in COGNITIVE_DIMENSIONS["Language Processing"]:
        if roi in roi_indices:
            predictions[:, roi_indices[roi]] *= 3.0
elif content_type == "Low engagement":
    predictions *= 0.2

# --- Score ---
averages, timeline = score_cognitive_load(predictions, roi_indices, tr_seconds)

# --- Display ---
col1, col2, col3, col4, col5 = st.columns(5)
dims = ["Overall", "Visual Complexity", "Auditory Demand", "Language Processing", "Executive Load"]
cols = [col1, col2, col3, col4, col5]
for col, dim in zip(cols, dims):
    val = averages.get(dim, 0.0)
    col.metric(dim, f"{val:.2f}", delta=None)

st.divider()

# --- Timeline ---
st.subheader("Cognitive Load Timeline")
timeline_df = pd.DataFrame(timeline)

fig = go.Figure()
colors = {"Visual Complexity": "#00D2FF", "Auditory Demand": "#FF6B6B", "Language Processing": "#A29BFE", "Executive Load": "#FFEAA7"}
for dim, color in colors.items():
    fig.add_trace(go.Scatter(
        x=timeline_df["time"],
        y=timeline_df[dim],
        name=dim,
        line=dict(color=color, width=2),
        mode="lines",
    ))

fig.update_layout(
    xaxis_title="Time (seconds)",
    yaxis_title="Cognitive Load (normalized)",
    yaxis_range=[0, 1.05],
    height=400,
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

# --- Dimension Breakdown ---
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.subheader("Dimension Breakdown")
    dim_data = {k: v for k, v in averages.items() if k != "Overall"}
    fig2 = go.Figure(go.Bar(
        x=list(dim_data.values()),
        y=list(dim_data.keys()),
        orientation="h",
        marker_color=list(colors.values()),
    ))
    fig2.update_layout(
        xaxis_title="Score",
        xaxis_range=[0, 1],
        height=300,
        template="plotly_dark",
    )
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("Radar Chart")
    categories = list(dim_data.keys())
    values = list(dim_data.values()) + [list(dim_data.values())[0]]  # close the polygon

    fig3 = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(108, 92, 231, 0.3)",
        line=dict(color="#6C5CE7"),
    ))
    fig3.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=350,
        template="plotly_dark",
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True)
