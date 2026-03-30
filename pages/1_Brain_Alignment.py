"""Brain Alignment Benchmark page."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils import (
    ALIGNMENT_METHODS,
    make_roi_indices,
    generate_brain_predictions,
    generate_model_features,
    permutation_test,
)

st.set_page_config(page_title="Brain Alignment", page_icon="🎯", layout="wide")
st.title("🎯 Brain Alignment Benchmark")
st.markdown("Compare how well AI model representations align with predicted brain responses.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Configuration")
    n_stimuli = st.slider("Number of stimuli", 10, 200, 50)
    seed = st.number_input("Random seed", value=42, min_value=0)

    st.subheader("Models to compare")
    models_config = {
        "CLIP ViT-L/14": st.checkbox("CLIP ViT-L/14", value=True),
        "DINOv2 ViT-S": st.checkbox("DINOv2 ViT-S", value=True),
        "V-JEPA2 ViT-G": st.checkbox("V-JEPA2 ViT-G", value=True),
        "LLaMA 3.2-3B": st.checkbox("LLaMA 3.2-3B", value=False),
    }
    model_dims = {
        "CLIP ViT-L/14": 768,
        "DINOv2 ViT-S": 384,
        "V-JEPA2 ViT-G": 1024,
        "LLaMA 3.2-3B": 3072,
    }
    selected_models = [m for m, checked in models_config.items() if checked]

    methods = st.multiselect("Methods", ["RSA", "CKA", "Procrustes"], default=["RSA", "CKA"])
    run_stats = st.checkbox("Run permutation test", value=False)
    n_perm = st.slider("Permutations", 50, 1000, 200) if run_stats else 0

if not selected_models or not methods:
    st.warning("Select at least one model and one method.")
    st.stop()

# --- Generate Data ---
roi_indices, n_vertices = make_roi_indices()
brain_pred = generate_brain_predictions(n_stimuli, n_vertices, seed)

model_features = {}
for i, name in enumerate(selected_models):
    model_features[name] = generate_model_features(n_stimuli, model_dims[name], seed + i + 1)

# --- Run Benchmark ---
with st.spinner("Computing alignment scores..."):
    results = []
    for model_name, features in model_features.items():
        for method_name in methods:
            score_fn = ALIGNMENT_METHODS[method_name]
            score = score_fn(features, brain_pred)
            row = {"Model": model_name, "Method": method_name, "Score": score}

            if run_stats:
                _, p_val = permutation_test(features, brain_pred, score_fn, n_perm, seed)
                row["p-value"] = p_val
                row["Significant"] = "Yes" if p_val < 0.05 else "No"

            results.append(row)

df = pd.DataFrame(results)

# --- Display Results ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Alignment Scores")
    fig = px.bar(
        df,
        x="Model",
        y="Score",
        color="Method",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        yaxis_title="Alignment Score",
        height=450,
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Results Table")
    display_df = df.copy()
    display_df["Score"] = display_df["Score"].map(lambda x: f"{x:.4f}")
    if "p-value" in display_df.columns:
        display_df["p-value"] = display_df["p-value"].map(lambda x: f"{x:.4f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# --- Per-ROI Analysis ---
st.divider()
st.subheader("Per-ROI Alignment (RSA)")

if "RSA" in methods and len(selected_models) >= 1:
    from utils import ROI_GROUPS, rsa_score

    roi_data = []
    for model_name, features in model_features.items():
        for group_name, rois in ROI_GROUPS.items():
            group_scores = []
            for roi in rois:
                if roi in roi_indices:
                    verts = roi_indices[roi]
                    valid = verts[verts < brain_pred.shape[1]]
                    if len(valid) >= 2:
                        s = rsa_score(features, brain_pred[:, valid])
                        group_scores.append(s)
            if group_scores:
                roi_data.append({
                    "Model": model_name,
                    "Region": group_name,
                    "RSA Score": float(np.mean(group_scores)),
                })

    if roi_data:
        roi_df = pd.DataFrame(roi_data)
        fig2 = px.bar(
            roi_df,
            x="Region",
            y="RSA Score",
            color="Model",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig2.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
