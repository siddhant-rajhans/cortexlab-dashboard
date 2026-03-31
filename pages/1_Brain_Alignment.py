"""Brain Alignment Benchmark - Research Grade."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from session import init_session, log_analysis, carry_rois, download_csv_button, show_analysis_log
from utils import (
    ALIGNMENT_METHODS, ROI_GROUPS, make_roi_indices,
    permutation_test, bootstrap_ci, fdr_correction, noise_ceiling,
    compute_rdm,
)
from synthetic import generate_realistic_predictions, generate_correlated_features

st.set_page_config(page_title="Brain Alignment", page_icon="🎯", layout="wide")
init_session()
show_analysis_log()

st.title("🎯 Brain Alignment Benchmark")
st.markdown("Score how well AI model representations align with predicted brain responses, with full statistical testing.")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    stimulus = st.selectbox("Stimulus type", ["visual", "auditory", "language", "multimodal"],
                            index=["visual", "auditory", "language", "multimodal"].index(st.session_state.get("stimulus_type", "visual")))
    n_timepoints = st.slider("Timepoints", 30, 200, st.session_state.get("n_timepoints", 80))
    seed = st.number_input("Seed", value=st.session_state.get("seed", 42), min_value=0)

    st.subheader("Models")
    model_configs = {
        "CLIP ViT-L/14": {"dim": 768, "alignment": st.slider("CLIP alignment", 0.0, 1.0, 0.6, 0.05)},
        "DINOv2 ViT-S": {"dim": 384, "alignment": st.slider("DINOv2 alignment", 0.0, 1.0, 0.3, 0.05)},
        "V-JEPA2 ViT-G": {"dim": 1024, "alignment": st.slider("V-JEPA2 alignment", 0.0, 1.0, 0.8, 0.05)},
    }

    st.subheader("Methods & Statistics")
    methods = st.multiselect("Methods", ["RSA", "CKA", "Procrustes"], default=["RSA", "CKA"])
    n_perm = st.slider("Permutations", 50, 1000, 200)
    n_boot = st.slider("Bootstrap samples", 100, 2000, 500)
    apply_fdr = st.checkbox("Apply FDR correction", value=True)

if not methods:
    st.warning("Select at least one method.")
    st.stop()

# --- Generate Data ---
roi_indices, n_vertices = make_roi_indices()
brain_pred = generate_realistic_predictions(n_timepoints, roi_indices, stimulus, seed=seed)

model_features = {}
for i, (name, cfg) in enumerate(model_configs.items()):
    model_features[name] = generate_correlated_features(
        brain_pred, cfg["alignment"], cfg["dim"], seed=seed + i + 1
    )

# --- Run Benchmark ---
with st.spinner("Computing alignment scores with statistical testing..."):
    results = []
    null_distributions = {}

    for model_name, features in model_features.items():
        for method_name in methods:
            score_fn = ALIGNMENT_METHODS[method_name]
            observed, p_val, null_dist = permutation_test(features, brain_pred, score_fn, n_perm, seed)
            point, ci_lo, ci_hi = bootstrap_ci(features, brain_pred, score_fn, n_boot, seed=seed)
            null_distributions[f"{model_name}_{method_name}"] = null_dist

            results.append({
                "Model": model_name,
                "Method": method_name,
                "Score": observed,
                "CI Lower": ci_lo,
                "CI Upper": ci_hi,
                "p-value": p_val,
            })

    df = pd.DataFrame(results)
    log_analysis(f"Brain alignment: {len(model_features)} models x {len(methods)} methods")

# --- Noise Ceiling ---
ceiling_scores = {}
for method_name in methods:
    score_fn = ALIGNMENT_METHODS[method_name]
    ceil_mean, ceil_std = noise_ceiling(brain_pred, score_fn, seed=seed)
    ceiling_scores[method_name] = ceil_mean

# --- Display: Alignment Scores with CIs ---
st.subheader("Alignment Scores")

col_chart, col_table = st.columns([2, 1])

with col_chart:
    fig = go.Figure()
    method_colors = {"RSA": "#00D2FF", "CKA": "#FF6B6B", "Procrustes": "#A29BFE"}
    x_positions = list(model_configs.keys())

    for method_name in methods:
        method_df = df[df["Method"] == method_name]
        fig.add_trace(go.Bar(
            name=method_name,
            x=method_df["Model"],
            y=method_df["Score"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=(method_df["CI Upper"] - method_df["Score"]).tolist(),
                arrayminus=(method_df["Score"] - method_df["CI Lower"]).tolist(),
            ),
            marker_color=method_colors.get(method_name, "#888"),
        ))
        # Noise ceiling line
        if method_name in ceiling_scores:
            fig.add_hline(
                y=ceiling_scores[method_name],
                line_dash="dash", line_color=method_colors.get(method_name, "#888"),
                opacity=0.4,
                annotation_text=f"{method_name} ceiling",
                annotation_position="top right",
            )

    fig.update_layout(
        barmode="group", yaxis_title="Alignment Score",
        height=450, template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_table:
    st.subheader("Results")
    display_df = df.copy()
    for col in ["Score", "CI Lower", "CI Upper", "p-value"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    download_csv_button(df, "brain_alignment_results.csv")

# --- Null Distribution ---
with st.expander("Null Distributions (Permutation Tests)", expanded=False):
    st.markdown("The histogram shows the distribution of scores under the null hypothesis (no alignment). "
                "The red line marks the observed score. If it falls far to the right, alignment is significant.")
    cols = st.columns(min(len(null_distributions), 3))
    for i, (key, null_dist) in enumerate(null_distributions.items()):
        model_name, method_name = key.rsplit("_", 1)
        row = df[(df["Model"] == model_name) & (df["Method"] == method_name)].iloc[0]
        with cols[i % len(cols)]:
            fig_null = go.Figure()
            fig_null.add_trace(go.Histogram(x=null_dist, nbinsx=30, marker_color="rgba(100,100,100,0.6)", name="Null"))
            fig_null.add_vline(x=row["Score"], line_color="red", line_width=2, annotation_text=f"Observed")
            fig_null.update_layout(
                title=f"{model_name} ({method_name})",
                xaxis_title="Score", yaxis_title="Count",
                height=250, template="plotly_dark", showlegend=False,
                margin=dict(t=40, b=30, l=30, r=10),
            )
            st.plotly_chart(fig_null, use_container_width=True)
            st.caption(f"p = {row['p-value']:.4f}")

# --- RDM Visualization ---
with st.expander("Representational Dissimilarity Matrices", expanded=False):
    st.markdown("RDMs show pairwise dissimilarity between stimulus representations. "
                "Similar RDM structure between model and brain indicates representational alignment.")
    rdm_model_name = st.selectbox("Model for RDM", list(model_features.keys()))
    col_brain, col_model = st.columns(2)

    brain_rdm = compute_rdm(brain_pred)
    model_rdm = compute_rdm(model_features[rdm_model_name])

    with col_brain:
        fig_rdm = go.Figure(go.Heatmap(z=brain_rdm, colorscale="Viridis", colorbar=dict(title="Dissimilarity")))
        fig_rdm.update_layout(title="Brain RDM", height=350, template="plotly_dark", xaxis_title="Stimulus", yaxis_title="Stimulus")
        st.plotly_chart(fig_rdm, use_container_width=True)

    with col_model:
        fig_rdm2 = go.Figure(go.Heatmap(z=model_rdm, colorscale="Viridis", colorbar=dict(title="Dissimilarity")))
        fig_rdm2.update_layout(title=f"{rdm_model_name} RDM", height=350, template="plotly_dark", xaxis_title="Stimulus", yaxis_title="Stimulus")
        st.plotly_chart(fig_rdm2, use_container_width=True)

# --- Per-ROI Analysis with FDR ---
st.divider()
st.subheader("Per-ROI Alignment")

roi_method = st.selectbox("Method for ROI analysis", methods, key="roi_method")
score_fn = ALIGNMENT_METHODS[roi_method]

roi_data = []
roi_p_values = []
top_model = df[df["Method"] == roi_method].sort_values("Score", ascending=False).iloc[0]["Model"]
features = model_features[top_model]

for group_name, rois in ROI_GROUPS.items():
    for roi in rois:
        if roi in roi_indices:
            verts = roi_indices[roi]
            valid = verts[verts < brain_pred.shape[1]]
            if len(valid) >= 2:
                s = score_fn(features, brain_pred[:, valid])
                _, p, _ = permutation_test(features, brain_pred[:, valid], score_fn, n_perm=50, seed=seed)
                roi_data.append({"ROI": roi, "Group": group_name, "Score": s, "p-value": p})
                roi_p_values.append(p)

if roi_data:
    roi_df = pd.DataFrame(roi_data)
    if apply_fdr and len(roi_p_values) > 1:
        corrected_p, significant = fdr_correction(roi_p_values)
        roi_df["FDR p-value"] = corrected_p
        roi_df["Significant"] = significant
        roi_df["Label"] = roi_df.apply(lambda r: f"{r['ROI']} *" if r["Significant"] else r["ROI"], axis=1)
    else:
        roi_df["Label"] = roi_df["ROI"]
        roi_df["Significant"] = roi_df["p-value"] < 0.05

    group_colors = {"Visual": "#00D2FF", "Auditory": "#FF6B6B", "Language": "#A29BFE", "Executive": "#FFEAA7"}
    fig_roi = px.bar(roi_df, x="Label", y="Score", color="Group",
                     color_discrete_map=group_colors)
    fig_roi.update_layout(height=400, template="plotly_dark", xaxis_tickangle=45)
    st.plotly_chart(fig_roi, use_container_width=True)
    st.caption(f"Model: {top_model} | * = significant after FDR correction (q < 0.05)" if apply_fdr else f"Model: {top_model}")

    # Carry ROIs button
    sig_rois = roi_df[roi_df["Significant"]]["ROI"].tolist() if "Significant" in roi_df.columns else []
    if sig_rois:
        if st.button(f"Carry {len(sig_rois)} significant ROIs to other pages"):
            carry_rois(sig_rois, "Temporal Dynamics / Connectivity")
            st.success(f"Carried {len(sig_rois)} ROIs: {', '.join(sig_rois[:5])}{'...' if len(sig_rois) > 5 else ''}")

# --- Methodology ---
with st.expander("Methodology", expanded=False):
    st.markdown("""
**Representational Similarity Analysis (RSA)** compares the geometry of two representation
spaces by computing pairwise dissimilarity matrices (RDMs) and correlating their upper triangles
via Spearman rank correlation. Range: [-1, 1]. Values > 0.1 are typically meaningful.
*Kriegeskorte et al., 2008, Frontiers in Systems Neuroscience.*

**Centered Kernel Alignment (CKA)** measures similarity between representations using
HSIC (Hilbert-Schmidt Independence Criterion) normalized by self-similarities. Invariant to
orthogonal transformations and isotropic scaling. Range: [0, 1].
*Kornblith et al., 2019, ICML.*

**Procrustes** finds the optimal rotation mapping one space onto another and measures
residual distance. Score = 1 - normalized Procrustes distance. Range: [0, 1].
*Ding et al., 2021, NeurIPS.*

**Noise ceiling** estimates the maximum achievable alignment score given the noise in the
brain data, computed via split-half reliability.

**FDR correction** (Benjamini-Hochberg) controls the false discovery rate when testing
multiple ROIs simultaneously.
""")
