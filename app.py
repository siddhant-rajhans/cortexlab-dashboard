"""CortexLab Dashboard - Futuristic Landing Page."""

import streamlit as st
import numpy as np

from theme import inject_theme, hero_header, glow_card, section_header, feature_card
from session import init_session, show_analysis_log
from utils import make_roi_indices
from brain_mesh import load_fsaverage_mesh, generate_sample_activations, render_interactive_3d

st.set_page_config(page_title="CortexLab", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")
init_session()
inject_theme()

# --- Hero ---
hero_header(
    "CortexLab",
    "Enhanced multimodal fMRI brain encoding toolkit built on Meta's TRIBE v2",
)

# --- 3D Brain Hero ---
st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

with st.spinner("Rendering brain..."):
    coords, faces = load_fsaverage_mesh("left", "fsaverage5")
    n_verts = coords.shape[0]
    roi_indices, _ = make_roi_indices()
    mesh_roi = {name: np.clip((idx * n_verts // 580).astype(int), 0, n_verts - 1) for name, idx in roi_indices.items()}
    activations = generate_sample_activations(n_verts, mesh_roi, "multimodal", seed=42)

    fig = render_interactive_3d(
        coords, faces, activations, cmap="Inferno", vmin=0, vmax=0.8,
        bg_color="#050510", initial_view="Lateral Left",
        roi_indices=mesh_roi, show_labels=False,
    )
    if fig is not None:
        fig.update_layout(
            height=500,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Stats Bar ---
st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
with c1: glow_card("Tests", "89", "All passing", "#10B981")
with c2: glow_card("Analysis Modules", "10", "Brain encoding", "#7C3AED")
with c3: glow_card("ROIs", "29", "HCP MMP1.0", "#3B82F6")
with c4: glow_card("Contributors", "4", "Open source", "#EC4899")
with c5: glow_card("License", "CC BY-NC", "Non-commercial", "#F59E0B")

# --- Feature Grid ---
st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
section_header("Analysis Tools", "Everything you need for computational neuroscience research")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(feature_card(
        "🎯", "Brain Alignment Benchmark",
        "Score any AI model against brain responses. RSA, CKA, Procrustes with permutation tests, bootstrap CIs, and FDR correction.",
        "#7C3AED"
    ), unsafe_allow_html=True)
    st.page_link("pages/1_Brain_Alignment.py", label="Open Brain Alignment")

with col2:
    st.markdown(feature_card(
        "📊", "Cognitive Load Scorer",
        "Predict cognitive demand across visual, auditory, language, and executive dimensions with confidence bands.",
        "#3B82F6"
    ), unsafe_allow_html=True)
    st.page_link("pages/2_Cognitive_Load.py", label="Open Cognitive Load")

with col3:
    st.markdown(feature_card(
        "⏱️", "Temporal Dynamics",
        "Peak response latency, lag correlations, sustained vs transient decomposition, cross-ROI lag matrix.",
        "#06B6D4"
    ), unsafe_allow_html=True)
    st.page_link("pages/3_Temporal_Dynamics.py", label="Open Temporal Dynamics")

col4, col5, col6 = st.columns(3)
with col4:
    st.markdown(feature_card(
        "🔗", "ROI Connectivity",
        "Functional connectivity, partial correlation, network clustering, modularity, degree and betweenness centrality.",
        "#10B981"
    ), unsafe_allow_html=True)
    st.page_link("pages/4_Connectivity.py", label="Open Connectivity")

with col5:
    st.markdown(feature_card(
        "🧠", "3D Brain Viewer",
        "Interactive rotatable brain surface with activation overlays, publication-quality multi-view panels, ROI highlighting.",
        "#EC4899"
    ), unsafe_allow_html=True)
    st.page_link("pages/5_Brain_Viewer.py", label="Open Brain Viewer")

with col6:
    st.markdown(feature_card(
        "⚡", "Streaming Inference",
        "Real-time sliding-window predictions for BCI pipelines. Cross-subject adaptation with minimal calibration data.",
        "#F59E0B"
    ), unsafe_allow_html=True)
    st.markdown(f"""
    <a href="https://github.com/siddhant-rajhans/cortexlab" target="_blank" style="
        display: inline-block; padding: 0.4rem 1rem;
        color: #F59E0B; font-size: 0.85rem;
        text-decoration: none;
    ">View on GitHub &rarr;</a>
    """, unsafe_allow_html=True)

# --- Data Config (collapsed) ---
st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
with st.expander("Data Configuration", expanded=False):
    from session import data_summary_widget, upload_npy_widget
    col_src, col_params = st.columns([1, 2])

    with col_src:
        source = st.radio("Data source", ["Synthetic (realistic)", "Upload your data"], index=0)
        st.session_state["data_source"] = "synthetic" if "Synthetic" in source else "uploaded"

    with col_params:
        if st.session_state["data_source"] == "synthetic":
            c1, c2, c3, c4 = st.columns(4)
            st.session_state["stimulus_type"] = c1.selectbox("Stimulus", ["visual", "auditory", "language", "multimodal"])
            st.session_state["n_timepoints"] = c2.slider("TRs", 30, 200, 80)
            st.session_state["tr_seconds"] = c3.slider("TR (s)", 0.5, 2.0, 1.0, 0.1)
            st.session_state["seed"] = c4.number_input("Seed", value=42, min_value=0)

            roi_indices_full, n_vertices = make_roi_indices()
            st.session_state["roi_indices"] = roi_indices_full
            st.session_state["n_vertices"] = n_vertices

            from synthetic import generate_realistic_predictions
            predictions = generate_realistic_predictions(
                st.session_state["n_timepoints"], roi_indices_full,
                st.session_state["stimulus_type"], st.session_state["tr_seconds"],
                seed=st.session_state["seed"],
            )
            st.session_state["brain_predictions"] = predictions
        else:
            uploaded = upload_npy_widget("Upload predictions (.npy)", "upload_home")
            if uploaded is not None:
                st.session_state["brain_predictions"] = uploaded
                roi_indices_full, _ = make_roi_indices()
                st.session_state["roi_indices"] = roi_indices_full

    preds = st.session_state.get("brain_predictions")
    roi_idx = st.session_state.get("roi_indices")
    if preds is not None and roi_idx is not None:
        data_summary_widget(preds, roi_idx)

# --- Footer ---
show_analysis_log()
st.markdown("""
<div style="text-align: center; padding: 2rem 0 1rem 0; color: #475569; font-size: 0.75rem;">
    Built on <a href="https://github.com/facebookresearch/tribev2" style="color: #7C3AED; text-decoration: none;">Meta's TRIBE v2</a>
    &nbsp;&bull;&nbsp;
    <a href="https://github.com/siddhant-rajhans/cortexlab" style="color: #3B82F6; text-decoration: none;">GitHub</a>
    &nbsp;&bull;&nbsp;
    <a href="https://huggingface.co/SID2000/cortexlab" style="color: #06B6D4; text-decoration: none;">HuggingFace</a>
    &nbsp;&bull;&nbsp;
    CC BY-NC 4.0
</div>
""", unsafe_allow_html=True)
