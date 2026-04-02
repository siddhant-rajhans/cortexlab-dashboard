"""CortexLab Dashboard - Home Page with Data Management."""

import streamlit as st
import numpy as np

from session import init_session, data_summary_widget, show_analysis_log, upload_npy_widget
from utils import make_roi_indices

st.set_page_config(page_title="CortexLab Dashboard", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
init_session()

st.title("CortexLab Dashboard")
st.markdown("**Research-grade analysis toolkit for multimodal fMRI brain encoding**")

# --- Data Source ---
st.divider()
st.subheader("Data Configuration")

col_src, col_params = st.columns([1, 2])

with col_src:
    source = st.radio("Data source", ["Synthetic (realistic)", "Upload your data"], index=0)
    st.session_state["data_source"] = "synthetic" if "Synthetic" in source else "uploaded"

with col_params:
    if st.session_state["data_source"] == "synthetic":
        c1, c2, c3, c4 = st.columns(4)
        st.session_state["stimulus_type"] = c1.selectbox("Stimulus type", ["visual", "auditory", "language", "multimodal"])
        st.session_state["n_timepoints"] = c2.slider("Duration (TRs)", 30, 200, 80)
        st.session_state["tr_seconds"] = c3.slider("TR (seconds)", 0.5, 2.0, 1.0, 0.1)
        st.session_state["seed"] = c4.number_input("Seed", value=42, min_value=0)

        # Generate on config change
        roi_indices, n_vertices = make_roi_indices()
        st.session_state["roi_indices"] = roi_indices
        st.session_state["n_vertices"] = n_vertices

        from synthetic import generate_realistic_predictions
        predictions = generate_realistic_predictions(
            st.session_state["n_timepoints"], roi_indices,
            st.session_state["stimulus_type"], st.session_state["tr_seconds"],
            seed=st.session_state["seed"],
        )
        st.session_state["brain_predictions"] = predictions
    else:
        uploaded = upload_npy_widget("Upload brain predictions (.npy, shape: timepoints x vertices)", "upload_predictions")
        if uploaded is not None:
            st.session_state["brain_predictions"] = uploaded
            roi_indices, _ = make_roi_indices()
            st.session_state["roi_indices"] = roi_indices

# --- Data Summary ---
roi_indices = st.session_state.get("roi_indices")
predictions = st.session_state.get("brain_predictions")
if predictions is not None and roi_indices is not None:
    data_summary_widget(predictions, roi_indices)

    # Show HRF-convolved signal preview
    with st.expander("Data Preview", expanded=False):
        import plotly.graph_objects as go
        from utils import ROI_GROUPS

        fig = go.Figure()
        t = np.arange(predictions.shape[0]) * st.session_state.get("tr_seconds", 1.0)
        colors = {"Visual": "#00D2FF", "Auditory": "#FF6B6B", "Language": "#A29BFE", "Executive": "#FFEAA7"}
        for group, rois in ROI_GROUPS.items():
            vals = []
            for roi in rois:
                if roi in roi_indices:
                    verts = roi_indices[roi]
                    valid = verts[verts < predictions.shape[1]]
                    if len(valid) > 0:
                        vals.append(np.abs(predictions[:, valid]).mean(axis=1))
            if vals:
                mean_tc = np.mean(vals, axis=0)
                fig.add_trace(go.Scatter(x=t, y=mean_tc, name=group, line=dict(color=colors.get(group, "#888"), width=2)))

        fig.update_layout(
            xaxis_title="Time (seconds)", yaxis_title="Mean |activation|",
            height=300, template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Mean absolute activation per functional group. Note the hemodynamic response shape and modality-specific activation patterns.")

# --- Navigation ---
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.subheader("Analysis Tools")
    st.page_link("pages/1_Brain_Alignment.py", label="Brain Alignment Benchmark", icon="🎯")
    st.caption("RSA, CKA, Procrustes with permutation tests, bootstrap CIs, FDR correction, noise ceiling, and RDM visualization")

    st.page_link("pages/2_Cognitive_Load.py", label="Cognitive Load Scorer", icon="📊")
    st.caption("Timeline with confidence bands, dimension correlation, per-ROI breakdown, comparison mode")

with col2:
    st.subheader("Advanced Analysis")
    st.page_link("pages/3_Temporal_Dynamics.py", label="Temporal Dynamics", icon="⏱️")
    st.caption("Raw timecourses, peak latency hierarchy, optimal lag analysis, cross-ROI lag matrix")

    st.page_link("pages/4_Connectivity.py", label="ROI Connectivity", icon="🔗")
    st.caption("Partial correlation, modularity, betweenness centrality, dendrogram, network graph")

# --- Analysis Log ---
show_analysis_log()

st.divider()
st.caption("[GitHub](https://github.com/siddhant-rajhans/cortexlab) | [HuggingFace](https://huggingface.co/SID2000/cortexlab) | [Dashboard Repo](https://github.com/siddhant-rajhans/cortexlab-dashboard)")
