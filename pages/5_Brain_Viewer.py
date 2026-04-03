"""Interactive 3D Brain Viewer - Publication Quality + Explorer."""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from session import init_session, show_analysis_log, upload_npy_widget
from brain_mesh import (
    load_fsaverage_mesh,
    load_sulcal_map,
    generate_sample_activations,
    highlight_rois,
    blend_with_sulcal,
    render_publication_views,
    render_interactive_3d,
    roi_summary_table,
    VIEWS,
    ACTIVATION_PATTERNS,
)
from utils import ROI_GROUPS, make_roi_indices

st.set_page_config(page_title="3D Brain Viewer", page_icon="🧠", layout="wide")
init_session()
show_analysis_log()

st.title("🧠 Interactive 3D Brain Viewer")
st.markdown("Explore brain activation patterns on the cortical surface. Publication-quality multi-view panels + interactive 3D rotation.")

# --- Sidebar ---
with st.sidebar:
    st.header("Brain Viewer")

    hemi = st.selectbox("Hemisphere", ["left", "right"], index=0)
    resolution = st.selectbox("Mesh resolution", ["fsaverage5", "fsaverage4"], index=0,
                              help="fsaverage5: 10k vertices (detailed). fsaverage4: 2.5k vertices (fast).")

    st.subheader("Data")
    data_source = st.radio("Data source", ["Sample activations", "From current analysis", "Upload .npy"])

    if data_source == "Sample activations":
        pattern = st.selectbox("Activation pattern", list(ACTIVATION_PATTERNS.keys()),
                               help="Modality-specific activation: visual lights up V1/V2/MT, language lights up Broca's/Wernicke's, etc.")
        seed = st.number_input("Seed", value=42, min_value=0)

    st.subheader("Appearance")
    cmap = st.selectbox("Colormap", ["Hot", "Inferno", "Plasma", "Viridis", "RdBu_r", "Coolwarm"], index=0)
    vmin, vmax = st.slider("Data range", 0.0, 1.0, (0.0, 1.0), 0.05)
    bg_color = st.selectbox("Background", ["#0E1117", "#000000", "#1A1A2E"], index=0,
                            format_func=lambda x: {"#0E1117": "Dark", "#000000": "Black", "#1A1A2E": "Navy"}[x])

    st.subheader("ROI Highlighting")
    roi_groups_selected = st.multiselect("Region groups", list(ROI_GROUPS.keys()))
    available_rois = []
    for g in roi_groups_selected:
        available_rois.extend(ROI_GROUPS[g])
    selected_rois = st.multiselect("Specific ROIs", available_rois, default=available_rois[:5] if available_rois else [])
    show_labels = st.checkbox("Show ROI labels", value=True)

# --- Load Mesh ---
with st.spinner(f"Loading {resolution} brain mesh ({hemi} hemisphere)..."):
    coords, faces = load_fsaverage_mesh(hemi, resolution)
    n_vertices = coords.shape[0]

# --- Load/Generate Data ---
roi_indices, _ = make_roi_indices()

# Map ROI indices to actual mesh vertices (scale to mesh size)
# Since our ROI indices are synthetic (0-580), map them proportionally to actual mesh
mesh_roi_indices = {}
for name, idx in roi_indices.items():
    scaled = (idx * n_vertices // 580).astype(int)
    scaled = scaled[scaled < n_vertices]
    mesh_roi_indices[name] = scaled

if data_source == "Sample activations":
    vertex_data = generate_sample_activations(n_vertices, mesh_roi_indices, pattern, seed)
elif data_source == "Upload .npy":
    uploaded = upload_npy_widget(f"Upload vertex data (.npy, {n_vertices} vertices)", "brain_upload")
    if uploaded is not None and len(uploaded) == n_vertices:
        vertex_data = uploaded
    elif uploaded is not None:
        st.warning(f"Expected {n_vertices} vertices, got {len(uploaded)}. Using sample data.")
        vertex_data = generate_sample_activations(n_vertices, mesh_roi_indices, "visual", 42)
    else:
        vertex_data = generate_sample_activations(n_vertices, mesh_roi_indices, "visual", 42)
elif data_source == "From current analysis":
    preds = st.session_state.get("brain_predictions")
    if preds is not None:
        # Average across timepoints, take first n_vertices
        avg = np.abs(preds).mean(axis=0)
        if len(avg) >= n_vertices:
            vertex_data = avg[:n_vertices]
        else:
            vertex_data = np.pad(avg, (0, n_vertices - len(avg)))
        # Normalize to [0, 1]
        vd_range = vertex_data.max() - vertex_data.min()
        if vd_range > 0:
            vertex_data = (vertex_data - vertex_data.min()) / vd_range
    else:
        st.info("No analysis data in session. Go to Home page to generate data, or use sample activations.")
        vertex_data = generate_sample_activations(n_vertices, mesh_roi_indices, "visual", 42)

# Apply ROI highlighting
if selected_rois:
    vertex_data = highlight_rois(vertex_data, mesh_roi_indices, selected_rois, boost=1.8)

# Blend with sulcal map for anatomical context
try:
    sulc = load_sulcal_map(hemi, resolution)
    vertex_data_display = blend_with_sulcal(vertex_data, sulc)
except Exception:
    vertex_data_display = vertex_data

# --- Publication Views ---
st.subheader("Publication Views")
st.caption("Four standard neuroimaging views. Right-click any panel to save as image.")

fig_pub = render_publication_views(coords, faces, vertex_data_display, cmap, vmin, vmax, bg_color)
st.plotly_chart(fig_pub, use_container_width=True)

# --- Interactive 3D ---
st.divider()
st.subheader("Interactive 3D Explorer")
st.caption("Rotate: drag | Zoom: scroll | Pan: shift+drag")

col_view, col_space = st.columns([1, 3])
with col_view:
    initial_view = st.selectbox("Initial view", list(VIEWS.keys()), index=0)

result = render_interactive_3d(
    coords, faces, vertex_data_display, cmap, vmin, vmax,
    bg_color, initial_view, mesh_roi_indices,
    roi_labels=selected_rois, show_labels=show_labels,
)
if result is not None:
    st.plotly_chart(result, use_container_width=True)

# --- ROI Summary ---
if selected_rois:
    st.divider()
    col_table, col_hist = st.columns([1, 1])

    with col_table:
        st.subheader("ROI Summary")
        summary = roi_summary_table(vertex_data, mesh_roi_indices, selected_rois)
        if summary is not None:
            st.dataframe(
                summary.style.format({"Mean": "{:.4f}", "Std": "{:.4f}", "Min": "{:.4f}", "Max": "{:.4f}"}),
                use_container_width=True, hide_index=True,
            )

    with col_hist:
        st.subheader("Activation Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=vertex_data, nbinsx=50,
            marker_color="rgba(108, 92, 231, 0.7)",
            name="All vertices",
        ))
        # Overlay selected ROI distributions
        group_colors = {"Visual": "#00D2FF", "Auditory": "#FF6B6B", "Language": "#A29BFE", "Executive": "#FFEAA7"}
        for roi in selected_rois[:3]:  # limit to 3 for clarity
            if roi in mesh_roi_indices:
                valid = mesh_roi_indices[roi]
                valid = valid[valid < len(vertex_data)]
                if len(valid) > 0:
                    group = "Other"
                    for g, rois in ROI_GROUPS.items():
                        if roi in rois:
                            group = g
                            break
                    fig_hist.add_trace(go.Histogram(
                        x=vertex_data[valid], nbinsx=20,
                        marker_color=group_colors.get(group, "#888"),
                        name=roi, opacity=0.6,
                    ))
        fig_hist.update_layout(
            xaxis_title="Activation", yaxis_title="Count",
            height=350, template="plotly_dark",
            barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# --- Stats ---
st.divider()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Vertices", f"{n_vertices:,}")
col2.metric("Mean Activation", f"{vertex_data.mean():.4f}")
col3.metric("Active Vertices", f"{(vertex_data > 0.1).sum():,} ({100 * (vertex_data > 0.1).mean():.0f}%)")
col4.metric("Peak", f"{vertex_data.max():.4f}")

# --- Methodology ---
with st.expander("About the 3D Brain Viewer", expanded=False):
    st.markdown("""
**Surface Mesh**: The brain surface is the fsaverage template from FreeSurfer, loaded via
nilearn. fsaverage5 has 10,242 vertices per hemisphere; fsaverage4 has 2,562.

**Activation Overlay**: Vertex-level scalar data is projected onto the mesh surface as a
colormap. The data is blended with the sulcal depth map (anatomical grooves) to provide
spatial context.

**Sample Activations**: Modality-specific patterns assign activation weights to HCP MMP1.0
ROIs based on established functional neuroanatomy. Visual stimuli activate V1/V2/MT,
auditory stimuli activate A1/belt areas, language stimuli activate Broca's (area 44/45)
and Wernicke's (TPOJ1/2).

**ROI Highlighting**: Selected ROIs are amplified (1.8x) to make them visually distinct.
The summary table shows descriptive statistics for highlighted regions.

**Publication Views**: Four standard views (lateral left, lateral right, medial, dorsal)
match the conventions used in neuroimaging journals. Right-click to save individual panels.

**Interactive View**: Supports rotation (drag), zoom (scroll), and pan (shift+drag).
Uses PyVista when available, falls back to Plotly mesh3d.

**References**:
- Fischl, 2012, *NeuroImage* (FreeSurfer surface reconstruction)
- Glasser et al., 2016, *Nature* (HCP MMP1.0 parcellation)
""")
