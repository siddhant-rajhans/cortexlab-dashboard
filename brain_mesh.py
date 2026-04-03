"""3D brain mesh loading, data projection, and rendering utilities.

Supports both publication-quality multi-view panels (Plotly) and
interactive 3D exploration (PyVista/stpyvista with Plotly fallback).
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import ROI_GROUPS

# --- Camera Presets ---

VIEWS = {
    "Lateral Left": dict(eye=dict(x=-1.7, y=0, z=0.3), up=dict(x=0, y=0, z=1)),
    "Lateral Right": dict(eye=dict(x=1.7, y=0, z=0.3), up=dict(x=0, y=0, z=1)),
    "Medial": dict(eye=dict(x=1.5, y=0.3, z=0.2), up=dict(x=0, y=0, z=1)),
    "Dorsal": dict(eye=dict(x=0, y=0, z=2.2), up=dict(x=0, y=1, z=0)),
    "Ventral": dict(eye=dict(x=0, y=0, z=-2.2), up=dict(x=0, y=1, z=0)),
    "Anterior": dict(eye=dict(x=0, y=1.7, z=0.3), up=dict(x=0, y=0, z=1)),
    "Posterior": dict(eye=dict(x=0, y=-1.7, z=0.3), up=dict(x=0, y=0, z=1)),
}

# --- Modality activation patterns ---

ACTIVATION_PATTERNS = {
    "visual": {
        "V1": 1.0, "V2": 0.9, "V3": 0.8, "V4": 0.7,
        "MT": 0.75, "MST": 0.65, "FFC": 0.6, "VVC": 0.55,
        "A1": 0.05, "LBelt": 0.04, "44": 0.08, "45": 0.07,
        "46": 0.25, "FEF": 0.35,
    },
    "auditory": {
        "A1": 1.0, "LBelt": 0.9, "MBelt": 0.85, "PBelt": 0.8,
        "A4": 0.7, "A5": 0.65,
        "V1": 0.03, "44": 0.12, "45": 0.1, "TPOJ1": 0.25,
        "46": 0.15,
    },
    "language": {
        "44": 1.0, "45": 0.95, "IFJa": 0.85, "IFJp": 0.8,
        "TPOJ1": 0.9, "TPOJ2": 0.85, "STV": 0.75, "PSL": 0.7,
        "V1": 0.05, "A1": 0.25, "46": 0.45,
    },
    "multimodal": {
        "V1": 0.6, "V2": 0.55, "MT": 0.5,
        "A1": 0.6, "LBelt": 0.55,
        "44": 0.55, "45": 0.5, "TPOJ1": 0.5,
        "46": 0.35, "FEF": 0.3,
    },
}


# --- Mesh Loading ---

@st.cache_resource
def load_fsaverage_mesh(hemi="left", resolution="fsaverage5"):
    """Load fsaverage brain mesh via nilearn. Returns (coords, faces)."""
    from nilearn.datasets import fetch_surf_fsaverage
    from nilearn.surface import load_surf_mesh

    fsaverage = fetch_surf_fsaverage(mesh=resolution)
    key = f"pial_{hemi}"
    coords, faces = load_surf_mesh(fsaverage[key])
    return np.array(coords, dtype=np.float32), np.array(faces, dtype=np.int32)


@st.cache_resource
def load_sulcal_map(hemi="left", resolution="fsaverage5"):
    """Load sulcal depth map for anatomical background."""
    from nilearn.datasets import fetch_surf_fsaverage
    from nilearn.surface import load_surf_data

    fsaverage = fetch_surf_fsaverage(mesh=resolution)
    sulc = load_surf_data(fsaverage[f"sulc_{hemi}"])
    return np.array(sulc, dtype=np.float32)


# --- Data Projection ---

def generate_sample_activations(n_vertices, roi_indices, pattern="visual", seed=42):
    """Generate demo activation data with modality-specific patterns.

    Returns vertex-level activation array of shape (n_vertices,).
    """
    rng = np.random.default_rng(seed)
    weights = ACTIVATION_PATTERNS.get(pattern, ACTIVATION_PATTERNS["visual"])

    data = rng.standard_normal(n_vertices) * 0.05  # low baseline noise

    for roi_name, vertices in roi_indices.items():
        w = weights.get(roi_name, 0.02)
        valid = vertices[vertices < n_vertices]
        if len(valid) > 0:
            # Smooth activation with per-vertex jitter
            data[valid] = w + rng.standard_normal(len(valid)) * 0.05

    return np.clip(data, 0, 1)


def highlight_rois(vertex_data, roi_indices, selected_rois, boost=1.5):
    """Amplify activation in selected ROIs for visual highlighting."""
    data = vertex_data.copy()
    for roi in selected_rois:
        if roi in roi_indices:
            valid = roi_indices[roi]
            valid = valid[valid < len(data)]
            if len(valid) > 0:
                data[valid] = np.clip(data[valid] * boost, 0, 1)
    return data


def blend_with_sulcal(vertex_data, sulcal_map, data_opacity=0.85):
    """Blend activation data with sulcal background for anatomical context."""
    sulc_norm = (sulcal_map - sulcal_map.min()) / (sulcal_map.max() - sulcal_map.min() + 1e-8)
    bg = 0.25 + sulc_norm * 0.3  # gray range 0.25-0.55

    # Where activation is low, show more background
    alpha = np.clip(vertex_data * 3, 0, data_opacity)
    blended = alpha * vertex_data + (1 - alpha) * bg
    return blended


# --- Plotly Rendering ---

def _make_mesh3d(coords, faces, vertex_data, cmap, vmin, vmax, opacity=1.0, name=""):
    """Create a Plotly Mesh3d trace."""
    return go.Mesh3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=vertex_data,
        intensitymode="vertex",
        colorscale=cmap,
        cmin=vmin, cmax=vmax,
        opacity=opacity,
        lighting=dict(ambient=0.4, diffuse=0.6, specular=0.3, roughness=0.5, fresnel=0.2),
        lightposition=dict(x=100, y=200, z=300),
        showscale=False,
        name=name,
        hovertemplate="Vertex: %{pointNumber}<br>Value: %{intensity:.3f}<extra></extra>",
    )


def _scene_layout(camera, bg_color="#0E1117"):
    """Create a Plotly 3D scene layout."""
    return dict(
        camera=camera,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor=bg_color,
        aspectmode="data",
    )


def render_publication_views(coords, faces, vertex_data, cmap="Hot", vmin=0, vmax=1, bg_color="#0E1117"):
    """Render 4-panel publication-quality brain views.

    Returns a Plotly figure with lateral left, lateral right, medial, and dorsal views.
    """
    view_keys = ["Lateral Left", "Lateral Right", "Medial", "Dorsal"]

    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "scene"}] * 4],
        subplot_titles=view_keys,
        horizontal_spacing=0.01,
    )

    for i, view_name in enumerate(view_keys, 1):
        mesh = _make_mesh3d(coords, faces, vertex_data, cmap, vmin, vmax, name=view_name)
        fig.add_trace(mesh, row=1, col=i)
        fig.update_layout(**{f"scene{i if i > 1 else ''}": _scene_layout(VIEWS[view_name], bg_color)})

    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor=bg_color,
        font=dict(color="white"),
        showlegend=False,
    )

    # Add colorbar as a separate invisible trace
    fig.add_trace(go.Mesh3d(
        x=[0], y=[0], z=[0], i=[0], j=[0], k=[0],
        intensity=[0], colorscale=cmap, cmin=vmin, cmax=vmax,
        showscale=True,
        colorbar=dict(
            title=dict(text="Activation", side="right"),
            len=0.8, thickness=15, x=1.02,
            tickfont=dict(color="white"),
        ),
        opacity=0,
        hoverinfo="none",
    ))

    return fig


def render_interactive_3d(coords, faces, vertex_data, cmap="Hot", vmin=0, vmax=1,
                          bg_color="#0E1117", initial_view="Lateral Left",
                          roi_indices=None, roi_labels=None, show_labels=False):
    """Render an interactive rotatable 3D brain.

    First attempts PyVista via stpyvista, falls back to Plotly mesh3d.
    """
    # Try stpyvista first
    try:
        return _render_pyvista(coords, faces, vertex_data, cmap, vmin, vmax,
                               bg_color, initial_view, roi_indices, show_labels)
    except Exception:
        pass

    # Fallback: Plotly mesh3d (always works)
    return _render_plotly(coords, faces, vertex_data, cmap, vmin, vmax,
                          bg_color, initial_view, roi_indices, roi_labels, show_labels)


def _render_pyvista(coords, faces, vertex_data, cmap, vmin, vmax,
                    bg_color, initial_view, roi_indices, show_labels):
    """Render with PyVista via stpyvista."""
    import pyvista as pv
    from stpyvista import stpyvista
    from stpyvista.utils import start_xvfb

    if "IS_XVFB_RUNNING" not in st.session_state:
        try:
            start_xvfb()
        except Exception:
            pass
        st.session_state.IS_XVFB_RUNNING = True

    pv_faces = np.column_stack([np.full(len(faces), 3), faces]).ravel()
    mesh = pv.PolyData(coords, pv_faces)
    mesh.point_data["activation"] = vertex_data

    cmap_map = {"Hot": "hot", "Inferno": "inferno", "Plasma": "plasma",
                "Viridis": "viridis", "RdBu_r": "RdBu_r", "Coolwarm": "coolwarm"}
    pv_cmap = cmap_map.get(cmap, "hot")

    plotter = pv.Plotter(window_size=[900, 600], off_screen=True)
    plotter.add_mesh(
        mesh, scalars="activation", cmap=pv_cmap,
        clim=[vmin, vmax], smooth_shading=True,
        ambient=0.4, diffuse=0.6, specular=0.3,
        show_scalar_bar=True,
    )

    if show_labels and roi_indices:
        for name, vertices in roi_indices.items():
            valid = vertices[vertices < len(coords)]
            if len(valid) > 0:
                center = coords[valid].mean(axis=0)
                plotter.add_point_labels(
                    center.reshape(1, 3), [name],
                    font_size=10, shape_opacity=0.3,
                    text_color="white",
                )

    r, g, b = int(bg_color[1:3], 16), int(bg_color[3:5], 16), int(bg_color[5:7], 16)
    plotter.background_color = (r / 255, g / 255, b / 255)

    stpyvista(plotter, key="brain_3d_viewer")
    return None  # stpyvista renders directly


def _render_plotly(coords, faces, vertex_data, cmap, vmin, vmax,
                   bg_color, initial_view, roi_indices, roi_labels, show_labels):
    """Render with Plotly mesh3d (fallback)."""
    fig = go.Figure()

    fig.add_trace(_make_mesh3d(coords, faces, vertex_data, cmap, vmin, vmax))

    # Add ROI labels as scatter3d annotations
    if show_labels and roi_indices:
        label_x, label_y, label_z, label_text = [], [], [], []
        for name, vertices in roi_indices.items():
            valid = vertices[vertices < len(coords)]
            if len(valid) > 0:
                center = coords[valid].mean(axis=0)
                label_x.append(center[0])
                label_y.append(center[1])
                label_z.append(center[2])
                label_text.append(name)

        fig.add_trace(go.Scatter3d(
            x=label_x, y=label_y, z=label_z,
            mode="text",
            text=label_text,
            textfont=dict(size=9, color="white"),
            hoverinfo="text",
            showlegend=False,
        ))

    camera = VIEWS.get(initial_view, VIEWS["Lateral Left"])
    fig.update_layout(
        scene=_scene_layout(camera, bg_color),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=bg_color,
    )

    return fig


# --- ROI Helpers ---

def make_vertex_roi_indices(n_vertices_per_roi=20):
    """Create ROI -> vertex index mapping matching utils.make_roi_indices."""
    from utils import ALL_ROIS
    indices = {}
    offset = 0
    for roi in ALL_ROIS:
        indices[roi] = np.arange(offset, offset + n_vertices_per_roi)
        offset += n_vertices_per_roi
    return indices, offset


def roi_summary_table(vertex_data, roi_indices, selected_rois):
    """Compute summary stats for selected ROIs."""
    import pandas as pd
    rows = []
    for roi in selected_rois:
        if roi in roi_indices:
            valid = roi_indices[roi]
            valid = valid[valid < len(vertex_data)]
            if len(valid) > 0:
                vals = vertex_data[valid]
                group = "Other"
                for g, rois in ROI_GROUPS.items():
                    if roi in rois:
                        group = g
                        break
                rows.append({
                    "ROI": roi,
                    "Group": group,
                    "Mean": float(vals.mean()),
                    "Std": float(vals.std()),
                    "Min": float(vals.min()),
                    "Max": float(vals.max()),
                    "Vertices": len(valid),
                })
    return pd.DataFrame(rows) if rows else None
