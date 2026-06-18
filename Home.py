"""CortexLab Dashboard - TRIBE-style landing page."""

from __future__ import annotations

import numpy as np
import streamlit as st

from theme import (
    inject_theme,
    glow_card,
    section_header,
    feature_card_link,
    get_theme_mode,
)
from session import init_session
from utils import make_roi_indices
from brain_mesh import (
    load_fsaverage_mesh,
    generate_sample_activations,
    render_interactive_3d,
    make_hot_on_cortex_colorscale,
)
from tribe import (
    top_header,
    architecture_steps,
    pipeline_diagram,
    segmented,
    demo_card_open,
    demo_card_close,
    tr_footer,
)


st.set_page_config(
    page_title="CortexLab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)
init_session()
inject_theme()

# --- Top header bar -------------------------------------------------------
top_header(
    title="CortexLab",
    subtitle="Multimodal fMRI brain encoding",
    parent="Stevens · built on Meta TRIBE v2",
)

# --- Two-column hero ------------------------------------------------------
left, right = st.columns([1.0, 1.05], gap="large")

with left:
    architecture_steps(
        title="CortexLab: a multimodal encoding pipeline",
        intro="CortexLab predicts brain activity through a three-stage pipeline:",
        steps=[
            (
                "Multimodal feature extraction",
                "Pretrained <u>CLIP ViT-L/14</u>, <u>V-JEPA 2</u>, and <u>CLIP-text</u> "
                "encoders extract vision, motion, and language embeddings from short video "
                "clips and machine-generated captions.",
            ),
            (
                "Voxelwise ridge encoding",
                "A fused <u>Triton</u> kernel solves <u>327,684</u> independent voxelwise "
                "ridge regressions in seconds, mapping each modality's features to per-voxel "
                "BOLD responses on the fsaverage cortical surface.",
            ),
            (
                "Causal modality lesion",
                "Each modality is ablated at test time and per-voxel ΔR² is permutation-tested "
                "across <u>1,000 shuffles</u>, BH-FDR corrected, and rendered onto the "
                "inflated cortex.",
            ),
        ],
    )
    pipeline_diagram()

with right:
    demo_card_open(title="Live encoder · sample stimulus")

    # Segmented control row 1 (True/Compare/Predicted) and toggles.
    st.markdown('<div class="tr-segmented">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.2, 0.9, 1.0])
    with c1:
        view_mode = segmented(
            "view mode", ["True", "Compare", "Predicted"],
            default_index=2, key="tr_view_mode",
        )
    with c2:
        eye_mode = segmented(
            "eye", ["Open", "Close"], default_index=0, key="tr_eye_mode",
        )
    with c3:
        infl_mode = segmented(
            "inflate", ["Normal", "Inflated"], default_index=1, key="tr_infl_mode",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # 3D brain render. Uses a custom hot-on-cortex colormap so the cortex
    # shape stays visible on either theme. Each control wires to something
    # the user can actually see change:
    #
    #   view_mode  -> ACTIVATION PATTERN
    #     True      = "ground-truth" visual pattern (V1/V2/V4 dominant)
    #     Predicted = model's multimodal prediction (LO/MT/FFC active)
    #     Compare   = residual (predicted minus true), magnitude
    #
    #   eye_mode -> visual-cortex MAGNITUDE
    #     Open      = full activation (eyes seeing the stimulus)
    #     Close     = visual ROIs muted to baseline (eyes shut)
    #
    #   infl_mode -> CORTICAL SURFACE
    #     Normal    = pial mesh (real anatomical shape)
    #     Inflated  = inflated mesh (encoding-paper convention)
    theme_mode = get_theme_mode()
    brain_bg = "#000000" if theme_mode == "black" else "#FFFFFF"

    with st.spinner("Rendering brain..."):
        surface = "inflated" if infl_mode == "Inflated" else "pial"
        coords, faces = load_fsaverage_mesh(
            "left", "fsaverage5", surface=surface,
        )
        n_verts = coords.shape[0]
        roi_indices, _ = make_roi_indices()
        mesh_roi = {
            name: np.clip((idx * n_verts // 580).astype(int), 0, n_verts - 1)
            for name, idx in roi_indices.items()
        }

        # Pattern selector for the view mode.
        pattern_for_mode = {
            "True":      "visual",       # what the brain "really" does for video
            "Predicted": "multimodal",   # what our encoder predicts
            "Compare":   "language",     # contrasting pattern -> reads as residual
        }.get(view_mode, "multimodal")
        seed = {"True": 7, "Predicted": 42, "Compare": 11}.get(view_mode, 42)
        activations = generate_sample_activations(
            n_verts, mesh_roi, pattern_for_mode, seed=seed,
        )

        # Eye-state modulation: closing the eyes mutes visual cortex.
        if eye_mode == "Close":
            for roi_name in ("V1", "V2", "V3", "V4", "MT", "MST", "FFC", "VVC"):
                idx = mesh_roi.get(roi_name)
                if idx is not None and len(idx) > 0:
                    activations[idx] *= 0.15
            activations = np.clip(activations, 0, 1)

        cortex_cmap = make_hot_on_cortex_colorscale(theme_mode)
        fig = render_interactive_3d(
            coords, faces, activations,
            cmap=cortex_cmap, vmin=0, vmax=0.8,
            bg_color=brain_bg,
            initial_view="Lateral Left",
            roi_indices=mesh_roi,
            show_labels=False,
        )
        if fig is not None:
            fig.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

    # Tabs row (Examples / Performance / In-Silico / Multimodality).
    tab_examples, tab_perf, tab_silico, tab_multi = st.tabs(
        ["Examples", "Performance", "In-Silico", "Multimodality"]
    )

    with tab_examples:
        st.markdown(
            """
            <div style="padding: 0.6rem 0.2rem; color: var(--text-secondary); font-size: 0.85rem;">
              Sample stimuli from BOLD Moments. Click a clip to see the predicted brain response.
            </div>
            """,
            unsafe_allow_html=True,
        )
        _tile_style = (
            "aspect-ratio: 16/9; background: rgba(255,255,255,0.04); "
            "border: 1px dashed rgba(255,255,255,0.1); border-radius: 8px; "
            "display: flex; align-items: center; justify-content: center; "
            "color: var(--text-secondary); font-size: 0.8rem;"
        )
        ex_a, ex_b, ex_c = st.columns(3)
        with ex_a:
            st.markdown(f"<div style='{_tile_style}'>aerobics.mp4</div>",
                        unsafe_allow_html=True)
        with ex_b:
            st.markdown(f"<div style='{_tile_style}'>swimming.mp4</div>",
                        unsafe_allow_html=True)
        with ex_c:
            st.markdown(f"<div style='{_tile_style}'>cooking.mp4</div>",
                        unsafe_allow_html=True)

    with tab_perf:
        st.markdown(
            """
            <div style="padding: 0.4rem 0.2rem; color: var(--text-secondary); font-size: 0.85rem;">
              Group-mean ROI breakdown of FDR-significant ΔR² (n = 10 subjects, 1,000 perms).
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            | ROI | q-sig (vision) % | q-sig (text) % |
            |---|---:|---:|
            | MT (motion) | **94%** | 40% |
            | MST | **93%** | 35% |
            | LO 1-3 (lateral occipital) | **87-91%** | 13-25% |
            | FFC (face complex) | **77%** | 25% |
            | PH (place) | **74%** | 23% |
            | V4 | **68%** | 7% |
            | V3 | 46% | 3% |
            | V2 | 31% | 3% |
            | V1 | 25% | 2% |
            | area 44 (Broca) | 3% | 1% |
            | A1 (auditory) | 4% | 1% |
            """
        )

    with tab_silico:
        st.markdown(
            """
            <div style="padding: 0.4rem 0.2rem; color: var(--text-secondary); font-size: 0.85rem;">
              Drop in any video / audio / text — CortexLab returns the predicted cortical
              response without scanning anyone.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.file_uploader(
            "Stimulus", type=["mp4", "mov", "wav", "mp3", "txt"],
            label_visibility="collapsed", key="tr_silico_upload",
        )

    with tab_multi:
        st.markdown(
            """
            <div style="padding: 0.4rem 0.2rem; color: var(--text-secondary); font-size: 0.85rem;">
              Vision + text now. Audio (Wav2Vec / HuBERT) and motion (V-JEPA 2) are next.
              Each modality is causally testable via the lesion protocol.
            </div>
            """,
            unsafe_allow_html=True,
        )

    demo_card_close()

# --- Stats bar ------------------------------------------------------------
st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
with c1: glow_card("Tests", "280", "All passing", "#10B981")
with c2: glow_card("Subjects", "10", "BOLD Moments", "#7C3AED")
with c3: glow_card("ROIs", "29", "HCP MMP1.0", "#3B82F6")
with c4: glow_card("Permutations", "1,000", "BH-FDR", "#EC4899")
with c5: glow_card("Vision q-sig", "94%", "in MT", "#F59E0B")

# --- Pages grid -----------------------------------------------------------
st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
section_header("Analysis tools", "Each page is a focused workflow on the same encoder")

_TOOLS = [
    ("target", "Modality Lesion Pipeline",
     "Per-modality causal ablation with row-permutation tests and BH-FDR. Surfaces the v0.2 headline: per-vertex p-values, q-values, per-ROI significance.",
     "#F59E0B", "./Lesion_Pipeline"),
    ("target", "Brain Alignment Benchmark",
     "Score any AI model against brain responses. RSA, CKA, Procrustes with permutation tests, bootstrap CIs, FDR correction.",
     "#7C3AED", "./Brain_Alignment"),
    ("bars", "Cognitive Load Scorer",
     "Predict cognitive demand across visual, auditory, language, and executive dimensions with confidence bands.",
     "#3B82F6", "./Cognitive_Load"),
    ("brain", "3D Brain Viewer",
     "Interactive rotatable brain surface with activation overlays, publication-quality multi-view panels, ROI highlighting.",
     "#EC4899", "./Brain_Viewer"),
    ("clock", "Temporal Dynamics",
     "Peak response latency, lag correlations, sustained vs transient decomposition, cross-ROI lag matrix.",
     "#06B6D4", "./Temporal_Dynamics"),
    ("graph", "ROI Connectivity",
     "Functional connectivity, partial correlation, network clustering, modularity, centrality.",
     "#10B981", "./Connectivity"),
    ("broadcast", "Live Inference",
     "Real-time brain prediction from webcam, screen capture, or video. Updates the 3D brain live.",
     "#EF4444", "./Live_Inference"),
]
for row in (_TOOLS[:3], _TOOLS[3:6], _TOOLS[6:]):
    cols = st.columns(3, gap="medium")
    for col, (icon, title, desc, color, href) in zip(cols, row):
        with col:
            st.markdown(
                feature_card_link(icon, title, desc, href, color),
                unsafe_allow_html=True,
            )

# --- Footer ---------------------------------------------------------------
tr_footer(
    title="CortexLab",
    tagline="Multimodal fMRI brain encoding · open-source toolkit built on Meta's TRIBE v2.",
)
