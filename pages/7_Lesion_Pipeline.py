"""Modality Lesion Pipeline - the v0.2 headline feature.

Surfaces what `cortexlab.analysis.lesion.run_modality_lesion` produces:

- per-vertex full-model R^2,
- per-vertex delta R^2 for each modality (vision / audio / text),
- per-vertex p-values from row-permutation tests (Phipson-Smyth +1 smoothing),
- per-vertex BH-FDR q-values,
- per-ROI summary with frac-significant at user-chosen alpha.

Works in two modes:

1. Synthetic (default): biologically-plausible delta R^2 maps generated
   so the dashboard is a fair preview of the real pipeline without
   requiring a Jarvis-class GPU.
2. Real-data: upload an `subject_XX_lesion.npz` produced by
   `experiments.causal_modality_ablation` and the same UI renders against
   the actual cortexlab output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from brain_mesh import load_fsaverage_mesh, render_interactive_3d
from cortexlab_bridge import cortexlab_status, load_lesion_npz
from session import init_session
from synthetic import generate_lesion_results, roi_summary_from_lesion
from theme import inject_theme, section_header
from utils import ROI_GROUPS, make_roi_indices

st.set_page_config(
    page_title="Lesion Pipeline",
    page_icon="🧪",
    layout="wide",
)
init_session()
inject_theme()

# ---------------------------------------------------------------- header ---

st.title("🧪 Modality Lesion Pipeline")
st.markdown(
    "Per-modality causal ablation with row-permutation tests and "
    "Benjamini-Hochberg FDR. Mirrors the v0.2 "
    "`cortexlab.analysis.lesion.run_modality_lesion` output."
)

# --------------------------------------------------------------- sidebar ---

bridge = cortexlab_status()

with st.sidebar:
    st.header("Pipeline")

    mode_label_default = (
        "Synthetic preview"
        if not bridge.available
        else "Synthetic preview"
    )
    mode = st.radio(
        "Data source",
        ["Synthetic preview", "Upload real .npz"],
        index=0,
        help=(
            "Synthetic mirrors what a real run produces using "
            "biologically-plausible ROI assignments. Upload .npz to feed "
            "the page a real `subject_XX_lesion.npz` from "
            "`experiments.causal_modality_ablation`."
        ),
    )

    if mode == "Synthetic preview":
        seed = int(st.number_input("Seed", value=42, min_value=0, max_value=10_000))
        n_perm = int(
            st.slider(
                "Permutations (simulated)",
                min_value=100, max_value=2000, value=1000, step=100,
                help="Sets the p-value floor 1/(B+1). Mirrors the "
                     "`--permutations` flag on the real orchestrator.",
            )
        )
        uploaded_npz = None
    else:
        seed = 42
        n_perm = 1000
        uploaded_npz = st.file_uploader(
            "subject_XX_lesion.npz",
            type=["npz"],
            help=(
                "From `experiments/causal_modality_ablation.py --output`. "
                "Needs `full_r2`, `delta_<modality>`, and `p_<modality>` "
                "arrays. Q-values are recomputed via "
                "`cortexlab.analysis.stats.bh_fdr` at load time."
            ),
        )

    st.subheader("Significance")
    alpha = float(
        st.slider(
            "BH-FDR threshold (q)",
            min_value=0.001, max_value=0.20,
            value=0.05, step=0.005,
            format="%.3f",
            help=(
                "Move this to see how many vertices survive correction. "
                "Tighter q is the publication standard (q < 0.05); "
                "looser thresholds (q < 0.10, 0.15) are sometimes "
                "reported for exploratory analyses."
            ),
        )
    )

    st.subheader("Display")
    hemi = st.selectbox("Hemisphere", ["left", "right"], index=0)
    mesh_resolution = st.selectbox(
        "Mesh resolution",
        ["fsaverage5", "fsaverage4"],
        index=0,
    )
    apply_q_mask = st.checkbox(
        "Mask map to q < threshold",
        value=True,
        help="Off: show raw delta R^2 everywhere. On: zero out "
             "vertices where q >= threshold (the publication recipe).",
    )

# --------------------------------------------------------------- runtime ---

with st.spinner("Loading mesh..."):
    coords, faces = load_fsaverage_mesh(hemi, mesh_resolution)
    n_mesh_v = coords.shape[0]

roi_indices_580, n_roi_v = make_roi_indices()
mesh_roi_indices = {}
for name, idx in roi_indices_580.items():
    scaled = (idx * n_mesh_v // n_roi_v).astype(int)
    scaled = scaled[scaled < n_mesh_v]
    mesh_roi_indices[name] = scaled

if mode == "Synthetic preview":
    lesion = generate_lesion_results(
        n_vertices=n_mesh_v,
        roi_indices=mesh_roi_indices,
        n_permutations=n_perm,
        seed=seed,
    )
    source_caption = (
        f"Synthetic preview · {n_mesh_v:,} vertices · "
        f"{lesion['n_permutations']:,} permutations · seed {seed}"
    )
elif uploaded_npz is None:
    st.info(
        "Upload a `subject_XX_lesion.npz` in the sidebar to populate this "
        "page from a real lesion run, or switch to synthetic preview."
    )
    st.stop()
else:
    try:
        lesion = load_lesion_npz(uploaded_npz)
        source_caption = (
            f"Real run · {lesion['full_r2'].shape[0]:,} vertices · "
            f"{lesion['n_permutations']:,} permutations · "
            f"modalities: {', '.join(lesion['modality_order'])}"
        )
    except Exception as e:
        st.error(f"Could not parse uploaded .npz: {e}")
        st.stop()

st.caption(source_caption)

# ----------------------------------------------------------- summary row ---

modalities = lesion["modality_order"]

cols = st.columns(len(modalities) + 1)

cols[0].metric(
    "Full-model mean R²",
    f"{float(lesion['full_r2'].mean()):.3f}",
    help="Encoder R² averaged over all vertices.",
)

palette = {
    "vision": "#06B6D4",
    "video":  "#06B6D4",
    "audio":  "#F472B6",
    "text":   "#A78BFA",
}

for i, m in enumerate(modalities, start=1):
    d = lesion["delta_r2"][m]
    q = lesion["q_values"].get(m)
    if q is not None and q.size:
        frac_sig = float((q < alpha).mean())
    else:
        frac_sig = float("nan")
    cols[i].metric(
        f"Δ R² ({m})",
        f"+{d.mean():.4f}",
        delta=f"{frac_sig:.1%} vertices q<{alpha:.3f}",
        delta_color="normal" if frac_sig > 0 else "off",
        help=f"Mean lesion drop when {m} is ablated. The delta line "
             f"reports the fraction of vertices that survive BH-FDR at "
             f"q < {alpha:.3f}.",
    )

# ------------------------------------------------------------- main plots ---

section_header(
    "Per-modality cortical maps",
    "Each panel shows Δ R² for ablating one modality. Hot vertices "
    "lose the most predictive power when that modality is removed.",
)

map_tabs = st.tabs([f"Δ R² ({m})" for m in modalities])

for m, tab in zip(modalities, map_tabs):
    with tab:
        d = lesion["delta_r2"][m].copy()
        q = lesion["q_values"].get(m)

        if apply_q_mask and q is not None and q.size == d.size:
            d_display = d.copy()
            d_display[q >= alpha] = 0.0
        else:
            d_display = d

        # Normalise to [0, 1] for the existing mesh colormap pipeline.
        rng_max = max(float(np.nanmax(np.abs(d_display))), 1e-6)
        d_norm = np.clip(d_display / rng_max, -1.0, 1.0)
        d_norm = (d_norm - d_norm.min()) / (d_norm.max() - d_norm.min() + 1e-9)

        c1, c2 = st.columns([3, 1], gap="medium")
        with c1:
            fig = render_interactive_3d(
                coords, faces, d_norm.astype(np.float32),
                "Hot", 0.0, 1.0,
                "#0E1117",
                "Lateral Left" if hemi == "left" else "Lateral Right",
                mesh_roi_indices,
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown(f"**Modality: `{m}`**")
            st.markdown(
                f"- Vertices with Δ > 0: **{(d > 0).sum():,}** / "
                f"{d.size:,}"
            )
            if q is not None and q.size:
                st.markdown(
                    f"- Survive q < {alpha:.3f}: **{(q < alpha).sum():,}**"
                )
                st.markdown(
                    f"- Median q: **{float(np.median(q)):.3f}**"
                )
            st.markdown(
                f"- Top-quintile Δ R²: **+{float(np.quantile(d, 0.8)):.4f}**"
            )

# -------------------------------------------------------- per-ROI table ---

section_header(
    "Per-ROI summary",
    "One row per (ROI, modality). Sorted by Δ R² descending within "
    "the currently-selected modality. Highlights the rows where "
    "fraction-significant > 0 at the chosen threshold.",
)

active_modality = st.selectbox(
    "Rank ROIs by Δ R² for modality",
    modalities,
    index=0,
    key="lesion_rank_modality",
)

rows = roi_summary_from_lesion(lesion, mesh_roi_indices, alpha=alpha)
df = pd.DataFrame(rows)

if not df.empty:
    df_view = df[df["modality"] == active_modality].copy()
    df_view = df_view.sort_values("delta_R2_mean", ascending=False)
    df_view = df_view.rename(columns={
        "ROI": "ROI",
        "n_voxels": "n vert",
        "full_R2": "full R²",
        "delta_R2_mean": "Δ R² mean",
        "delta_R2_top20": "Δ R² top-20%",
        "p_median": "p median",
        "q_median": "q median",
        "frac_q_sig": f"frac q<{alpha:.3f}",
    })
    df_view = df_view.drop(columns=["modality"])

    st.dataframe(
        df_view.style.format({
            "full R²": "{:.3f}",
            "Δ R² mean": "{:+.4f}",
            "Δ R² top-20%": "{:+.4f}",
            "p median": "{:.3f}",
            "q median": "{:.3f}",
            f"frac q<{alpha:.3f}": "{:.1%}",
        }).background_gradient(
            subset=["Δ R² mean"],
            cmap="RdBu_r",
            vmin=-0.10, vmax=0.10,
        ),
        use_container_width=True,
        hide_index=True,
        height=420,
    )

# ------------------------------------------------------------ distribution ---

section_header(
    "Δ R² distribution by modality",
    "Histogram of per-vertex Δ R² across the whole cortex. "
    "Synthetic data should show a tight null centred on zero with a "
    "right-tail of driving ROIs.",
)

fig_hist = go.Figure()
for m in modalities:
    fig_hist.add_trace(go.Histogram(
        x=lesion["delta_r2"][m],
        nbinsx=60,
        name=m,
        marker_color=palette.get(m, "#FBBF24"),
        opacity=0.65,
    ))
fig_hist.update_layout(
    xaxis_title="Δ R² per vertex",
    yaxis_title="Vertex count",
    barmode="overlay",
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    height=380,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig_hist, use_container_width=True)

# ----------------------------------------------------------------- methods ---

with st.expander("About the lesion pipeline", expanded=False):
    st.markdown(
        f"""
**Pipeline steps** (`cortexlab.analysis.lesion.run_modality_lesion`):

1. Fit a GPU voxelwise ridge encoder on train features → brain
   responses. Compute baseline R² on held-out test.
2. For each modality, zero out that modality's feature block at test
   time, re-predict (no refit), and compute Δ R² = R²_full − R²_ablated
   per vertex.
3. Repeat row-permutation of the same modality's test rows. For each
   permutation, recompute Δ R² and accumulate. One-sided p-value per
   vertex = (count(perm Δ R² ≥ observed Δ R²) + 1) / (B + 1).
4. Apply Benjamini-Hochberg FDR (`cortexlab.analysis.stats.bh_fdr`) to
   the per-vertex p-values for each modality.

**Significance reporting** at the per-ROI level. For each ROI we
compute median p, median q, and `frac_q_sig` = fraction of ROI
vertices with q < α. A row of `frac_q_sig > 0` for vision in V1
means "V1 vertices that survive {alpha:.3f} BH-FDR after ablating
vision are non-trivial."

**Synthetic vs real**. Synthetic mode generates delta R² maps where
driving ROIs (V1/V2/V4/MT for vision, A1 and belt areas for audio,
Broca's and STV for text) carry the signal. P-values for non-driving
vertices are sampled uniform on [{1/(n_perm+1):.4f}, 1] which is what
a real row-permutation test produces under the null.

**To replace synthetic with real**: upload an
`subject_XX_lesion.npz` produced by
`experiments/causal_modality_ablation.py`. The dashboard reads the
same keys (`full_r2`, `delta_<modality>`, `p_<modality>`).

**Cortexlab bridge**: `{bridge.available}` ({bridge.version or 'n/a'}).
The page never hard-imports cortexlab; it probes at startup and falls
back to synthetic when missing.

**References**:
- Phipson & Smyth (2010), Permutation P-values should never be zero.
- Benjamini & Hochberg (1995), Controlling the false discovery rate.
- Lahner et al. (2024), BOLD Moments dataset.
"""
    )
