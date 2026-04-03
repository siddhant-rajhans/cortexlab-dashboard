"""Live Brain Prediction - Real-Time Inference from Webcam, Screen, or Video."""

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from session import init_session, show_analysis_log
from theme import inject_theme, glow_card, section_header
from utils import make_roi_indices, COGNITIVE_DIMENSIONS

st.set_page_config(page_title="Live Inference", page_icon="🔴", layout="wide")
init_session()
inject_theme()
show_analysis_log()

st.title("🔴 Live Brain Prediction")
st.markdown("Real-time brain activation prediction from webcam, screen capture, or video file.")

# --- Check Dependencies ---
deps_ok = True
missing = []

try:
    from live_capture import WebcamCapture, ScreenCapture, FileStreamer, get_capture_source
    from live_engine import LiveInferenceEngine, CORTEXLAB_AVAILABLE
except ImportError as e:
    deps_ok = False
    missing.append(str(e))

# --- Sidebar ---
with st.sidebar:
    st.header("Live Inference")

    source_type = st.selectbox("Source", ["webcam", "screen", "file"],
                               format_func={"webcam": "Webcam + Mic", "screen": "Screen Capture", "file": "Video File"}.get)

    if source_type == "file":
        uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mkv", "mov", "webm"])

    st.subheader("Settings")
    capture_fps = st.slider("Capture FPS", 0.5, 5.0, 1.0, 0.5,
                            help="Frames per second. Higher = more responsive but more CPU/GPU load.")

    if CORTEXLAB_AVAILABLE:
        device = st.selectbox("Device", ["auto", "cuda", "cpu"])
        st.success("CortexLab detected. Real inference available.")
    else:
        device = "cpu"
        st.warning("CortexLab not installed. Running in **simulation mode** (predictions from image statistics).")
        with st.expander("Install CortexLab"):
            st.code("pip install -e ../cortexlab[analysis]", language="bash")

    st.subheader("Display")
    show_brain_3d = st.checkbox("Show 3D brain", value=True)
    show_timeline = st.checkbox("Show cognitive load timeline", value=True)
    timeline_window = st.slider("Timeline window (seconds)", 10, 120, 60)

# --- Initialize Engine ---
roi_indices, n_vertices = make_roi_indices()

if "live_engine" not in st.session_state:
    st.session_state["live_engine"] = None
if "live_running" not in st.session_state:
    st.session_state["live_running"] = False

# --- Controls ---
col_start, col_stop, col_status = st.columns([1, 1, 2])

with col_start:
    start_clicked = st.button("▶ Start", type="primary", use_container_width=True,
                              disabled=st.session_state.get("live_running", False))

with col_stop:
    stop_clicked = st.button("⬛ Stop", use_container_width=True,
                             disabled=not st.session_state.get("live_running", False))

# Handle Start
if start_clicked and deps_ok:
    # Create capture source
    if source_type == "webcam":
        capture = WebcamCapture(fps=capture_fps)
    elif source_type == "screen":
        capture = ScreenCapture(fps=capture_fps)
    elif source_type == "file":
        if uploaded_file is not None:
            import tempfile, os
            tmp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())
            capture = FileStreamer(file_path=tmp_path, fps=capture_fps)
        else:
            st.error("Upload a video file first.")
            st.stop()

    # Create and start engine
    engine = LiveInferenceEngine(
        n_vertices=n_vertices,
        roi_indices=roi_indices,
        device=device,
    )
    engine.start(capture)
    st.session_state["live_engine"] = engine
    st.session_state["live_running"] = True
    st.rerun()

# Handle Stop
if stop_clicked:
    engine = st.session_state.get("live_engine")
    if engine:
        engine.stop()
    st.session_state["live_running"] = False
    st.rerun()

# --- Status Bar ---
with col_status:
    engine = st.session_state.get("live_engine")
    if engine and st.session_state.get("live_running"):
        metrics = engine.get_metrics()
        st.markdown(f"""
        <div style="display: flex; gap: 1.5rem; align-items: center; padding: 0.5rem;">
            <span style="color: #EF4444; font-size: 1.2rem;">● LIVE</span>
            <span style="color: #94A3B8;">Mode: <b style="color: #06B6D4;">{metrics.mode}</b></span>
            <span style="color: #94A3B8;">FPS: <b style="color: #10B981;">{metrics.fps:.1f}</b></span>
            <span style="color: #94A3B8;">Predictions: <b style="color: #A29BFE;">{metrics.total_predictions}</b></span>
            <span style="color: #94A3B8;">Latency: <b style="color: #FFEAA7;">{metrics.avg_latency_ms:.0f}ms</b></span>
        </div>
        """, unsafe_allow_html=True)
    elif not st.session_state.get("live_running"):
        st.markdown('<span style="color: #64748B;">Ready. Select a source and click Start.</span>', unsafe_allow_html=True)

st.divider()

# --- Live Display ---
if st.session_state.get("live_running") and engine:
    predictions = engine.get_predictions(timeline_window)

    if predictions:
        latest = predictions[-1]

        # --- Cognitive Load Metrics ---
        cog = latest.cognitive_load
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: glow_card("Overall", f"{cog.get('Overall', 0):.2f}", "", "#7C3AED")
        with c2: glow_card("Visual", f"{cog.get('Visual Complexity', 0):.2f}", "", "#00D2FF")
        with c3: glow_card("Auditory", f"{cog.get('Auditory Demand', 0):.2f}", "", "#FF6B6B")
        with c4: glow_card("Language", f"{cog.get('Language Processing', 0):.2f}", "", "#A29BFE")
        with c5: glow_card("Executive", f"{cog.get('Executive Load', 0):.2f}", "", "#FFEAA7")

        col_brain, col_timeline = st.columns([1, 1])

        # --- 3D Brain ---
        if show_brain_3d:
            with col_brain:
                section_header("Brain Activation", f"t = {latest.timestamp:.1f}s")
                try:
                    from brain_mesh import (
                        load_fsaverage_mesh, render_interactive_3d,
                    )
                    coords, faces = load_fsaverage_mesh("left", "fsaverage4")  # Fast mesh for live
                    n_mesh = coords.shape[0]

                    # Map vertex data to mesh size
                    vd = latest.vertex_data
                    if len(vd) < n_mesh:
                        vd = np.interp(np.linspace(0, len(vd) - 1, n_mesh), np.arange(len(vd)), vd)
                    elif len(vd) > n_mesh:
                        vd = vd[:n_mesh]

                    fig_brain = render_interactive_3d(
                        coords, faces, vd, cmap="Inferno", vmin=0, vmax=0.8,
                        bg_color="#050510", initial_view="Lateral Left",
                    )
                    if fig_brain:
                        fig_brain.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
                        st.plotly_chart(fig_brain, use_container_width=True)
                except Exception as e:
                    st.warning(f"Brain render error: {e}")

        # --- Cognitive Load Timeline ---
        if show_timeline:
            with col_timeline:
                section_header("Cognitive Load Timeline", f"{len(predictions)} data points")

                fig_tl = go.Figure()
                timestamps = [p.timestamp for p in predictions]
                dim_colors = {
                    "Visual Complexity": "#00D2FF",
                    "Auditory Demand": "#FF6B6B",
                    "Language Processing": "#A29BFE",
                    "Executive Load": "#FFEAA7",
                }

                for dim, color in dim_colors.items():
                    values = [p.cognitive_load.get(dim, 0) for p in predictions]
                    fig_tl.add_trace(go.Scatter(
                        x=timestamps, y=values, name=dim.split()[0],
                        line=dict(color=color, width=2), mode="lines",
                    ))

                fig_tl.update_layout(
                    xaxis_title="Time (seconds)", yaxis_title="Load",
                    yaxis_range=[0, 1.05], height=400,
                    template="plotly_dark",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(l=40, r=10, t=10, b=40),
                )
                st.plotly_chart(fig_tl, use_container_width=True)

        # --- Store latest predictions for other pages ---
        all_vertex_data = np.array([p.vertex_data for p in predictions])
        st.session_state["brain_predictions"] = all_vertex_data
        st.session_state["roi_indices"] = roi_indices
        st.session_state["data_source"] = "live_inference"

        # --- Navigation ---
        st.divider()
        st.markdown("**Explore live predictions in other tools:**")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.page_link("pages/5_Brain_Viewer.py", label="Brain Viewer", icon="🧠")
        with c2: st.page_link("pages/2_Cognitive_Load.py", label="Cognitive Load", icon="📊")
        with c3: st.page_link("pages/3_Temporal_Dynamics.py", label="Temporal Dynamics", icon="⏱️")
        with c4: st.page_link("pages/4_Connectivity.py", label="Connectivity", icon="🔗")

    # --- Auto-refresh ---
    time.sleep(1.0)
    st.rerun()

else:
    # --- Not running: show instructions ---
    st.markdown("""
    <div style="
        text-align: center; padding: 3rem 2rem;
        background: rgba(15, 15, 40, 0.4);
        border: 1px solid rgba(100, 100, 255, 0.15);
        border-radius: 16px; margin: 1rem 0;
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🧠</div>
        <h3 style="color: #F1F5F9; margin-bottom: 0.5rem;">Ready for Live Brain Prediction</h3>
        <p style="color: #94A3B8; max-width: 600px; margin: 0 auto;">
            Select a source (webcam, screen capture, or video file) from the sidebar,
            then click <b>Start</b> to begin real-time brain activation prediction.
        </p>
        <div style="margin-top: 1.5rem; display: flex; justify-content: center; gap: 2rem;">
            <div style="text-align: center;">
                <div style="font-size: 1.5rem;">📹</div>
                <div style="color: #06B6D4; font-size: 0.85rem; font-weight: 600;">Webcam</div>
                <div style="color: #64748B; font-size: 0.75rem;">Live camera feed</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem;">🖥️</div>
                <div style="color: #7C3AED; font-size: 0.85rem; font-weight: 600;">Screen</div>
                <div style="color: #64748B; font-size: 0.75rem;">Capture display</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem;">🎬</div>
                <div style="color: #EC4899; font-size: 0.85rem; font-weight: 600;">Video File</div>
                <div style="color: #64748B; font-size: 0.75rem;">Frame-by-frame</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show last predictions if available
    if st.session_state.get("brain_predictions") is not None and st.session_state.get("data_source") == "live_inference":
        st.info(f"Previous session predictions available ({st.session_state['brain_predictions'].shape[0]} timepoints). Navigate to analysis pages to explore them.")

# --- Methodology ---
with st.expander("About Live Inference", expanded=False):
    st.markdown(f"""
**Mode: {'Real (CortexLab)' if CORTEXLAB_AVAILABLE else 'Simulation'}**

{'**Real Inference**: Uses TRIBE v2 to extract features (V-JEPA2, Wav2Vec-BERT, LLaMA 3.2) and predict fMRI brain activation at each captured frame. Requires GPU for interactive speed.' if CORTEXLAB_AVAILABLE else '**Simulation Mode**: CortexLab is not installed. Predictions are generated from image statistics (brightness, contrast, color variance) mapped to brain ROIs. This demonstrates the pipeline without requiring GPU or model weights.'}

**Sources:**
- **Webcam**: Captures frames via OpenCV. Requires `pip install opencv-python`.
- **Screen Capture**: Captures display via mss. Requires `pip install mss Pillow`.
- **Video File**: Reads uploaded video frame-by-frame at the specified FPS.

**Cognitive Load Dimensions** are computed from predicted vertex activations
grouped by HCP MMP1.0 ROIs (same method as the Cognitive Load Scorer page).

**Performance:**
- Simulation mode: ~1-5ms per frame (CPU)
- Real inference with GPU: ~50-200ms per frame
- Real inference with CPU: ~5-30s per frame (not recommended)

**To enable real inference:**
```bash
pip install -e path/to/cortexlab[analysis]
```
""")
