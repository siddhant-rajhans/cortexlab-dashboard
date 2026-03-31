"""Temporal Dynamics - Research Grade."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from session import init_session, log_analysis, get_carried_rois, download_csv_button, show_analysis_log
from utils import make_roi_indices, peak_latency, temporal_correlation, decompose_response, ROI_GROUPS, ALL_ROIS
from synthetic import generate_realistic_predictions, generate_correlated_features

st.set_page_config(page_title="Temporal Dynamics", page_icon="⏱️", layout="wide")
init_session()
show_analysis_log()

st.title("⏱️ Temporal Dynamics")
st.markdown("Analyze how brain responses evolve over time, including processing hierarchy and temporal coupling with model features.")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    stim_type = st.selectbox("Stimulus type", ["visual", "auditory", "language", "multimodal"],
                             index=["visual", "auditory", "language", "multimodal"].index(st.session_state.get("stimulus_type", "visual")))
    n_timepoints = st.slider("Duration (TRs)", 30, 200, 80)
    tr_seconds = st.slider("TR (seconds)", 0.5, 2.0, 1.0, 0.1)
    seed = st.number_input("Seed", value=42, min_value=0)

    st.subheader("ROI Selection")
    carried = get_carried_rois()
    use_carried = False
    if carried:
        use_carried = st.checkbox(f"Use {len(carried)} ROIs from Brain Alignment", value=True)

    if use_carried and carried:
        selected_rois = carried
        st.caption(f"Using: {', '.join(selected_rois[:5])}{'...' if len(selected_rois) > 5 else ''}")
    else:
        selected_group = st.selectbox("Region group", list(ROI_GROUPS.keys()))
        available_rois = ROI_GROUPS[selected_group]
        selected_rois = st.multiselect("ROIs to analyze", available_rois, default=available_rois[:4])

    max_lag = st.slider("Max correlation lag (TRs)", 5, 30, 15)
    cutoff = st.slider("Decomposition cutoff (seconds)", 1.0, 10.0, 4.0, 0.5)

if not selected_rois:
    st.warning("Select at least one ROI.")
    st.stop()

# --- Generate Data ---
roi_indices, n_vertices = make_roi_indices()
predictions = generate_realistic_predictions(n_timepoints, roi_indices, stim_type, tr_seconds, seed=seed)
features = generate_correlated_features(predictions, alignment_strength=0.5, feature_dim=64, seed=seed + 1)
log_analysis(f"Temporal dynamics: {stim_type}, {len(selected_rois)} ROIs")

time_axis = np.arange(n_timepoints) * tr_seconds
colors = ["#00D2FF", "#FF6B6B", "#A29BFE", "#FFEAA7", "#55EFC4", "#FD79A8", "#74B9FF", "#E17055"]

# --- Raw ROI Timecourses ---
st.subheader("Raw ROI Timecourses")
st.markdown("Mean absolute activation over time for each selected ROI. Note the hemodynamic response shape after stimulus events.")

fig_raw = go.Figure()
for i, roi in enumerate(selected_rois):
    if roi in roi_indices:
        verts = roi_indices[roi]
        valid = verts[verts < predictions.shape[1]]
        if len(valid) > 0:
            tc = np.abs(predictions[:, valid]).mean(axis=1)
            fig_raw.add_trace(go.Scatter(
                x=time_axis, y=tc, name=roi,
                line=dict(color=colors[i % len(colors)], width=2),
            ))

fig_raw.update_layout(
    xaxis_title="Time (seconds)", yaxis_title="Mean |activation|",
    height=400, template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_raw, use_container_width=True)

# --- Peak Latency (sorted = processing hierarchy) ---
st.divider()
st.subheader("Peak Response Latency (Processing Hierarchy)")
st.markdown("ROIs sorted by peak latency reveal the cortical processing hierarchy: early sensory areas respond first, association cortex later.")

latency_data = []
for roi in selected_rois:
    if roi in roi_indices:
        lat = peak_latency(predictions, roi_indices, roi, tr_seconds)
        # Determine functional group
        group = "Other"
        for g, rois in ROI_GROUPS.items():
            if roi in rois:
                group = g
                break
        latency_data.append({"ROI": roi, "Peak Latency (s)": lat, "Group": group})

lat_df = pd.DataFrame(latency_data).sort_values("Peak Latency (s)")
group_colors = {"Visual": "#00D2FF", "Auditory": "#FF6B6B", "Language": "#A29BFE", "Executive": "#FFEAA7", "Other": "#888"}

col1, col2 = st.columns([2, 1])
with col1:
    fig_lat = go.Figure(go.Bar(
        x=lat_df["Peak Latency (s)"], y=lat_df["ROI"],
        orientation="h",
        marker_color=[group_colors.get(g, "#888") for g in lat_df["Group"]],
    ))
    fig_lat.update_layout(
        xaxis_title="Time to peak (seconds)", height=max(250, len(selected_rois) * 30),
        template="plotly_dark", yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_lat, use_container_width=True)

with col2:
    st.dataframe(lat_df[["ROI", "Peak Latency (s)", "Group"]], use_container_width=True, hide_index=True)
    download_csv_button(lat_df, "peak_latencies.csv")

# --- Lag Correlation with Significance ---
st.divider()
st.subheader("Temporal Correlation (Brain vs Model Features)")
st.markdown("Pearson correlation at different time lags. The peak indicates optimal temporal alignment. "
            "Gray band shows 95% null range from shuffled data.")

lags = np.arange(-max_lag, max_lag + 1) * tr_seconds
fig_corr = go.Figure()

# Null band (shuffle features, compute correlation envelope)
rng = np.random.default_rng(seed)
null_corrs = []
for _ in range(50):
    shuffled = features[rng.permutation(len(features))]
    for roi in selected_rois[:1]:  # Use first ROI for null band
        if roi in roi_indices:
            nc = temporal_correlation(predictions, shuffled, roi_indices, roi, max_lag)
            null_corrs.append(nc)
if null_corrs:
    null_arr = np.array(null_corrs)
    null_hi = np.percentile(null_arr, 97.5, axis=0)
    null_lo = np.percentile(null_arr, 2.5, axis=0)
    fig_corr.add_trace(go.Scatter(x=lags, y=null_hi, mode="lines", line=dict(width=0), showlegend=False))
    fig_corr.add_trace(go.Scatter(x=lags, y=null_lo, mode="lines", line=dict(width=0),
                                   fill="tonexty", fillcolor="rgba(150,150,150,0.2)",
                                   name="95% null range"))

# Actual correlations
optimal_lags = []
for i, roi in enumerate(selected_rois):
    if roi in roi_indices:
        corr = temporal_correlation(predictions, features, roi_indices, roi, max_lag)
        fig_corr.add_trace(go.Scatter(
            x=lags, y=corr, name=roi,
            line=dict(color=colors[i % len(colors)], width=2),
        ))
        opt_idx = np.argmax(np.abs(corr))
        optimal_lags.append({"ROI": roi, "Optimal Lag (s)": lags[opt_idx], "Max |r|": float(np.abs(corr[opt_idx]))})

fig_corr.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
fig_corr.update_layout(
    xaxis_title="Lag (seconds)", yaxis_title="Pearson Correlation",
    height=400, template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_corr, use_container_width=True)

# --- Optimal Lag Summary ---
if optimal_lags:
    st.subheader("Optimal Lag Summary")
    opt_df = pd.DataFrame(optimal_lags).sort_values("Max |r|", ascending=False)
    st.dataframe(opt_df, use_container_width=True, hide_index=True)
    download_csv_button(opt_df, "optimal_lags.csv")

# --- Cross-ROI Lag Matrix ---
if len(selected_rois) >= 2:
    st.divider()
    st.subheader("Cross-ROI Lag Matrix")
    st.markdown("Optimal lag between each pair of ROIs. Positive values mean the row ROI leads the column ROI.")

    n_rois = len(selected_rois)
    lag_matrix = np.zeros((n_rois, n_rois))
    for i, roi_a in enumerate(selected_rois):
        if roi_a not in roi_indices:
            continue
        verts_a = roi_indices[roi_a]
        valid_a = verts_a[verts_a < predictions.shape[1]]
        if len(valid_a) == 0:
            continue
        tc_a = np.abs(predictions[:, valid_a]).mean(axis=1)
        for j, roi_b in enumerate(selected_rois):
            if i == j or roi_b not in roi_indices:
                continue
            verts_b = roi_indices[roi_b]
            valid_b = verts_b[verts_b < predictions.shape[1]]
            if len(valid_b) == 0:
                continue
            tc_b = np.abs(predictions[:, valid_b]).mean(axis=1)
            # Cross-correlation to find optimal lag
            corrs_ab = temporal_correlation(predictions, tc_b, roi_indices, roi_a, max_lag)
            opt_idx = np.argmax(np.abs(corrs_ab))
            lag_matrix[i, j] = lags[opt_idx]

    fig_lagmat = go.Figure(go.Heatmap(
        z=lag_matrix, x=selected_rois, y=selected_rois,
        colorscale="RdBu_r", zmid=0,
        colorbar=dict(title="Lag (s)"),
        text=np.round(lag_matrix, 1), texttemplate="%{text}",
    ))
    fig_lagmat.update_layout(height=400, template="plotly_dark")
    st.plotly_chart(fig_lagmat, use_container_width=True)

# --- Sustained vs Transient ---
st.divider()
st.subheader("Sustained vs Transient Decomposition")
st.markdown("Moving-average filter separates slow sustained responses from fast transient spikes.")

roi_for_decomp = st.selectbox("ROI for decomposition", selected_rois)
sustained, transient = decompose_response(predictions, roi_indices, roi_for_decomp, cutoff, tr_seconds)
original = sustained + transient

fig_decomp = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                           subplot_titles=("Original Signal", "Sustained Component", "Transient Component"))

fig_decomp.add_trace(go.Scatter(x=time_axis, y=original, line=dict(color="#888", width=1.5)), row=1, col=1)
fig_decomp.add_trace(go.Scatter(x=time_axis, y=sustained, line=dict(color="#6C5CE7", width=2)), row=2, col=1)
fig_decomp.add_trace(go.Scatter(x=time_axis, y=transient, line=dict(color="#FF6B6B", width=1.5)), row=3, col=1)

fig_decomp.update_xaxes(title_text="Time (seconds)", row=3, col=1)
fig_decomp.update_layout(height=550, template="plotly_dark", showlegend=False)
st.plotly_chart(fig_decomp, use_container_width=True)

# --- Methodology ---
with st.expander("Methodology", expanded=False):
    st.markdown("""
**Peak Latency** is the time at which mean absolute activation reaches its maximum
within an ROI. In real fMRI, early sensory cortex (V1, A1) peaks at ~5-6s post-stimulus
due to the hemodynamic response, while association cortex (dlPFC, angular gyrus) peaks
~1-3s later reflecting higher-order processing.

**Temporal Correlation** computes Pearson correlation between the ROI timecourse and model
feature timecourse at each lag in ``[-max_lag, +max_lag]`` TRs. The lag at maximum absolute
correlation reveals the temporal offset at which model and brain are best aligned.

**Null significance band** is estimated by shuffling the model features 50 times and
computing the lag correlation each time. The 95% envelope of these null correlations
provides a significance threshold.

**Sustained vs Transient Decomposition** uses a moving-average filter with the specified
cutoff period. The sustained component captures slow, maintained responses (e.g., block
design activations), while the transient component captures fast, event-related responses.

**Cross-ROI Lag Matrix** shows the optimal temporal offset between every pair of ROIs,
revealing directional information flow (positive lag = row ROI leads column ROI).

**References**:
- Boynton et al., 1996, *J Neuroscience* (hemodynamic response function)
- Friston et al., 1998, *NeuroImage* (temporal basis functions in fMRI)
""")
