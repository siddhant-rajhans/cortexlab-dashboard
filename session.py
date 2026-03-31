"""Shared session state management and data I/O utilities.

Manages cross-page state (selected ROIs, predictions, analysis log)
and provides upload/download widgets.
"""

import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


def init_session():
    """Initialize session state with defaults. Safe to call multiple times."""
    defaults = {
        "brain_predictions": None,
        "model_features": {},
        "roi_indices": None,
        "n_vertices": 0,
        "selected_rois": [],
        "data_source": "synthetic",
        "stimulus_type": "visual",
        "tr_seconds": 1.0,
        "n_timepoints": 80,
        "seed": 42,
        "analysis_log": [],
        "carry_rois": [],  # ROIs carried from another page
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def log_analysis(description):
    """Append an entry to the analysis log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {description}"
    if "analysis_log" not in st.session_state:
        st.session_state["analysis_log"] = []
    st.session_state["analysis_log"].append(entry)


def carry_rois(rois, target_page=""):
    """Store selected ROIs for cross-page workflow."""
    st.session_state["carry_rois"] = list(rois)
    log_analysis(f"Carried {len(rois)} ROIs to {target_page}")


def get_carried_rois():
    """Retrieve ROIs carried from another page."""
    return st.session_state.get("carry_rois", [])


def get_or_generate_data(roi_indices):
    """Get brain predictions from session or generate new synthetic data."""
    from synthetic import generate_realistic_predictions

    params_key = (
        st.session_state.get("n_timepoints", 80),
        st.session_state.get("stimulus_type", "visual"),
        st.session_state.get("seed", 42),
    )

    # Check if we need to regenerate
    if (
        st.session_state.get("brain_predictions") is None
        or st.session_state.get("_data_params") != params_key
        or st.session_state.get("data_source") == "synthetic"
    ):
        if st.session_state.get("data_source") == "uploaded" and st.session_state.get("brain_predictions") is not None:
            return st.session_state["brain_predictions"]

        predictions = generate_realistic_predictions(
            n_timepoints=st.session_state["n_timepoints"],
            roi_indices=roi_indices,
            stimulus_type=st.session_state["stimulus_type"],
            tr_seconds=st.session_state["tr_seconds"],
            seed=st.session_state["seed"],
        )
        st.session_state["brain_predictions"] = predictions
        st.session_state["_data_params"] = params_key

    return st.session_state["brain_predictions"]


def upload_npy_widget(label, key):
    """File uploader for .npy arrays with validation."""
    uploaded = st.file_uploader(label, type=["npy"], key=key)
    if uploaded is not None:
        try:
            data = np.load(io.BytesIO(uploaded.read()))
            st.success(f"Loaded: shape {data.shape}, dtype {data.dtype}")
            return data
        except Exception as e:
            st.error(f"Failed to load file: {e}")
    return None


def download_csv_button(df, filename, label="Download CSV"):
    """Download button for a pandas DataFrame as CSV."""
    csv = df.to_csv(index=False)
    st.download_button(label, csv, filename, "text/csv")


def download_json_button(data, filename, label="Download JSON"):
    """Download button for a dict as JSON."""
    json_str = json.dumps(data, indent=2, default=str)
    st.download_button(label, json_str, filename, "application/json")


def show_analysis_log():
    """Display the analysis log in the sidebar."""
    log = st.session_state.get("analysis_log", [])
    if log:
        with st.sidebar:
            with st.expander("Analysis Log", expanded=False):
                for entry in reversed(log[-20:]):
                    st.caption(entry)


def data_summary_widget(predictions, roi_indices):
    """Show a summary of the current data."""
    if predictions is None:
        st.info("No data loaded. Generate synthetic data or upload your own.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Timepoints", predictions.shape[0])
    col2.metric("Vertices", predictions.shape[1])
    col3.metric("ROIs", len(roi_indices))
    col4.metric("Source", st.session_state.get("data_source", "synthetic").title())
