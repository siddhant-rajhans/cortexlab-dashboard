"""CortexLab Dashboard - Home Page."""

import streamlit as st

st.set_page_config(
    page_title="CortexLab Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("CortexLab Dashboard")
st.markdown("**Interactive analysis toolkit for multimodal fMRI brain encoding**")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Analysis Tools")
    st.page_link("pages/1_Brain_Alignment.py", label="Brain Alignment Benchmark", icon="🎯")
    st.markdown("Score how brain-like any AI model's representations are using RSA, CKA, or Procrustes")

    st.page_link("pages/2_Cognitive_Load.py", label="Cognitive Load Scorer", icon="📊")
    st.markdown("Predict cognitive demand across visual, auditory, language, and executive dimensions")

with col2:
    st.subheader("Advanced Analysis")
    st.page_link("pages/3_Temporal_Dynamics.py", label="Temporal Dynamics", icon="⏱️")
    st.markdown("Analyze peak response latency, lag correlations, and sustained vs transient components")

    st.page_link("pages/4_Connectivity.py", label="ROI Connectivity", icon="🔗")
    st.markdown("Compute functional connectivity matrices, cluster networks, and graph metrics")

st.divider()

st.subheader("About")
st.markdown(
    """
CortexLab is an enhanced toolkit built on [Meta's TRIBE v2](https://github.com/facebookresearch/tribev2)
for predicting how the human brain responds to video, audio, and text.

This dashboard runs on **synthetic data** by default - no GPU or real fMRI data required.
All analysis tools mirror the CortexLab Python API.

[GitHub](https://github.com/siddhant-rajhans/cortexlab)
&nbsp; | &nbsp;
[HuggingFace](https://huggingface.co/SID2000/cortexlab)
"""
)
