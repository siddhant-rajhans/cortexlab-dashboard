# CortexLab Dashboard

Futuristic interactive analysis dashboard for [CortexLab](https://github.com/siddhant-rajhans/cortexlab) - multimodal fMRI brain encoding toolkit built on Meta's TRIBE v2.

Glassmorphism dark theme with 3D brain visualization, real-time inference, and research-grade analysis tools.

## Pages

| Page | Description |
|---|---|
| **Brain Alignment Benchmark** | Score AI models against brain responses with RSA, CKA, Procrustes + permutation tests, bootstrap CIs, FDR correction, noise ceiling, RDM visualization |
| **Cognitive Load Scorer** | Predict cognitive demand across 4 dimensions with confidence bands, comparison mode, per-ROI breakdown |
| **Temporal Dynamics** | Raw timecourses, peak latency hierarchy, lag correlation with null bands, cross-ROI lag matrix, sustained/transient decomposition |
| **ROI Connectivity** | Partial correlation, dendrogram, modularity, degree/betweenness centrality, edge weight distribution, network graph |
| **3D Brain Viewer** | Interactive rotatable fsaverage brain with activation overlays, publication-quality 4-panel views, ROI highlighting, sulcal depth blending |
| **Live Inference** | Real-time brain prediction from webcam, screen capture, or video file with live-updating 3D brain, cognitive load timeline, and metrics |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run Home.py
```

Runs on **biologically realistic synthetic data** by default (HRF convolution, modality-specific ROI activation, spatial smoothing). No GPU or real fMRI data required.

## Live Inference (Local Only)

For real-time brain prediction from webcam, screen, or video:

```bash
# Install optional capture dependencies
pip install opencv-python mss Pillow

# For real TRIBE v2 inference (needs GPU):
pip install -e ../cortexlab[analysis]

# Start dashboard
streamlit run Home.py
# Navigate to Live Inference page
```

Without CortexLab installed, live inference runs in **simulation mode** - predictions are generated from image statistics (brightness, contrast, color) mapped to brain ROIs.

## Features

- **Futuristic UI**: Glassmorphism dark theme, neon accents, gradient headings, glowing metric cards, animated borders
- **3D Brain Hero**: Rotatable fsaverage brain mesh on the home page
- **Biologically Realistic Data**: HRF-convolved synthetic data with modality-specific activation patterns
- **Statistical Rigor**: Permutation tests, bootstrap CIs, FDR correction, noise ceiling estimation
- **Cross-Page State**: ROI selections carry between pages, shared session predictions
- **File Upload**: Upload .npy predictions from real CortexLab runs
- **CSV/JSON Export**: Download results from every analysis page
- **Methodology Docs**: Every page has an expandable methodology section with references

## Deployment

### HuggingFace Spaces

Live at: [huggingface.co/spaces/SID2000/cortexlab-dashboard](https://huggingface.co/spaces/SID2000/cortexlab-dashboard)

Docker-based deployment. Live inference page shows simulation mode (no webcam/GPU access in Spaces).

### Local

```bash
git clone https://github.com/siddhant-rajhans/cortexlab-dashboard.git
cd cortexlab-dashboard
pip install -r requirements.txt
streamlit run Home.py
```

## Links

- [CortexLab Library](https://github.com/siddhant-rajhans/cortexlab)
- [CortexLab on HuggingFace](https://huggingface.co/SID2000/cortexlab)
- [Live Demo](https://huggingface.co/spaces/SID2000/cortexlab-dashboard)

## License

CC BY-NC 4.0
