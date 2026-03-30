# CortexLab Dashboard

Interactive analysis dashboard for [CortexLab](https://github.com/siddhant-rajhans/cortexlab) - multimodal fMRI brain encoding toolkit.

## Pages

- **Brain Alignment Benchmark** - Compare AI model representations against brain responses (RSA, CKA, Procrustes)
- **Cognitive Load Scorer** - Visualize cognitive demand across visual, auditory, language, and executive dimensions
- **Temporal Dynamics** - Peak latency, lag correlations, sustained vs transient response decomposition
- **ROI Connectivity** - Correlation matrices, network clustering, degree centrality, graph visualization

## Quick Start

```bash
pip install -r requirements.txt
streamlit run Home.py
```

Runs on **synthetic data** by default - no GPU or real fMRI data required.

## Links

- [CortexLab Library](https://github.com/siddhant-rajhans/cortexlab)
- [CortexLab on HuggingFace](https://huggingface.co/SID2000/cortexlab)

## License

CC BY-NC 4.0
