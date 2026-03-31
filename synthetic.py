"""Biologically realistic synthetic fMRI data generation.

Generates data with hemodynamic response convolution, modality-specific
activation patterns, spatial autocorrelation, temporal noise structure,
and scanner drift - mimicking real fMRI recordings.
"""

import numpy as np

# --- Hemodynamic Response Function ---

def generate_hrf(tr_seconds=1.0, duration=30.0):
    """Canonical double-gamma hemodynamic response function.

    Models the BOLD signal: a positive peak at ~5-6s followed by a
    smaller negative undershoot at ~15s.
    """
    t = np.arange(0, duration, tr_seconds)
    # Double gamma parameters (SPM canonical)
    a1, b1 = 6.0, 1.0   # positive peak
    a2, b2 = 16.0, 1.0   # undershoot
    c = 1.0 / 6.0        # undershoot ratio

    from scipy.stats import gamma as gamma_dist
    h = gamma_dist.pdf(t, a1, scale=b1) - c * gamma_dist.pdf(t, a2, scale=b2)
    h = h / np.max(np.abs(h))  # normalize to [-1, 1]
    return h


def generate_stimulus_events(n_timepoints, tr_seconds=1.0, n_events=5, seed=42):
    """Generate random stimulus onset times as a binary event train.

    Returns a (n_timepoints,) array with 1s at stimulus onsets.
    Events are spaced at least 8 seconds apart.
    """
    rng = np.random.default_rng(seed)
    total_seconds = n_timepoints * tr_seconds
    min_gap = 8.0  # minimum inter-stimulus interval

    events = np.zeros(n_timepoints)
    onsets = []
    attempts = 0
    while len(onsets) < n_events and attempts < 1000:
        t = rng.uniform(2.0, total_seconds - 10.0)
        if all(abs(t - o) > min_gap for o in onsets):
            onsets.append(t)
        attempts += 1

    for onset in onsets:
        idx = int(onset / tr_seconds)
        if 0 <= idx < n_timepoints:
            events[idx] = 1.0

    return events


# --- Modality-Specific Activation Weights ---

# Weight for each ROI given a stimulus modality (0 = no response, 1 = maximum)
MODALITY_WEIGHTS = {
    "visual": {
        # Strong visual cortex activation
        "V1": 1.0, "V2": 0.95, "V3": 0.85, "V4": 0.8,
        "MT": 0.75, "MST": 0.7, "FFC": 0.65, "VVC": 0.6,
        # Weak cross-modal
        "A1": 0.05, "LBelt": 0.04, "MBelt": 0.03, "PBelt": 0.03, "A4": 0.02, "A5": 0.02,
        # Minimal language
        "44": 0.08, "45": 0.07, "IFJa": 0.06, "IFJp": 0.05,
        "TPOJ1": 0.1, "TPOJ2": 0.08, "STV": 0.07, "PSL": 0.06,
        # Moderate executive (attention)
        "46": 0.3, "9-46d": 0.25, "8Av": 0.35, "8Ad": 0.3,
        "FEF": 0.4, "p32pr": 0.15, "a32pr": 0.12,
    },
    "auditory": {
        "V1": 0.03, "V2": 0.03, "V3": 0.02, "V4": 0.02,
        "MT": 0.02, "MST": 0.01, "FFC": 0.01, "VVC": 0.01,
        "A1": 1.0, "LBelt": 0.95, "MBelt": 0.9, "PBelt": 0.85, "A4": 0.75, "A5": 0.7,
        "44": 0.15, "45": 0.12, "IFJa": 0.1, "IFJp": 0.08,
        "TPOJ1": 0.25, "TPOJ2": 0.2, "STV": 0.3, "PSL": 0.2,
        "46": 0.2, "9-46d": 0.15, "8Av": 0.12, "8Ad": 0.1,
        "FEF": 0.08, "p32pr": 0.1, "a32pr": 0.08,
    },
    "language": {
        "V1": 0.05, "V2": 0.04, "V3": 0.03, "V4": 0.03,
        "MT": 0.02, "MST": 0.02, "FFC": 0.1, "VVC": 0.08,
        "A1": 0.3, "LBelt": 0.25, "MBelt": 0.2, "PBelt": 0.15, "A4": 0.2, "A5": 0.15,
        "44": 1.0, "45": 0.95, "IFJa": 0.85, "IFJp": 0.8,
        "TPOJ1": 0.9, "TPOJ2": 0.85, "STV": 0.75, "PSL": 0.7,
        "46": 0.5, "9-46d": 0.45, "8Av": 0.3, "8Ad": 0.25,
        "FEF": 0.15, "p32pr": 0.35, "a32pr": 0.3,
    },
    "multimodal": {
        "V1": 0.7, "V2": 0.65, "V3": 0.55, "V4": 0.5,
        "MT": 0.5, "MST": 0.45, "FFC": 0.4, "VVC": 0.35,
        "A1": 0.7, "LBelt": 0.65, "MBelt": 0.55, "PBelt": 0.5, "A4": 0.45, "A5": 0.4,
        "44": 0.65, "45": 0.6, "IFJa": 0.5, "IFJp": 0.45,
        "TPOJ1": 0.6, "TPOJ2": 0.55, "STV": 0.5, "PSL": 0.45,
        "46": 0.4, "9-46d": 0.35, "8Av": 0.3, "8Ad": 0.25,
        "FEF": 0.3, "p32pr": 0.25, "a32pr": 0.2,
    },
}


def generate_realistic_predictions(
    n_timepoints,
    roi_indices,
    stimulus_type="visual",
    tr_seconds=1.0,
    n_events=5,
    snr=2.0,
    seed=42,
):
    """Generate biologically realistic fMRI-like predictions.

    Parameters
    ----------
    n_timepoints : int
        Number of TRs.
    roi_indices : dict[str, np.ndarray]
        ROI name -> vertex indices mapping.
    stimulus_type : str
        One of "visual", "auditory", "language", "multimodal".
    tr_seconds : float
        Repetition time in seconds.
    n_events : int
        Number of stimulus events.
    snr : float
        Signal-to-noise ratio (higher = cleaner signal).
    seed : int
        Random seed.
    """
    rng = np.random.default_rng(seed)
    n_vertices = max(max(v) for v in roi_indices.values()) + 1
    predictions = np.zeros((n_timepoints, n_vertices))

    # 1. Generate stimulus-evoked signal
    events = generate_stimulus_events(n_timepoints, tr_seconds, n_events, seed)
    hrf = generate_hrf(tr_seconds)

    # Convolve events with HRF
    bold_signal = np.convolve(events, hrf)[:n_timepoints]

    # 2. Apply modality-specific weights per ROI
    weights = MODALITY_WEIGHTS.get(stimulus_type, MODALITY_WEIGHTS["multimodal"])
    for roi_name, vertices in roi_indices.items():
        w = weights.get(roi_name, 0.1)
        # Add per-ROI latency jitter (higher-order areas respond later)
        latency_shift = 0
        if roi_name in ["44", "45", "IFJa", "IFJp", "46", "9-46d"]:
            latency_shift = int(2.0 / tr_seconds)  # ~2s later for association cortex
        elif roi_name in ["TPOJ1", "TPOJ2", "STV", "PSL"]:
            latency_shift = int(1.5 / tr_seconds)

        shifted = np.roll(bold_signal, latency_shift) * w
        # Add per-vertex variation within ROI
        for v in vertices:
            if v < n_vertices:
                vertex_scale = 0.8 + 0.4 * rng.random()
                predictions[:, v] = shifted * vertex_scale

    # 3. Add temporal autocorrelation (AR(1) noise)
    ar_coeff = 0.5
    noise = rng.standard_normal(predictions.shape)
    for t in range(1, n_timepoints):
        noise[t] += ar_coeff * noise[t - 1]

    # 4. Add scanner drift (low-frequency sinusoidal)
    t_axis = np.arange(n_timepoints) * tr_seconds
    drift = 0.1 * np.sin(2 * np.pi * t_axis / (n_timepoints * tr_seconds * 0.8))
    drift = drift[:, np.newaxis]

    # 5. Combine signal + noise + drift
    signal_power = np.std(predictions[predictions != 0]) if np.any(predictions != 0) else 1.0
    noise_power = signal_power / max(snr, 0.1)
    predictions = predictions + noise * noise_power + drift

    # 6. Spatial smoothing (average with neighbors within same ROI)
    for roi_name, vertices in roi_indices.items():
        valid = vertices[vertices < n_vertices]
        if len(valid) > 1:
            roi_data = predictions[:, valid].copy()
            kernel = np.ones(min(3, len(valid))) / min(3, len(valid))
            for t in range(n_timepoints):
                predictions[t, valid] = np.convolve(roi_data[t], kernel, mode="same")

    return predictions


def generate_correlated_features(
    brain_predictions,
    alignment_strength=0.5,
    feature_dim=512,
    seed=42,
):
    """Generate model features with controllable correlation to brain data.

    Parameters
    ----------
    brain_predictions : np.ndarray
        Brain data of shape (n_stimuli, n_vertices).
    alignment_strength : float
        0.0 = random features, 1.0 = perfectly correlated with brain.
    feature_dim : int
        Output feature dimensionality.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Features of shape (n_stimuli, feature_dim).
    """
    rng = np.random.default_rng(seed)
    n_stimuli = brain_predictions.shape[0]

    # Project brain data to feature_dim via random projection
    n_vertices = brain_predictions.shape[1]
    projection = rng.standard_normal((n_vertices, feature_dim)) / np.sqrt(n_vertices)
    brain_projected = brain_predictions @ projection

    # Generate random features
    random_features = rng.standard_normal((n_stimuli, feature_dim))

    # Mix: strength controls brain-alignment vs randomness
    strength = np.clip(alignment_strength, 0.0, 1.0)
    features = strength * brain_projected + (1 - strength) * random_features

    # Standardize
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    return features
