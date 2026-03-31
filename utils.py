"""Shared utilities for the CortexLab dashboard.

Provides synthetic data generation and analysis functions that mirror
CortexLab's API without requiring the full library or GPU.
"""

import numpy as np
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster


# --- ROI Definitions ---

ROI_GROUPS = {
    "Executive": ["46", "9-46d", "8Av", "8Ad", "FEF", "p32pr", "a32pr"],
    "Visual": ["V1", "V2", "V3", "V4", "MT", "MST", "FFC", "VVC"],
    "Auditory": ["A1", "LBelt", "MBelt", "PBelt", "A4", "A5"],
    "Language": ["44", "45", "IFJa", "IFJp", "TPOJ1", "TPOJ2", "STV", "PSL"],
}

ALL_ROIS = [roi for group in ROI_GROUPS.values() for roi in group]


def make_roi_indices(n_vertices_per_roi=20):
    """Create ROI -> vertex index mapping."""
    indices = {}
    offset = 0
    for roi in ALL_ROIS:
        indices[roi] = np.arange(offset, offset + n_vertices_per_roi)
        offset += n_vertices_per_roi
    return indices, offset


# --- Brain Alignment ---

def compute_rdm(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normalised = features / norms
    return 1.0 - normalised @ normalised.T


def rsa_score(model_features, brain_features):
    rdm_m = compute_rdm(model_features)
    rdm_b = compute_rdm(brain_features)
    idx = np.triu_indices(rdm_m.shape[0], k=1)
    corr, _ = spearmanr(rdm_m[idx], rdm_b[idx])
    return float(corr) if not np.isnan(corr) else 0.0


def cka_score(X, Y):
    n = X.shape[0]
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    XX = X @ X.T
    YY = Y @ Y.T
    hsic_xy = np.trace(XX @ YY) / (n - 1) ** 2
    hsic_xx = np.trace(XX @ XX) / (n - 1) ** 2
    hsic_yy = np.trace(YY @ YY) / (n - 1) ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 1e-12 else 0.0


def procrustes_score(X, Y):
    min_dim = min(X.shape[1], Y.shape[1])
    X, Y = X[:, :min_dim], Y[:, :min_dim]
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    nx, ny = np.linalg.norm(X), np.linalg.norm(Y)
    if nx < 1e-12 or ny < 1e-12:
        return 0.0
    X, Y = X / nx, Y / ny
    U, _, Vt = np.linalg.svd(Y.T @ X, full_matrices=False)
    rotated = Y @ (U @ Vt)
    return float(max(0.0, 1.0 - np.linalg.norm(X - rotated)))


ALIGNMENT_METHODS = {"RSA": rsa_score, "CKA": cka_score, "Procrustes": procrustes_score}


def permutation_test(model_feat, brain_pred, method_fn, n_perm=500, seed=42):
    """Returns (observed_score, p_value, null_distribution)."""
    rng = np.random.default_rng(seed)
    observed = method_fn(model_feat, brain_pred)
    null_dist = []
    for _ in range(n_perm):
        perm_score = method_fn(model_feat[rng.permutation(len(model_feat))], brain_pred)
        null_dist.append(perm_score)
    null_dist = np.array(null_dist)
    count = np.sum(null_dist >= observed)
    p_value = (count + 1) / (n_perm + 1)
    return observed, p_value, null_dist


def bootstrap_ci(model_feat, brain_pred, method_fn, n_boot=500, confidence=0.95, seed=42):
    """Returns (point_estimate, ci_lower, ci_upper)."""
    rng = np.random.default_rng(seed)
    n = model_feat.shape[0]
    point = method_fn(model_feat, brain_pred)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        scores.append(method_fn(model_feat[idx], brain_pred[idx]))
    scores = np.array(scores)
    alpha = 1 - confidence
    return point, float(np.percentile(scores, 100 * alpha / 2)), float(np.percentile(scores, 100 * (1 - alpha / 2)))


def fdr_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns corrected p-values and significance mask."""
    p = np.array(p_values)
    n = len(p)
    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]
    corrected = np.empty(n)
    corrected[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        corrected[sorted_idx[i]] = min(corrected[sorted_idx[i + 1]], sorted_p[i] * n / (i + 1))
    return corrected, corrected < alpha


def noise_ceiling(brain_pred, method_fn, n_splits=20, seed=42):
    """Estimate noise ceiling via split-half reliability."""
    rng = np.random.default_rng(seed)
    n = brain_pred.shape[0]
    scores = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        half = n // 2
        s = method_fn(brain_pred[idx[:half]], brain_pred[idx[half:half * 2]])
        scores.append(s)
    return float(np.mean(scores)), float(np.std(scores))


def partial_correlation(predictions, roi_indices):
    """Compute partial correlation matrix (correlation controlling for mean signal)."""
    names = list(roi_indices.keys())
    n = len(names)
    T = predictions.shape[0]
    timecourses = np.zeros((n, T))
    for i, name in enumerate(names):
        verts = roi_indices[name]
        valid = verts[verts < predictions.shape[1]]
        if len(valid) > 0:
            timecourses[i] = predictions[:, valid].mean(axis=1)

    # Partial correlation via precision matrix
    cov = np.cov(timecourses)
    try:
        prec = np.linalg.inv(cov + 1e-6 * np.eye(n))
        d = np.sqrt(np.diag(prec))
        d[d == 0] = 1
        partial = -prec / np.outer(d, d)
        np.fill_diagonal(partial, 1.0)
    except np.linalg.LinAlgError:
        partial = np.eye(n)
    return np.nan_to_num(partial, nan=0.0), names


def betweenness_centrality(corr_matrix, roi_names, threshold=0.3):
    """Compute betweenness centrality from thresholded connectivity."""
    import networkx as nx
    n = corr_matrix.shape[0]
    G = nx.Graph()
    for i, name in enumerate(roi_names):
        G.add_node(name)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix[i, j]) > threshold:
                G.add_edge(roi_names[i], roi_names[j], weight=abs(corr_matrix[i, j]))
    bc = nx.betweenness_centrality(G)
    return {name: bc.get(name, 0.0) for name in roi_names}


def modularity_score(corr_matrix, labels):
    """Compute Newman's modularity Q for a given partition."""
    n = corr_matrix.shape[0]
    adj = np.abs(corr_matrix).copy()
    np.fill_diagonal(adj, 0)
    m = adj.sum() / 2
    if m == 0:
        return 0.0
    Q = 0.0
    k = adj.sum(axis=1)
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                Q += adj[i, j] - k[i] * k[j] / (2 * m)
    return float(Q / (2 * m))


# --- Cognitive Load ---

COGNITIVE_DIMENSIONS = {
    "Executive Load": ["46", "9-46d", "8Av", "8Ad", "FEF", "p32pr", "a32pr"],
    "Visual Complexity": ["V1", "V2", "V3", "V4", "MT", "MST", "FFC", "VVC"],
    "Auditory Demand": ["A1", "LBelt", "MBelt", "PBelt", "A4", "A5"],
    "Language Processing": ["44", "45", "IFJa", "IFJp", "TPOJ1", "TPOJ2", "STV", "PSL"],
}


def score_cognitive_load(predictions, roi_indices, tr_seconds=1.0):
    baseline = max(float(np.median(np.abs(predictions))), 1e-8)
    timeline = []
    dim_scores = {d: [] for d in COGNITIVE_DIMENSIONS}

    for t in range(predictions.shape[0]):
        row = {}
        for dim, rois in COGNITIVE_DIMENSIONS.items():
            vals = []
            for roi in rois:
                if roi in roi_indices:
                    verts = roi_indices[roi]
                    valid = verts[verts < predictions.shape[1]]
                    if len(valid) > 0:
                        vals.append(np.abs(predictions[t, valid]).mean())
            score = min(float(np.mean(vals)) / baseline, 1.0) if vals else 0.0
            dim_scores[dim].append(score)
            row[dim] = score
        row["time"] = t * tr_seconds
        timeline.append(row)

    averages = {d: float(np.mean(v)) for d, v in dim_scores.items()}
    averages["Overall"] = float(np.mean(list(averages.values())))
    return averages, timeline


# --- Temporal Dynamics ---

def peak_latency(predictions, roi_indices, roi_name, tr_seconds=1.0):
    verts = roi_indices.get(roi_name, np.array([]))
    valid = verts[verts < predictions.shape[1]]
    if len(valid) == 0:
        return 0.0
    tc = np.abs(predictions[:, valid]).mean(axis=1)
    return float(np.argmax(tc) * tr_seconds)


def temporal_correlation(predictions, features, roi_indices, roi_name, max_lag=10):
    verts = roi_indices.get(roi_name, np.array([]))
    valid = verts[verts < predictions.shape[1]]
    if len(valid) == 0:
        return np.zeros(2 * max_lag + 1)
    brain_tc = np.abs(predictions[:, valid]).mean(axis=1)
    model_tc = features.mean(axis=1) if features.ndim > 1 else features
    n = min(len(brain_tc), len(model_tc))
    brain_tc, model_tc = brain_tc[:n], model_tc[:n]

    corrs = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            b, m = brain_tc[lag:], model_tc[:n - lag]
        else:
            b, m = brain_tc[:n + lag], model_tc[-lag:]
        if len(b) < 2:
            corrs.append(0.0)
            continue
        bz, mz = b - b.mean(), m - m.mean()
        denom = np.sqrt((bz ** 2).sum() * (mz ** 2).sum())
        corrs.append(float((bz * mz).sum() / denom) if denom > 1e-12 else 0.0)
    return np.array(corrs)


def decompose_response(predictions, roi_indices, roi_name, cutoff_seconds=4.0, tr_seconds=1.0):
    verts = roi_indices.get(roi_name, np.array([]))
    valid = verts[verts < predictions.shape[1]]
    if len(valid) == 0:
        return np.zeros(predictions.shape[0]), np.zeros(predictions.shape[0])
    tc = np.abs(predictions[:, valid]).mean(axis=1)
    window = max(1, int(cutoff_seconds / tr_seconds))
    sustained = np.convolve(tc, np.ones(window) / window, mode="same")
    return sustained, tc - sustained


# --- Connectivity ---

def compute_connectivity(predictions, roi_indices):
    names = list(roi_indices.keys())
    n = len(names)
    T = predictions.shape[0]
    timecourses = np.zeros((n, T))
    for i, name in enumerate(names):
        verts = roi_indices[name]
        valid = verts[verts < predictions.shape[1]]
        if len(valid) > 0:
            timecourses[i] = predictions[:, valid].mean(axis=1)
    corr = np.corrcoef(timecourses) if T >= 2 else np.eye(n)
    return np.nan_to_num(corr, nan=0.0), names


def cluster_rois(corr_matrix, roi_names, n_clusters=4):
    n = corr_matrix.shape[0]
    n_clusters = min(n_clusters, n)
    dist = 1.0 - np.abs(corr_matrix)
    np.fill_diagonal(dist, 0.0)
    condensed = [dist[i, j] for i in range(n) for j in range(i + 1, n)]
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    clusters = {}
    for name, cid in zip(roi_names, labels):
        clusters.setdefault(int(cid), []).append(name)
    return clusters, labels


def graph_metrics(corr_matrix, roi_names, threshold=0.3):
    n = corr_matrix.shape[0]
    adj = (np.abs(corr_matrix) > threshold).astype(float)
    np.fill_diagonal(adj, 0.0)
    degree = adj.sum(axis=1)
    max_d = max(n - 1, 1)
    return {name: float(degree[i] / max_d) for i, name in enumerate(roi_names)}


# --- Synthetic Data Generators ---

def generate_brain_predictions(n_timepoints=60, n_vertices=580, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_timepoints, n_vertices))


def generate_model_features(n_stimuli=60, feature_dim=512, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_stimuli, feature_dim))
