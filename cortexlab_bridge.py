"""Optional bridge to the installed `cortexlab` library.

The dashboard works end-to-end on synthetic data, but when `cortexlab` is
installed and the user has a real lesion-pipeline result on disk, this
bridge wires up the real path. Public surface kept tiny on purpose so a
caller can just write:

    bridge = cortexlab_status()
    if bridge.available:
        # use bridge.load_lesion_npz(...) and bridge.bh_fdr(...)
    else:
        # fall back to synthetic generators

Designed so the dashboard never hard-imports cortexlab; we probe in a
try/except and store a single status object the rest of the code reads.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BridgeStatus:
    available: bool
    version: str | None
    reason: str | None


def cortexlab_status() -> BridgeStatus:
    """Probe for the cortexlab library without raising."""
    try:
        import cortexlab  # noqa: F401
    except ImportError as e:
        return BridgeStatus(False, None, f"not installed: {e}")
    version = getattr(__import__("cortexlab"), "__version__", "unknown")
    return BridgeStatus(True, version, None)


def load_lesion_npz(path: str) -> dict:
    """Read a `subject_XX_lesion.npz` produced by the v0.2 orchestrator.

    Returns the same dict shape as
    :func:`synthetic.generate_lesion_results` so downstream rendering
    code is identical for real and synthetic data.

    Modality names are recovered from any `delta_<m>` keys present in
    the npz. P-values and q-values are read when present; q-values are
    recomputed from p-values via cortexlab.analysis.stats.bh_fdr when
    only p is present (matches the postprocess_roi.py contract).
    """
    import numpy as np

    npz = np.load(path)
    keys = list(npz.files)
    modalities = sorted(
        k.replace("delta_", "") for k in keys if k.startswith("delta_")
    )
    if not modalities:
        raise ValueError(
            f"{path}: no `delta_<m>` arrays found; expected the v0.2 "
            "subject_XX_lesion.npz schema."
        )

    delta_r2 = {m: npz[f"delta_{m}"] for m in modalities}

    p_values: dict[str, np.ndarray] = {}
    for m in modalities:
        key = f"p_{m}"
        if key in keys:
            p_values[m] = npz[key]

    q_values: dict[str, np.ndarray] = {}
    if p_values:
        try:
            from cortexlab.analysis.stats import bh_fdr
        except ImportError:
            from synthetic import _bh_fdr as bh_fdr  # type: ignore[import-not-found,assignment]
        for m, p in p_values.items():
            q_values[m] = bh_fdr(p).astype(np.float32)

    return {
        "full_r2": npz["full_r2"],
        "delta_r2": delta_r2,
        "p_values": p_values,
        "q_values": q_values,
        "n_permutations": int(npz["n_permutations"]) if "n_permutations" in keys else 0,
        "modality_order": modalities,
    }
