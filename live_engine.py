"""Real-time brain prediction engine.

Runs in a background thread, consuming frames from a capture source,
extracting features, and producing brain predictions via TRIBE v2.

When CortexLab is not installed, falls back to a simulation mode that
generates synthetic predictions from frame statistics.
"""

from __future__ import annotations

import time
import threading
import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from live_capture import BaseCapture, MediaFrame

logger = logging.getLogger(__name__)

# Check if CortexLab is available
try:
    from cortexlab.inference.predictor import TribeModel
    CORTEXLAB_AVAILABLE = True
except ImportError:
    CORTEXLAB_AVAILABLE = False


@dataclass
class LivePrediction:
    """A single prediction with metadata."""
    vertex_data: np.ndarray  # (n_vertices,)
    timestamp: float
    cognitive_load: dict[str, float] = field(default_factory=dict)
    processing_time_ms: float = 0.0


@dataclass
class LiveMetrics:
    """Aggregated metrics from the live engine."""
    fps: float = 0.0
    total_frames: int = 0
    total_predictions: int = 0
    avg_latency_ms: float = 0.0
    is_running: bool = False
    mode: str = "simulation"  # "simulation" or "cortexlab"


class LiveInferenceEngine:
    """Background engine for real-time brain prediction.

    Consumes frames from a capture source and produces brain predictions.
    If CortexLab is installed and a GPU is available, uses the real TRIBE v2
    model. Otherwise, falls back to simulation mode that generates plausible
    predictions from frame statistics.
    """

    def __init__(
        self,
        n_vertices: int = 580,
        roi_indices: dict | None = None,
        buffer_size: int = 120,
        checkpoint: str = "facebook/tribev2",
        device: str = "auto",
        cache_folder: str = "./cache",
    ):
        self.n_vertices = n_vertices
        self.roi_indices = roi_indices or {}
        self.buffer_size = buffer_size
        self.checkpoint = checkpoint
        self.device = device
        self.cache_folder = cache_folder

        self._predictions: deque[LivePrediction] = deque(maxlen=buffer_size)
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._model = None
        self._metrics = LiveMetrics()
        self._capture: BaseCapture | None = None

    def start(self, capture: BaseCapture):
        """Start the inference engine with a media capture source."""
        if self._running:
            return

        self._capture = capture
        self._running = True
        self._metrics = LiveMetrics(is_running=True)

        # Try to load CortexLab model
        if CORTEXLAB_AVAILABLE:
            try:
                logger.info("Loading TRIBE v2 model...")
                self._model = TribeModel.from_pretrained(
                    self.checkpoint, device=self.device, cache_folder=self.cache_folder
                )
                self._metrics.mode = "cortexlab"
                logger.info("Model loaded. Using real inference.")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Using simulation mode.")
                self._model = None
                self._metrics.mode = "simulation"
        else:
            self._metrics.mode = "simulation"

        capture.start()
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the engine and capture source."""
        self._running = False
        if self._capture:
            self._capture.stop()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._metrics.is_running = False

    def get_latest_prediction(self) -> LivePrediction | None:
        with self._lock:
            return self._predictions[-1] if self._predictions else None

    def get_predictions(self, n: int = 60) -> list[LivePrediction]:
        with self._lock:
            return list(self._predictions)[-n:]

    def get_metrics(self) -> LiveMetrics:
        return self._metrics

    def _inference_loop(self):
        """Main loop: consume frames, produce predictions."""
        frame_times = deque(maxlen=30)
        last_frame_count = 0

        while self._running:
            frame = self._capture.get_latest_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            # Skip if we already processed this frame
            current_count = self._capture.frame_count
            if current_count == last_frame_count:
                time.sleep(0.05)
                continue
            last_frame_count = current_count

            start = time.time()

            if self._model is not None and self._metrics.mode == "cortexlab":
                prediction = self._run_real_inference(frame)
            else:
                prediction = self._run_simulation(frame)

            elapsed_ms = (time.time() - start) * 1000
            prediction.processing_time_ms = elapsed_ms

            with self._lock:
                self._predictions.append(prediction)

            # Update metrics
            frame_times.append(time.time())
            self._metrics.total_predictions += 1
            self._metrics.total_frames = current_count
            self._metrics.avg_latency_ms = elapsed_ms
            if len(frame_times) >= 2:
                self._metrics.fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])

        # Check if capture stopped (file ended)
        if not self._capture.is_running:
            self._running = False
            self._metrics.is_running = False

    def _run_real_inference(self, frame: MediaFrame) -> LivePrediction:
        """Run actual TRIBE v2 inference on a frame.

        For real-time, we skip the full pipeline (get_events_dataframe)
        and use a simplified feature extraction path.
        """
        import tempfile
        import os

        try:
            # Save frame as temporary video (1 frame)
            import cv2
            tmp_path = os.path.join(tempfile.gettempdir(), "cortexlab_live_frame.mp4")
            h, w = frame.video_frame.shape[:2]
            out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (w, h))
            out.write(cv2.cvtColor(frame.video_frame, cv2.COLOR_RGB2BGR))
            out.release()

            events = self._model.get_events_dataframe(video_path=tmp_path)
            preds, _ = self._model.predict(events, verbose=False)
            vertex_data = preds.mean(axis=0) if preds.ndim == 2 else preds

            # Normalize to [0, 1]
            vmin, vmax = vertex_data.min(), vertex_data.max()
            if vmax > vmin:
                vertex_data = (vertex_data - vmin) / (vmax - vmin)

            os.unlink(tmp_path)
        except Exception as e:
            logger.warning(f"Inference failed: {e}. Falling back to simulation.")
            return self._run_simulation(frame)

        cog_load = self._compute_cognitive_load(vertex_data)
        return LivePrediction(
            vertex_data=vertex_data,
            timestamp=frame.timestamp,
            cognitive_load=cog_load,
        )

    def _run_simulation(self, frame: MediaFrame) -> LivePrediction:
        """Generate plausible predictions from frame statistics.

        Uses frame brightness/color as proxy for visual complexity,
        creating biologically-inspired activation patterns.
        """
        rng = np.random.default_rng(int(frame.timestamp * 1000) % (2**31))

        # Base noise
        vertex_data = rng.standard_normal(self.n_vertices) * 0.03

        if frame.video_frame is not None:
            img = frame.video_frame.astype(np.float32) / 255.0

            # Visual complexity from image statistics
            brightness = img.mean()
            contrast = img.std()
            color_variance = img.var(axis=(0, 1)).mean()

            # Map to ROI activations
            for roi_name, vertices in self.roi_indices.items():
                valid = vertices[vertices < self.n_vertices]
                if len(valid) == 0:
                    continue

                # Visual ROIs respond to brightness/contrast
                if roi_name in ["V1", "V2", "V3", "V4", "MT", "MST", "FFC", "VVC"]:
                    activation = contrast * 0.8 + color_variance * 0.5
                # Auditory ROIs get low baseline
                elif roi_name in ["A1", "LBelt", "MBelt", "PBelt", "A4", "A5"]:
                    activation = 0.05 + rng.random() * 0.1
                # Language ROIs moderate
                elif roi_name in ["44", "45", "IFJa", "IFJp", "TPOJ1", "TPOJ2"]:
                    activation = brightness * 0.3
                # Executive ROIs track change
                elif roi_name in ["46", "9-46d", "8Av", "8Ad", "FEF"]:
                    activation = contrast * 0.5
                else:
                    activation = 0.1

                vertex_data[valid] = activation + rng.standard_normal(len(valid)) * 0.05

        vertex_data = np.clip(vertex_data, 0, 1)
        cog_load = self._compute_cognitive_load(vertex_data)

        return LivePrediction(
            vertex_data=vertex_data,
            timestamp=frame.timestamp,
            cognitive_load=cog_load,
        )

    def _compute_cognitive_load(self, vertex_data: np.ndarray) -> dict[str, float]:
        """Compute cognitive load dimensions from vertex data."""
        from utils import COGNITIVE_DIMENSIONS

        baseline = max(float(np.median(np.abs(vertex_data))), 1e-8)
        scores = {}
        for dim, rois in COGNITIVE_DIMENSIONS.items():
            vals = []
            for roi in rois:
                if roi in self.roi_indices:
                    verts = self.roi_indices[roi]
                    valid = verts[verts < len(vertex_data)]
                    if len(valid) > 0:
                        vals.append(np.abs(vertex_data[valid]).mean())
            scores[dim] = min(float(np.mean(vals)) / baseline, 1.0) if vals else 0.0
        scores["Overall"] = float(np.mean(list(scores.values()))) if scores else 0.0
        return scores
