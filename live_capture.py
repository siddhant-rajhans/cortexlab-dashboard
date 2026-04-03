"""Media capture sources for live brain prediction.

Provides webcam, screen capture, and file streaming sources that
yield frames at a controlled rate for real-time inference.
"""

from __future__ import annotations

import time
import threading
import logging
from pathlib import Path
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MediaFrame:
    """A single frame from any media source."""
    video_frame: np.ndarray | None = None  # (H, W, 3) RGB
    audio_chunk: np.ndarray | None = None  # (samples,) float32
    timestamp: float = 0.0


class BaseCapture:
    """Base class for media capture sources."""

    def __init__(self, fps: float = 1.0):
        self.fps = fps
        self._running = False
        self._buffer: deque[MediaFrame] = deque(maxlen=300)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def get_latest_frame(self) -> MediaFrame | None:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def get_all_frames(self) -> list[MediaFrame]:
        with self._lock:
            frames = list(self._buffer)
            return frames

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_count(self) -> int:
        return len(self._buffer)

    def _capture_loop(self):
        raise NotImplementedError


class WebcamCapture(BaseCapture):
    """Capture frames from webcam using OpenCV."""

    def __init__(self, camera_index: int = 0, fps: float = 1.0, resolution: tuple = (640, 480)):
        super().__init__(fps)
        self.camera_index = camera_index
        self.resolution = resolution

    def _capture_loop(self):
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV not installed. Run: pip install opencv-python")
            self._running = False
            return

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error(f"Cannot open camera {self.camera_index}")
            self._running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        start_time = time.time()
        interval = 1.0 / self.fps

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break
                # BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                media_frame = MediaFrame(
                    video_frame=frame_rgb,
                    timestamp=time.time() - start_time,
                )
                with self._lock:
                    self._buffer.append(media_frame)
                time.sleep(interval)
        finally:
            cap.release()


class ScreenCapture(BaseCapture):
    """Capture screen frames using mss."""

    def __init__(self, fps: float = 1.0, region: dict | None = None):
        super().__init__(fps)
        self.region = region  # {"left": 0, "top": 0, "width": 1920, "height": 1080}

    def _capture_loop(self):
        try:
            import mss
            from PIL import Image
        except ImportError:
            logger.error("mss/PIL not installed. Run: pip install mss Pillow")
            self._running = False
            return

        start_time = time.time()
        interval = 1.0 / self.fps

        with mss.mss() as sct:
            monitor = self.region or sct.monitors[1]  # Primary monitor
            while self._running:
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                frame = np.array(img)
                media_frame = MediaFrame(
                    video_frame=frame,
                    timestamp=time.time() - start_time,
                )
                with self._lock:
                    self._buffer.append(media_frame)
                time.sleep(interval)


class FileStreamer(BaseCapture):
    """Stream a video file frame-by-frame at real-time speed."""

    def __init__(self, file_path: str, fps: float = 1.0):
        super().__init__(fps)
        self.file_path = file_path

    def _capture_loop(self):
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV not installed. Run: pip install opencv-python")
            self._running = False
            return

        cap = cv2.VideoCapture(self.file_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {self.file_path}")
            self._running = False
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        # Skip frames to match our target FPS
        frame_skip = max(1, int(video_fps / self.fps))
        frame_idx = 0
        start_time = time.time()
        interval = 1.0 / self.fps

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    self._running = False
                    break
                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                media_frame = MediaFrame(
                    video_frame=frame_rgb,
                    timestamp=time.time() - start_time,
                )
                with self._lock:
                    self._buffer.append(media_frame)
                time.sleep(interval)
        finally:
            cap.release()


def get_capture_source(source_type: str, **kwargs) -> BaseCapture:
    """Factory function to create a capture source."""
    sources = {
        "webcam": WebcamCapture,
        "screen": ScreenCapture,
        "file": FileStreamer,
    }
    if source_type not in sources:
        raise ValueError(f"Unknown source: {source_type}. Choose from {list(sources)}")
    return sources[source_type](**kwargs)
