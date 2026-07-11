"""
drowsiness_mediapipe.py
=======================
Fast EAR / MAR landmark detection using MediaPipe Face Mesh.

Replaces the dlib shape-predictor pipeline:
  - No .dat model file required
  - Runs entirely on CPU (or GPU if available) via TFLite
  - Typically 3-5× faster than dlib on the same hardware
  - Returns the same data shapes that processor.py expects

Public API
----------
    detector = MediaPipeDrowsinessDetector()
    result   = detector.process_frame(bgr_frame)

    result keys:
        found          bool   – at least one face was detected
        left_eye       list[(x,y)] × 6   – pixel coords
        right_eye      list[(x,y)] × 6
        mouth          list[(x,y)] × 20
        EAR            float
        MAR            float
        bbox           (x, y, w, h) | None
"""

import cv2
import mediapipe as mp
import numpy as np

from utils import (
    MP_LEFT_EYE,
    MP_RIGHT_EYE,
    MP_MOUTH,
    compute_EAR,
    compute_MAR,
    extract_mp_eye_points,
    extract_mp_mouth_points,
)

# ---------------------------------------------------------------------------
# Module-level MediaPipe objects (constructed once per process)
# ---------------------------------------------------------------------------
_mp_face_mesh = mp.solutions.face_mesh


class MediaPipeDrowsinessDetector:
    """
    Wraps MediaPipe FaceMesh for per-frame EAR/MAR extraction.

    Parameters
    ----------
    max_num_faces : int
        Maximum faces to detect (1 is fastest for a driver-facing camera).
    refine_landmarks : bool
        Enable iris + attention-mesh refinements (slightly more accurate lips).
    min_detection_confidence : float
    min_tracking_confidence  : float
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._face_mesh = _mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_frame(self, bgr_frame: np.ndarray) -> dict:
        """
        Run face-mesh on one BGR frame.

        Returns a dict with keys described in the module docstring.
        All coordinates are in pixel space matching the input frame dimensions.
        """
        h, w = bgr_frame.shape[:2]

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Make the array read-only – avoids an unnecessary copy inside MP
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        rgb.flags.writeable = True

        empty = {
            "found":     False,
            "left_eye":  None,
            "right_eye": None,
            "mouth":     None,
            "EAR":       1.0,
            "MAR":       0.0,
            "bbox":      None,
        }

        if not results.multi_face_landmarks:
            return empty

        # Use the first (most prominent) face
        face_landmarks = results.multi_face_landmarks[0].landmark

        # --- Eye points (6 each) ---
        left_eye  = extract_mp_eye_points(face_landmarks, MP_LEFT_EYE,  w, h)
        right_eye = extract_mp_eye_points(face_landmarks, MP_RIGHT_EYE, w, h)

        # --- Mouth points (20) ---
        mouth = extract_mp_mouth_points(face_landmarks, MP_MOUTH, w, h)

        # --- EAR / MAR ---
        EAR_val = (compute_EAR(left_eye) + compute_EAR(right_eye)) / 2.0
        MAR_val = compute_MAR(mouth)

        # --- Bounding box from all 468 landmarks ---
        xs = [lm.x * w for lm in face_landmarks]
        ys = [lm.y * h for lm in face_landmarks]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        # Add a small margin
        pad_x = int((x_max - x_min) * 0.10)
        pad_y = int((y_max - y_min) * 0.10)
        bx = max(0, x_min - pad_x)
        by = max(0, y_min - pad_y)
        bw = min(w - bx, (x_max - x_min) + 2 * pad_x)
        bh = min(h - by, (y_max - y_min) + 2 * pad_y)

        return {
            "found":     True,
            "left_eye":  left_eye,
            "right_eye": right_eye,
            "mouth":     mouth,
            "EAR":       float(EAR_val),
            "MAR":       float(MAR_val),
            "bbox":      (bx, by, bw, bh),
        }

    def close(self):
        """Release MediaPipe resources."""
        self._face_mesh.close()

    # Support `with` statement
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
