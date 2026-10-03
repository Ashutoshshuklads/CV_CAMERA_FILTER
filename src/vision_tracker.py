# D:\snapchat_filters\src\vision_tracker.py
import math
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.config import FACE_MODEL_PATH, HAND_MODEL_PATH, PINCH_THRESHOLD_NORM

@dataclass
class FaceData:
    detected: bool = False
    landmarks_px: Optional[np.ndarray] = None  # shape (478, 2) in pixels
    center: Tuple[int, int] = (0, 0)
    width: float = 0.0
    height: float = 0.0
    angle_deg: float = 0.0
    mouth_open_ratio: float = 0.0
    left_eye: Tuple[int, int] = (0, 0)
    right_eye: Tuple[int, int] = (0, 0)
    nose_tip: Tuple[int, int] = (0, 0)
    forehead: Tuple[int, int] = (0, 0)
    chin: Tuple[int, int] = (0, 0)

@dataclass
class HandData:
    wrist: Tuple[int, int] = (0, 0)
    palm_center: Tuple[int, int] = (0, 0)
    index_tip: Tuple[int, int] = (0, 0)
    thumb_tip: Tuple[int, int] = (0, 0)
    pinch_dist_norm: float = 1.0
    pinch_dist_px: float = 100.0
    is_pinching: bool = False
    landmarks_px: Optional[np.ndarray] = None  # shape (21, 2)

class VisionTracker:
    def __init__(self, face_model_path=FACE_MODEL_PATH, hand_model_path=HAND_MODEL_PATH):
        # Setup Face Landmarker
        base_face = python.BaseOptions(model_asset_path=face_model_path)
        options_face = vision.FaceLandmarkerOptions(
            base_options=base_face,
            output_face_blendshapes=True,
            num_faces=1
        )
        self.face_detector = vision.FaceLandmarker.create_from_options(options_face)

        # Setup Hand Detector (same strong settings as the virtual-mouse project)
        self._init_hand_detector(hand_model_path)

    # ------------------------------------------------------------------
    # Hand detection - ported from virtual_mouse.py
    #   * tracking mode (static_image_mode=False) instead of re-detecting
    #     the hand from scratch on every frame
    #   * full model (model_complexity=1) and 0.65 confidence thresholds
    #   * runs on the FULL-resolution frame (not the 0.5x downscaled one)
    #   * single hand -> no jumping between two hands
    # ------------------------------------------------------------------
    def _init_hand_detector(self, hand_model_path):
        self._last_ts_ms = 0
        try:
            # Legacy Solutions API (exactly what virtual_mouse.py uses)
            self._hands_legacy = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.65,
            )
            self._hand_backend = "legacy"
        except AttributeError:
            # Newer MediaPipe removed mp.solutions -> Tasks API in VIDEO
            # (tracking) mode with the very same thresholds.
            base_hand = python.BaseOptions(model_asset_path=hand_model_path)
            options_hand = vision.HandLandmarkerOptions(
                base_options=base_hand,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.65,
                min_hand_presence_confidence=0.65,
                min_tracking_confidence=0.65,
            )
            self.hand_detector = vision.HandLandmarker.create_from_options(options_hand)
            self._hand_backend = "tasks"
        print(f"[Hands] Backend: {self._hand_backend} (tracking mode, conf 0.65, full-res)")

    def _detect_hands(self, frame_bgr: np.ndarray):
        """Return a list of hands; each hand is a list of 21 landmarks (.x/.y normalized)."""
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if self._hand_backend == "legacy":
            rgb.flags.writeable = False
            res = self._hands_legacy.process(rgb)
            if res.multi_hand_landmarks:
                return [h.landmark for h in res.multi_hand_landmarks]
            return []
        ts_ms = int(time.monotonic() * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        mp_full = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.hand_detector.detect_for_video(mp_full, ts_ms)
        return list(res.hand_landmarks) if res.hand_landmarks else []

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[FaceData, List[HandData]]:
        h, w, _ = frame_bgr.shape

        # Downscale for ultra-fast landmark inference (guarantees 50+ FPS)
        scale = 0.5
        infer_w = int(w * scale)
        infer_h = int(h * scale)
        infer_bgr = cv2.resize(frame_bgr, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
        infer_rgb = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=infer_rgb)

        # 1. Face Inference
        face_res = self.face_detector.detect(mp_img)
        face_data = FaceData()

        if face_res.face_landmarks and len(face_res.face_landmarks) > 0:
            lms = face_res.face_landmarks[0]
            pts = np.zeros((len(lms), 2), dtype=np.int32)
            for i, lm in enumerate(lms):
                pts[i] = [int(lm.x * w), int(lm.y * h)]
            face_data.landmarks_px = pts
            face_data.detected = True

            # Key anchor landmarks
            # 1: Nose tip, 10: Forehead, 152: Chin, 33: Left eye outer, 263: Right eye outer
            # 133: Left eye inner, 362: Right eye inner
            face_data.nose_tip = tuple(pts[1])
            face_data.forehead = tuple(pts[10])
            face_data.chin = tuple(pts[152])

            # Eye centers
            l_eye_center = ((pts[33][0] + pts[133][0]) // 2, (pts[33][1] + pts[133][1]) // 2)
            r_eye_center = ((pts[263][0] + pts[362][0]) // 2, (pts[263][1] + pts[362][1]) // 2)
            face_data.left_eye = l_eye_center
            face_data.right_eye = r_eye_center

            # Head angle (tilt / roll) in degrees
            dx = r_eye_center[0] - l_eye_center[0]
            dy = r_eye_center[1] - l_eye_center[1]
            face_data.angle_deg = math.degrees(math.atan2(dy, dx))

            # Width and height
            face_data.width = float(np.linalg.norm(np.array(pts[454]) - np.array(pts[234])))
            face_data.height = float(np.linalg.norm(np.array(pts[152]) - np.array(pts[10])))
            face_data.center = (int((pts[234][0] + pts[454][0]) // 2), int((pts[10][1] + pts[152][1]) // 2))

            # Mouth Openness: Lip distance vs Mouth width
            mouth_w = np.linalg.norm(np.array(pts[291]) - np.array(pts[61])) + 1e-5
            lip_dist = np.linalg.norm(np.array(pts[14]) - np.array(pts[13]))
            face_data.mouth_open_ratio = float(lip_dist / mouth_w)

        # 2. Hand Inference (full-resolution frame, tracking mode)
        hand_list = self._detect_hands(frame_bgr)
        hands_data: List[HandData] = []

        if hand_list:
            for hand_lms in hand_list:
                hpts = np.zeros((len(hand_lms), 2), dtype=np.int32)
                for j, lm in enumerate(hand_lms):
                    hpts[j] = [int(lm.x * w), int(lm.y * h)]

                hd = HandData()
                hd.landmarks_px = hpts
                hd.wrist = tuple(hpts[0])
                hd.palm_center = tuple(hpts[9])
                hd.thumb_tip = tuple(hpts[4])
                hd.index_tip = tuple(hpts[8])

                # Normalized distance between thumb tip (4) and index tip (8)
                t_norm = np.array([hand_lms[4].x, hand_lms[4].y])
                i_norm = np.array([hand_lms[8].x, hand_lms[8].y])
                dist_norm = float(np.linalg.norm(t_norm - i_norm))
                dist_px = float(np.linalg.norm(np.array(hd.thumb_tip) - np.array(hd.index_tip)))

                hd.pinch_dist_norm = dist_norm
                hd.pinch_dist_px = dist_px
                hd.is_pinching = (dist_norm < PINCH_THRESHOLD_NORM)
                hands_data.append(hd)

        return face_data, hands_data
