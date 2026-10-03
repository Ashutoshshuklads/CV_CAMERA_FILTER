# D:\snapchat_filters\src\filters\retro_filter.py
import math
import cv2
import numpy as np
from src.filters.base_filter import BaseFilter
from src.vision_tracker import FaceData, HandData
from typing import List

class RetroFilter(BaseFilter):
    def __init__(self):
        super().__init__("Retro Shades")

    def apply(self, frame: np.ndarray, face_data: FaceData, hands_data: List[HandData]) -> np.ndarray:
        h, w, _ = frame.shape

        # 1. Vintage 90s Color Grade (Warm nostalgic VHS vibe)
        # Boost red/yellow and slightly crush blacks
        b, g, r = cv2.split(frame)
        r = cv2.add(r, 18)
        g = cv2.add(g, 8)
        b = cv2.subtract(b, 10)
        frame = cv2.merge([b, g, r])

        # VHS REC timestamp overlay
        cv2.putText(frame, "REC [PLAY]  00:19:94  SP", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 240, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (20, 38), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        if not face_data.detected or face_data.landmarks_px is None:
            return frame

        pts = face_data.landmarks_px
        ang = face_data.angle_deg
        fw = face_data.width

        eye_center = (
            int((face_data.left_eye[0] + face_data.right_eye[0]) // 2),
            int((face_data.left_eye[1] + face_data.right_eye[1]) // 2)
        )

        rad = math.radians(ang)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        # 2. Classic Pixelated Black Sunglasses
        gw = int(fw * 0.95)
        gh = int(fw * 0.28)
        half_w = gw // 2
        half_h = gh // 2

        # Draw left and right dark lenses
        for is_right in [False, True]:
            # Lens center
            eye_pt = face_data.right_eye if is_right else face_data.left_eye
            lens_w = int(fw * 0.42)
            lens_h = int(fw * 0.24)
            self.draw_rotated_ellipse(frame, eye_pt, (lens_w // 2, lens_h // 2), int(ang), (15, 15, 15), -1)
            self.draw_rotated_ellipse(frame, eye_pt, (lens_w // 2, lens_h // 2), int(ang), (220, 220, 220), 2)

            # Iconic white glare reflection slash
            glare_start = (
                int(eye_pt[0] - lens_w * 0.25 * cos_a - lens_h * 0.3 * sin_a),
                int(eye_pt[1] - lens_w * 0.25 * sin_a + lens_h * 0.3 * cos_a)
            )
            glare_end = (
                int(eye_pt[0] + lens_w * 0.15 * cos_a + lens_h * 0.25 * sin_a),
                int(eye_pt[1] + lens_w * 0.15 * sin_a - lens_h * 0.25 * cos_a)
            )
            cv2.line(frame, glare_start, glare_end, (255, 255, 255), max(2, int(fw * 0.015)), lineType=cv2.LINE_AA)

        # Bridge connecting shades
        bridge_p1 = (
            int(face_data.left_eye[0] + (fw * 0.15) * cos_a),
            int(face_data.left_eye[1] + (fw * 0.15) * sin_a)
        )
        bridge_p2 = (
            int(face_data.right_eye[0] - (fw * 0.15) * cos_a),
            int(face_data.right_eye[1] - (fw * 0.15) * sin_a)
        )
        cv2.line(frame, bridge_p1, bridge_p2, (15, 15, 15), max(4, int(fw * 0.025)), lineType=cv2.LINE_AA)

        # 3. Handlebar Retro Mustache (above top lip landmark 164 / 13)
        stache_center = pts[164]
        mw = int(fw * 0.45)
        mh = int(fw * 0.12)

        # Draw left and right mustache wings
        l_wing = (
            int(stache_center[0] - (mw * 0.28) * cos_a),
            int(stache_center[1] - (mw * 0.28) * sin_a)
        )
        r_wing = (
            int(stache_center[0] + (mw * 0.28) * cos_a),
            int(stache_center[1] + (mw * 0.28) * sin_a)
        )
        self.draw_rotated_ellipse(frame, l_wing, (mw // 3, mh // 2), int(ang - 15), (20, 20, 20), -1)
        self.draw_rotated_ellipse(frame, r_wing, (mw // 3, mh // 2), int(ang + 15), (20, 20, 20), -1)

        return frame
