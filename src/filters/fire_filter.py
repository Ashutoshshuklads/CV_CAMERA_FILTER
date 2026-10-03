# D:\snapchat_filters\src\filters\fire_filter.py
import math
import cv2
import numpy as np
from src.filters.base_filter import BaseFilter
from src.vision_tracker import FaceData, HandData
from src.particle_system import EmbersSystem
from typing import List

class FireFilter(BaseFilter):
    def __init__(self):
        super().__init__("Fire Demon")
        self.embers = EmbersSystem(max_particles=75)

    def apply(self, frame: np.ndarray, face_data: FaceData, hands_data: List[HandData]) -> np.ndarray:
        h, w, _ = frame.shape

        # 1. Rising Fiery Embers
        self.embers.update_and_draw(frame, h, w)

        if not face_data.detected or face_data.landmarks_px is None:
            return frame

        pts = face_data.landmarks_px
        ang = face_data.angle_deg
        fw = face_data.width

        rad = math.radians(ang)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        # 2. Glowing Fiery Eyes
        for eye_pt in [face_data.left_eye, face_data.right_eye]:
            # Outer flame glow
            cv2.circle(frame, eye_pt, int(fw * 0.08), (0, 80, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(frame, eye_pt, int(fw * 0.05), (0, 180, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, eye_pt, int(fw * 0.02), (255, 255, 255), -1, lineType=cv2.LINE_AA)

        # 3. Fiery Demon Horns on Forehead (Landmarks 103 and 332)
        horn_h = int(fw * 0.55)
        horn_w = int(fw * 0.16)

        up_rad = math.radians(ang - 90)

        for is_right in [False, True]:
            anchor = pts[332] if is_right else pts[103]
            curve_sign = 1 if is_right else -1

            # Base, mid, and tip points for curved horn
            horn_pts = []
            num_steps = 6
            for step in range(num_steps + 1):
                t = step / num_steps
                # Curve outward as it goes up
                offset_up = horn_h * t
                offset_out = curve_sign * (t ** 1.8) * (horn_w * 1.6)

                px = int(anchor[0] + offset_up * math.cos(up_rad) + offset_out * cos_a)
                py = int(anchor[1] + offset_up * math.sin(up_rad) + offset_out * sin_a)
                horn_pts.append((px, py))

            # Draw tapering horn segments with fire colors
            for step in range(num_steps):
                p1 = horn_pts[step]
                p2 = horn_pts[step + 1]
                t = step / num_steps
                thickness = int(max(2, horn_w * (1.0 - t * 0.85)))
                # Gradient: Yellow at base -> Orange -> Crimson at tip
                r = int(255 * (1.0 - t * 0.2))
                g = int(220 * (1.0 - t))
                b = 0
                cv2.line(frame, p1, p2, (b, g, r), thickness, lineType=cv2.LINE_AA)

            # Glowing tip
            tip = horn_pts[-1]
            cv2.circle(frame, tip, 4, (0, 240, 255), -1, lineType=cv2.LINE_AA)

        return frame
