# D:\snapchat_filters\src\filters\blizzard_filter.py
import math
import cv2
import numpy as np
from src.filters.base_filter import BaseFilter
from src.vision_tracker import FaceData, HandData
from src.particle_system import SnowSystem
from typing import List

class BlizzardFilter(BaseFilter):
    def __init__(self):
        super().__init__("Blizzard")
        self.snow_system = SnowSystem(max_particles=90)

    def apply(self, frame: np.ndarray, face_data: FaceData, hands_data: List[HandData]) -> np.ndarray:
        h, w, _ = frame.shape

        # 1. Falling dynamic snowflakes
        self.snow_system.update_and_draw(frame, h, w)

        # 2. Frosty Vignette around border
        # Cool ice-blue tone shift
        b, g, r = cv2.split(frame)
        b = cv2.add(b, 16)
        r = cv2.subtract(r, 6)
        frame = cv2.merge([b, g, r])

        # Border frost vignette
        cv2.rectangle(frame, (0, 0), (w, h), (255, 230, 200), 4)

        if not face_data.detected or face_data.landmarks_px is None:
            return frame

        pts = face_data.landmarks_px
        ang = face_data.angle_deg
        fw = face_data.width

        # 3. Soft Rosy Cold-Blush Cheeks (landmarks 205 and 425)
        blush_radius = int(fw * 0.12)
        blush_overlay = frame.copy()
        for is_right in [False, True]:
            cheek = pts[425] if is_right else pts[205]
            cv2.circle(blush_overlay, (int(cheek[0]), int(cheek[1])), blush_radius, (100, 100, 255), -1, lineType=cv2.LINE_AA)

        cv2.addWeighted(blush_overlay, 0.28, frame, 0.72, 0, dst=frame)

        # 4. Cute Frost Crown / Snowflake on Forehead (Landmark 10)
        forehead = pts[10]
        rad = math.radians(ang)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        crown_sz = int(fw * 0.08)
        cx, cy = int(forehead[0]), int(forehead[1] - fw * 0.1)

        # Draw 6-arm snowflake
        for i in range(6):
            arm_rad = math.radians(ang + i * 60)
            x2 = int(cx + math.cos(arm_rad) * crown_sz)
            y2 = int(cy + math.sin(arm_rad) * crown_sz)
            cv2.line(frame, (cx, cy), (x2, y2), (255, 255, 255), 2, lineType=cv2.LINE_AA)
            # Branch tips
            bx1 = int(x2 + math.cos(arm_rad + 0.5) * (crown_sz * 0.3))
            by1 = int(y2 + math.sin(arm_rad + 0.5) * (crown_sz * 0.3))
            bx2 = int(x2 + math.cos(arm_rad - 0.5) * (crown_sz * 0.3))
            by2 = int(y2 + math.sin(arm_rad - 0.5) * (crown_sz * 0.3))
            cv2.line(frame, (x2, y2), (bx1, by1), (255, 230, 200), 1, lineType=cv2.LINE_AA)
            cv2.line(frame, (x2, y2), (bx2, by2), (255, 230, 200), 1, lineType=cv2.LINE_AA)

        cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        return frame
