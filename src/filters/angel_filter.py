# D:\snapchat_filters\src\filters\angel_filter.py
import math
import time
import cv2
import numpy as np
from src.filters.base_filter import BaseFilter
from src.vision_tracker import FaceData, HandData
from src.particle_system import GoldenSparkleSystem
from typing import List

class AngelFilter(BaseFilter):
    def __init__(self):
        super().__init__("Angel Halo")
        self.sparkle_system = GoldenSparkleSystem(max_sparkles=35)
        self.pulse_phase = 0.0

    def apply(self, frame: np.ndarray, face_data: FaceData, hands_data: List[HandData]) -> np.ndarray:
        h, w, _ = frame.shape

        # Golden Dreamy Warmth
        b, g, r = cv2.split(frame)
        r = cv2.add(r, 14)
        g = cv2.add(g, 10)
        frame = cv2.merge([b, g, r])

        anchor = face_data.forehead if face_data.detected else (w // 2, h // 3)
        self.sparkle_system.update_and_draw(frame, anchor, h, w)

        if not face_data.detected or face_data.landmarks_px is None:
            return frame

        pts = face_data.landmarks_px
        ang = face_data.angle_deg
        fw = face_data.width

        # 1. Floating Glowing Angel Halo
        forehead = pts[10]
        up_rad = math.radians(ang - 90)
        halo_dist = int(fw * 0.48)
        halo_center = (
            int(forehead[0] + math.cos(up_rad) * halo_dist),
            int(forehead[1] + math.sin(up_rad) * halo_dist)
        )

        halo_rx = int(fw * 0.42)
        halo_ry = int(fw * 0.12)

        self.pulse_phase = (self.pulse_phase + 0.08) % (math.pi * 2)
        pulse = math.sin(self.pulse_phase) * 3

        # Outer soft golden glow ring
        self.draw_rotated_ellipse(frame, halo_center, (int(halo_rx + pulse + 4), int(halo_ry + pulse + 3)), int(ang), (0, 160, 255), 6)
        # Bright inner gold core
        self.draw_rotated_ellipse(frame, halo_center, (int(halo_rx + pulse), int(halo_ry + pulse)), int(ang), (0, 230, 255), 3)
        # White hot highlight ring
        self.draw_rotated_ellipse(frame, halo_center, (int(halo_rx + pulse), int(halo_ry + pulse)), int(ang), (255, 255, 255), 1)

        # 2. Cheek Starbursts
        t = time.time()
        for is_right in [False, True]:
            cheek = pts[425] if is_right else pts[205]
            star_sz = int(max(4, fw * 0.04 + math.sin(t * 5 + (1 if is_right else 0)) * 2))
            cx, cy = int(cheek[0]), int(cheek[1])

            # 4-point golden star
            cv2.line(frame, (cx - star_sz, cy), (cx + star_sz, cy), (0, 240, 255), 2, lineType=cv2.LINE_AA)
            cv2.line(frame, (cx, cy - star_sz), (cx, cy + star_sz), (0, 240, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        return frame
