# D:\snapchat_filters\src\filters\cyberpunk_filter.py
import math
import time
import cv2
import numpy as np
from src.filters.base_filter import BaseFilter
from src.vision_tracker import FaceData, HandData
from src.particle_system import CyberMatrixSystem
from typing import List

class CyberpunkFilter(BaseFilter):
    def __init__(self):
        super().__init__("Cyberpunk")
        self.matrix_system = CyberMatrixSystem(num_columns=28)
        self.scan_phase = 0.0

    def apply(self, frame: np.ndarray, face_data: FaceData, hands_data: List[HandData]) -> np.ndarray:
        h, w, _ = frame.shape

        # 1. Environment Effect: Falling Cyber Grid / Matrix Beams
        self.matrix_system.update_and_draw(frame, h, w)

        if not face_data.detected or face_data.landmarks_px is None:
            return frame

        pts = face_data.landmarks_px
        ang = face_data.angle_deg
        fw = face_data.width

        # 2. Neon Visor over eyes
        # Temples: 127 (left), 356 (right); Brow: 9 (forehead low); Nose bridge: 168
        eye_y = int((face_data.left_eye[1] + face_data.right_eye[1]) // 2)
        eye_x = int((face_data.left_eye[0] + face_data.right_eye[0]) // 2)

        v_w = int(fw * 1.05)
        v_h = int(fw * 0.32)

        # Create overlay for visor
        overlay = frame.copy()
        rad = math.radians(ang)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        # 4 corners of rotated visor box
        half_w = v_w // 2
        half_h = v_h // 2
        corners = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (int(half_w * 0.9), half_h),
            (int(-half_w * 0.9), half_h)
        ]
        poly_pts = []
        for cx, cy in corners:
            rx = int(eye_x + cx * cos_a - cy * sin_a)
            ry = int(eye_y + cx * sin_a + cy * cos_a)
            poly_pts.append([rx, ry])
        poly_arr = np.array([poly_pts], dtype=np.int32)

        # Dark tinted semi-transparent cyber glass
        cv2.fillPoly(overlay, poly_arr, (120, 40, 20)) # Deep cyber navy
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, dst=frame)

        # Glowing Neon Cyan Visor Border
        cv2.polylines(frame, poly_arr, isClosed=True, color=(255, 240, 0), thickness=3, lineType=cv2.LINE_AA)
        cv2.polylines(frame, poly_arr, isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        # Dynamic Scanline moving across visor
        self.scan_phase = (self.scan_phase + 0.08) % (math.pi * 2)
        scan_y_offset = math.sin(self.scan_phase) * (half_h * 0.7)
        scan_p1 = (
            int(eye_x - half_w * cos_a - scan_y_offset * sin_a),
            int(eye_y - half_w * sin_a + scan_y_offset * cos_a)
        )
        scan_p2 = (
            int(eye_x + half_w * cos_a - scan_y_offset * sin_a),
            int(eye_y + half_w * sin_a + scan_y_offset * cos_a)
        )
        cv2.line(frame, scan_p1, scan_p2, (0, 255, 255), 2, lineType=cv2.LINE_AA)

        # 3. Target Reticles & HUD Data over eyes
        for eye_pt, label in [(face_data.left_eye, "L_OPT"), (face_data.right_eye, "R_OPT")]:
            cv2.circle(frame, eye_pt, int(fw * 0.07), (0, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.circle(frame, eye_pt, int(fw * 0.02), (0, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.line(frame, (eye_pt[0] - 15, eye_pt[1]), (eye_pt[0] + 15, eye_pt[1]), (255, 255, 255), 1)
            cv2.line(frame, (eye_pt[0], eye_pt[1] - 15), (eye_pt[0], eye_pt[1] + 15), (255, 255, 255), 1)

        # HUD Digital readout
        text_pt = (int(poly_pts[0][0] + 10), int(poly_pts[0][1] - 8))
        cv2.putText(frame, "CYBER-HUD v2.4 // SYNC 98%", text_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # 4. Facial Circuit Lines on cheeks (landmarks 234 -> 132 -> 58, and 454 -> 361 -> 288)
        left_circuit = [pts[234], pts[132], pts[58], pts[172]]
        right_circuit = [pts[454], pts[361], pts[288], pts[397]]

        for circuit in [left_circuit, right_circuit]:
            for i in range(len(circuit) - 1):
                p1 = tuple(circuit[i])
                p2 = tuple(circuit[i+1])
                cv2.line(frame, p1, p2, (255, 240, 0), 2, lineType=cv2.LINE_AA)
                cv2.circle(frame, p1, 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, tuple(circuit[-1]), 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        return frame
