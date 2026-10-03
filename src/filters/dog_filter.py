# D:\snapchat_filters\src\filters\dog_filter.py
import math
import cv2
import numpy as np
from src.filters.base_filter import BaseFilter
from src.vision_tracker import FaceData, HandData
from typing import List
 
class DogFilter(BaseFilter):
    def __init__(self):
        super().__init__("Puppy Dog")
 
    def apply(self, frame: np.ndarray, face_data: FaceData, hands_data: List[HandData]) -> np.ndarray:
        if not face_data.detected or face_data.landmarks_px is None:
            return frame
 
        h, w, _ = frame.shape
        pts = face_data.landmarks_px
        ang = face_data.angle_deg
        fw = face_data.width
 
        ear_w = int(fw * 0.28)
        ear_h = int(fw * 0.55)
 
        # 1. Puppy Ears
        # Anchor from the forehead/hairline point (landmark 10 = top of forehead,
        # the highest reliable point on the face mesh) instead of the temple
        # landmarks (21 / 251), which sit too low near eye level and made the
        # ears appear next to the eyes instead of on top of the head.
        forehead = pts[10]
 
        # Direction vectors based on head tilt:
        # - "right" vector points along the face's horizontal axis
        # - "up" vector points perpendicular to it, i.e. straight up from the head
        rad = math.radians(ang)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
 
        up_rad = math.radians(ang - 90)
        cos_up = math.cos(up_rad)
        sin_up = math.sin(up_rad)
 
        # How far apart the ears sit (sideways) and how far above the
        # forehead point they float (upwards) - both scaled to face width
        # so the alignment holds steady across distances/face sizes.
        side_dist = fw * 0.40
        up_dist = fw * 0.62
 
        l_ear_center = (
            int(forehead[0] - side_dist * cos_a + up_dist * cos_up),
            int(forehead[1] - side_dist * sin_a + up_dist * sin_up)
        )
        r_ear_center = (
            int(forehead[0] + side_dist * cos_a + up_dist * cos_up),
            int(forehead[1] + side_dist * sin_a + up_dist * sin_up)
        )
 
        # Draw outer brown ear
        outer_color = (42, 92, 168)  # Warm Dog Brown (BGR)
        inner_color = (180, 160, 255) # Soft Pink Inner Ear (BGR)
 
        # Left Ear (floppy tilted outwards)
        self.draw_rotated_ellipse(frame, l_ear_center, (ear_w, ear_h), int(ang - 25), outer_color, -1)
        self.draw_rotated_ellipse(frame, l_ear_center, (int(ear_w * 0.6), int(ear_h * 0.7)), int(ang - 25), inner_color, -1)
 
        # Right Ear
        self.draw_rotated_ellipse(frame, r_ear_center, (ear_w, ear_h), int(ang + 25), outer_color, -1)
        self.draw_rotated_ellipse(frame, r_ear_center, (int(ear_w * 0.6), int(ear_h * 0.7)), int(ang + 25), inner_color, -1)
 
        # 2. Puppy Snout & Nose (Landmark 1 is nose tip)
        nose_pt = pts[1]
        snout_r = int(fw * 0.12)
        # Cute black button nose
        self.draw_rotated_ellipse(frame, (int(nose_pt[0]), int(nose_pt[1] - snout_r * 0.2)), (int(snout_r * 1.2), int(snout_r * 0.85)), int(ang), (20, 20, 20), -1)
        # Cute white shine highlight
        shine_pt = (int(nose_pt[0] - snout_r * 0.35), int(nose_pt[1] - snout_r * 0.45))
        cv2.circle(frame, shine_pt, max(2, int(snout_r * 0.22)), (255, 255, 255), -1, lineType=cv2.LINE_AA)
 
        # Whisker dots on cheeks (landmarks 205, 425)
        for offset_x in [-1, 1]:
            cheek_pt = pts[205] if offset_x == -1 else pts[425]
            for dx, dy in [(-10, -5), (-5, 6), (-12, 10)]:
                dot_x = int(cheek_pt[0] + dx * offset_x)
                dot_y = int(cheek_pt[1] + dy)
                cv2.circle(frame, (dot_x, dot_y), max(2, int(fw * 0.012)), (40, 40, 40), -1, lineType=cv2.LINE_AA)
 
        # 3. Dynamic Puppy Tongue (when mouth opens)
        if face_data.mouth_open_ratio > 0.18:
            lower_lip = pts[14]
            chin = pts[152]
            # Tongue length scales with mouth openness
            t_len = min(int(fw * 0.65), int(fw * 0.25 + (face_data.mouth_open_ratio - 0.18) * fw * 1.5))
            t_w = int(fw * 0.16)
 
            # Downward direction
            down_rad = math.radians(ang + 90)
            t_center = (
                int(lower_lip[0] + math.cos(down_rad) * (t_len * 0.5)),
                int(lower_lip[1] + math.sin(down_rad) * (t_len * 0.5))
            )
 
            # Pink tongue base
            tongue_color = (130, 110, 255) # Bright Pink BGR
            tongue_tip_color = (110, 90, 235)
            self.draw_rotated_ellipse(frame, t_center, (t_w, int(t_len * 0.6)), int(ang), tongue_color, -1)
 
            # Tongue tip curve
            tip_center = (
                int(lower_lip[0] + math.cos(down_rad) * (t_len * 0.85)),
                int(lower_lip[1] + math.sin(down_rad) * (t_len * 0.85))
            )
            self.draw_rotated_ellipse(frame, tip_center, (int(t_w * 0.95), int(t_w * 0.6)), int(ang), tongue_tip_color, -1)
 
            # Central tongue crease
            t_start = (int(lower_lip[0]), int(lower_lip[1]))
            t_end = (int(lower_lip[0] + math.cos(down_rad) * (t_len * 0.75)), int(lower_lip[1] + math.sin(down_rad) * (t_len * 0.75)))
            cv2.line(frame, t_start, t_end, (90, 70, 200), max(2, int(fw * 0.012)), lineType=cv2.LINE_AA)
 
        return frame
