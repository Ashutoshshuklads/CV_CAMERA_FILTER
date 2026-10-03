# D:\snapchat_filters\src\filters\base_filter.py
from abc import ABC, abstractmethod
import math
import cv2
import numpy as np
from src.vision_tracker import FaceData, HandData
from typing import List

class BaseFilter(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def apply(self, frame: np.ndarray, face_data: FaceData, hands_data: List[HandData]) -> np.ndarray:
        pass

    @staticmethod
    def draw_rotated_ellipse(img, center, axes, angle_deg, color, thickness=-1):
        # Draw anti-aliased rotated ellipse
        cv2.ellipse(img, center, axes, angle_deg, 0, 360, color, thickness, lineType=cv2.LINE_AA)

    @staticmethod
    def draw_glowing_circle(img, center, radius, color, glow_color, glow_radius):
        cv2.circle(img, center, glow_radius, glow_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, center, radius, color, -1, lineType=cv2.LINE_AA)
