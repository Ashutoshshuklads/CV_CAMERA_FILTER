# D:\snapchat_filters\src\ui_renderer.py
import math
import time
import cv2
import numpy as np
from typing import List, Optional
from src.config import FILTERS
from src.vision_tracker import HandData

class UIRenderer:
    def __init__(self):
        self.pulse_time = 0.0
        self.pinch_anim_start = 0.0
        self.pinch_anim_pos = None

    def trigger_pinch_effect(self, pos: tuple):
        self.pinch_anim_start = time.time()
        self.pinch_anim_pos = pos

    def render(
        self,
        frame: np.ndarray,
        selected_idx: int,
        applied_idx: int,
        hands_data: List[HandData],
        gesture_feedback: Optional[str] = None,
        fps: float = 0.0
    ) -> np.ndarray:
        h, w, _ = frame.shape
        self.pulse_time = time.time()

        # 1. Top HUD Bar
        self._render_top_bar(frame, w, applied_idx, fps)

        # 2. Hand Tracking Overlay & Pinch Meter
        self._render_hand_hud(frame, hands_data)

        # 3. Bottom Snapchat-Style Carousel Dock
        self._render_carousel(frame, w, h, selected_idx, applied_idx)

        # 4. Gesture Feedback Toast Notification
        if gesture_feedback:
            self._render_toast(frame, w, h, gesture_feedback)

        # 5. Pinch Confirmation Ripple Animation
        self._render_pinch_ripple(frame)

        return frame

    def _render_top_bar(self, frame: np.ndarray, w: int, applied_idx: int, fps: float):
        # Translucent top bar
        bar_h = 44
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, dst=frame)
        cv2.line(frame, (0, bar_h), (w, bar_h), (60, 60, 70), 1)

        # FPS
        fps_text = f"{int(fps)} FPS"
        cv2.putText(frame, fps_text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2, cv2.LINE_AA)

        # Controls Hint
        hint = "SWIPE: [<-] Prev | [->] Next   |   PINCH: Confirm & Apply"
        (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, hint, ((w - tw) // 2, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        # Active Filter Badge
        active_name = FILTERS[applied_idx]["name"]
        badge_text = f"ACTIVE: {active_name.upper()}"
        (bw, _), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        bx = w - bw - 25
        cv2.putText(frame, badge_text, (bx, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 2, cv2.LINE_AA)

    def _render_hand_hud(self, frame: np.ndarray, hands_data: List[HandData]):
        if not hands_data:
            return

        for hand in hands_data:
            # Draw connecting pinch line
            idx_pt = hand.index_tip
            thb_pt = hand.thumb_tip
            mid_pt = ((idx_pt[0] + thb_pt[0]) // 2, (idx_pt[1] + thb_pt[1]) // 2)

            # Color changes from Cyan (open) to Bright Gold/Green (pinched)
            line_col = (0, 255, 100) if hand.is_pinching else (255, 220, 0)

            # Fingertip glowing dots
            cv2.circle(frame, idx_pt, 7, line_col, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, idx_pt, 11, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            cv2.circle(frame, thb_pt, 7, line_col, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, thb_pt, 11, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            cv2.line(frame, idx_pt, thb_pt, line_col, 2, lineType=cv2.LINE_AA)

            # Circular Pinch Progress Meter around midpoint
            meter_radius = 22
            pinch_ratio = max(0.0, min(1.0, 1.0 - (hand.pinch_dist_norm / 0.12)))
            angle_end = int(pinch_ratio * 360)
            if angle_end > 5:
                cv2.ellipse(frame, mid_pt, (meter_radius, meter_radius), -90, 0, angle_end, (0, 255, 120), 2, lineType=cv2.LINE_AA)

            if hand.is_pinching:
                cv2.circle(frame, mid_pt, 10, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                cv2.putText(frame, "PINCHED!", (mid_pt[0] + 15, mid_pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    def _render_carousel(self, frame: np.ndarray, w: int, h: int, selected_idx: int, applied_idx: int):
        num_filters = len(FILTERS)
        dock_w = min(int(w * 0.88), num_filters * 105 + 60)
        dock_h = 100
        dock_x = (w - dock_w) // 2
        dock_y = h - dock_h - 18

        # Frosted glass dock background
        overlay = frame.copy()
        cv2.rectangle(overlay, (dock_x, dock_y), (dock_x + dock_w, dock_y + dock_h), (25, 25, 30), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, dst=frame)

        # Dock border with subtle glow
        cv2.rectangle(frame, (dock_x, dock_y), (dock_x + dock_w, dock_y + dock_h), (90, 90, 100), 1, lineType=cv2.LINE_AA)

        # Spacing for items
        item_spacing = dock_w / num_filters
        center_y = dock_y + 42

        for i, item in enumerate(FILTERS):
            cx = int(dock_x + (i + 0.5) * item_spacing)
            is_selected = (i == selected_idx)
            is_applied = (i == applied_idx)

            base_r = 24
            if is_selected:
                base_r = 30 # Magnified candidate card

            # Card Background
            card_col = (50, 50, 60)
            if is_applied and is_selected:
                card_col = (30, 80, 40)
            elif is_applied:
                card_col = (20, 60, 35)
            elif is_selected:
                card_col = (60, 50, 20)

            cv2.circle(frame, (cx, center_y), base_r, card_col, -1, lineType=cv2.LINE_AA)

            # Icon text or abbreviation
            icon_text = item["icon"]
            (tw, th), _ = cv2.getTextSize(icon_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42 if is_selected else 0.35, 1)
            cv2.putText(frame, icon_text, (cx - tw // 2, center_y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42 if is_selected else 0.35, (255, 255, 255), 1, cv2.LINE_AA)

            # Rings & Indicators
            if is_selected:
                # Pulsing animated neon cyan/gold selection ring
                pulse = int(math.sin(self.pulse_time * 8.0) * 2)
                ring_r = base_r + 4 + pulse
                ring_col = (0, 255, 120) if is_applied else (255, 230, 0)
                cv2.circle(frame, (cx, center_y), ring_r, ring_col, 3, lineType=cv2.LINE_AA)

                # Cursor arrow above selected item
                arrow_y = dock_y - 6
                arrow_pts = np.array([[cx, arrow_y + 6], [cx - 8, arrow_y - 4], [cx + 8, arrow_y - 4]], dtype=np.int32)
                cv2.fillPoly(frame, [arrow_pts], ring_col)

            if is_applied:
                # Green checkmark badge on applied card
                badge_pt = (cx + base_r - 6, center_y - base_r + 6)
                cv2.circle(frame, badge_pt, 7, (0, 200, 0), -1, lineType=cv2.LINE_AA)
                cv2.circle(frame, badge_pt, 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                cv2.putText(frame, "v", (badge_pt[0] - 4, badge_pt[1] + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

            # Filter Name Label below card
            name = item["name"]
            label_col = (255, 255, 255) if is_selected else (160, 160, 160)
            (lw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.38 if is_selected else 0.32, 1)
            cv2.putText(frame, name, (cx - lw // 2, dock_y + dock_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38 if is_selected else 0.32, label_col, 1, cv2.LINE_AA)

    def _render_toast(self, frame: np.ndarray, w: int, h: int, feedback: str):
        # Semi-transparent toast in lower-middle screen above dock
        (tw, th), _ = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        box_w = tw + 40
        box_h = th + 24
        box_x = (w - box_w) // 2
        box_y = h - 180

        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (10, 10, 15), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, dst=frame)

        # Border
        toast_col = (0, 255, 120) if "PINCH" in feedback else (255, 210, 0)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), toast_col, 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, feedback, (box_x + 20, box_y + box_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, toast_col, 2, cv2.LINE_AA)

    def _render_pinch_ripple(self, frame: np.ndarray):
        if self.pinch_anim_pos is None:
            return

        elapsed = time.time() - self.pinch_anim_start
        if elapsed > 0.45:
            self.pinch_anim_pos = None
            return

        progress = elapsed / 0.45
        radius = int(15 + progress * 60)
        alpha = 1.0 - progress
        col = tuple(int(c * alpha) for c in (0, 255, 120))

        cv2.circle(frame, self.pinch_anim_pos, radius, col, 3, lineType=cv2.LINE_AA)
        cv2.circle(frame, self.pinch_anim_pos, max(2, int(radius * 0.6)), col, 2, lineType=cv2.LINE_AA)
