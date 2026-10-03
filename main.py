import sys
import time
import datetime
import cv2
import numpy as np
from pathlib import Path
 
# Ensure src package is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))
 
from src.config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_BACKEND,
    CAPTURES_DIR,
    FILTERS
)
from src.vision_tracker import VisionTracker
from src.gesture_engine import GestureEngine
from src.ui_renderer import UIRenderer
 
# Filters
from src.filters.dog_filter import DogFilter
from src.filters.cyberpunk_filter import CyberpunkFilter
from src.filters.retro_filter import RetroFilter
from src.filters.fire_filter import FireFilter
from src.filters.angel_filter import AngelFilter
from src.filters.blizzard_filter import BlizzardFilter
 
def initialize_camera(index=CAMERA_INDEX):
    print(f"[Camera] Initializing webcam (index={index}) with DirectShow for zero latency...")
    cap = cv2.VideoCapture(index, CAMERA_BACKEND)
    if not cap.isOpened():
        print(f"[Camera] DirectShow failed, trying default backend...")
        cap = cv2.VideoCapture(index)
 
    if not cap.isOpened():
        print(f"[Error] Could not open camera {index}!")
        return None
 
    # Ultra-low latency settings
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
 
    # Warmup
    for _ in range(5):
        cap.read()
 
    print("[Camera] Webcam initialized successfully!")
    return cap
 
def save_snapshot(frame: np.ndarray) -> Path:
    """Save the given (already-filtered, HUD-free) frame as a timestamped PNG."""
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"snap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.png"
    save_path = CAPTURES_DIR / filename
    cv2.imwrite(str(save_path), frame)
    return save_path
 
def main():
    print("=" * 65)
    print("  SNAPCHAT AR FILTERS with HAND GESTURE & FACE DETECTION")
    print("=" * 65)
    print(" Controls:")
    print("  * SWIPE LEFT   -> Previous Filter candidate")
    print("  * SWIPE RIGHT  -> Next Filter candidate")
    print("  * PINCH INDEX  -> Apply / Lock in Selected Filter")
    print(" Keyboard Controls:")
    print("  * [<-] or [A]  -> Previous Filter")
    print("  * [->] or [D]  -> Next Filter")
    print("  * [Enter/Space]-> Apply Selected Filter")
    print("  * [0 - 6]      -> Quick Switch Filter")
    print("  * [C] or [P]   -> Capture Photo")
    print("  * [Q] or [ESC] -> Quit Application")
    print("=" * 65)
 
    cap = initialize_camera()
    if cap is None:
        print("[Error] No camera found. Please check your laptop webcam.")
        sys.exit(1)
 
    print("[AI Models] Loading MediaPipe Face and Hand Landmarkers...")
    tracker = VisionTracker()
    gesture_engine = GestureEngine()
    ui = UIRenderer()
 
    # Filter instances
    filters_map = {
        0: None, # Clean Camera
        1: DogFilter(),
        2: CyberpunkFilter(),
        3: RetroFilter(),
        4: FireFilter(),
        5: AngelFilter(),
        6: BlizzardFilter(),
    }
 
    selected_idx = 1  # Highlighted in bottom carousel
    applied_idx = 0   # Active on face (starts clean)
 
    fps = 30.0
    prev_time = time.time()
    window_name = "Snapchat AR Filters - Hand & Face Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
 
    # Photo-capture state
    capture_requested = False
    flash_alpha = 0.0
    last_capture_path = None
    capture_toast_until = 0.0
 
    print("[Ready] Running main loop. Press 'q' in video window to exit.")
 
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
 
            # Mirror frame horizontally for natural selfie perspective
            frame = cv2.flip(frame, 1)
 
            # 1. Real-time Face & Hand Tracking
            face_data, hands_data = tracker.process_frame(frame)
 
            # 2. Hand Gesture Processing
            gesture_event = gesture_engine.process(hands_data)
 
            if gesture_event == "SWIPE_LEFT":
                selected_idx = (selected_idx - 1) % len(FILTERS)
            elif gesture_event == "SWIPE_RIGHT":
                selected_idx = (selected_idx + 1) % len(FILTERS)
            elif gesture_event == "PINCH":
                applied_idx = selected_idx
                if gesture_engine.active_pinch_pos:
                    ui.trigger_pinch_effect(gesture_engine.active_pinch_pos)
 
            # 3. Keyboard Input Fallback
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]: # 27 is ESC
                break
            elif key in [81, 2424832, ord('a'), ord('A'), ord('[')]: # Left arrow
                selected_idx = (selected_idx - 1) % len(FILTERS)
            elif key in [83, 2555904, ord('d'), ord('D'), ord(']')]: # Right arrow
                selected_idx = (selected_idx + 1) % len(FILTERS)
            elif key in [13, 32]: # Enter or Space
                applied_idx = selected_idx
                h, w, _ = frame.shape
                ui.trigger_pinch_effect((w // 2, h // 2))
            elif key in [ord('c'), ord('C'), ord('p'), ord('P')]: # Capture photo
                capture_requested = True
            elif ord('0') <= key <= ord('6'):
                idx = key - ord('0')
                if idx < len(FILTERS):
                    selected_idx = idx
                    applied_idx = idx
 
            # 4. Apply Active Filter
            active_filter = filters_map.get(applied_idx)
            if active_filter is not None:
                frame = active_filter.apply(frame, face_data, hands_data)
 
            # 4b. Handle Photo Capture (save the clean, filtered frame
            # BEFORE the HUD/carousel/FPS overlay gets drawn on it, so the
            # saved photo looks like the filter alone, not the app's UI)
            if capture_requested:
                clean_frame = frame.copy()
                last_capture_path = save_snapshot(clean_frame)
                print(f"[Capture] Photo saved -> {last_capture_path}")
                capture_requested = False
                flash_alpha = 0.65
                capture_toast_until = time.time() + 1.2
 
            # 5. Render UI, Carousel, & HUD
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                cur_fps = 1.0 / dt
                fps = fps * 0.9 + cur_fps * 0.1
 
            capture_feedback = None
            if last_capture_path is not None and time.time() < capture_toast_until:
                capture_feedback = f"PHOTO SAVED: {last_capture_path.name}"
 
            frame = ui.render(
                frame=frame,
                selected_idx=selected_idx,
                applied_idx=applied_idx,
                hands_data=hands_data,
                gesture_feedback=gesture_engine.get_feedback() or capture_feedback,
                fps=fps
            )
 
            # 5b. Camera-shutter style flash on top of everything, fading out
            # over a few frames so a photo capture feels like a real click.
            if flash_alpha > 0.02:
                white = np.full_like(frame, 255)
                frame = cv2.addWeighted(white, flash_alpha, frame, 1 - flash_alpha, 0)
                flash_alpha *= 0.55
            else:
                flash_alpha = 0.0
 
            # 6. Display Output
            cv2.imshow(window_name, frame)
 
    except KeyboardInterrupt:
        print("\n[Exit] User interrupted via keyboard.")
    finally:
        print("[Cleanup] Releasing camera and closing windows...")
        cap.release()
        cv2.destroyAllWindows()
        print("[Cleanup] Done. Goodbye!")
 
if __name__ == "__main__":
    main()
 