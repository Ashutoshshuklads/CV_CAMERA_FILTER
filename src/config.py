
from pathlib import Path
import cv2
 
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
 
# Folder where captured photos are saved (created automatically if missing)
CAPTURES_DIR = BASE_DIR / "captures"
 
FACE_MODEL_PATH = str(MODELS_DIR / "face_landmarker.task")
HAND_MODEL_PATH = str(MODELS_DIR / "hand_landmarker.task")
 
# Camera Configuration
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_BACKEND = cv2.CAP_DSHOW
 
# Gesture Thresholds
SWIPE_MIN_DX = 80          # Minimum horizontal travel in pixels
SWIPE_MAX_DY = 140         # Hard cap on vertical deviation (pixels)
SWIPE_MAX_DY_RATIO = 0.7   # Vertical movement must be < 70% of horizontal (arcs are OK)
SWIPE_WINDOW_SEC = 0.5     # Time window (seconds) for trajectory tracking
SWIPE_HAND_LOST_GRACE_SEC = 0.3  # Keep trajectory if tracking drops briefly mid-swipe
SWIPE_COOLDOWN_SEC = 0.45  # Cooldown between consecutive swipes
PINCH_THRESHOLD_NORM = 0.065 # Normalized distance between index & thumb tip
PINCH_COOLDOWN_SEC = 0.5   # Cooldown after filter activation
 
# Filter Definitions
FILTERS = [
    {
        "id": 0,
        "name": "Normal",
        "tagline": "Clean Camera",
        "icon": "[Raw]",
        "color": (200, 200, 200)
    },
    {
        "id": 1,
        "name": "Puppy Dog",
        "tagline": "Ears & Tongue",
        "icon": "[Dog]",
        "color": (60, 180, 255)
    },
    {
        "id": 2,
        "name": "Cyberpunk",
        "tagline": "Neon Visor & Grid",
        "icon": "[Cyber]",
        "color": (255, 230, 0)
    },
    {
        "id": 3,
        "name": "Retro Shades",
        "tagline": "Vintage & Stache",
        "icon": "[Retro]",
        "color": (50, 220, 100)
    },
    {
        "id": 4,
        "name": "Fire Demon",
        "tagline": "Horns & Embers",
        "icon": "[Fire]",
        "color": (30, 70, 255)
    },
    {
        "id": 5,
        "name": "Angel Halo",
        "tagline": "Halo & Bloom",
        "icon": "[Halo]",
        "color": (0, 235, 255)
    },
    {
        "id": 6,
        "name": "Blizzard",
        "tagline": "Frost & Snow",
        "icon": "[Snow]",
        "color": (255, 200, 100)
    }
]
 