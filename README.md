# Snapchat AR Filters with Hand & Face Detection

Real-time computer vision project featuring 6 Snapchat-like face and environment filters, controlled with hand swipe and pinch gestures in front of your laptop camera.

---

## 🎯 Features

1. **Ultra-Low Latency Camera**: Instant startup using DirectShow (`CAP_DSHOW`) and buffer size 1 with 50+ FPS real-time tracking.
2. **Real-Time Hand Gestures**:
   - **Swipe Hand Left**: Selects the **Previous** filter candidate in the carousel.
   - **Swipe Hand Right**: Selects the **Next** filter candidate in the carousel.
   - **Pinch Index Finger & Thumb**: **Applies / Locks in** the selected filter onto your face and screen.
3. **Snapchat-Style Bottom-Center Carousel**:
   - Centered dock with interactive circular filter cards.
   - Animated glowing cyan/yellow pulsing indicator for the selected candidate.
   - Green checkmark badge on the currently applied filter.
   - Visual ripple burst on pinch activation.
4. **6 Face & Environment Filters**:
   - **Puppy Dog**: Floppy reactive ears, cute button nose, whisker dots, and interactive pink tongue when you open your mouth!
   - **Cyberpunk**: Neon visor across eyes, animated scanlines, eye reticles, facial cyber circuit lines, and matrix digital rain environment.
   - **Retro Shades**: 8-bit black sunglasses with glare reflection, classic handlebar mustache, and 90s vintage VHS color grading with timestamp.
   - **Fire Demon**: Glowing fiery horns on forehead, flaming blazing eye auras, and dynamic floating embers rising in the background.
   - **Angel Halo**: Hovering golden glowing halo above head, cheek starbursts, and dreamy golden hour ambient bloom.
   - **Blizzard**: Frost vignette border, cold rosy-blushed cheeks, forehead snowflake tiara, and dynamic falling snowflakes.
   - *(Filter 0: Clean Camera feed)*

---

## 🚀 How to Run

### Option 1: Double Click
Double click `run.bat` in `D:\snapchat_filters`.

### Option 2: Command Line
Open terminal and run:
```bash
cd D:\snapchat_filters
python main.py
```

---

## ⌨️ Keyboard Fallback Controls
- `[<-]` or `[A]` or `[`: Select Previous Filter
- `[->]` or `[D]` or `]`: Select Next Filter
- `[Enter]` or `[Space]`: Apply Selected Filter
- `[0] - [6]`: Instantly jump to a specific filter
- `[Q]` or `[ESC]`: Exit
