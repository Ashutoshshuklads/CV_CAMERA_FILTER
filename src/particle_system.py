# D:\snapchat_filters\src\particle_system.py
import math
import random
import time
import numpy as np
import cv2

class EmbersSystem:
    def __init__(self, max_particles=60):
        self.max_particles = max_particles
        self.particles = []  # dict with x, y, vx, vy, size, life, max_life, color

    def update_and_draw(self, frame: np.ndarray, h: int, w: int):
        # Spawn new particles
        while len(self.particles) < self.max_particles:
            self.particles.append({
                'x': random.uniform(0, w),
                'y': random.uniform(h * 0.7, h),
                'vx': random.uniform(-1.2, 1.2),
                'vy': random.uniform(-4.5, -2.0),
                'size': random.uniform(2.5, 6.0),
                'life': 1.0,
                'decay': random.uniform(0.015, 0.035),
                'color': random.choice([
                    (0, 140, 255),   # Orange
                    (0, 80, 255),    # Red-Orange
                    (0, 220, 255),   # Yellow-Gold
                    (50, 50, 255)    # Deep Red
                ])
            })

        # Update & draw
        alive = []
        overlay = np.zeros_like(frame)

        for p in self.particles:
            p['x'] += p['vx'] + math.sin(p['y'] * 0.05) * 0.8
            p['y'] += p['vy']
            p['life'] -= p['decay']

            if p['life'] > 0 and 0 <= p['x'] < w and p['y'] > 0:
                alive.append(p)
                radius = int(max(1, p['size'] * p['life']))
                # Glowing center and soft halo
                col = tuple(int(c * p['life']) for c in p['color'])
                cv2.circle(overlay, (int(p['x']), int(p['y'])), radius, col, -1)
                if radius > 2:
                    cv2.circle(overlay, (int(p['x']), int(p['y'])), radius + 2, tuple(int(c * 0.4) for c in col), 1)

        self.particles = alive
        cv2.add(frame, overlay, dst=frame)

class SnowSystem:
    def __init__(self, max_particles=100):
        self.max_particles = max_particles
        self.particles = []

    def update_and_draw(self, frame: np.ndarray, h: int, w: int):
        while len(self.particles) < self.max_particles:
            self.particles.append({
                'x': random.uniform(0, w),
                'y': random.uniform(-20, 0),
                'vy': random.uniform(1.8, 4.5),
                'phase': random.uniform(0, 6.28),
                'size': random.uniform(2.0, 5.5),
                'opacity': random.uniform(0.5, 0.95)
            })

        alive = []
        overlay = np.zeros_like(frame)
        t = time.time()

        for p in self.particles:
            p['y'] += p['vy']
            # Gentle horizontal sway
            x = p['x'] + math.sin(t * 2.0 + p['phase']) * 15.0

            if p['y'] < h and 0 <= x < w:
                alive.append(p)
                r = int(p['size'])
                val = int(255 * p['opacity'])
                # Soft white/ice cyan tint
                cv2.circle(overlay, (int(x), int(p['y'])), r, (val, val, val), -1)

        self.particles = alive
        cv2.addWeighted(frame, 1.0, overlay, 0.85, 0, dst=frame)

class CyberMatrixSystem:
    def __init__(self, num_columns=25):
        self.num_columns = num_columns
        self.drops = None

    def update_and_draw(self, frame: np.ndarray, h: int, w: int):
        if self.drops is None or len(self.drops) != self.num_columns:
            step = w // self.num_columns
            self.drops = [{
                'x': i * step + step // 2,
                'y': random.uniform(-h, 0),
                'speed': random.uniform(7.0, 16.0),
                'len': random.randint(30, 90)
            } for i in range(self.num_columns)]

        overlay = np.zeros_like(frame)

        for d in self.drops:
            d['y'] += d['speed']
            if d['y'] - d['len'] > h:
                d['y'] = random.uniform(-80, -10)
                d['speed'] = random.uniform(7.0, 16.0)

            x = int(d['x'])
            head_y = int(d['y'])
            tail_y = int(d['y'] - d['len'])

            # Glowing neon cyber beam
            if 0 <= head_y < h:
                cv2.circle(overlay, (x, head_y), 3, (255, 255, 255), -1)
            cv2.line(overlay, (x, max(0, tail_y)), (x, min(h - 1, head_y)), (255, 230, 0), 2)

        cv2.addWeighted(frame, 1.0, overlay, 0.45, 0, dst=frame)

class GoldenSparkleSystem:
    def __init__(self, max_sparkles=30):
        self.max_sparkles = max_sparkles
        self.sparkles = []

    def update_and_draw(self, frame: np.ndarray, anchor_center: tuple, h: int, w: int):
        cx, cy = anchor_center
        while len(self.sparkles) < self.max_sparkles:
            angle = random.uniform(0, 6.28)
            dist = random.uniform(20, 120)
            self.sparkles.append({
                'x': cx + math.cos(angle) * dist,
                'y': cy + math.sin(angle) * dist * 0.6,
                'vx': random.uniform(-0.5, 0.5),
                'vy': random.uniform(-1.0, -0.2),
                'life': 1.0,
                'decay': random.uniform(0.02, 0.05),
                'size': random.uniform(3, 7)
            })

        alive = []
        overlay = np.zeros_like(frame)

        for s in self.sparkles:
            s['x'] += s['vx']
            s['y'] += s['vy']
            s['life'] -= s['decay']

            if s['life'] > 0 and 0 <= s['x'] < w and 0 <= s['y'] < h:
                alive.append(s)
                sz = int(s['size'] * s['life'])
                x, y = int(s['x']), int(s['y'])
                col = (int(100 * s['life']), int(220 * s['life']), int(255 * s['life']))
                # 4-point star
                cv2.line(overlay, (x - sz, y), (x + sz, y), col, 1)
                cv2.line(overlay, (x, y - sz), (x, y + sz), col, 1)
                cv2.circle(overlay, (x, y), max(1, sz // 2), (255, 255, 255), -1)

        self.sparkles = alive
        cv2.add(frame, overlay, dst=frame)
