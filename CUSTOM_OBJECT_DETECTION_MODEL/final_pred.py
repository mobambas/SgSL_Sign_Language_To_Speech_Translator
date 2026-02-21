"""
Sign Language to Text Conversion Application

The CNN model takes 400x400 RGB images (white background with hand skeleton drawn)
and outputs 8 group classes, which are then refined using landmark rules.
"""

import math
import os
import threading
import time
import traceback
from collections import deque
from string import ascii_uppercase
from typing import List

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

#fallback
try:
    import enchant
    ENCHANT_AVAILABLE = True
except ImportError:
    ENCHANT_AVAILABLE = False
    print("Warning: enchant not available. Word suggestions disabled.")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: pyttsx3 not available. Text-to-speech disabled.")

try:
    from cvzone.HandTrackingModule import HandDetector
    CVZONE_AVAILABLE = True
except ImportError:
    CVZONE_AVAILABLE = False
    print("Error: cvzone not available. Hand detection will not work.")

try:
    from cvzone.FaceMeshModule import FaceMeshDetector
    FACEMESH_AVAILABLE = True
except ImportError:
    FACEMESH_AVAILABLE = False
    print("Warning: cvzone FaceMesh not available. Non-manual markers disabled.")

try:
    from keras.models import load_model
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Error: keras not available. Model prediction will not work.")

# Configuration
os.environ.setdefault("THEANO_FLAGS", "device=cuda, assert_no_cpu_op=True")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn8grps_rad1_model.h5")
CAMERA_INDEX = 0
OFFSET = 29
WHITE_BG_SIZE = 400

# Timing configuration
COOLDOWN_FRAMES = 30  # Frames to wait after adding a character (~1 second at 30fps)
STABLE_COUNT_REQUIRED = 3  # Consecutive same predictions needed

# Hand detection configuration
HAND_DETECTION_CONFIDENCE = 0.6
HAND_TRACKING_CONFIDENCE = 0.5

# Non-manual marker (NMM) configuration
NMM_STABLE_FRAMES = 4
NMM_MISSING_RESET = 6
NMM_BROW_RAISE_DELTA = 0.18
NMM_BROW_FURROW_DELTA = 0.12
NMM_BROW_BASELINE_TOL = 0.08
NMM_MOUTH_OPEN_RATIO = 0.25
HEAD_MOVEMENT_WINDOW = 12
HEAD_SHAKE_THRESHOLD = 0.08
HEAD_NOD_THRESHOLD = 0.08
HEAD_EVENT_COOLDOWN = 20
HEAD_TILT_DEG = 12


def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class HandSignClassifier:
    """Classifies hand signs using CNN + landmark rules."""
    
    def __init__(self, model_path: str):
        if not KERAS_AVAILABLE:
            raise RuntimeError("Keras not available")
        print(f"Loading model from: {model_path}")
        self.model = load_model(model_path)
        print("Model loaded successfully")
    
    def predict(self, white_image: np.ndarray, pts: List) -> str:
        """
        Predict letter from white background image with skeleton.
        
        Args:
            white_image: 400x400x3 BGR image with hand skeleton
            pts: List of 21 landmarks, each as [x, y] or [x, y, z]
        """
        try:
            # Model input: (1, 400, 400, 3)
            model_input = white_image.reshape(1, WHITE_BG_SIZE, WHITE_BG_SIZE, 3)
            predictions = self.model.predict(model_input, verbose=0)
            prob = np.array(predictions[0], dtype='float32')
            
            # Get top-2 groups
            ch1 = int(np.argmax(prob))
            prob[ch1] = 0
            ch2 = int(np.argmax(prob))
            
            # Apply rules
            ch1 = self._apply_group_rules(ch1, ch2, pts)
            result = self._apply_subgroup_rules(ch1, pts)
            
            return result
        except Exception as e:
            print(f"Prediction error: {e}")
            return ""
    
    def _apply_group_rules(self, ch1: int, ch2: int, pts: List) -> int:
        """Apply 8-group classification rules based on landmarks."""
        d = distance
        pl = [ch1, ch2]

        # Group 0 rules
        l = [[5,2],[5,3],[3,5],[3,6],[3,0],[3,2],[6,4],[6,1],[6,2],[6,6],[6,7],[6,0],[6,5],
             [4,1],[1,0],[1,1],[6,3],[1,6],[5,6],[5,1],[4,5],[1,4],[1,5],[2,0],[2,6],[4,6],
             [1,0],[5,7],[1,6],[6,1],[7,6],[2,5],[7,1],[5,4],[7,0],[7,5],[7,2]]
        if pl in l:
            if pts[6][1]<pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]:
                ch1 = 0

        l = [[2,2],[2,1]]
        if pl in l:
            if pts[5][0] < pts[4][0]:
                ch1 = 0

        # Group 2 rules
        l = [[0,0],[0,6],[0,2],[0,5],[0,1],[0,7],[5,2],[7,6],[7,1]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[0][0]>pts[8][0] and pts[0][0]>pts[4][0] and pts[0][0]>pts[12][0] and 
                pts[0][0]>pts[16][0] and pts[0][0]>pts[20][0]) and pts[5][0]>pts[4][0]:
                ch1 = 2

        l = [[6,0],[6,6],[6,2]]
        pl = [ch1, ch2]
        if pl in l:
            if d(pts[8], pts[16]) < 52:
                ch1 = 2

        # Group 3 rules
        l = [[1,4],[1,5],[1,6],[1,3],[1,0]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[6][1]>pts[8][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1] and 
                pts[0][0]<pts[8][0] and pts[0][0]<pts[12][0] and pts[0][0]<pts[16][0] and pts[0][0]<pts[20][0]):
                ch1 = 3

        l = [[4,6],[4,1],[4,5],[4,3],[4,7]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[4][0] > pts[0][0]:
                ch1 = 3

        l = [[5,3],[5,0],[5,7],[5,4],[5,2],[5,1],[5,5]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[2][1] + 15 < pts[16][1]:
                ch1 = 3

        # Group 4 rules (L)
        l = [[6,4],[6,1],[6,2]]
        pl = [ch1, ch2]
        if pl in l:
            if d(pts[4], pts[11]) > 55:
                ch1 = 4

        l = [[1,4],[1,6],[1,1]]
        pl = [ch1, ch2]
        if pl in l:
            if d(pts[4], pts[11]) > 50 and (pts[6][1]>pts[8][1] and pts[10][1]<pts[12][1] and 
                pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]):
                ch1 = 4

        l = [[3,6],[3,4]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[4][0] < pts[0][0]:
                ch1 = 4

        l = [[2,2],[2,5],[2,4]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[1][0] < pts[12][0]:
                ch1 = 4

        # Group 5 rules
        l = [[3,6],[3,5],[3,4]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[6][1]>pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and 
                pts[18][1]<pts[20][1]) and pts[4][1] > pts[10][1]:
                ch1 = 5

        l = [[3,2],[3,1],[3,6]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[4][1]+17>pts[8][1] and pts[4][1]+17>pts[12][1] and pts[4][1]+17>pts[16][1] and pts[4][1]+17>pts[20][1]:
                ch1 = 5

        l = [[4,4],[4,5],[4,2],[7,5],[7,6],[7,0]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[4][0] > pts[0][0]:
                ch1 = 5

        l = [[0,2],[0,6],[0,1],[0,5],[0,0],[0,7],[0,4],[0,3],[2,7]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[0][0]<pts[8][0] and pts[0][0]<pts[12][0] and pts[0][0]<pts[16][0] and pts[0][0]<pts[20][0]:
                ch1 = 5

        # Group 7 rules
        l = [[5,7],[5,2],[5,6]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[3][0] < pts[0][0]:
                ch1 = 7

        l = [[4,6],[4,2],[4,4],[4,1],[4,5],[4,7]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[6][1] < pts[8][1]:
                ch1 = 7

        l = [[6,7],[0,7],[0,1],[0,0],[6,4],[6,6],[6,5],[6,1]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[18][1] > pts[20][1]:
                ch1 = 7

        # Group 6 rules
        l = [[0,4],[0,2],[0,3],[0,1],[0,6]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[5][0] > pts[16][0]:
                ch1 = 6

        l = [[7,2]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[18][1]<pts[20][1] and pts[8][1]<pts[10][1]:
                ch1 = 6

        l = [[2,1],[2,2],[2,6],[2,7],[2,0]]
        pl = [ch1, ch2]
        if pl in l:
            if d(pts[8], pts[16]) > 50:
                ch1 = 6

        l = [[4,6],[4,2],[4,1],[4,4]]
        pl = [ch1, ch2]
        if pl in l:
            if d(pts[4], pts[11]) < 60:
                ch1 = 6

        l = [[1,4],[1,6],[1,0],[1,2]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[5][0] - pts[4][0] - 15 > 0:
                ch1 = 6

        # Group 1 rules
        l = [[5,0],[5,1],[5,4],[5,5],[5,6],[6,1],[7,6],[0,2],[7,1],[7,4],[6,6],[7,2],[5,0],[6,3],[6,4],[7,5],[7,2]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1] and pts[18][1]>pts[20][1]:
                ch1 = 1

        l = [[6,1],[6,0],[0,3],[6,4],[2,2],[0,6],[6,2],[7,6],[4,6],[4,1],[4,2],[0,2],[7,1],[7,4],[6,6],[7,2],[7,5],[7,2]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[6][1]<pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1] and pts[18][1]>pts[20][1]:
                ch1 = 1

        l = [[6,1],[6,0],[4,2],[4,1],[4,6],[4,4]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1] and pts[18][1]>pts[20][1]:
                ch1 = 1

        l = [[5,0],[3,4],[3,0],[3,1],[3,5],[5,5],[5,4],[5,1],[7,6]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[6][1]>pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and 
                pts[18][1]<pts[20][1]) and pts[2][0]<pts[0][0] and pts[4][1]>pts[14][1]:
                ch1 = 1

        l = [[4,1],[4,2],[4,4]]
        pl = [ch1, ch2]
        if pl in l:
            if d(pts[4], pts[11]) < 50 and (pts[6][1]>pts[8][1] and pts[10][1]<pts[12][1] and 
                pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]):
                ch1 = 1

        l = [[3,4],[3,0],[3,1],[3,5],[3,6]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[6][1]>pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and 
                pts[18][1]<pts[20][1]) and pts[2][0]<pts[0][0] and pts[14][1]<pts[4][1]:
                ch1 = 1

        l = [[6,6],[6,4],[6,1],[6,2]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[5][0] - pts[4][0] - 15 < 0:
                ch1 = 1

        l = [[5,4],[5,5],[5,1],[0,3],[0,7],[5,0],[0,2],[6,2],[7,5],[7,1],[7,6],[7,7]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[6][1]<pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]>pts[20][1]:
                ch1 = 1

        l = [[1,5],[1,7],[1,1],[1,6],[1,3],[1,0]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[4][0]<pts[5][0]+15 and (pts[6][1]<pts[8][1] and pts[10][1]<pts[12][1] and 
                pts[14][1]<pts[16][1] and pts[18][1]>pts[20][1]):
                ch1 = 7

        l = [[5,5],[5,0],[5,4],[5,1],[4,6],[4,1],[7,6],[3,0],[3,5]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]<pts[16][1] and 
                pts[18][1]<pts[20][1]) and pts[4][1]>pts[14][1]:
                ch1 = 1

        l = [[3,5],[3,0],[3,6],[5,1],[4,1],[2,0],[5,0],[5,5]]
        pl = [ch1, ch2]
        if pl in l:
            fg = 13
            if not (pts[0][0]+fg<pts[8][0] and pts[0][0]+fg<pts[12][0] and pts[0][0]+fg<pts[16][0] and pts[0][0]+fg<pts[20][0]) and \
               not (pts[0][0]>pts[8][0] and pts[0][0]>pts[12][0] and pts[0][0]>pts[16][0] and pts[0][0]>pts[20][0]) and \
               d(pts[4], pts[11]) < 50:
                ch1 = 1

        l = [[5,0],[5,5],[0,1]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1]:
                ch1 = 1

        return ch1

    def _apply_subgroup_rules(self, ch1: int, pts: List) -> str:
        """Determine specific letter within a group."""
        d = distance
        
        if ch1 == 0:
            result = 'S'
            if pts[4][0]<pts[6][0] and pts[4][0]<pts[10][0] and pts[4][0]<pts[14][0] and pts[4][0]<pts[18][0]:
                result = 'A'
            if (pts[4][0]>pts[6][0] and pts[4][0]<pts[10][0] and pts[4][0]<pts[14][0] and 
                pts[4][0]<pts[18][0] and pts[4][1]<pts[14][1] and pts[4][1]<pts[18][1]):
                result = 'T'
            if pts[4][1]>pts[8][1] and pts[4][1]>pts[12][1] and pts[4][1]>pts[16][1] and pts[4][1]>pts[20][1]:
                result = 'E'
            if pts[4][0]>pts[6][0] and pts[4][0]>pts[10][0] and pts[4][0]>pts[14][0] and pts[4][1]<pts[18][1]:
                result = 'M'
            if pts[4][0]>pts[6][0] and pts[4][0]>pts[10][0] and pts[4][1]<pts[18][1] and pts[4][1]<pts[14][1]:
                result = 'N'
            return result

        if ch1 == 2:
            return 'C' if d(pts[12], pts[4]) > 42 else 'O'

        if ch1 == 3:
            return 'G' if d(pts[8], pts[12]) > 72 else 'H'

        if ch1 == 7:
            return 'Y' if d(pts[8], pts[4]) > 42 else 'J'

        if ch1 == 4:
            return 'L'

        if ch1 == 6:
            return 'X'

        if ch1 == 5:
            if pts[4][0]>pts[12][0] and pts[4][0]>pts[16][0] and pts[4][0]>pts[20][0]:
                return 'Z' if pts[8][1]<pts[5][1] else 'Q'
            return 'P'

        if ch1 == 1:
            if pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1] and pts[18][1]>pts[20][1]:
                return 'B'
            if pts[6][1]>pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]:
                return 'D'
            if pts[6][1]<pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1] and pts[18][1]>pts[20][1]:
                return 'F'
            if pts[6][1]<pts[8][1] and pts[10][1]<pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]>pts[20][1]:
                return 'I'
            if pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1] and pts[18][1]<pts[20][1]:
                return 'W'
            if (pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]<pts[16][1] and 
                pts[18][1]<pts[20][1] and pts[4][1]<pts[9][1]):
                return 'K'
            if (d(pts[8], pts[12]) - d(pts[6], pts[10])) < 8 and (pts[6][1]>pts[8][1] and 
                pts[10][1]>pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]):
                return 'U'
            if (d(pts[8], pts[12]) - d(pts[6], pts[10])) >= 8 and (pts[6][1]>pts[8][1] and 
                pts[10][1]>pts[12][1] and pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]) and pts[4][1]>pts[9][1]:
                return 'V'
            if pts[8][0]>pts[12][0] and (pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and 
                pts[14][1]<pts[16][1] and pts[18][1]<pts[20][1]):
                return 'R'
            return 'B'

        return ascii_uppercase[ch1] if 0 <= ch1 < 26 else ""


class NonManualMarkerDetector:
    """Detects a small set of ASL non-manual markers using face landmarks."""

    def __init__(self):
        if not FACEMESH_AVAILABLE:
            raise RuntimeError("Face mesh not available")
        self.detector = FaceMeshDetector(maxFaces=1)
        self.nose_positions = deque(maxlen=HEAD_MOVEMENT_WINDOW)
        self.event_cooldown = 0
        self.brow_eye_baseline = None

    def _avg_point(self, pts, indices):
        total_x = 0.0
        total_y = 0.0
        for idx in indices:
            total_x += pts[idx][0]
            total_y += pts[idx][1]
        count = float(len(indices))
        return total_x / count, total_y / count

    def _update_brow_baseline(self, ratio):
        if self.brow_eye_baseline is None:
            self.brow_eye_baseline = ratio
            return
        if abs(ratio - self.brow_eye_baseline) < NMM_BROW_BASELINE_TOL:
            self.brow_eye_baseline = (self.brow_eye_baseline * 0.9) + (ratio * 0.1)

    def detect(self, frame):
        if self.event_cooldown > 0:
            self.event_cooldown -= 1

        _, faces = self.detector.findFaceMesh(frame, draw=False)
        if not faces:
            self.nose_positions.clear()
            return None

        pts = faces[0]
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        face_w = max_x - min_x
        face_h = max_y - min_y
        if face_w <= 0 or face_h <= 0:
            return None

        left_brow_idx = [70, 63, 105, 66, 107]
        right_brow_idx = [336, 296, 334, 293, 300]
        left_eye_top_idx = [159, 160]
        right_eye_top_idx = [386, 387]
        left_eye_bottom_idx = [145, 144]
        right_eye_bottom_idx = [374, 373]

        _, left_brow_y = self._avg_point(pts, left_brow_idx)
        _, right_brow_y = self._avg_point(pts, right_brow_idx)
        _, left_eye_top_y = self._avg_point(pts, left_eye_top_idx)
        _, right_eye_top_y = self._avg_point(pts, right_eye_top_idx)
        _, left_eye_bottom_y = self._avg_point(pts, left_eye_bottom_idx)
        _, right_eye_bottom_y = self._avg_point(pts, right_eye_bottom_idx)

        eye_height = (abs(left_eye_bottom_y - left_eye_top_y) + abs(right_eye_bottom_y - right_eye_top_y)) / 2.0
        brow_eye_dist = ((left_eye_top_y - left_brow_y) + (right_eye_top_y - right_brow_y)) / 2.0
        brow_eye_ratio = brow_eye_dist / max(eye_height, 1.0)
        if brow_eye_ratio > 0:
            self._update_brow_baseline(brow_eye_ratio)

        brow_state = "neutral"
        if self.brow_eye_baseline is not None:
            if brow_eye_ratio > self.brow_eye_baseline * (1.0 + NMM_BROW_RAISE_DELTA):
                brow_state = "raise"
            elif brow_eye_ratio < self.brow_eye_baseline * (1.0 - NMM_BROW_FURROW_DELTA):
                brow_state = "furrow"

        mouth_open = abs(pts[13][1] - pts[14][1])
        mouth_width = abs(pts[78][0] - pts[308][0])
        mouth_ratio = mouth_open / max(mouth_width, 1.0)
        mouth_state = "open" if mouth_ratio > NMM_MOUTH_OPEN_RATIO else "neutral"

        left_eye = pts[33]
        right_eye = pts[263]
        eye_angle = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        if eye_angle > HEAD_TILT_DEG:
            head_tilt = "right"
        elif eye_angle < -HEAD_TILT_DEG:
            head_tilt = "left"
        else:
            head_tilt = "neutral"

        nose_tip = pts[1]
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        norm_x = (nose_tip[0] - center_x) / face_w
        norm_y = (nose_tip[1] - center_y) / face_h
        self.nose_positions.append((norm_x, norm_y))

        head_shake = False
        head_nod = False
        if len(self.nose_positions) >= max(4, HEAD_MOVEMENT_WINDOW // 2):
            xs = [p[0] for p in self.nose_positions]
            ys = [p[1] for p in self.nose_positions]
            range_x = max(xs) - min(xs)
            range_y = max(ys) - min(ys)
            if self.event_cooldown == 0:
                if range_x > HEAD_SHAKE_THRESHOLD and range_x > range_y * 1.2:
                    head_shake = True
                    self.event_cooldown = HEAD_EVENT_COOLDOWN
                elif range_y > HEAD_NOD_THRESHOLD and range_y > range_x * 1.2:
                    head_nod = True
                    self.event_cooldown = HEAD_EVENT_COOLDOWN

        return {
            "brow_state": brow_state,
            "mouth_state": mouth_state,
            "head_tilt": head_tilt,
            "head_shake": head_shake,
            "head_nod": head_nod,
        }


class SuggestionEngine:
    """Provides word suggestions."""
    
    def __init__(self):
        self.dictionary = enchant.Dict("en-US") if ENCHANT_AVAILABLE else None
    
    def get_suggestions(self, text: str) -> List[str]:
        if not self.dictionary or not text:
            return []
        words = text.strip().split()
        if not words:
            return []
        try:
            return self.dictionary.suggest(words[-1])[:4]
        except:
            return []


class TextToSpeech:
    """Thread-safe text-to-speech handler.
    
    Reinitializes engine for each speak call to avoid pyttsx3 state issues.
    """
    
    def __init__(self):
        self.speaking = False
        self.available = TTS_AVAILABLE
        if self.available:
            print("Text-to-speech available")
    
    def speak(self, text: str):
        """Speak text in a separate thread to avoid blocking GUI."""
        if not self.available or not text.strip() or self.speaking:
            print(f"TTS: Cannot speak - available={self.available}, text='{text}', speaking={self.speaking}")
            return
        
        def _speak_thread():
            self.speaking = True
            try:
                # Create fresh engine for each speak call (fixes reuse issues)
                engine = pyttsx3.init()
                engine.setProperty("rate", 150)
                voices = engine.getProperty("voices")
                if voices:
                    engine.setProperty("voice", voices[0].id)
                
                print(f"TTS: Speaking '{text}'")
                engine.say(text)
                engine.runAndWait()
                
                # Clean up engine
                engine.stop()
                del engine
                
                print("TTS: Done speaking")
            except Exception as e:
                print(f"TTS error: {e}")
            finally:
                self.speaking = False
        
        thread = threading.Thread(target=_speak_thread, daemon=True)
        thread.start()


class SignLanguageApp:
    """Main application."""
    
    def __init__(self):
        # Video capture
        self.vs = cv2.VideoCapture(CAMERA_INDEX)
        
        # TWO separate hand detectors (like original code)
        if CVZONE_AVAILABLE:
            self.hd = HandDetector(
                maxHands=1,
                detectionCon=HAND_DETECTION_CONFIDENCE,
                minTrackCon=HAND_TRACKING_CONFIDENCE,
            )  # For full frame
            self.hd2 = HandDetector(
                maxHands=1,
                detectionCon=HAND_DETECTION_CONFIDENCE,
                minTrackCon=HAND_TRACKING_CONFIDENCE,
            )  # For cropped image
        else:
            self.hd = self.hd2 = None
        
        # Classifier
        self.classifier = None
        if KERAS_AVAILABLE:
            try:
                self.classifier = HandSignClassifier(MODEL_PATH)
            except Exception as e:
                print(f"Classifier error: {e}")
        
        # Suggestion engine
        self.suggestion_engine = SuggestionEngine()
        
        # Text-to-speech (new threaded implementation)
        self.tts = TextToSpeech()

        # State
        self.current_text = ""
        self.current_symbol = ""
        self.prev_char = ""
        self.same_char_count = 0
        self.blank_count = 0
        self.cooldown_counter = 0  # NEW: Cooldown after adding character
        self.pts = None

        # Non-manual markers
        self.nmm_detector = None
        if CVZONE_AVAILABLE and FACEMESH_AVAILABLE:
            try:
                self.nmm_detector = NonManualMarkerDetector()
            except Exception as e:
                print(f"NMM detector error: {e}")
        self.nmm_brow_state_raw = "neutral"
        self.nmm_brow_state = "neutral"
        self.nmm_brow_count = 0
        self.nmm_mouth_state_raw = "neutral"
        self.nmm_mouth_state = "neutral"
        self.nmm_mouth_count = 0
        self.nmm_head_tilt = "neutral"
        self.nmm_missing_frames = 0
        self.nmm_event_display = ""
        self.nmm_event_display_frames = 0
        self.active_nmm_tags = []
        self.active_nmm_applied = False
        self.pending_nmm_tags = []
        
        # Suggestions
        self.word1 = self.word2 = self.word3 = self.word4 = ""
        
        # Build GUI and start
        self._build_gui()
        self.video_loop()
    
    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1300x800")
        
        tk.Label(self.root, text="Sign Language To Text Conversion", 
                 font=("Courier", 28, "bold")).place(x=60, y=10)
        
        self.panel_camera = tk.Label(self.root, bg='black')
        self.panel_camera.place(x=50, y=60, width=580, height=500)
        
        self.panel_hand = tk.Label(self.root, bg='gray')
        self.panel_hand.place(x=700, y=120, width=400, height=400)

        # Hand type indicator (Left/Right)
        self.label_hand_type = tk.Label(self.root, text="", font=("Courier", 14), fg='green')
        self.label_hand_type.place(x=700, y=95)

        # Non-manual marker indicator
        self.label_nmm = tk.Label(self.root, text="", font=("Courier", 12), fg='darkgreen')
        self.label_nmm.place(x=700, y=70)
        if not self.nmm_detector:
            self.label_nmm.configure(text="NMM: unavailable")
        
        tk.Label(self.root, text="Character:", font=("Courier", 24, "bold")).place(x=50, y=580)
        self.label_char = tk.Label(self.root, text="", font=("Courier", 28, "bold"), fg='blue')
        self.label_char.place(x=250, y=580)
        
        # Cooldown indicator
        self.label_status = tk.Label(self.root, text="Ready", font=("Courier", 14), fg='green')
        self.label_status.place(x=350, y=588)
        
        tk.Label(self.root, text="Sentence:", font=("Courier", 24, "bold")).place(x=50, y=630)
        # Editable text entry - user can click and edit the text
        self.sentence_var = tk.StringVar()
        self.entry_sentence = tk.Entry(self.root, textvariable=self.sentence_var, 
                                       font=("Courier", 18), bg='white', width=55)
        self.entry_sentence.place(x=250, y=632, height=35)
        
        tk.Label(self.root, text="Suggestions:", font=("Courier", 22, "bold"), fg='red').place(x=50, y=690)
        
        self.btn1 = tk.Button(self.root, text="Word1", font=("Courier", 16), command=lambda: self._use_suggestion(0))
        self.btn1.place(x=280, y=685, width=130, height=45)
        self.btn2 = tk.Button(self.root, text="Word2", font=("Courier", 16), command=lambda: self._use_suggestion(1))
        self.btn2.place(x=430, y=685, width=130, height=45)
        self.btn3 = tk.Button(self.root, text="Word3", font=("Courier", 16), command=lambda: self._use_suggestion(2))
        self.btn3.place(x=580, y=685, width=130, height=45)
        self.btn4 = tk.Button(self.root, text="Word4", font=("Courier", 16), command=lambda: self._use_suggestion(3))
        self.btn4.place(x=730, y=685, width=130, height=45)
        
        tk.Button(self.root, text="Speak", font=("Courier", 18, "bold"), bg='#4CAF50', fg='white',
                  command=self.speak_text).place(x=1000, y=685, width=120, height=45)
        tk.Button(self.root, text="Clear", font=("Courier", 18, "bold"), bg='#f44336', fg='white',
                  command=self.clear_text).place(x=1140, y=685, width=120, height=45)

    def _extract_hands(self, hands_result):
        if not hands_result:
            return []
        if isinstance(hands_result, tuple):
            for item in hands_result:
                if isinstance(item, list):
                    return item
                if isinstance(item, dict):
                    return [item]
            return []
        if isinstance(hands_result, list):
            return hands_result
        if isinstance(hands_result, dict):
            return [hands_result]
        return []

    def _select_hand(self, hands):
        for hand in hands:
            if isinstance(hand, dict) and 'bbox' in hand:
                return hand
        return None

    def _landmarks_to_crop(self, lm_list, x1, y1, crop_w, flip_horizontal):
        converted = []
        for lm in lm_list:
            x = lm[0] - x1
            y = lm[1] - y1
            if flip_horizontal:
                x = crop_w - x
            x = int(round(x))
            y = int(round(y))
            if len(lm) >= 3:
                converted.append([x, y, lm[2]])
            else:
                converted.append([x, y])
        return converted
    
    def _flip_landmarks_horizontal(self, pts, img_width):
        """Flip landmarks horizontally for left hand -> right hand conversion."""
        flipped = []
        for pt in pts:
            # Flip x coordinate, keep y (and z if present)
            if len(pt) >= 3:
                flipped.append([img_width - pt[0], pt[1], pt[2]])
            else:
                flipped.append([img_width - pt[0], pt[1]])
        return flipped
    
    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok:
                self.root.after(30, self.video_loop)
                return
            
            # Flip for mirror
            frame = cv2.flip(frame, 1)
            frame_copy = np.array(frame)
            
            # Display camera feed
            self._show_image(self.panel_camera, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Non-manual markers from face mesh
            if self.nmm_detector:
                markers = self.nmm_detector.detect(frame)
                self._process_nmm(markers)
            
            # Decrement cooldown counter
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                self.label_status.configure(text=f"Cooldown: {self.cooldown_counter}", fg='orange')
            else:
                self.label_status.configure(text="Ready", fg='green')
            
            if self.hd:
                # Find hands in FULL frame using FIRST detector
                hands_raw = self.hd.findHands(frame, draw=False, flipType=True)
                hands = self._extract_hands(hands_raw)
                hand_info = self._select_hand(hands)
                
                if hand_info:
                    x, y, w, h = hand_info['bbox']
                    
                    # Check hand type - IMPORTANT: Because we flipped the frame for mirror display,
                    # cvzone's detection is INVERTED from the user's perspective:
                    # - cvzone "Left" = user's actual RIGHT hand (no flip needed)
                    # - cvzone "Right" = user's actual LEFT hand (needs flip)
                    cvzone_type = hand_info.get('type', 'Right')
                    
                    # Invert to get the user's ACTUAL hand
                    actual_hand = "Right" if cvzone_type == "Left" else "Left"
                    is_actual_left_hand = (actual_hand == "Left")
                    
                    self.label_hand_type.configure(text=f"Your {actual_hand} Hand")
                    
                    # Crop hand region from the COPY
                    y1, y2 = max(0, y - OFFSET), min(frame_copy.shape[0], y + h + OFFSET)
                    x1, x2 = max(0, x - OFFSET), min(frame_copy.shape[1], x + w + OFFSET)
                    hand_crop = frame_copy[y1:y2, x1:x2]
                    
                    if hand_crop.size > 0:
                        # If user's actual LEFT hand, flip the crop to make it look like RIGHT hand
                        # (The model was trained on right hands)
                        if is_actual_left_hand:
                            hand_crop = cv2.flip(hand_crop, 1)  # Horizontal flip
                        
                        crop_w = hand_crop.shape[1]
                        pts = None
                        if 'lmList' in hand_info and hand_info['lmList']:
                            pts = self._landmarks_to_crop(
                                hand_info['lmList'],
                                x1,
                                y1,
                                crop_w,
                                is_actual_left_hand,
                            )
                        
                        # Create white background
                        white = np.ones((WHITE_BG_SIZE, WHITE_BG_SIZE, 3), dtype=np.uint8) * 255
                        
                        if pts is None and self.hd2:
                            # Find landmarks in CROPPED image using SECOND detector
                            handz_raw = self.hd2.findHands(hand_crop, draw=False, flipType=True)
                            handz = self._extract_hands(handz_raw)
                            for item in handz:
                                if isinstance(item, dict) and 'lmList' in item:
                                    pts = item['lmList']
                                    break
                        
                        if pts:
                            self.pts = pts
                            
                            # Calculate offset to center on white background
                            os_x = ((WHITE_BG_SIZE - w) // 2) - 15
                            os_y = ((WHITE_BG_SIZE - h) // 2) - 15
                            
                            # Draw skeleton on white background
                            self._draw_skeleton(white, self.pts, os_x, os_y)
                            
                            # Show white background with skeleton
                            self._show_image(self.panel_hand, cv2.cvtColor(white, cv2.COLOR_BGR2RGB))
                            
                            # Make prediction (only if not in cooldown)
                            if self.classifier and self.cooldown_counter == 0:
                                result = self.classifier.predict(white, self.pts)
                                if result:
                                    self._handle_prediction(result)
                            
                            self.blank_count = 0
                        else:
                            # No landmarks - show cropped hand
                            resized = cv2.resize(hand_crop, (WHITE_BG_SIZE, WHITE_BG_SIZE))
                            self._show_image(self.panel_hand, cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                else:
                    # No hand detected
                    self.label_hand_type.configure(text="No hand")
                    self.blank_count += 1
                    if self.blank_count > 15:
                        self._handle_blank()
                        self.blank_count = 0
        
        except Exception as e:
            print(f"Video loop error: {e}")
            traceback.print_exc()
        
        self.root.after(10, self.video_loop)
    
    def _draw_skeleton(self, img, pts, os_x, os_y):
        """Draw hand skeleton on image."""
        # Finger connections (same as original)
        for t in range(0, 4):
            cv2.line(img, (pts[t][0]+os_x, pts[t][1]+os_y), 
                    (pts[t+1][0]+os_x, pts[t+1][1]+os_y), (0, 255, 0), 3)
        for t in range(5, 8):
            cv2.line(img, (pts[t][0]+os_x, pts[t][1]+os_y), 
                    (pts[t+1][0]+os_x, pts[t+1][1]+os_y), (0, 255, 0), 3)
        for t in range(9, 12):
            cv2.line(img, (pts[t][0]+os_x, pts[t][1]+os_y), 
                    (pts[t+1][0]+os_x, pts[t+1][1]+os_y), (0, 255, 0), 3)
        for t in range(13, 16):
            cv2.line(img, (pts[t][0]+os_x, pts[t][1]+os_y), 
                    (pts[t+1][0]+os_x, pts[t+1][1]+os_y), (0, 255, 0), 3)
        for t in range(17, 20):
            cv2.line(img, (pts[t][0]+os_x, pts[t][1]+os_y), 
                    (pts[t+1][0]+os_x, pts[t+1][1]+os_y), (0, 255, 0), 3)
        # Palm connections
        cv2.line(img, (pts[5][0]+os_x, pts[5][1]+os_y), (pts[9][0]+os_x, pts[9][1]+os_y), (0, 255, 0), 3)
        cv2.line(img, (pts[9][0]+os_x, pts[9][1]+os_y), (pts[13][0]+os_x, pts[13][1]+os_y), (0, 255, 0), 3)
        cv2.line(img, (pts[13][0]+os_x, pts[13][1]+os_y), (pts[17][0]+os_x, pts[17][1]+os_y), (0, 255, 0), 3)
        cv2.line(img, (pts[0][0]+os_x, pts[0][1]+os_y), (pts[5][0]+os_x, pts[5][1]+os_y), (0, 255, 0), 3)
        cv2.line(img, (pts[0][0]+os_x, pts[0][1]+os_y), (pts[17][0]+os_x, pts[17][1]+os_y), (0, 255, 0), 3)
    
    def _show_image(self, panel, rgb_img):
        """Display RGB image on panel."""
        try:
            img = Image.fromarray(rgb_img)
            imgtk = ImageTk.PhotoImage(image=img)
            panel.imgtk = imgtk
            panel.configure(image=imgtk)
        except:
            pass
    
    def _handle_prediction(self, result):
        """Handle prediction with stability and cooldown."""
        self.current_symbol = result
        self.label_char.configure(text=result)
        
        if result == self.prev_char:
            self.same_char_count += 1
        else:
            self.same_char_count = 1
            self.prev_char = result
        
        # Add character after stable predictions AND if not in cooldown
        if self.same_char_count >= STABLE_COUNT_REQUIRED and self.cooldown_counter == 0:
            # Sync from Entry in case user edited it manually
            self.current_text = self.sentence_var.get()
            self.current_text += result
            self.sentence_var.set(self.current_text)
            self._update_suggestions()
            
            # Move cursor to end of Entry
            self.entry_sentence.icursor(tk.END)
            
            # Start cooldown to prevent rapid repeated characters
            self.cooldown_counter = COOLDOWN_FRAMES
            self.same_char_count = 0
            self.prev_char = ""  # Reset so next different char can be detected
    
    def _handle_blank(self):
        """Handle no hand - add space."""
        # Sync from Entry in case user edited it manually
        self.current_text = self.sentence_var.get()

        self.current_text, _ = self._apply_pending_nmm_tags(self.current_text)
        if self.current_text and not self.current_text.endswith(" "):
            self.current_text += " "
        if self.current_text:
            self.sentence_var.set(self.current_text)
            self._update_suggestions()
        
        self.prev_char = ""
        self.same_char_count = 0
        self.cooldown_counter = 0  # Reset cooldown when hand is removed
    
    def _update_suggestions(self):
        suggestions = self.suggestion_engine.get_suggestions(self.current_text)
        self.word1 = suggestions[0] if len(suggestions) > 0 else ""
        self.word2 = suggestions[1] if len(suggestions) > 1 else ""
        self.word3 = suggestions[2] if len(suggestions) > 2 else ""
        self.word4 = suggestions[3] if len(suggestions) > 3 else ""
        self.btn1.configure(text=self.word1 or "Word1")
        self.btn2.configure(text=self.word2 or "Word2")
        self.btn3.configure(text=self.word3 or "Word3")
        self.btn4.configure(text=self.word4 or "Word4")
    
    def _use_suggestion(self, idx):
        words = [self.word1, self.word2, self.word3, self.word4]
        if idx < len(words) and words[idx]:
            # Sync from Entry in case user edited it
            self.current_text = self.sentence_var.get()
            
            last_space = self.current_text.rfind(" ")
            if last_space >= 0:
                self.current_text = self.current_text[:last_space+1] + words[idx] + " "
            else:
                self.current_text = words[idx] + " "
            
            self.current_text, _ = self._apply_pending_nmm_tags(self.current_text)
            self.sentence_var.set(self.current_text)
            self._update_suggestions()
            
            # Move cursor to end of Entry
            self.entry_sentence.icursor(tk.END)
    
    def speak_text(self):
        """Speak the current text using TTS (reads from Entry in case user edited it)."""
        # Get text from Entry widget (user may have edited it)
        text_to_speak = self.sentence_var.get()
        # Sync our internal state with what's in the Entry
        self.current_text = text_to_speak
        print(f"Speak button clicked. Text: '{text_to_speak}'")
        self.tts.speak(text_to_speak)
    
    def clear_text(self):
        self.current_text = ""
        self.current_symbol = ""
        self.prev_char = ""
        self.same_char_count = 0
        self.cooldown_counter = 0
        self.word1 = self.word2 = self.word3 = self.word4 = ""
        self.active_nmm_tags = []
        self.active_nmm_applied = False
        self.pending_nmm_tags = []
        self.label_char.configure(text="")
        self.sentence_var.set("")
        self.label_status.configure(text="Ready", fg='green')
        self._update_suggestions()

    def _apply_pending_nmm_tags(self, text):
        tags_to_apply = []
        if self.active_nmm_tags and not self.active_nmm_applied:
            tags_to_apply.extend(self.active_nmm_tags)
            self.active_nmm_applied = True
        if self.pending_nmm_tags:
            tags_to_apply.extend(self.pending_nmm_tags)
            self.pending_nmm_tags = []
        if not tags_to_apply:
            return text, False
        updated = text.rstrip(" ")
        if updated:
            updated += " "
        updated += " ".join(f"[{tag}]" for tag in tags_to_apply)
        updated += " "
        return updated, True

    def _process_nmm(self, markers):
        if not markers:
            self.nmm_missing_frames += 1
            if self.nmm_missing_frames >= NMM_MISSING_RESET:
                self.nmm_brow_state = "neutral"
                self.nmm_brow_state_raw = "neutral"
                self.nmm_brow_count = 0
                self.nmm_mouth_state = "neutral"
                self.nmm_mouth_state_raw = "neutral"
                self.nmm_mouth_count = 0
                self.nmm_head_tilt = "neutral"
                self.active_nmm_tags = []
                self.active_nmm_applied = False
            self._update_nmm_label()
            return

        self.nmm_missing_frames = 0
        brow_state = markers["brow_state"]
        if brow_state == self.nmm_brow_state_raw:
            self.nmm_brow_count += 1
        else:
            self.nmm_brow_state_raw = brow_state
            self.nmm_brow_count = 1
        if self.nmm_brow_count >= NMM_STABLE_FRAMES:
            self.nmm_brow_state = brow_state

        mouth_state = markers["mouth_state"]
        if mouth_state == self.nmm_mouth_state_raw:
            self.nmm_mouth_count += 1
        else:
            self.nmm_mouth_state_raw = mouth_state
            self.nmm_mouth_count = 1
        if self.nmm_mouth_count >= NMM_STABLE_FRAMES:
            self.nmm_mouth_state = mouth_state

        self.nmm_head_tilt = markers["head_tilt"]

        new_active_tags = []
        if self.nmm_brow_state == "raise":
            new_active_tags.append("Q-YN")
        elif self.nmm_brow_state == "furrow":
            new_active_tags.append("Q-WH")
        if self.nmm_mouth_state == "open":
            new_active_tags.append("MOUTH-OPEN")
        if new_active_tags != self.active_nmm_tags:
            self.active_nmm_tags = new_active_tags
            self.active_nmm_applied = False

        if markers["head_shake"]:
            self.pending_nmm_tags.append("NEG")
            self.nmm_event_display = "HEAD_SHAKE"
            self.nmm_event_display_frames = 10
        if markers["head_nod"]:
            self.pending_nmm_tags.append("AFFIRM")
            self.nmm_event_display = "HEAD_NOD"
            self.nmm_event_display_frames = 10

        self._update_nmm_label()

    def _update_nmm_label(self):
        if not hasattr(self, "label_nmm"):
            return
        if not self.nmm_detector:
            self.label_nmm.configure(text="NMM: unavailable")
            return
        parts = []
        if self.nmm_brow_state == "raise":
            parts.append("BROW_RAISE")
        elif self.nmm_brow_state == "furrow":
            parts.append("BROW_FURROW")
        if self.nmm_mouth_state == "open":
            parts.append("MOUTH_OPEN")
        if self.nmm_head_tilt != "neutral":
            parts.append(f"TILT_{self.nmm_head_tilt.upper()}")
        if self.nmm_event_display_frames > 0:
            parts.append(self.nmm_event_display)
            self.nmm_event_display_frames -= 1
        text = "NMM: " + (" ".join(parts) if parts else "neutral")
        self.label_nmm.configure(text=text)

    def destructor(self):
        print("Closing Application...")
        try:
            self.vs.release()
            cv2.destroyAllWindows()
        except:
            pass
        self.root.destroy()
    
    def run(self):
        print("Starting Application...")
        self.root.mainloop()


if __name__ == "__main__":
    app = SignLanguageApp()
    app.run()
