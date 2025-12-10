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
from string import ascii_uppercase
from typing import List

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Optional imports with graceful fallback
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
            self.hd = HandDetector(maxHands=1, detectionCon=0.8)   # For full frame
            self.hd2 = HandDetector(maxHands=1, detectionCon=0.8)  # For cropped image
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
            
            # Decrement cooldown counter
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                self.label_status.configure(text=f"Cooldown: {self.cooldown_counter}", fg='orange')
            else:
                self.label_status.configure(text="Ready", fg='green')
            
            if self.hd:
                # Find hands in FULL frame using FIRST detector
                hands = self.hd.findHands(frame, draw=False, flipType=True)
                
                # Handle cvzone return format
                if isinstance(hands, tuple):
                    hands = hands[0] if hands[0] else []
                hands = hands or []
                
                if hands and 'bbox' in hands[0]:
                    hand_info = hands[0]
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
                        
                        # Create white background
                        white = np.ones((WHITE_BG_SIZE, WHITE_BG_SIZE, 3), dtype=np.uint8) * 255
                        
                        # Find landmarks in CROPPED image using SECOND detector
                        handz = self.hd2.findHands(hand_crop, draw=False, flipType=True)
                        
                        if isinstance(handz, tuple):
                            handz = handz[0] if handz[0] else []
                        handz = handz or []
                        
                        if handz and 'lmList' in handz[0]:
                            self.pts = handz[0]['lmList']
                            
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
        
        if self.current_text and not self.current_text.endswith(" "):
            self.current_text += " "
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
        self.label_char.configure(text="")
        self.sentence_var.set("")
        self.label_status.configure(text="Ready", fg='green')
        self._update_suggestions()

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
