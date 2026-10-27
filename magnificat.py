"""MAGNIFICAT - Sanctuaire de surveillance baroque (spectacle lacte/saint Francois).

Base : architecture de nude_criticism.py (qui marche), enrichie :
- OpenNSFW : indice de lubricite en direct (lazy, comme NudeNet)
- YOLO-World : vaches, pis, biberons (relique) + slider de biais inquisitorial
- MediaPipe : hallelujia si main gauche levee avec 2 doigts (lazy)
- Halo SANCTA autour des tetes
- Jauge lactee

Design : eglise catholique baroque (or, pourpre).
pygame : charge uniquement au moment du hallelujia (jamais a l'import).
"""

import os
import tempfile
import time

import cv2
import numpy as np
import tkinter as tk

try:
    import opennsfw2
    NSFW_OK = True
except Exception:
    opennsfw2, NSFW_OK = None, False

try:
    from ultralytics import YOLOWorld
    YOLO_OK = True
except Exception:
    YOLOWorld, YOLO_OK = None, False

try:
    import mediapipe as mp
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
        mp, MP_OK = None, False
    else:
        MP_OK = True
except Exception:
    mp, MP_OK = None, False

GOLD = (30, 175, 235)          # BGR or baroque
DETECT_INTERVAL = 1.0
NSFW_INTERVAL = 2.0
YOLO_INTERVAL = 1.5
HALLELUJAH_COOLDOWN = 6.0
FONT = cv2.FONT_HERSHEY_SIMPLEX
TMP_DIR = tempfile.gettempdir()
NSFW_TMP = os.path.join(TMP_DIR, "magnificat_frame.jpg")
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
YOLOR_CLASSES = ["cow", "udder", "baby bottle"]


def draw_text_bg(frame, text, org, scale, color, thickness=1, bg=(0, 0, 0)):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 3, y - th - 3), (x + tw + 3, y + base + 3), bg, -1)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness)


def play_hallelujah():
    """pygame initialise uniquement ici, jamais a l'import."""
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
    except Exception:
        print("HALLELUJIA ! (audio indisponible)")
        return
    try:
        mp3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "hallelujah.mp3")
        if os.path.exists(mp3):
            pygame.mixer.music.load(mp3)
            pygame.mixer.music.play()
            return
    except Exception:
        pass
    try:
        sr = 44100
        t = np.linspace(0, 0.9, int(sr * 0.9), endpoint=False)
        wave = sum(0.22 * np.sin(2 * np.pi * f * t) * np.exp(-2.2 * t)
                   for f in (523.25, 659.25, 783.99, 1046.5))
        wave = (wave * 32767 * 0.8).astype(np.int16)
        stereo = np.ascontiguousarray(np.column_stack([wave, wave]))
        pygame.sndarray.make_sound(stereo).play()
    except Exception:
        print("HALLELUJIA !")


def fingers_up(hand):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return sum(1 for tip, pip in zip(tips, pips)
               if hand.landmark[tip].y < hand.landmark[pip].y - 0.03)


class App:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.window = "MAGNIFICAT - Sanctuaire de Surveillance"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 960, 540)

        self.features = {"nsfw": True, "vaches": True, "musique": True,
                         "biberon": True, "halo": True, "jauge": True}
        self.bias = 0.0
        self.milk = 0.0

        self.nsfw_detector = None          # lazy, comme NudeNet dans nude_criticism
        self.nsfw_score = 0.0
        self.last_nsfw = 0.0
        self.yolo_model = None
        self.yolo_dets = []
        self.last_yolo = 0.0
        self.hands_detector = None
        self.last_hallelujah = 0.0
        self.running = False
        self.gui = Gui(self)
        if NSFW_OK:
            self.gui.set_nsfw_state("chargement au 1er appel...")
        else:
            self.gui.set_nsfw_state("absent (pip install opennsfw2)")

    # ---- modeles paresseux (pattern nude_criticism) ----

    def get_nsfw(self):
        if self.nsfw_detector is None and NSFW_OK:
            self.nsfw_detector = True
            self.gui.set_nsfw_state("ACTIF")
        return self.nsfw_detector if self.nsfw_detector else None

    def get_yolo(self):
        if self.yolo_model is None and YOLO_OK:
            self.gui.set_yolo_state("CHARGEMENT du modele...")
            try:
                model = YOLOWorld("yolov8s-worldv2")
                model.set_classes(YOLOR_CLASSES)
                self.yolo_model = model
                self.gui.set_yolo_state("ACTIF")
            except Exception as exc:
                self.yolo_model = False
                self.gui.set_yolo_state(f"ERREUR : {exc}")
        return self.yolo_model if self.yolo_model else None

    def get_hands(self):
        if self.hands_detector is None and MP_OK:
            try:
                self.hands_detector = mp.solutions.hands.Hands(
                    max_num_hands=2, min_detection_confidence=0.6)
                self.gui.set_hands_state("ACTIF")
            except Exception as exc:
                self.hands_detector = False
                self.gui.set_hands_state(f"ERREUR : {exc}")
        return self.hands_detector if self.hands_detector else None

    # ---- analyses ----

    def analyze_nsfw(self, frame):
        if not self.get_nsfw():
            return self.nsfw_score
        try:
            cv2.imwrite(NSFW_TMP, frame)
            return float(opennsfw2.predict_image(NSFW_TMP))
        except Exception as exc:
            self.gui.set_nsfw_state(f"ERREUR : {exc}")
            return 0.0

    def analyze_yolo(self, frame):
        model = self.get_yolo()
        if model is None:
            return []
        try:
            thresh = max(0.02, 0.5 * (1.0 - self.bias))
            res = model.predict(frame, conf=thresh, verbose=False)[0]
            dets = []
            for b in res.boxes:
                cls = res.names[int(b.cls)]
                conf = min(1.0, float(b.conf) * (1.0 + 0.6 * self.bias))
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                dets.append((cls, conf, (x1, y1, x2 - x1, y2 - y1)))
            return dets
        except Exception as exc:
            self.gui.set_yolo_state(f"ERREUR : {exc}")
            return []

    def analyze_hands(self, frame, now):
        hands = self.get_hands()
        if hands is None:
            return False
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            if not res.multi_hand_landmarks:
                return False
            for hand, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = handed.classification[0].label
                wrist_y = hand.landmark[0].y
                if label == "Right" and fingers_up(hand) == 2 and wrist_y < 0.45 \
                        and now - self.last_hallelujah > HALLELUJAH_COOLDOWN:
                    self.last_hallelujah = now
                    return True
        except Exception as exc:
            self.gui.set_hands_state(f"ERREUR : {exc}")
        return False

    # ---- dessins ----

    def draw_yolo(self, frame):
        for cls, conf, (x, y, w, h) in self.yolo_dets:
            if cls == "cow":
                color, txt = (200, 150, 255), f"VACHE {conf:.0%}"
            elif cls == "udder":
                color, txt = (255, 220, 240), f"PIS {conf:.0%}"
            elif cls == "baby bottle" and self.features["biberon"]:
                color, txt = (60, 215, 245), f"RELIQUE {conf:.0%}"
            else:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            draw_text_bg(frame, txt, (x, max(20, y - 8)), 0.5, color, 1)

    def draw_nsfw(self, frame):
        s = self.nsfw_score
        color = GOLD if s < 0.2 else (0, 165, 255) if s < 0.5 else (0, 0, 255)
        draw_text_bg(frame, f"INDICE DE LUBRICITE : {s:.1%}", (10, 30), 0.6, color, 2)
        bar_w = int(300 * min(1.0, s))
        cv2.rectangle(frame, (10, 40), (310, 52), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, 40), (10 + bar_w, 52), color, -1)

    def draw_halo(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(70, 70))
        for (x, y, w, h) in faces:
            cx = x + w // 2
            cy = max(14, y - int(0.18 * h))
            r = max(24, w // 2)
            cv2.ellipse(frame, (cx, cy), (r, int(r * 0.38)), 0, 0, 360, GOLD, 3)
            cv2.ellipse(frame, (cx, cy), (r + 6, int(r * 0.45)), 0, 0, 360, (10, 120, 170), 1)
            draw_text_bg(frame, "SANCTA", (max(4, cx - r), max(16, cy - r // 2 - 6)), 0.5, GOLD, 1)

    def draw_milk(self, frame):
        self.milk = max(0.0, min(100.0, self.milk - 0.01))
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = w - 90, h - 220, w - 40, h - 40
        cv2.rectangle(frame, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), GOLD, 2)
        fill = int((y1 - y0) * self.milk / 100.0)
        cv2.rectangle(frame, (x0, y1 - fill), (x1, y1), (235, 250, 255), -1)
        for i in range(1, 4):
            yy = y0 + (y1 - y0) * i // 4
            cv2.line(frame, (x0, yy), (x0 + 10, yy), GOLD, 1)
        draw_text_bg(frame, f"LAIT {self.milk:.0f}%", (x0 - 5, y0 - 14), 0.5, GOLD, 1)

    def draw_baroque(self, frame):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 34), (20, 12, 60), -1)
        cv2.rectangle(frame, (0, 0), (w, 34), GOLD, 1)
        draw_text_bg(frame, "+ MAGNIFICAT +  SANCTUAIRE DE SURVEILLANCE LACTEE  +",
                     (w // 2 - 285, 24), 0.55, GOLD, 1, bg=(20, 12, 60))
        cv2.rectangle(frame, (0, h - 3), (w, h), GOLD, 2)

    # ---- boucle (meme structure que nude_criticism) ----

    def tick(self):
        ret, frame = self.cap.read()
        now = time.time()
        if not ret:
            self.gui.set_nsfw_state("Flux webcam interrompu.")
            self.quit()
            return

        # analyses cadencees (comme analyze() dans nude_criticism)
        if self.features["nsfw"] and now - self.last_nsfw >= NSFW_INTERVAL:
            self.last_nsfw = now
            self.nsfw_score = self.analyze_nsfw(frame)
        if (self.features["vaches"] or self.features["biberon"]) and now - self.last_yolo >= YOLO_INTERVAL:
            self.last_yolo = now
            self.yolo_dets = self.analyze_yolo(frame)
        if self.features["musique"] and self.analyze_hands(frame, now):
            play_hallelujah()
            self.milk = min(100.0, self.milk + 5)

        # dessins (toujours, meme pendant les chargements)
        if self.features["nsfw"] and NSFW_OK:
            self.draw_nsfw(frame)
        if self.features["vaches"] or self.features["biberon"]:
            self.draw_yolo(frame)
            if self.features["jauge"] and any(c == "udder" for c, _, _ in self.yolo_dets):
                self.milk = min(100.0, self.milk + 0.02)
        if self.features["halo"]:
            self.draw_halo(frame)
        if self.features["jauge"]:
            self.draw_milk(frame)
        self.draw_baroque(frame)
        cv2.imshow(self.window, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.quit()
            return
        if self.running:
            self.gui.root.after(33, self.tick)

    def run(self):
        if not self.cap.isOpened():
            print("Erreur : webcam inaccessible.")
            self.cap.release()
            return
        self.running = True
        self.gui.root.after(33, self.tick)
        self.gui.root.mainloop()

    def quit(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.gui.root.destroy()


class Gui:
    """Sacristie : interrupteurs des miracles, etat des modules, slider de biais."""

    def __init__(self, app):
        self.app = app
        self.root = tk.Tk()
        self.root.title("MAGNIFICAT - Sacristie de Controle")
        self.root.geometry("460x640")
        BG = "#1a0a2e"
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", app.quit)

        for col in ("#d4af37", "#5c0a1e", "#d4af37"):
            tk.Label(self.root, bg=col, height=1).pack(fill=tk.X)
        tk.Label(self.root, text="+  M A G N I F I C A T  +", bg=BG, fg="#d4af37",
                 font=("Times", 17, "bold")).pack(pady=(5, 0))
        tk.Label(self.root, text="OFFICE DE SURVEILLANCE - ARCHIDIOCESE TECHNOLOGIQUE",
                 bg=BG, fg="#c09090", font=("Times", 9)).pack()

        etat_frame = tk.Frame(self.root, bg="#12041f", relief="sunken")
        etat_frame.pack(fill=tk.X, padx=10, pady=6)
        self.nsfw_lbl = tk.Label(etat_frame, text=f"{'OpenNSFW  '}: {'chargement...' if NSFW_OK else 'absent (pip install opennsfw2)'}",
                                bg="#12041f", fg="#e8d8b0", font=("Courier", 8), anchor="w")
        self.nsfw_lbl.pack(fill=tk.X, padx=6, pady=1)
        self.yolo_lbl = tk.Label(etat_frame, text=f"{'YOLO-World'}: {'chargement...' if YOLO_OK else 'absent (pip install ultralytics)'}",
                                bg="#12041f", fg="#e8d8b0", font=("Courier", 8), anchor="w")
        self.yolo_lbl.pack(fill=tk.X, padx=6, pady=1)
        self.hands_lbl = tk.Label(etat_frame, text=f"{'MediaPipe '}: {'chargement...' if MP_OK else 'API incompatible (pip install mediapipe==0.10.14)'}",
                                 bg="#12041f", fg="#e8d8b0", font=("Courier", 8), anchor="w")
        self.hands_lbl.pack(fill=tk.X, padx=6, pady=1)

        tk.Label(self.root, text="MIRACLES ACTIVABLES :", bg=BG, fg="#d4af37",
                 font=("Times", 11, "bold")).pack(anchor="w", padx=12)
        self.vars = {}
        for key, label in [
            ("nsfw", "Indice de lubricite (OpenNSFW)"),
            ("vaches", "Vaches et pis sacres (YOLO)"),
            ("musique", "Hallelujia - main gauche, 2 doigts"),
            ("biberon", "Relique (biberon)"),
            ("halo", "Halo SANCTA"),
            ("jauge", "Jauge lactee"),
        ]:
            var = tk.BooleanVar(value=app.features[key])
            self.vars[key] = var
            tk.Checkbutton(self.root, text=label, variable=var, bg=BG, fg="#f0e6d0",
                           activebackground="#2a0a3e", activeforeground="#d4af37",
                           selectcolor="#12041f", font=("Times", 11),
                           command=self._toggle).pack(anchor="w", padx=24, pady=1)

        tk.Label(self.root, text="BIAIS INQUISITORIAL DU MODELE VACHES :",
                 bg=BG, fg="#d4af37", font=("Times", 10, "bold")).pack(anchor="w", padx=12, pady=(8, 0))
        self.bias_var = tk.DoubleVar(value=0.0)
        tk.Scale(self.root, variable=self.bias_var, from_=0, to=100, orient=tk.HORIZONTAL,
                 bg=BG, fg="#d4af37", troughcolor="#12041f", highlightthickness=0,
                 font=("Times", 9), length=380,
                 command=lambda v: setattr(app, "bias", float(v) / 100.0)).pack(padx=16)
        tk.Label(self.root, text="0 : le modele ne voit que la verite.\n"
                                 "100 : le modele voit des vaches partout (foi renforcee).",
                 bg=BG, fg="#c09090", font=("Times", 8)).pack()
        tk.Label(self.root, text="GLOIRE AU LAIT ETERNEL - (C) 2028 ARCHIDIOCESE",
                 bg=BG, fg="#8a6a3a", font=("Times", 8)).pack(side=tk.BOTTOM, pady=6)

    def _toggle(self):
        for key, var in self.vars.items():
            self.app.features[key] = var.get()

    def _set_state(self, lbl, txt):
        lbl.config(text=lbl.cget("text").split(":")[0] + f": {txt}",
                   fg="#00ff9d" if txt == "ACTIF" else "#e8a0a0")

    def set_nsfw_state(self, txt):
        self._set_state(self.nsfw_lbl, txt)

    def set_yolo_state(self, txt):
        self._set_state(self.yolo_lbl, txt)

    def set_hands_state(self, txt):
        self._set_state(self.hands_lbl, txt)


if __name__ == "__main__":
    app = App()
    app.run()
