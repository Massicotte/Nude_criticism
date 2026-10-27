"""MAGNIFICAT - Sanctuaire de surveillance baroque (spectacle lacte/saint Francois).

Fonctionnalites (chacune activable/desactivable depuis la Sacristie) :
- OpenNSFW : indice de lubricite en direct (via fichier temp, comme NudeNet)
- YOLO-World : vaches, pis, biberons (relique) + slider de biais inquisitorial
- MediaPipe (API solutions) : hallelujia si main gauche levee avec 2 doigts
- Halo SANCTA autour des tetes
- Jauge lactee

Chaque module est independant : s'il plante, les autres continuent.
Design : eglise catholique baroque (or, pourpre).
"""

import os
import tempfile
import time

import cv2
import numpy as np
import tkinter as tk

# ------------------ imports optionnels ------------------

try:
    import opennsfw2
    NSFW_OK = True
except Exception:
    opennsfw2 = None
    NSFW_OK = False

try:
    from ultralytics import YOLOWorld
    YOLO_OK = True
except Exception:
    YOLOWorld = None
    YOLO_OK = False

try:
    import mediapipe as mp
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
        mp = None
        MP_OK = False
    else:
        MP_OK = True
except Exception:
    mp = None
    MP_OK = False

try:
    import pygame
    pygame.mixer.init()
    PYGAME_OK = True
except Exception:
    pygame = None
    PYGAME_OK = False

# ------------------ constantes ------------------

GOLD = (30, 175, 235)          # BGR
NSFW_INTERVAL = 2.0
YOLO_INTERVAL = 1.5
HALLELUJAH_COOLDOWN = 6.0
FONT = cv2.FONT_HERSHEY_SIMPLEX
TMP_DIR = tempfile.gettempdir()
NSFW_TMP = os.path.join(TMP_DIR, "magnificat_frame.jpg")
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def draw_text_bg(frame, text, org, scale, color, thickness=1, bg=(0, 0, 0)):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 3, y - th - 3), (x + tw + 3, y + base + 3), bg, -1)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness)


def play_hallelujah():
    if not PYGAME_OK:
        print("HALLELUJIA !")
        return
    mp3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "hallelujah.mp3")
    try:
        if os.path.exists(mp3):
            pygame.mixer.music.load(mp3)
            pygame.mixer.music.play()
            return
    except Exception:
        pass
    # accord de Do majeur synthetise
    sr = 44100
    t = np.linspace(0, 0.9, int(sr * 0.9), endpoint=False)
    wave = sum(0.22 * np.sin(2 * np.pi * f * t) * np.exp(-2.2 * t)
               for f in (523.25, 659.25, 783.99, 1046.5))
    wave = (wave * 32767 * 0.8).astype(np.int16)
    stereo = np.ascontiguousarray(np.column_stack([wave, wave]))
    pygame.sndarray.make_sound(stereo).play()


def fingers_up(hand):
    """Doigts longs tendus : pointe plus haute que l'articulation."""
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return sum(1 for tip, pip in zip(tips, pips)
               if hand.landmark[tip].y < hand.landmark[pip].y - 0.03)


# ------------------ modules ------------------

class NsfwModule:
    """Indice de lubricite : frame -> fichier temp -> opennsfw2."""

    def __init__(self):
        self.score = 0.0
        self.last = 0.0
        self.error = None
        self.active = NSFW_OK

    def step(self, frame, now):
        if not self.active or now - self.last < NSFW_INTERVAL:
            return
        self.last = now
        try:
            cv2.imwrite(NSFW_TMP, frame)
            self.score = float(opennsfw2.predict_image(NSFW_TMP))
            self.error = None
        except Exception as exc:
            self.error = str(exc)
            self.score = 0.0

    def draw(self, frame):
        if not self.active:
            draw_text_bg(frame, "OPENNSFW ABSENT (pip install opennsfw2)", (10, 30), 0.5, (150, 150, 150), 1)
            return
        s = self.score
        color = GOLD if s < 0.2 else (0, 165, 255) if s < 0.5 else (0, 0, 255)
        draw_text_bg(frame, f"INDICE DE LUBRICITE : {s:.1%}", (10, 30), 0.6, color, 2)
        bar_w = int(300 * min(1.0, s))
        cv2.rectangle(frame, (10, 40), (310, 52), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, 40), (10 + bar_w, 52), color, -1)
        if self.error:
            draw_text_bg(frame, self.error[:60], (10, 70), 0.4, (0, 0, 255), 1)


class YoloModule:
    """Vaches, pis, biberon-relique + biais inquisitorial."""

    def __init__(self, gui):
        self.model = None
        self.dets = []
        self.last = 0.0
        self.error = None
        self.active = YOLO_OK
        self.gui = gui

    def load(self):
        if self.model is None:
            self.gui.status("yolo", "CHARGEMENT...")
            try:
                m = YOLOWorld("yolov8s-worldv2")
                m.set_classes(["cow", "udder", "baby bottle"])
                self.model = m
                self.gui.status("yolo", "OK")
            except Exception as exc:
                self.model = False
                self.error = str(exc)
                self.gui.status("yolo", f"ERREUR : {exc}")

    def step(self, frame, now, bias):
        if not self.active:
            return
        self.load()
        if self.model is None or self.model is False:
            return
        if now - self.last < YOLO_INTERVAL:
            return
        self.last = now
        try:
            thresh = max(0.02, 0.5 * (1.0 - bias))
            res = self.model.predict(frame, conf=thresh, verbose=False)[0]
            dets = []
            for b in res.boxes:
                cls = res.names[int(b.cls)]
                conf = float(b.conf) * (1.0 + 0.6 * bias)
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                dets.append((cls, min(conf, 1.0), (x1, y1, x2 - x1, y2 - y1)))
            self.dets = dets
            self.error = None
        except Exception as exc:
            self.error = str(exc)
            self.gui.status("yolo", f"ERREUR : {exc}")

    def draw(self, frame, show_relique, milk):
        for cls, conf, (x, y, w, h) in self.dets:
            if cls == "cow":
                color, txt = (200, 150, 255), f"VACHE {conf:.0%}"
            elif cls == "udder":
                color, txt = (255, 220, 240), f"PIS {conf:.0%}"
            elif cls == "baby bottle" and show_relique:
                color, txt = (60, 215, 245), f"RELIQUE {conf:.0%}"
            else:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            draw_text_bg(frame, txt, (x, max(20, y - 8)), 0.5, color, 1)


class HandsModule:
    """Hallelujia : main gauche levee, 2 doigts (API solutions, vue miroir)."""

    def __init__(self, gui):
        self.detector = None
        self.last_hallelujah = 0.0
        self.error = None
        self.active = MP_OK
        self.gui = gui

    def load(self):
        if self.detector is None and self.active:
            try:
                self.detector = mp.solutions.hands.Hands(
                    max_num_hands=2, min_detection_confidence=0.6)
                self.gui.status("hands", "OK")
            except Exception as exc:
                self.detector = False
                self.error = str(exc)
                self.gui.status("hands", f"ERREUR : {exc}")

    def step(self, frame, now, on_hallelujah):
        if not self.active:
            return
        self.load()
        if not self.detector:
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.detector.process(rgb)
            if not res.multi_hand_landmarks:
                return
            for hand, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = handed.classification[0].label
                wrist_y = hand.landmark[0].y
                if label == "Right" and fingers_up(hand) == 2 and wrist_y < 0.45 \
                        and now - self.last_hallelujah > HALLELUJAH_COOLDOWN:
                    self.last_hallelujah = now
                    on_hallelujah()
                    return
        except Exception as exc:
            self.error = str(exc)
            self.gui.status("hands", f"ERREUR : {exc}")


class HaloModule:
    """Halo SANCTA autour des tetes (Haar cascade, aucune dependance)."""

    @staticmethod
    def draw(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(70, 70))
        for (x, y, w, h) in faces:
            cx = x + w // 2
            cy = max(14, y - int(0.18 * h))
            r = max(24, w // 2)
            cv2.ellipse(frame, (cx, cy), (r, int(r * 0.38)), 0, 0, 360, GOLD, 3)
            cv2.ellipse(frame, (cx, cy), (r + 6, int(r * 0.45)), 0, 0, 360, (10, 120, 170), 1)
            draw_text_bg(frame, "SANCTA", (max(4, cx - r), max(16, cy - r // 2 - 6)), 0.5, GOLD, 1)


class MilkModule:
    """Jauge lactee : +5 par hallelujia, croit a la vue d'un pis, s'evapore."""

    def __init__(self):
        self.level = 0.0

    def add(self, amount):
        self.level = min(100.0, self.level + amount)

    def step(self, yolo_dets):
        if any(cls == "udder" for cls, _, _ in yolo_dets):
            self.level = min(100.0, self.level + 0.02)

    def draw(self, frame):
        self.level = max(0.0, self.level - 0.01)
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = w - 90, h - 220, w - 40, h - 40
        cv2.rectangle(frame, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), GOLD, 2)
        fill = int((y1 - y0) * self.level / 100.0)
        cv2.rectangle(frame, (x0, y1 - fill), (x1, y1), (235, 250, 255), -1)
        for i in range(1, 4):
            yy = y0 + (y1 - y0) * i // 4
            cv2.line(frame, (x0, yy), (x0 + 10, yy), GOLD, 1)
        draw_text_bg(frame, f"LAIT {self.level:.0f}%", (x0 - 5, y0 - 14), 0.5, GOLD, 1)


# ------------------ application ------------------

class App:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        cv2.namedWindow("MAGNIFICAT", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MAGNIFICAT", 960, 540)
        self.features = {"nsfw": True, "vaches": True, "musique": True,
                         "biberon": True, "halo": True, "jauge": True}
        self.bias = 0.0
        self.running = False
        self.gui = Gui(self)
        self.nsfw = NsfwModule()
        self.yolo = YoloModule(self.gui)
        self.hands = HandsModule(self.gui)
        self.milk = MilkModule()

    def tick(self):
        ret, frame = self.cap.read()
        now = time.time()
        if not ret:
            self.quit()
            return
        if self.features["nsfw"]:
            self.nsfw.step(frame, now)
            self.nsfw.draw(frame)
        if self.features["vaches"] or self.features["biberon"]:
            self.yolo.step(frame, now, self.bias)
            self.yolo.draw(frame, self.features["biberon"], self.milk)
        if self.features["musique"]:
            self.hands.step(frame, now, lambda: (play_hallelujah(), self.milk.add(5)))
        if self.features["halo"]:
            HaloModule.draw(frame)
        if self.features["jauge"]:
            self.milk.step(self.yolo.dets)
            self.milk.draw(frame)
        # cadre baroque
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 34), (20, 12, 60), -1)
        cv2.rectangle(frame, (0, 0), (w, 34), GOLD, 1)
        draw_text_bg(frame, "+ MAGNIFICAT +  SANCTUAIRE DE SURVEILLANCE LACTEE  +",
                     (w // 2 - 285, 24), 0.55, GOLD, 1, bg=(20, 12, 60))
        cv2.rectangle(frame, (0, h - 3), (w, h), GOLD, 2)
        cv2.imshow("MAGNIFICAT", frame)
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

        etats = {
            "OpenNSFW  ": "OK" if NSFW_OK else "absent (pip install opennsfw2)",
            "YOLO-World": "OK" if YOLO_OK else "absent (pip install ultralytics)",
            "MediaPipe ": "OK" if MP_OK else "API incompatible (pip install mediapipe==0.10.14)",
            "Audio     ": "OK" if PYGAME_OK else "absent (bip systeme)",
        }
        self.status_labels = {}
        frame = tk.Frame(self.root, bg="#12041f", relief="sunken")
        frame.pack(fill=tk.X, padx=10, pady=6)
        for name, etat in etats.items():
            key = {"OpenNSFW  ": "nsfw", "YOLO-World": "yolo",
                   "MediaPipe ": "hands", "Audio     ": "audio"}[name]
            lbl = tk.Label(frame, text=f"{name}: {etat}", bg="#12041f",
                           fg="#00ff9d" if etat == "OK" else "#e8a0a0",
                           font=("Courier", 8), anchor="w", wraplength=420, justify=tk.LEFT)
            lbl.pack(fill=tk.X, padx=6, pady=1)
            self.status_labels[key] = lbl

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
                           command=lambda k=key: self._toggle(k, self.app)).pack(anchor="w", padx=24, pady=1)

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

    def _toggle(self, key, app):
        app.features[key] = self.vars[key].get()

    def status(self, key, txt):
        if key in self.status_labels:
            name = {"nsfw": "OpenNSFW  ", "yolo": "YOLO-World", "hands": "MediaPipe ",
                    "audio": "Audio     "}[key]
            self.status_labels[key].config(
                text=f"{name}: {txt}",
                fg="#00ff9d" if txt == "OK" else "#e8a0a0")


if __name__ == "__main__":
    app = App()
    app.run()
