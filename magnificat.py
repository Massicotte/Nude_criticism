"""MAGNIFICAT - Sanctuaire de surveillance baroque (spectacle lacte/saint Francois).

Fonctionnalites (chacune activable/desactivable depuis le pupitre) :
- OpenNSFW : indice de lubricite en direct
- YOLO-World : detection de vaches, pis et biberons (relique) + slider de biais
- MediaPipe : hallelujah quand la main gauche est levee avec 2 doigts
- Halo dore autour de la tete des comediens
- Jauge laitiere

Design : eglise catholique baroque (or, pourpre, ornaments).
"""

import time

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

try:
    import opennsfw2
    NSFW_OK = True
except ImportError:
    opennsfw2, NSFW_OK = None, False

try:
    from ultralytics import YOLOWorld
    YOLO_OK = True
except ImportError:
    YOLOWorld, YOLO_OK = None, False

try:
    import mediapipe as mp
    MP_OK = True
except ImportError:
    mp, MP_OK = None, False

try:
    import pygame
    pygame.mixer.init()
    PYGAME_OK = True
except Exception:
    pygame, PYGAME_OK = None, False

GOLD = (30, 175, 235)          # BGR or
PURPLE = (150, 20, 90)
FONT = cv2.FONT_HERSHEY_SIMPLEX
NSFW_INTERVAL = 2.0
YOLO_INTERVAL = 1.5
HALLELUJAH_COOLDOWN = 6.0
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def draw_text_bg(frame, text, org, scale, color, thickness=1, bg=(0, 0, 0)):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 3, y - th - 3), (x + tw + 3, y + base + 3), bg, -1)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness)


def synth_hallelujah():
    """Accord majeur synthetise (fallback sans fichier mp3)."""
    if not PYGAME_OK:
        return
    sr = 44100
    notes = [523.25, 659.25, 783.99, 1046.5]  # Do majeur ascendant (alleluia)
    t = np.linspace(0, 0.9, int(sr * 0.9), endpoint=False)
    wave = sum(0.22 * np.sin(2 * np.pi * f * t) * np.exp(-2.2 * t) for f in notes)
    wave = (wave * 32767 * 0.8).astype(np.int16)
    stereo = np.column_stack([wave, wave])
    pygame.sndarray.make_sound(stereo).play()


def play_hallelujah():
    if not PYGAME_OK:
        print("\a HALLELUJIA !")
        return
    try:
        import os
        if os.path.exists("assets/hallelujah.mp3"):
            pygame.mixer.music.load("assets/hallelujah.mp3")
            pygame.mixer.music.play()
            return
    except Exception:
        pass
    synth_hallelujah()


def fingers_up(hand_landmarks, mp_hands):
    """Nombre de doigts tendus (pouce exclu si serre)."""
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]
    count = 0
    for tip, pip in zip(tips, pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y - 0.02:
            count += 1
    return count


class App:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.window = "MAGNIFICAT - Sanctuaire de Surveillance"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 960, 540)

        self.features = {
            "nsfw": True, "vaches": True, "musique": True,
            "biberon": True, "halo": True, "jauge": True,
        }
        self.nsfw_score = 0.0
        self.nsfw_error = None
        self.last_nsfw = 0.0
        self.yolo = None
        self.yolo_dets = []
        self.last_yolo = 0.0
        self.bias = 0.0  # 0..1 : biais inquisitorial
        self.hands = None
        self.last_hallelujah = 0.0
        self.milk = 0.0
        self.running = False
        self.gui = Gui(self)

    # ---- modeles paresseux ----

    def get_yolo(self):
        if self.yolo is None and YOLO_OK:
            self.gui.set_module_status("yolo", "CHARGEMENT du modele (telechargement possible)...")
            try:
                model = YOLOWorld("yolov8s-worldv2")
                model.set_classes(["cow", "udder", "baby bottle"])
                self.yolo = model
                self.gui.set_module_status("yolo", "OK")
            except Exception as exc:
                self.yolo = False
                self.gui.set_module_status("yolo", f"ERREUR : {exc}")
        return self.yolo or None

    def get_hands(self):
        if self.hands is None and MP_OK:
            try:
                self.hands = mp.solutions.hands.Hands(
                    max_num_hands=2, min_detection_confidence=0.6)
                self.gui.set_module_status("hands", "OK")
            except Exception as exc:
                self.hands = False
                self.gui.set_module_status("hands", f"ERREUR : {exc}")
        return self.hands or None

    # ---- features ----

    def run_nsfw(self, frame, now):
        if now - self.last_nsfw < NSFW_INTERVAL:
            return
        self.last_nsfw = now
        if not NSFW_OK:
            return
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.nsfw_score = float(opennsfw2.predict_image(img))
            self.nsfw_error = None
        except Exception as exc:
            self.nsfw_error = str(exc)
            self.nsfw_score = 0.0
            self.gui.set_module_status("nsfw", f"ERREUR : {exc}")

    def run_yolo(self, frame, now):
        if now - self.last_yolo < YOLO_INTERVAL:
            return
        self.last_yolo = now
        model = self.get_yolo()
        if model is None:
            return
        thresh = max(0.02, 0.5 * (1.0 - self.bias))
        try:
            res = model.predict(frame, conf=thresh, verbose=False)[0]
            self.yolo_dets = []
            for b in res.boxes:
                cls = res.names[int(b.cls)]
                conf = float(b.conf) * (1.0 + 0.6 * self.bias)  # le biais gonfle la certitude
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                self.yolo_dets.append((cls, min(conf, 1.0), (x1, y1, x2 - x1, y2 - y1)))
        except Exception:
            pass

    def run_hands(self, frame, now):
        hands = self.get_hands()
        if hands is None:
            return False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            return False
        triggered = False
        for hand, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
            label = handed.classification[0].label  # "Left"/"Right" (vue miroir)
            wrist_y = hand.landmark[0].y
            n = fingers_up(hand, mp.solutions.hands)
            if label == "Left" and n == 2 and wrist_y < 0.45 and now - self.last_hallelujah > HALLELUJAH_COOLDOWN:
                self.last_hallelujah = now
                triggered = True
                if self.features["musique"]:
                    play_hallelujah()
                    self.milk += 5
        return triggered

    def draw_halo(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(70, 70))
        for (x, y, w, h) in faces:
            cx = x + w // 2
            cy = y - int(0.18 * h)
            r = max(24, w // 2)
            overlay = frame.copy()
            cv2.ellipse(overlay, (cx, cy), (r, int(r * 0.38)), 0, 0, 360, GOLD, 3)
            cv2.ellipse(overlay, (cx, cy), (r + 6, int(r * 0.45)), 0, 0, 360, (10, 120, 170), 1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            draw_text_bg(frame, "SANCTA", (cx - r, cy - int(r * 0.5) - 6), 0.5, GOLD, 1)

    def draw_yolo(self, frame):
        for cls, conf, (x, y, w, h) in self.yolo_dets:
            if cls == "cow":
                color = (200, 150, 255)
                txt = f"VACHE {conf:.0%}"
            elif cls == "udder":
                color = (255, 220, 240)
                txt = f"PIS {conf:.0%}"
                if self.features["jauge"]:
                    self.milk += 0.02
            elif cls == "baby bottle" and self.features["biberon"]:
                color = (60, 215, 245)
                txt = f"RELIQUE {conf:.0%}"
            else:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            draw_text_bg(frame, txt, (x, y - 8), 0.5, color, 1)

    def draw_nsfw(self, frame):
        s = self.nsfw_score
        color = GOLD if s < 0.2 else (0, 165, 255) if s < 0.5 else (0, 0, 255)
        draw_text_bg(frame, f"INDICE DE LUBRICITE : {s:.1%}", (10, 30), 0.6, color, 2)
        bar_w = int(300 * min(1.0, s))
        cv2.rectangle(frame, (10, 40), (310, 52), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, 40), (10 + bar_w, 52), color, -1)

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

    def draw_baroque_frame(self, frame):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 34), (20, 12, 60), -1)
        cv2.rectangle(frame, (0, 0), (w, 34), GOLD, 1)
        draw_text_bg(frame, "\u271d MAGNIFICAT \u271d  SANCTUAIRE DE SURVEILLANCE LACTEE  \u271d",
                     (w // 2 - 300, 24), 0.55, GOLD, 1, bg=(20, 12, 60))
        cv2.rectangle(frame, (0, h - 3), (w, h), GOLD, 2)

    # ---- boucle ----

    def tick(self):
        ret, frame = self.cap.read()
        now = time.time()
        if not ret:
            self.quit()
            return
        if self.features["nsfw"] and NSFW_OK:
            self.run_nsfw(frame, now)
            self.draw_nsfw(frame)
        if self.features["vaches"] or self.features["biberon"]:
            self.run_yolo(frame, now)
            self.draw_yolo(frame)
        if self.features["musique"] and MP_OK:
            self.run_hands(frame, now)
        if self.features["halo"]:
            self.draw_halo(frame)
        if self.features["jauge"]:
            self.draw_milk(frame)
        self.draw_baroque_frame(frame)
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
    """Pupitre baroque : un interrupteur par miracle."""

    def __init__(self, app):
        self.app = app
        self.root = tk.Tk()
        self.root.title("MAGNIFICAT - Sacristie de Controle")
        self.root.geometry("460x620")
        self.root.configure(bg="#1a0a2e")
        self.root.protocol("WM_DELETE_WINDOW", app.quit)
        BG = "#1a0a2e"

        for col in ("#d4af37", "#5c0a1e", "#d4af37"):
            tk.Label(self.root, bg=col, height=1).pack(fill=tk.X)
        tk.Label(self.root, text="\u271d  M A G N I F I C A T  \u271d", bg=BG, fg="#d4af37",
                 font=("Times", 17, "bold")).pack(pady=(5, 0))
        tk.Label(self.root, text="OFFICE DE SURVEILLANCE - ARCHIDIOCESE TECHNOLOGIQUE",
                 bg=BG, fg="#c09090", font=("Times", 9)).pack()

        self.module_status = {
            "nsfw": "OK" if NSFW_OK else "absent (pip install opennsfw2)",
            "yolo": "OK" if YOLO_OK else "absent (pip install ultralytics)",
            "hands": "OK" if MP_OK else "absent (pip install mediapipe)",
            "audio": "OK" if PYGAME_OK else "absent (bip systeme)",
        }
        self.status_frame = tk.Frame(self.root, bg="#12041f", relief="sunken")
        self.status_frame.pack(fill=tk.X, padx=10, pady=6)
        self.status_labels = {}
        for key, txt in self.module_status.items():
            name = {"nsfw": "OpenNSFW", "yolo": "YOLO-World", "hands": "MediaPipe", "audio": "Audio"}[key]
            lbl = tk.Label(self.status_frame, text=f"{name:<11}: {txt}", bg="#12041f",
                           fg="#00ff9d" if txt == "OK" else "#e8a0a0",
                           font=("Courier", 8), anchor="w", justify=tk.LEFT, wraplength=420)
            lbl.pack(fill=tk.X, padx=6, pady=1)
            self.status_labels[key] = lbl

        tk.Label(self.root, text="MIRACLES ACTIVABLES :", bg=BG, fg="#d4af37",
                 font=("Times", 11, "bold")).pack(anchor="w", padx=12)

        self.vars = {}
        labels = [
            ("nsfw", "Indice de lubricite (OpenNSFW)"),
            ("vaches", "Vaches et pis sacres (YOLO)"),
            ("musique", "Hallelujia - main gauche, 2 doigts"),
            ("biberon", "Relique (biberon)"),
            ("halo", "Halo de saintete"),
            ("jauge", "Jauge lactee"),
        ]
        for key, label in labels:
            var = tk.BooleanVar(value=app.features[key])
            self.vars[key] = var
            tk.Checkbutton(self.root, text=label, variable=var, bg=BG, fg="#f0e6d0",
                           activebackground="#2a0a3e", activeforeground="#d4af37",
                           selectcolor="#12041f", font=("Times", 11),
                           command=lambda k=key: self.toggle(k)).pack(anchor="w", padx=24, pady=1)

        tk.Label(self.root, text="BIAIS INQUISITORIAL DU MODELE VACHES :",
                 bg=BG, fg="#d4af37", font=("Times", 10, "bold")).pack(anchor="w", padx=12, pady=(8, 0))
        self.bias_var = tk.DoubleVar(value=0.0)
        self.bias_scale = tk.Scale(
            self.root, variable=self.bias_var, from_=0, to=100, orient=tk.HORIZONTAL,
            bg=BG, fg="#d4af37", troughcolor="#12041f", highlightthickness=0,
            font=("Times", 9), length=380,
            command=lambda v: setattr(app, "bias", float(v) / 100.0))
        self.bias_scale.pack(padx=16)
        tk.Label(self.root, text="0 : le modele ne voit que la verite.\n"
                                 "100 : le modele voit des vaches partout (foi renforcee).",
                 bg=BG, fg="#c09090", font=("Times", 8)).pack()

        tk.Label(self.root, text="GLOIRE AU LAIT ETERNEL - (C) 2028 ARCHIDIOCESE",
                 bg=BG, fg="#8a6a3a", font=("Times", 8)).pack(side=tk.BOTTOM, pady=6)

    def toggle(self, key):
        self.app.features[key] = self.vars[key].get()

    def set_module_status(self, key, txt):
        self.module_status[key] = txt
        if key in self.status_labels:
            self.status_labels[key].config(
                text=f"{ {'nsfw': 'OpenNSFW', 'yolo': 'YOLO-World', 'hands': 'MediaPipe', 'audio': 'Audio'}[key]:<11}: {txt}",
                fg="#00ff9d" if txt == "OK" else "#e8a0a0")


if __name__ == "__main__":
    app = App()
    app.run()
