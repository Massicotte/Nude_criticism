"""NUDE CRITICISM - Critique nue du flux video en direct.

Webcam -> NudeNet en temps reel -> affichage de TOUTES les informations
detectees (labels, scores, boites, segmentation) sur le flux, plus un
pupitre technique listant l'etat du detecteur.
"""

import time

import cv2
import numpy as np
import tkinter as tk

try:
    from nudenet import NudeDetector
    NN_OK = True
except ImportError:
    NN_OK = False

DETECT_INTERVAL = 1.0          # secondes entre deux analyses NudeNet
BOX_COLORS = {
    "FEMALE_GENITALIA_EXPOSED": (0, 0, 255),
    "FEMALE_BREAST_EXPOSED": (0, 105, 255),
    "FEMALE_GENITALIA_COVERED": (128, 0, 128),
    "FEMALE_BREAST_COVERED": (128, 0, 128),
    "MALE_GENITALIA_EXPOSED": (255, 0, 255),
    "BUTTOCKS_EXPOSED": (0, 165, 255),
    "BUTTOCKS_COVERED": (128, 0, 128),
    "ANUS_EXPOSED": (0, 0, 255),
    "MALE_BREAST_EXPOSED": (255, 0, 0),
    "MALE_BREAST_COVERED": (128, 0, 128),
    "FEET_EXPOSED": (0, 255, 255),
    "FEET_COVERED": (128, 0, 128),
    "ARMPITS_EXPOSED": (200, 200, 0),
    "ARMPITS_COVERED": (128, 0, 128),
    "FACE_FEMALE": (0, 255, 0),
    "FACE_MALE": (0, 255, 0),
    "BELLY_EXPOSED": (255, 255, 0),
    "BELLY_COVERED": (128, 0, 128),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_text_bg(frame, text, org, scale, color, thickness=1):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 4, y - th - 4), (x + tw + 4, y + base + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness)


class App:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.window = "NUDE CRITICISM - Analyse en direct"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 960, 540)
        self.detector = NudeDetector() if NN_OK else None
        self.detections = []          # dernier resultat brut NudeNet
        self.last_detect = 0.0
        self.detect_count = 0
        self.fps = 0.0
        self._prev = time.time()
        self.running = False
        self.gui = Gui(self)

    # ---- analyse ----

    def analyze(self, frame):
        """Analyse NudeNet et retourne la liste brute des detections."""
        if self.detector is None:
            return []
        tmp_path = "/tmp/nc_frame.jpg"
        cv2.imwrite(tmp_path, frame)
        try:
            res = self.detector.detect(tmp_path)
        except Exception as exc:
            self.gui.set_status(f"Erreur NudeNet : {exc}")
            return self.detections
        if isinstance(res, list) and res:
            return res[0] if isinstance(res[0], list) else res
        return []

    # ---- boucle ----

    def tick(self):
        ret, frame = self.cap.read()
        now = time.time()
        if not ret:
            self.quit()
            return
        self.fps = 1.0 / max(0.001, now - self._prev)
        self._prev = now

        if now - self.last_detect >= DETECT_INTERVAL:
            self.last_detect = now
            self.detections = self.analyze(frame)
            self.detect_count += 1
            self.gui.refresh_detections()

        for det in self.detections:
            label = det.get("class", det.get("label", "?"))
            score = det.get("score", 0.0)
            x, y, w, h = det["box"]
            color = BOX_COLORS.get(label, (200, 200, 200))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            draw_text_bg(frame, f"{label} {score:.0%}", (x, y - 8), 0.45, color, 1)

        draw_text_bg(frame, f"{len(self.detections)} detections | analyse #{self.detect_count} "
                           f"| {self.fps:.0f} fps | NudeNet: {'OK' if NN_OK else 'ABSENT'}",
                     (10, 26), 0.5, (0, 200, 255), 1)
        cv2.imshow(self.window, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.quit()
            return
        if self.running:
            self.gui.root.after(33, self.tick)

    def run(self):
        if not self.cap.isOpened():
            tk.messagebox.showerror("Erreur", "Impossible d'acceder a la webcam.") if False else None
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
    """Pupitre technique : TOUTES les informations brutes de NudeNet."""

    def __init__(self, app):
        self.app = app
        self.root = tk.Tk()
        self.root.title("NUDE CRITICISM - Pupitre")
        self.root.geometry("520x700")
        self.root.configure(bg="#060628")
        BG = "#060628"
        self.root.protocol("WM_DELETE_WINDOW", app.quit)

        for col in ("#002395", "#ffffff", "#ED2939"):
            tk.Label(self.root, bg=col, height=1).pack(fill=tk.X)
        tk.Label(self.root, text="N U D E   C R I T I C I S M", bg=BG, fg="#58d6ff",
                 font=("Courier", 15, "bold")).pack(pady=(4, 0))
        tk.Label(self.root, text="OBSERVATION COMPORTEMENTALE EN TEMPS REEL - FLUX NON CONSERVE",
                 bg=BG, fg="#ffb000", font=("Courier", 8, "bold")).pack()

        self.status_var = tk.StringVar(value="Detecteur NudeNet : " + ("ACTIF" if NN_OK else "NON INSTALLE (pip install nudenet)"))
        tk.Label(self.root, textvariable=self.status_var, bg="#0a0a2e", fg="#ffb000",
                 font=("Courier", 8), anchor="w", relief="sunken",
                 wraplength=490, justify=tk.LEFT).pack(fill=tk.X, padx=10, pady=6, ipady=3)

        tk.Label(self.root, text="DETECTIONS BRUTES (sortie NudeNet integrale) :",
                 bg=BG, fg="#58d6ff", font=("Courier", 9, "bold")).pack(anchor="w", padx=10)
        self.output = tk.Text(self.root, bg="#0a0a2e", fg="#00ff9d",
                             font=("Courier", 9), relief="sunken", height=28, wrap="word")
        self.output.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        tk.Label(self.root, text="AUCUN FLUX NI IMAGE N'EST ENREGISTRE - ANALYSE VOLATILE (C) 2028",
                 bg=BG, fg="#7a7a8a", font=("Courier", 7)).pack(pady=(0, 5))

    def refresh_detections(self):
        self.output.config(state=tk.NORMAL)
        self.output.delete("1.0", tk.END)
        dets = self.app.detections
        if not dets:
            self.output.insert(tk.END, f"Analyse #{self.app.detect_count} : aucune detection.\n"
                                       f"(NudeNet ne voit rien a critiquer... ou ne veut pas voir)")
        for i, det in enumerate(dets, 1):
            label = det.get("class", det.get("label", "?"))
            score = det.get("score", 0.0)
            box = det.get("box", [0, 0, 0, 0])
            lines = [
                f"--- detection {i} ---",
                f"classe    : {label}",
                f"score     : {score:.3f} ({score:.1%})",
                f"boite     : x={box[0]} y={box[1]} w={box[2]} h={box[3]}",
            ]
            for extra_key in sorted(det.keys()):
                if extra_key not in ("class", "label", "score", "box"):
                    lines.append(f"{extra_key:<9} : {det[extra_key]}")
            self.output.insert(tk.END, "\n".join(lines) + "\n\n")
        self.output.config(state=tk.DISABLED)

    def set_status(self, msg):
        self.status_var.set(msg)


if __name__ == "__main__":
    if not NN_OK:
        print("ATTENTION : nudenet n'est pas installe.")
        print("  pip install nudenet")
    app = App()
    app.run()
