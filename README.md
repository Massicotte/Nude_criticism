# NUDE CRITICISM

Analyse en direct du flux webcam par **NudeNet** — affichage intégral de
toutes les informations que le détecteur peut extraire. Spectacle
technocritique : rendre visible le regard de la machine.

## Installation (PowerShell)

```powershell
git clone https://github.com/Massicotte/Nude_criticism.git
cd Nude_criticism
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python nude_criticism.py
```

> NudeNet télécharge son modèle (~100 Mo) au premier lancement.

## Utilisation

- Fenêtre vidéo : boîtes colorées + label + score sur chaque zone détectée
  (seins, fesses, genitalia, pieds, aisselles, visages, ventres… versions
  *exposed* et *covered*).
- Pupitre (2ᵉ fenêtre) : **sortie brute intégrale de NudeNet** — classe,
  score, boîte, et toute autre clé renvoyée par le détecteur.
- `Q` pour quitter.
- Analyse toutes les 1 s (réglable via `DETECT_INTERVAL`), affichage ~30 fps.

**Aucun flux ni image n'est enregistré.** L'analyse est volatile : la machine
critique, mais ne garde aucune trace.

## Avertissement

Outil de spectacle destiné à une réflexion technocritique sur la
classification automatique des corps. Les scores de NudeNet sont des
probabilités de modèle, pas des vérités.

---

# MAGNIFICAT — le sanctuaire de surveillance baroque

Deuxième programme du spectacle : `python magnificat.py`. Design église
catholique baroque (or, pourpre). Chaque miracle est activable/désactivable
depuis la **Sacristie de Contrôle** :

| Miracle | Effet |
|---|---|
| **Indice de lubricité (OpenNSFW)** | score OpenNSFW2 en direct + barre |
| **Vaches et pis sacrés (YOLO)** | détection « cow » et « udder », signalée à l'écran |
| **Hallelujia** | musique sacrée quand la **main gauche** est levée avec **2 doigts** (cooldown 6 s) |
| **Relique (biberon)** | « baby bottle » détecté → carré doré marqué « RELIQUE » |
| **Halo de sainteté** | auréole dorée au-dessus de chaque visage détecté |
| **Jauge laitière** | jauge de lait à droite ; +5% par hallelujia, croît à la vue d'un pis |

**Biais inquisitorial** : le slider de la Sacristie (0 → 100) fait croire au
modèle YOLO de plus en plus qu'il voit des vaches — le seuil de détection
baisse et la confiance affichée est gonflée. À 100, la foi remplace la
perception. (C'est le propos technocritique : le réglage du seuil d'un
modèle change ce qu'il « voit ».)

## Dépendances MAGNIFICAT

```powershell
pip install opennsfw2 ultralytics mediapipe pygame
```

Chaque module est optionnel : absent → la fonctionnalité reste désactivable,
l'app tourne quand même (avec l'accord synthétisé si pygame manque).
`Q` pour quitter.
