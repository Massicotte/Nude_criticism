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
