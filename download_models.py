#!/usr/bin/env python3
"""Télécharge uniquement le modèle VAD et le modèle Parakeet (desktop) en local."""

import json
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# Pointe directement vers le dossier actuel (parakeet-web)
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
MODELS_JSON = APP_DIR / "models.json"


def download_file(url: str, dest: Path):
    """Télécharge un fichier avec une barre de progression[cite: 2]."""
    resp = requests.get(url, stream=True, timeout=(15, 60))
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))


def main():
    if not MODELS_JSON.exists():
        print(f"❌ Erreur : Le fichier {MODELS_JSON} est introuvable.")
        print("Merci de copier models.json dans le dossier actuel.")
        sys.exit(1)

    with open(MODELS_JSON) as f:
        config = json.load(f)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Téléchargement du VAD[cite: 2]
    print("=== Vérification du modèle VAD ===")
    vad = config["vad"]
    dest_vad = MODELS_DIR / vad["filename"]
    if dest_vad.exists() and dest_vad.stat().st_size > 0:
        print(f"✅ {vad['filename']} est déjà présent.")
    else:
        print("Téléchargement du modèle VAD...")
        download_file(vad["url"], dest_vad)

    # 2. Téléchargement de Parakeet (profil "desktop")[cite: 2]
    profile_id = "desktop"
    profile = config["profiles"][profile_id]
    profile_dir = MODELS_DIR / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Vérification du modèle {profile['name']} ({profile['size_mb']} MB) ===")

    for key, info in profile["files"].items():
        dest = profile_dir / info["filename"]
        if dest.exists() and dest.stat().st_size > 0:
            print(f"✅ {info['filename']} est déjà présent.")
            continue
        print(f"Téléchargement de {info['filename']}...")
        download_file(info["url"], dest)

    print(f"\n🎉 Terminé. Le modèle est prêt dans : {profile_dir}")


if __name__ == "__main__":
    main()
