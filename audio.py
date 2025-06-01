import librosa
import numpy as np
import os
import subprocess
import tempfile

def extract_audio(video_path):
    """Extract audio from video and detect siren-like sounds."""
    try:
        # Créer un fichier temporaire avec un nom unique
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            audio_path = temp_audio.name

        # Commande ffmpeg avec option -y pour forcer l'écrasement
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # Pas de vidéo
            '-acodec', 'pcm_s16le',  # Codec audio
            '-ar', '44100',  # Fréquence d'échantillonnage
            '-ac', '1',  # Mono
            '-y',  # Forcer l'écrasement
            audio_path
        ]
        
        # Exécuter la commande et capturer les erreurs
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        if result.returncode != 0:
            print(f"Erreur lors de l'extraction audio : {result.stderr}")
            os.remove(audio_path) if os.path.exists(audio_path) else None
            return False

        # Charger l'audio
        y, sr = librosa.load(audio_path, sr=44100)
        os.remove(audio_path)  # Nettoyer immédiatement après chargement

        # Extraire les caractéristiques MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Détection simple de sirène (basée sur le centroïde spectral)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        siren_present = np.mean(spectral_centroid) > 500 and np.mean(spectral_centroid) < 1500
        
        return siren_present
    except Exception as e:
        print(f"Erreur extracting audio : {e}")
        return False