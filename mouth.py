import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

# Chargement des variables d'environnement
load_dotenv()

# --- INITIALISATION CLIENT ELEVENLABS ---
try:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("⚠️ ATTENTION : ELEVENLABS_API_KEY introuvable dans le fichier .env")
        client = None
    else:
        client = ElevenLabs(api_key=api_key)
except Exception as e:
    print(f"🔴 Erreur d'initialisation ElevenLabs : {e}")
    client = None

# --- CONFIGURATION DES VOIX ---
VOICE_IDS = {
    "homme": "pNInz6obpgDQGcFmaJgB", # Adam
    "femme": "21m00Tcm4TlvDq8ikWAM"  # Rachel
}

# Variable globale pour stocker la voix actuelle (par défaut : homme)
current_voice_id = VOICE_IDS["homme"]

def set_voice_gender(gender):
    """
    Change l'ID de la voix utilisée.
    """
    global current_voice_id
    gender_key = gender.lower() if gender else "homme"
    
    if gender_key in VOICE_IDS:
        print(f"🔊 [MOUTH] Changement de voix vers : {gender_key}")
        current_voice_id = VOICE_IDS[gender_key]
    else:
        print(f"⚠️ [MOUTH] Genre inconnu '{gender}', la voix reste inchangée.")

def run_tts(text):
    """
    FONCTION PRINCIPALE POUR LE WEB.
    Génère l'audio avec ElevenLabs et renvoie les données brutes (bytes).
    Ne joue RIEN sur le serveur.
    """
    if not client:
        print("🔴 Erreur : Client ElevenLabs non initialisé.")
        return None
        
    if not text or not text.strip():
        return None

    try:
        # Appel API ElevenLabs
        # On utilise le modèle "turbo_v2_5" pour la rapidité
        audio_stream = client.text_to_speech.convert(
            voice_id=current_voice_id,
            model_id="eleven_turbo_v2_5", 
            text=text,
            output_format="mp3_44100_128",
            optimize_streaming_latency="4"
        )

        # L'API renvoie un générateur (flux). 
        # On doit le convertir en un seul bloc d'octets pour l'envoyer via HTTP.
        # C'est ce bloc que ton Javascript va lire.
        audio_bytes = b"".join(chunk for chunk in audio_stream if chunk)
        
        return audio_bytes

    except Exception as e:
        print(f"🔴 Erreur lors de la génération ElevenLabs : {e}")
        return None

# =================================================================
# FONCTIONS DE COMPATIBILITÉ (POUR EVITER LES ERREURS DANS SERVER.PY)
# =================================================================

def stop_speaking():
    """Inutile en mode Web API, le navigateur gère l'arrêt."""
    pass 

def is_busy():
    """Le serveur API est toujours disponible."""
    return False

def speak_streaming(text):
    """Ancienne fonction locale, désactivée."""
    pass