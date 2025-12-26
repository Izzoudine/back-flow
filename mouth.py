import edge_tts
import asyncio

# --- CONFIGURATION DES VOIX (Microsoft Neural) ---
# Ces voix sont gratuites, de haute qualité et ne nécessitent pas de clé API.
VOICES = {
    "homme": "fr-FR-HenriNeural",   # Voix masculine fluide
    "femme": "fr-FR-VivienneNeural" # Voix féminine douce
}

# Variable globale pour stocker le genre actuel (par défaut : homme)
CURRENT_GENDER = "homme"

def set_voice_gender(gender):
    """
    Change le genre de la voix (Homme/Femme).
    """
    global CURRENT_GENDER
    g = gender.lower() if gender else "homme"
    
    # Détection souple (ex: "Male", "Garçon", "M")
    if "fem" in g or "woman" in g or "fille" in g:
        CURRENT_GENDER = "femme"
    else:
        CURRENT_GENDER = "homme"
        
    print(f"🔊 [MOUTH] Voix définie sur : {CURRENT_GENDER} ({VOICES[CURRENT_GENDER]})")

async def run_tts(text):
    """
    FONCTION PRINCIPALE EDGE TTS.
    Génère l'audio à partir du texte et renvoie les octets (bytes) MP3.
    
    ⚠️ IMPORTANT : Cette fonction est ASYNCHRONE (async).
    """
    if not text or not text.strip():
        return None

    try:
        # Sélection de la voix
        voice = VOICES.get(CURRENT_GENDER, VOICES["homme"])
        
        # Création de l'objet de communication
        communicate = edge_tts.Communicate(text, voice)
        
        audio_data = b""
        
        # On récupère les morceaux d'audio au fur et à mesure
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
                
        return audio_data

    except Exception as e:
        print(f"🔴 Erreur Edge TTS : {e}")
        return None

# =================================================================
# FONCTIONS DE COMPATIBILITÉ (POUR NE PAS CASSER LES IMPORTS)
# =================================================================

def stop_speaking():
    pass 

def is_busy():
    return False

def speak_streaming(text):
    pass