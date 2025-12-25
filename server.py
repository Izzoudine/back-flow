import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv

# Import Deepgram SDK (Indispensable pour calculer les WPM et les pauses)
from deepgram import DeepgramClient

# Tes modules locaux
from brain import Brain
from mouth import run_tts, set_voice_gender

load_dotenv()

app = FastAPI()

# --- CONFIGURATION CORS (Pour autoriser le frontend) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INITIALISATION ---
lucas_brain = Brain()

# Deepgram est utilisé uniquement pour les STATISTIQUES du Pitch (WPM, Pauses)
# Car Gemini ne donne pas encore les timestamps précis des mots.
try:
    deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
except Exception as e:
    print(f"⚠️ Warning: Deepgram non configuré. L'analyse WPM ne marchera pas.")

# --- MODELES DE DONNÉES ---
class ConfigInput(BaseModel):
    character: str
    gender: str
    scenario: str
    behavior: str

# --- FONCTION UTILITAIRE (Métriques Pitch) ---
def calculate_metrics(dg_response):
    try:
        data = dg_response.results.channels[0].alternatives[0]
        transcript = data.transcript
        words = data.words 
        if not words: return {"wpm": 0, "fillers": 0, "transcript": "", "pauses": 0}
        
        # Durée en minutes
        duration_min = (words[-1].end - words[0].start) / 60
        if duration_min <= 0: duration_min = 0.1
        
        # Calcul WPM
        wpm = int(len(words) / duration_min)
        
        # Compteur de "Euh"
        filler_count = 0
        fillers_list = ["euh", "hum", "ben", "bah", "bon", "alors", "genre"]
        for w in words:
            if w.word.lower().replace('.', '') in fillers_list: filler_count += 1
            
        # Compteur de Pauses > 2 secondes
        long_pauses = 0
        for i in range(len(words) - 1):
            if (words[i+1].start - words[i].end) > 2.0: long_pauses += 1
            
        return {"wpm": wpm, "fillers": filler_count, "pauses": long_pauses, "transcript": transcript}
    except Exception as e:
        print(f"Erreur Metrics: {e}")
        return {"wpm": 0, "fillers": 0, "transcript": "Erreur calcul", "pauses": 0}


# =================================================================
# 🔥 ROUTES API
# =================================================================

@app.post("/config")
async def config_endpoint(config: ConfigInput):
    """Configure la personnalité de l'IA et le genre de la voix TTS"""
    print(f"⚙️ Config: {config.character} ({config.gender}) - {config.behavior}")
    set_voice_gender(config.gender)
    lucas_brain.update_persona(config.character, config.scenario, config.behavior)
    return {"status": "configured"}

@app.post("/chat_audio")
async def chat_audio_endpoint(audio_file: UploadFile = File(...)):
    """
    1. Reçoit l'audio (WebM) du navigateur.
    2. L'envoie à Gemini Flash (Multimodal).
    3. Reçoit la réponse TEXTE.
    4. Convertit le texte en AUDIO (TTS).
    5. Renvoie l'audio au navigateur.
    """
    unique_id = str(uuid.uuid4())[:8]
    temp_filename = f"temp_chat_{unique_id}.webm"
    print(f"🎤 [Chat] Audio reçu ({audio_file.filename})")

    try:
        # 1. Sauvegarde Temporaire
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        # 2. Cerveau (Gemini écoute l'audio)
        # ATTENTION: Il faut que ton brain.py ait la méthode think_from_audio
        response_text = lucas_brain.think_from_audio(temp_filename)
        print(f"🧠 [Chat] Réponse Gemini: {response_text}")

        # 3. Bouche (TTS)
        audio_bytes = run_tts(response_text)

        if not audio_bytes:
            raise Exception("Erreur génération audio TTS")

        # 4. Envoi
        return Response(content=audio_bytes, media_type="audio/mpeg")

    except Exception as e:
        print(f"🔴 Erreur Chat Audio: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    finally:
        # Nettoyage
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/analyze_audio")
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    """
    Analyse Pitch Hybride :
    - Deepgram pour les métriques précises (WPM, Pauses).
    - Gemini pour le coaching qualitatif (Structure, Persuasion).
    """
    unique_id = str(uuid.uuid4())[:8]
    temp_filename = f"temp_pitch_{unique_id}.webm"
    print(f"📊 [Pitch] Analyse demandée...")

    try:
        # Sauvegarde
        with open(temp_filename, "wb") as buffer: 
            shutil.copyfileobj(file.file, buffer) 
            
        # 1. Analyse Quantitative (Deepgram)
        with open(temp_filename, "rb") as audio:
            options = {
                "model": "nova-2", 
                "language": "fr", 
                "smart_format": True, 
                "filler_words": True, 
                "punctuate": True
            }
            dg_response = deepgram.listen.prerecorded.v("1").transcribe_file({"buffer": audio.read()}, options)
            
        metrics = calculate_metrics(dg_response)
        print(f"   -> WPM: {metrics['wpm']}, Euh: {metrics['fillers']}")
        
        # 2. Analyse Qualitative (Gemini)
        # On envoie la transcription à Gemini pour l'analyse QQOQCP
        prompt_analysis = f"""
        Voici la transcription d'un pitch d'entrainement : 
        "{metrics['transcript']}"
        
        Analyse-le selon la méthode QQOQCP et donne-moi un JSON strict.
        """
        gemini_json = lucas_brain.analyze_pitch(prompt_analysis)
        
        return {"status": "ok", "metrics": metrics, "advice": gemini_json}
        
    except Exception as e: 
        print(f"🔴 Erreur Analyse: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)

@app.post("/stop")
async def stop_endpoint():
    """Permet de reset l'historique ou arrêter les processus si besoin"""
    lucas_brain.clear_history() # Optionnel si tu veux reset le contexte
    return {"status": "stopped"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)