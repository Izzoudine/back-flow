import os
import shutil
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv

# Import Deepgram SDK (Uniquement pour l'analyse de pitch précise)
from deepgram import DeepgramClient

# Import de tes modules locaux
from brain import Brain
from mouth import run_tts, set_voice_gender, is_busy

load_dotenv()

app = FastAPI()

# --- CONFIGURATION CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation du Cerveau
lucas_brain = Brain()

# Initialisation Deepgram (Juste pour les stats du Pitch)
try:
    deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
except Exception as e:
    print(f"⚠️ Warning: Deepgram non configuré (Le Pitch Analyzer ne marchera pas, mais le Chat oui).")

# --- MODELES ---
class UserInput(BaseModel):
    text: str

class ConfigInput(BaseModel):
    character: str
    gender: str
    scenario: str
    behavior: str

# --- FONCTION UTILITAIRE (Metriques Pitch) ---
def calculate_metrics(dg_response):
    try:
        data = dg_response.results.channels[0].alternatives[0]
        transcript = data.transcript
        words = data.words 
        if not words: return {"wpm": 0, "fillers": 0, "transcript": "", "pauses": 0}
        
        duration_min = (words[-1].end - words[0].start) / 60
        if duration_min <= 0: duration_min = 0.1
        
        wpm = int(len(words) / duration_min)
        
        filler_count = 0
        fillers_list = ["euh", "hum", "ben", "bah", "bon", "alors"]
        for w in words:
            if w.word.lower().replace('.', '') in fillers_list: filler_count += 1
            
        long_pauses = 0
        for i in range(len(words) - 1):
            if (words[i+1].start - words[i].end) > 2.0: long_pauses += 1
            
        return {"wpm": wpm, "fillers": filler_count, "pauses": long_pauses, "transcript": transcript}
    except Exception: return {"wpm": 0, "fillers": 0, "transcript": "Erreur calcul", "pauses": 0}


# =================================================================
# 🔥 ROUTES API
# =================================================================

@app.post("/config")
async def config_endpoint(config: ConfigInput):
    """Configure la personnalité et la voix"""
    print(f"⚙️ Config: {config.character} ({config.gender})")
    set_voice_gender(config.gender)
    lucas_brain.update_persona(config.character, config.scenario, config.behavior)
    return {"status": "configured"}

@app.post("/chat_audio")
async def chat_audio_endpoint(file: UploadFile = File(...)):
    """
    NOUVELLE ROUTE MAGIQUE (Multimodale)
    1. Reçoit l'audio du user.
    2. L'envoie direct à Gemini.
    3. Reçoit la réponse texte.
    4. Convertit en audio TTS.
    5. Renvoie l'audio.
    """
    print(f"🎤 Audio reçu ({file.filename})...")
    
    temp_filename = f"temp_{file.filename}"
    
    try:
        # 1. Sauvegarde temporaire
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Cerveau (Audio -> Texte intelligent)
        response_text = lucas_brain.think_from_audio(temp_filename)
        
        # 3. Bouche (Texte -> Audio)
        audio_bytes = run_tts(response_text)

        if not audio_bytes:
            return {"status": "error", "message": "Erreur TTS"}

        # 4. Envoi
        return Response(content=audio_bytes, media_type="audio/mpeg")

    except Exception as e:
        print(f"🔴 Erreur Chat Audio: {e}")
        return {"status": "error", "message": str(e)}
    
    finally:
        # Nettoyage
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/chat")
async def chat_text_endpoint(input_data: UserInput):
    """Route de secours (Texte seul)"""
    print(f"👤 Texte reçu: {input_data.text}")
    
    # On utilise le mode streaming classique converti en texte complet
    full_response = ""
    for chunk in lucas_brain.think_streaming(input_data.text):
        full_response += chunk
    
    print(f"🤖 Réponse: {full_response}")
    audio_bytes = run_tts(full_response)
    return Response(content=audio_bytes, media_type="audio/mpeg")

@app.post("/analyze_audio")
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    """
    Analyse Pitch (Utilise Deepgram car on a besoin des timestamps précis pour les WPM)
    """
    print(f"📊 Analyse Pitch demandée...")
    temp_filename = f"analyze_{file.filename}"
    
    with open(temp_filename, "wb") as buffer: 
        shutil.copyfileobj(file.file, buffer) 
        
    try:
        with open(temp_filename, "rb") as audio:
            options = {
                "model": "nova-2", "language": "fr", "smart_format": True, 
                "filler_words": True, "punctuate": True
            }
            # Deepgram transcrit
            dg_response = deepgram.listen.prerecorded.v("1").transcribe_file({"buffer": audio.read()}, options)
            
        # On calcule les stats (WPM, Euh...)
        metrics = calculate_metrics(dg_response)
        
        # On demande à Gemini d'analyser la structure (QQOQCP)
        prompt = f"TRANSCRIPTION DU PITCH: '{metrics['transcript']}'"
        gemini_json = lucas_brain.analyze_pitch(prompt)
        
        return {"status": "ok", "metrics": metrics, "advice": gemini_json}
        
    except Exception as e: 
        print(f"🔴 Erreur Analyse: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)

@app.post("/stop")
async def stop_endpoint():
    return {"status": "stopped"}

@app.post("/status")
async def status_endpoint():
    return {"is_speaking": False}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)