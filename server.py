import os
import shutil
import uuid
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv

# Import Deepgram SDK (C'est notre moteur "Turbo" pour l'oreille)
from deepgram import DeepgramClient, PrerecordedOptions

# Tes modules locaux
from brain import Brain
from mouth import run_tts, set_voice_gender

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

# --- INITIALISATION ---
lucas_brain = Brain()

# Deepgram est maintenant CRITIQUE pour la vitesse
try:
    deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
except Exception as e:
    print(f"🔴 ERREUR CRITIQUE: Deepgram API Key manquante. Le système ne marchera pas.")

# --- MODELES ---
class ConfigInput(BaseModel):
    character: str
    gender: str
    scenario: str
    behavior: str

# --- FONCTION UTILITAIRE (Pour le mode Pitch uniquement) ---
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
        fillers_list = ["euh", "hum", "ben", "bah", "bon"]
        for w in words:
            if w.word.lower().replace('.', '') in fillers_list: filler_count += 1
            
        long_pauses = 0
        for i in range(len(words) - 1):
            if (words[i+1].start - words[i].end) > 2.0: long_pauses += 1
            
        return {"wpm": wpm, "fillers": filler_count, "pauses": long_pauses, "transcript": transcript}
    except Exception: return {"wpm": 0, "fillers": 0, "transcript": "", "pauses": 0}


# =================================================================
# 🔥 ROUTES API
# =================================================================

@app.post("/config")
async def config_endpoint(config: ConfigInput):
    print(f"⚙️ Config: {config.character}")
    set_voice_gender(config.gender)
    lucas_brain.update_persona(config.character, config.scenario, config.behavior)
    return {"status": "configured"}

# --- LA ROUTE QUI CHANGE TOUT (Version Turbo) ---
@app.post("/chat_audio")
async def chat_audio_endpoint(audio_file: UploadFile = File(...)):
    """
    Pipeline Ultra-Rapide :
    1. Audio -> Deepgram (STT) : 0.3s
    2. Texte -> Gemini (LLM) : 0.5s
    3. Texte -> TTS (Audio) : 1.0s
    Total : < 2.5s (hors réseau)
    """
    unique_id = str(uuid.uuid4())[:8]
    temp_filename = f"temp_chat_{unique_id}.webm"
    start_time = time.time() # Chrono pour mesurer la vitesse

    try:
        # 1. Sauvegarde (Obligatoire pour Deepgram File)
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
            
        # Sécurité fichier vide
        if os.path.getsize(temp_filename) < 1000:
            return JSONResponse(content={"error": "Audio vide"}, status_code=400)

        # 2. Transciption via Deepgram (C'est ici qu'on gagne 3 secondes !)
        with open(temp_filename, "rb") as audio:
            # Smart formatting met la ponctuation, essentiel pour que Gemini comprenne le sens
            options = {"model": "nova-2", "language": "fr", "smart_format": True}
            response = deepgram.listen.prerecorded.v("1").transcribe_file({"buffer": audio.read()}, options)
        
        user_text = response.results.channels[0].alternatives[0].transcript
        
        t_stt = time.time()
        print(f"⚡ [1] Transcription ({t_stt - start_time:.2f}s) : {user_text}")

        if not user_text.strip():
            return JSONResponse(content={"error": "Silence"}, status_code=400)

        # 3. Cerveau (Gemini reçoit du TEXTE maintenant)
        # Il faut utiliser la méthode 'think_text_only' qu'on a ajoutée dans brain.py
        ai_response_text = lucas_brain.think_text_only(user_text)
        
        t_llm = time.time()
        print(f"🧠 [2] Gemini ({t_llm - t_stt:.2f}s) : {ai_response_text[:50]}...")

        # 4. Bouche (TTS)
        audio_bytes = run_tts(ai_response_text)
        
        t_tts = time.time()
        print(f"🔊 [3] TTS ({t_tts - t_llm:.2f}s)")
        print(f"🚀 TOTAL SERVER: {t_tts - start_time:.2f}s")

        return Response(content=audio_bytes, media_type="audio/mpeg")

    except Exception as e:
        print(f"🔴 Erreur: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)

@app.post("/analyze_audio")
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    # ... (Code inchangé pour l'analyse Pitch) ...
    unique_id = str(uuid.uuid4())[:8]
    temp_filename = f"temp_pitch_{unique_id}.webm"
    try:
        with open(temp_filename, "wb") as buffer: shutil.copyfileobj(file.file, buffer) 
        with open(temp_filename, "rb") as audio:
            options = {"model": "nova-2", "language": "fr", "smart_format": True, "filler_words": True, "punctuate": True}
            dg_response = deepgram.listen.prerecorded.v("1").transcribe_file({"buffer": audio.read()}, options)
        metrics = calculate_metrics(dg_response)
        gemini_json = lucas_brain.analyze_pitch(f"Transcription: {metrics['transcript']}")
        return {"status": "ok", "metrics": metrics, "advice": gemini_json}
    except Exception as e: return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)

@app.post("/stop")
async def stop_endpoint():
    lucas_brain.clear_history()
    return {"status": "stopped"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)