import os
import shutil
import json
import asyncio
from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response # <--- AJOUT CRUCIAL POUR RENVOYER L'AUDIO
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv

# Import Deepgram SDK
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

# Import des modules locaux
from brain import Brain
# On importe run_tts (nouvelle fonction) et les autres pour la compatibilité
from mouth import run_tts, stop_speaking, set_voice_gender, is_busy

load_dotenv()

app = FastAPI()

# --- CONFIGURATION CORS ---
# Permet à Vercel (et n'importe qui pour l'instant) de contacter ton API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

lucas_brain = Brain()

# Initialisation Deepgram (pour le STT analyse)
try:
    deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
except Exception as e:
    print(f"🔴 Erreur API Key Deepgram : {e}")

# --- MODELES ---
class UserInput(BaseModel):
    text: str

class ConfigInput(BaseModel):
    character: str
    gender: str
    scenario: str
    behavior: str

# --- FONCTION METRIQUES (Inchangée) ---
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
        fillers_list = ["euh", "hum", "ben", "bah", "mmh", "euh...", "bon"]
        for w in words:
            if hasattr(w, 'fillers') and w.fillers: filler_count += 1
            elif w.word.lower().replace('.', '').replace(',', '') in fillers_list: filler_count += 1
        long_pauses = 0
        for i in range(len(words) - 1):
            if (words[i+1].start - words[i].end) > 2.0: long_pauses += 1
        return {"wpm": wpm, "fillers": filler_count, "pauses": long_pauses, "transcript": transcript}
    except Exception: return {"wpm": 0, "fillers": 0, "transcript": "Erreur", "pauses": 0}

# --- ROUTES API ---

@app.post("/config")
async def config_endpoint(config: ConfigInput):
    """Change la voix et la personnalité"""
    set_voice_gender(config.gender)
    lucas_brain.update_persona(config.character, config.scenario, config.behavior)
    return {"status": "configured"}

@app.post("/chat")
async def chat_endpoint(input_data: UserInput):
    """
    NOUVELLE LOGIQUE /CHAT :
    1. Reçoit le texte.
    2. Génère tout le texte de réponse (Gemini).
    3. Génère l'audio (ElevenLabs).
    4. RENVOIE LE FICHIER AUDIO (Blob) au navigateur.
    """
    print(f"👤 User: {input_data.text}")
    
    # 1. On récupère la réponse complète de l'IA (Brain)
    full_response_text = ""
    for chunk in lucas_brain.think_streaming(input_data.text):
        full_response_text += chunk
    
    print(f"🤖 IA: {full_response_text}")

    # 2. On génère les octets audio (Mouth)
    # run_tts retourne des bytes (b'...')
    audio_bytes = run_tts(full_response_text)

    if not audio_bytes:
        return {"status": "error", "message": "Echec génération audio"}

    # 3. On renvoie le fichier audio directement via HTTP
    # Le frontend va recevoir ce fichier et le jouer avec new Audio(blob)
    return Response(content=audio_bytes, media_type="audio/mpeg")

@app.post("/stop")
async def stop_endpoint():
    # En Web, c'est le JS qui stop l'audio, mais on garde la route
    return {"status": "stopped"}

@app.get("/status")
async def status_endpoint():
    # Stateless
    return {"is_speaking": False}

@app.post("/analyze_audio")
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    """Analyse du Pitch (Deepgram Prerecorded + Gemini)"""
    print(f"🎤 Pitch reçu : {file.filename}")
    temp_filename = f"temp_{file.filename}"
    
    with open(temp_filename, "wb") as buffer: 
        shutil.copyfileobj(file.file, buffer) 
        
    try:
        with open(temp_filename, "rb") as audio:
            # Options Deepgram optimisées pour l'analyse
            options = {
                "model": "nova-2", "language": "fr", "smart_format": True, 
                "diarize": False, "filler_words": True, "punctuate": True
            }
            response = deepgram.listen.prerecorded.v("1").transcribe_file({"buffer": audio.read()}, options)
            
        metrics = calculate_metrics(response)
        
        prompt = f"TRANSCRIPTION: '{metrics['transcript']}'\nSTATS: Vitesse {metrics['wpm']} mots/min, {metrics['fillers']} hésitations."
        gemini_json = lucas_brain.analyze_pitch(prompt)
        
        return {"status": "ok", "metrics": metrics, "advice": gemini_json}
        
    except Exception as e: 
        print(f"🔴 ERREUR PITCH: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)

@app.websocket("/ws/listen")
async def listen_websocket(websocket: WebSocket):
    """Streaming Audio Chat (Deepgram Live)"""
    await websocket.accept()
    print("🟢 WebSocket Connecté")
    try:
        deepgram_live = deepgram.listen.live.v("1")
        
        def on_message(result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) > 0 and result.is_final:
                # Renvoie le texte au Frontend
                asyncio.run_coroutine_threadsafe(websocket.send_text(sentence), asyncio.get_event_loop())
                
        deepgram_live.on(LiveTranscriptionEvents.Transcript, on_message)
        
        options = LiveOptions(model="nova-2", language="fr", smart_format=True, interim_results=False, filler_words=False)
        
        if deepgram_live.start(options) is False: return
        
        while True:
            data = await websocket.receive_bytes()
            deepgram_live.send(data)
            
    except Exception as e: print(f"Websocket closed: {e}")
    finally:
        deepgram_live.finish()
        try: await websocket.close()
        except: pass

if __name__ == "__main__":
    # Ecoute sur 0.0.0.0 pour être accessible depuis l'extérieur de la VM
    uvicorn.run(app, host="0.0.0.0", port=8000)