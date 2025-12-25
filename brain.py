import os
import google.generativeai as genai
from dotenv import load_dotenv
import time


load_dotenv()

class Brain:
    def __init__(self):
        # Configuration de l'API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("🔴 ERREUR : Pas de GEMINI_API_KEY dans .env")
        
        genai.configure(api_key=api_key)

        # Modèle : On utilise Flash 1.5 (Rapide + Multimodal Audio)
        self.model_name = "gemini-2.5-flash"
        
        # Variables d'état
        self.chat = None
        self.system_instruction = "Tu es un assistant utile."
        
        # Initialisation par défaut
        self.init_model()

    def init_model(self):
        """Initialise ou Réinitialise le modèle avec l'instruction actuelle"""
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name, 
                system_instruction=self.system_instruction
            )
            # On démarre une nouvelle session de chat (historique vide)
            self.chat = self.model.start_chat(history=[])
            print(f"🧠 Cerveau prêt : {self.model_name}")
        except Exception as e:
            print(f"🔴 Erreur chargement modèle : {e}")

    def update_persona(self, name, scenario, behavior):
        """Met à jour l'identité de l'IA et reset la conversation"""
        self.system_instruction = f"""
        Tu incarnes {name}.
        SCÉNARIO : {scenario}
        COMPORTEMENT : {behavior}
        
        RÈGLES IMPORTANTES :
        - Tu es dans une conversation ORALE (Audio).
        - Tes réponses seront lues par une voix TTS.
        - Ne fais JAMAIS de listes à puces, de markdown (gras/italique) ou d'émojis complexes.
        - Fais des phrases courtes, naturelles et percutantes.
        - Si l'utilisateur hésite, encourage-le.
        """
        print(f"🧠 Persona mise à jour : {name}")
        # On recharge le modèle pour appliquer la nouvelle instruction système
        self.init_model()

    def think_from_audio(self, audio_path):
        """
        Reçoit un fichier audio, l'envoie à Gemini, ATTEND qu'il soit prêt,
        et retourne la réponse.
        """
        try:
            print(f"👂 Brain écoute le fichier : {audio_path}")
            
            # 1. Upload
            audio_file = genai.upload_file(path=audio_path)
            
            # 2. ATTENTE ACTIVE (C'est ça qui corrige ton erreur 400)
            print("⏳ Attente du traitement audio par Google...")
            while audio_file.state.name == "PROCESSING":
                time.sleep(1)
                audio_file = genai.get_file(audio_file.name)
            
            if audio_file.state.name != "ACTIVE":
                raise Exception(f"Fichier audio refusé par Google : {audio_file.state.name}")

            print("✅ Audio prêt. Envoi au chat...")

            # 3. Envoi dans le CHAT
            response = self.chat.send_message([audio_file])
            
            text_response = response.text
            print(f"🧠 Réponse générée : {text_response[:50]}...")
            
            return text_response

        except Exception as e:
            print(f"🔴 Erreur Brain Audio : {e}")
            return "Désolé, j'ai eu un problème technique avec l'audio."
    def think_streaming(self, user_text):
        """Fonction de secours pour le chat textuel classique"""
        if not self.chat: return
        try:
            response = self.chat.send_message(user_text, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            print(f"🔴 Erreur Chat Texte : {e}")
            yield "Erreur technique."

    def analyze_pitch(self, prompt_context):
        """
        Analyse le pitch (Appel unique, pas besoin de mémoire de chat ici).
        On utilise generate_content directement.
        """
        print(f"📊 Analyse Pitch en cours...")
        
        analysis_instruction = """
        Tu es un expert en Pitch de Startup (Type Y-Combinator).
        Analyse ce pitch. Sois SÉVÈRE mais JUSTE.
        
        Réponds UNIQUEMENT avec ce JSON strict (sans Markdown) :
        {
            "note": "Note sur 100 (ex: 65/100)",
            "accroche_probleme": "Analyse du WHY/Problème (1 phrase)",
            "solution_cible": "Analyse du Solution/Cible (1 phrase)",
            "unicite_business": "Analyse Différenciation/Business (1 phrase)",
            "cta_action": "Analyse du Call to Action (1 phrase)",
            "elements_manquants": "Liste les éléments cruciaux oubliés (ou 'Aucun' si complet)",
            "conseil": "TON meilleur conseil pour améliorer ce pitch"
        }
        """
        
        try:
            # On utilise le modèle sans historique pour une analyse one-shot
            # On peut réutiliser self.model ou en instancier un temporaire
            analysis_model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=analysis_instruction
            )
            
            response = analysis_model.generate_content(prompt_context)
            
            # Nettoyage du Markdown json si Gemini en met
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            return clean_json
            
        except Exception as e:
            print(f"🔴 CRASH ANALYSE : {e}")
            # JSON de secours pour éviter que le frontend plante
            return '{"note": "0/100", "accroche_probleme": "Erreur", "solution_cible": "Erreur", "unicite_business": "Erreur", "cta_action": "Erreur", "elements_manquants": "Erreur technique IA", "conseil": "Vérifiez la connexion API."}'